"""Workflow orchestration utilities for the AI answer evaluation system.

The goal of this module is to keep app.py slim while centralising
stateful operations (question/answer ingestion, OCR, evaluation, PDF
creation).  We model the pipeline with dataclasses so we can reason
about each artefact explicitly and use deterministic, efficient data
structures for lookups.
"""

from __future__ import annotations

import hashlib
import os
import re
from bisect import insort
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Question:
    number: int
    text: str
    max_marks: float


@dataclass(frozen=True)
class Answer:
    number: int
    text: str


@dataclass(frozen=True)
class StudentAnswer:
    number: int
    text: str


class WorkflowError(Exception):
    """Domain specific exception that carries an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SortedLookup:
    """Maintain objects keyed by their question number in sorted order.

    Internally we keep both a dict for O(1) lookups and a sorted list for
    deterministic iteration.  This gives us O(log n) insertion because we
    only binary-search the index when a new key arrives.
    """

    def __init__(self):
        self._order: List[int] = []
        self._items: Dict[int, object] = {}
        self._lock = RLock()

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._order)

    def clear(self):
        with self._lock:
            self._order.clear()
            self._items.clear()

    def upsert(self, number: int, payload: object):
        with self._lock:
            if number not in self._items:
                insort(self._order, number)
            self._items[number] = payload

    def bulk_load(self, payloads: Iterable[object]):
        with self._lock:
            self._order.clear()
            self._items.clear()
            for payload in payloads:
                number = getattr(payload, "number", None)
                if number is None:
                    raise ValueError("Payload missing 'number' attribute")
                if number not in self._items:
                    insort(self._order, number)
                self._items[number] = payload

    def get(self, number: int) -> Optional[object]:
        with self._lock:
            return self._items.get(number)

    def as_list(self) -> List[object]:
        with self._lock:
            return [self._items[num] for num in self._order]


class BoundedLRUCache:
    """Simple OrderedDict-based LRU cache for OCR outputs."""

    def __init__(self, maxsize: int = 2):
        self._maxsize = maxsize
        self._store: OrderedDict[str, object] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Optional[object]:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: str, value: object):
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()


class EvaluationWorkflow:
    """High-level orchestration for the evaluation pipeline."""

    def __init__(
        self,
        *,
        pdf_processor,
        evaluator,
        result_pdf_generator,
        ocr_pdf_converter,
        upload_folder: str,
        result_folder: str,
    ):
        self.pdf_processor = pdf_processor
        self.evaluator = evaluator
        self.result_pdf_generator = result_pdf_generator
        self.ocr_pdf_converter = ocr_pdf_converter
        self.upload_folder = upload_folder
        self.result_folder = result_folder

        self._questions = SortedLookup()
        self._answers = SortedLookup()
        self._ocr_cache = BoundedLRUCache(maxsize=3)
        self._student_answer_cache = BoundedLRUCache(maxsize=3)
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_question_paper(self, filepath: str) -> List[Question]:
        questions_raw = self.pdf_processor.extract_questions_with_marks(filepath)
        if not questions_raw:
            raise WorkflowError(
                "No questions found in PDF. Please ensure numbering and marks are visible.",
                status_code=400,
            )

        questions = [
            Question(
                number=int(q["question_number"]),
                text=q["question_text"],
                max_marks=float(q["max_marks"]),
            )
            for q in questions_raw
        ]

        # Store in memory only
        self._questions.bulk_load(questions)
        print(f"✓ [WORKFLOW] Stored {len(self._questions)} questions in memory")
        print(f"✓ [WORKFLOW] Questions: {[q.number for q in questions[:5]]}")
        return questions

    def process_answer_key(self, filepath: str) -> List[Answer]:
        answers_raw = self.pdf_processor.extract_answers(filepath)
        if not answers_raw:
            raise WorkflowError(
                "No answers detected. Please upload a clear answer key PDF.", status_code=400
            )

        answers = [
            Answer(number=int(a["question_number"]), text=a["answer_text"])
            for a in answers_raw
        ]

        # Store in memory only
        self._answers.bulk_load(answers)
        print(f"✓ [WORKFLOW] Stored {len(self._answers)} answers in memory")
        print(f"✓ [WORKFLOW] Answers: {[a.number for a in answers[:5]]}")
        return answers

    def evaluate_student_paper(self, *, filepath: str, filename: str, student_name: str) -> Dict:
        print(f"🔍 [WORKFLOW] Starting evaluation for {filename}")
        print(f"🔍 [WORKFLOW] Current questions in memory: {len(self._questions)}")
        print(f"🔍 [WORKFLOW] Current answers in memory: {len(self._answers)}")
        
        if self.evaluator is None:
            raise WorkflowError(
                "Ollama evaluator is not available. Please start Ollama and pull the configured model.",
                status_code=500,
            )

        if not self._questions.as_list():
            raise WorkflowError("Upload a question paper before evaluating.", status_code=400)
        if not self._answers.as_list():
            raise WorkflowError("Upload an answer key before evaluating.", status_code=400)

        if self.pdf_processor.deepseek_ocr is None:
            raise WorkflowError(
                "DeepSeek-OCR is not available. Please ensure the model is initialized.",
                status_code=500,
            )
        
        print(f"✓ [WORKFLOW] Pre-flight checks passed")

        try:
            file_hash = self._hash_file(filepath)
            print(f"✓ [WORKFLOW] File hash: {file_hash[:16]}...")
        except Exception as e:
            raise WorkflowError(f"Failed to hash file: {str(e)}", status_code=500)
        
        extracted_text = self._ocr_cache.get(file_hash)
        if not extracted_text:
            print(f"🧠 [WORKFLOW] Running OCR extraction...")
            try:
                extracted_text = self.pdf_processor.deepseek_ocr.extract_text_from_pdf(filepath)
                print(f"✓ [WORKFLOW] OCR extracted {len(extracted_text) if extracted_text else 0} characters")
            except Exception as e:
                raise WorkflowError(f"OCR extraction failed: {str(e)}", status_code=500)
            
            if not extracted_text or len(extracted_text.strip()) < 20:
                raise WorkflowError(
                    "Insufficient text extracted from student PDF. Please upload a clearer scan.",
                    status_code=400,
                )
            self._ocr_cache.set(file_hash, extracted_text)
        else:
            print(f"✓ [WORKFLOW] Using cached OCR result")

        try:
            cleaned_text = self._clean_ocr_text(extracted_text)
            print(f"✓ [WORKFLOW] Cleaned text: {len(cleaned_text)} characters")
        except Exception as e:
            raise WorkflowError(f"Text cleaning failed: {str(e)}", status_code=500)
        if len(cleaned_text) < 20:
            raise WorkflowError(
                "OCR extraction produced too little text. Ensure handwriting is legible.",
                status_code=400,
            )

        ocr_pdf_filename = f"student_ocr_{filename}"
        ocr_pdf_path = os.path.join(self.upload_folder, ocr_pdf_filename)
        try:
            print(f"📄 [WORKFLOW] Creating searchable PDF...")
            self.ocr_pdf_converter.create_text_pdf(
                extracted_text=extracted_text,
                output_path=ocr_pdf_path,
                title=f"Student Answer Sheet - {student_name} (OCR Extracted)",
            )
            print(f"✓ [WORKFLOW] Searchable PDF created")
        except Exception as e:
            print(f"⚠ [WORKFLOW] Warning: Could not create OCR PDF: {str(e)}")
            # Non-fatal, continue evaluation

        student_answers_cached = self._student_answer_cache.get(file_hash)
        if student_answers_cached is None:
            print(f"📝 [WORKFLOW] Parsing student answers...")
            try:
                student_answers_cached = self.pdf_processor.extract_student_answers(extracted_text)
                print(f"✓ [WORKFLOW] Found {len(student_answers_cached) if student_answers_cached else 0} answers from text")
            except Exception as e:
                print(f"⚠ [WORKFLOW] Text parsing failed: {str(e)}")
                student_answers_cached = []
            
            if not student_answers_cached:
                print(f"🔄 [WORKFLOW] Retrying with direct PDF parsing...")
                try:
                    student_answers_cached = self.pdf_processor.extract_student_answers(filepath)
                    print(f"✓ [WORKFLOW] Found {len(student_answers_cached) if student_answers_cached else 0} answers from PDF")
                except Exception as e:
                    print(f"⚠ [WORKFLOW] PDF parsing also failed: {str(e)}")
                    student_answers_cached = []
            
            self._student_answer_cache.set(file_hash, student_answers_cached or [])
        else:
            print(f"✓ [WORKFLOW] Using cached student answers")

        if not student_answers_cached:
            raise WorkflowError(
                "No student answers detected. Please ensure each response is numbered (Q1, Q2, ...).",
                status_code=400,
            )

        student_answers = [
            {
                "question_number": int(entry["question_number"]),
                "student_answer": entry["student_answer"],
            }
            for entry in student_answers_cached
        ]

        questions_payload = [
            {
                "question_number": q.number,
                "question_text": q.text,
                "max_marks": q.max_marks,
            }
            for q in self._questions.as_list()
        ]

        answers_payload = [
            {
                "question_number": a.number,
                "answer_text": a.text,
            }
            for a in self._answers.as_list()
        ]

        print(f"🎓 [WORKFLOW] Starting AI evaluation with {len(questions_payload)} questions...")
        try:
            evaluation_result = self.evaluator.evaluate_all_answers(
                questions=questions_payload,
                answer_key=answers_payload,
                student_answers=student_answers,
            )
            print(f"✓ [WORKFLOW] Evaluation completed")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise WorkflowError(f"Evaluation failed: {str(e)}", status_code=500)

        result_filename = self._build_result_filename(student_name, filename)
        result_path = os.path.join(self.result_folder, result_filename)
        try:
            print(f"📊 [WORKFLOW] Generating result PDF...")
            self.result_pdf_generator.generate_result_pdf(
                evaluation_result=evaluation_result,
                student_name=student_name,
                output_path=result_path,
            )
            print(f"✓ [WORKFLOW] Result PDF generated: {result_filename}")
        except Exception as e:
            print(f"⚠ [WORKFLOW] Warning: PDF generation failed: {str(e)}")
            # Non-fatal, return evaluation results anyway

        question_wise_summary = [
            {
                "question_number": item["Question_Number"],
                "max_marks": item["Max_Marks"],
                "obtained_marks": item["Awarded_Marks"],
                "concept_match_score": item["Concept_Match_Score"],
                "feedback": item["Feedback"],
            }
            for item in evaluation_result["question_wise_results"]
        ]

        return {
            "result_filename": result_filename,
            "result_path": result_path,
            "ocr_pdf_filename": ocr_pdf_filename,
            "ocr_pdf_path": ocr_pdf_path,
            "question_wise_summary": question_wise_summary,
            "evaluation": evaluation_result,
        }

    def reset_state(self):
        """Clear caches and in-memory indices (vector DB is cleared by caller)."""
        self._questions.clear()
        self._answers.clear()
        self._ocr_cache.clear()
        self._student_answer_cache.clear()

    def status_snapshot(self) -> Dict[str, int]:
        return {
            "questions_count": len(self._questions),
            "answers_count": len(self._answers),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hash_file(self, filepath: str) -> str:
        sha1 = hashlib.sha1()
        with open(filepath, "rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                sha1.update(chunk)
        return sha1.hexdigest()

    def _clean_ocr_text(self, text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _build_result_filename(self, student_name: str, filename: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", student_name.strip() or "Student")
        return f"result_{safe_name}_{filename}"[:150]
