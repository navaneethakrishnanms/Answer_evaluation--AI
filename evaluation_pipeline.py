"""
Complete Answer Evaluation Pipeline
Orchestrates OCR extraction, LLM evaluation, and report generation
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from pdf_processor_fast import PDFProcessor
from deepseek_ocr import DeepSeekOCR
from ollama_evaluator import OllamaEvaluator
from pdf_generator import OCRtoPDFConverter, ResultPDFGenerator


class AnswerEvaluationPipeline:
    def __init__(self, ollama_model: Optional[str] = None):
        """
        Initialize the complete evaluation pipeline
        
        Args:
            ollama_model: Ollama model name to use for evaluation
        """
        print("\n" + "="*60)
        print("🚀 Initializing Answer Evaluation Pipeline")
        print("="*60 + "\n")
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.deepseek_ocr = None  # Lazy load when needed
        self.ollama_evaluator = OllamaEvaluator(model_name=ollama_model)
        self.pdf_converter = OCRtoPDFConverter()
        self.result_generator = ResultPDFGenerator()
        
        # Storage for extracted data
        self.questions = []
        self.answer_key = []
        self.student_answers = []
        self.student_text_pdf_path = None
        self.student_raw_text = ""
        
        print("✅ Pipeline initialized successfully!\n")
    
    def process_question_paper(self, pdf_path: str) -> List[Dict]:
        """
        Extract questions from question paper PDF
        
        Args:
            pdf_path: Path to question paper PDF
            
        Returns:
            List of question dictionaries
        """
        print(f"📋 Processing Question Paper: {pdf_path}")
        
        try:
            self.questions = self.pdf_processor.extract_questions(pdf_path)
            print(f"✅ Extracted {len(self.questions)} questions")
            
            for q in self.questions:
                print(f"   Q{q['question_number']}: {q['max_marks']} marks")
            
            return self.questions
            
        except Exception as e:
            print(f"❌ Error processing question paper: {str(e)}")
            raise
    
    def process_answer_key(self, pdf_path: str) -> List[Dict]:
        """
        Extract answers from answer key PDF
        
        Args:
            pdf_path: Path to answer key PDF
            
        Returns:
            List of answer dictionaries
        """
        print(f"\n✅ Processing Answer Key: {pdf_path}")
        
        try:
            self.answer_key = self.pdf_processor.extract_answers(pdf_path)
            print(f"✅ Extracted {len(self.answer_key)} answers")
            
            for ans in self.answer_key:
                preview = ans['answer_text'][:50] + "..." if len(ans['answer_text']) > 50 else ans['answer_text']
                print(f"   Answer {ans['question_number']}: {preview}")
            
            return self.answer_key
            
        except Exception as e:
            print(f"❌ Error processing answer key: {str(e)}")
            raise
    
    def process_student_answer(
        self,
        pdf_path: str,
        output_text_pdf: str = "student_text_answer.pdf",
        use_ocr: bool = True
    ) -> tuple[List[Dict], str]:
        """
        Process student answer PDF - detect if image-based and use OCR if needed
        
        Args:
            pdf_path: Path to student answer PDF
            output_text_pdf: Path to save OCR extracted text PDF
            use_ocr: Whether to use OCR for extraction
            
        Returns:
            Tuple of (student_answers list, path to text PDF)
        """
        print(f"\n✍️  Processing Student Answer: {pdf_path}")
        
        try:
            # Step 1: Check if PDF is image-based
            is_image_based = self._is_image_based_pdf(pdf_path)
            
            if is_image_based and use_ocr:
                print("📸 Detected image-based PDF - Using DeepSeek OCR...")
                
                # Initialize DeepSeek OCR if not already done
                if self.deepseek_ocr is None:
                    print("🔄 Loading DeepSeek OCR model...")
                    self.deepseek_ocr = DeepSeekOCR()
                
                # Extract text using OCR
                extracted_text = self.deepseek_ocr.extract_text_from_pdf(pdf_path)
                print(f"✅ OCR extraction complete ({len(extracted_text)} characters)")
                
                # Convert extracted text to PDF
                print(f"📄 Creating text-based PDF: {output_text_pdf}")
                self.pdf_converter.create_text_pdf(
                    extracted_text=extracted_text,
                    output_path=output_text_pdf,
                    title="Student Answer Sheet (OCR Extracted)"
                )
                
            else:
                print("📝 Text-based PDF detected - Using fast extraction...")
                extracted_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
                print(f"✅ Text extraction complete ({len(extracted_text)} characters)")
                
                # Still create a cleaned PDF
                self.pdf_converter.create_text_pdf(
                    extracted_text=extracted_text,
                    output_path=output_text_pdf,
                    title="Student Answer Sheet (Extracted)"
                )
            
            # Step 2: Parse student answers from extracted text
            self.student_answers = self.pdf_processor.extract_student_answers(extracted_text)
            self.student_raw_text = extracted_text
            self.student_text_pdf_path = output_text_pdf
            print(f"✅ Extracted {len(self.student_answers)} student answers")
            
            for ans in self.student_answers:
                preview = ans['student_answer'][:50] + "..." if len(ans['student_answer']) > 50 else ans['student_answer']
                print(f"   Q{ans['question_number']}: {preview}")
            
            return self.student_answers, output_text_pdf
            
        except Exception as e:
            print(f"❌ Error processing student answer: {str(e)}")
            raise
    
    def evaluate_answers(self) -> Dict:
        """
        Evaluate student answers using Ollama LLM
        
        Returns:
            Complete evaluation result dictionary
        """
        print(f"\n{'='*60}")
        print("🎓 Starting Concept-Based Evaluation")
        print(f"{'='*60}\n")
        
        if not self.questions:
            raise Exception("No questions loaded. Please process question paper first.")
        
        if not self.answer_key:
            raise Exception("No answer key loaded. Please process answer key first.")
        
        if not self.student_answers:
            raise Exception("No student answers loaded. Please process student answer first.")
        
        # Perform evaluation using Ollama
        evaluation_result = self.ollama_evaluator.evaluate_all_answers(
            questions=self.questions,
            answer_key=self.answer_key,
            student_answers=self.student_answers
        )
        
        return evaluation_result
    
    def generate_result_pdf(
        self,
        evaluation_result: Dict,
        output_path: str = "student_result.pdf",
        student_name: str = "Student"
    ) -> str:
        """
        Generate final result PDF
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save result PDF
            student_name: Student's name
            
        Returns:
            Path to generated PDF
        """
        print(f"\n📊 Generating Result PDF...")
        
        result_pdf_path = self.result_generator.generate_result_pdf(
            evaluation_result=evaluation_result,
            student_name=student_name,
            output_path=output_path
        )
        
        return result_pdf_path
    
    def save_evaluation_json(
        self,
        evaluation_result: Dict,
        output_path: str = "evaluation.json"
    ) -> str:
        """
        Save evaluation results as JSON
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save JSON file
            
        Returns:
            Path to saved JSON file
        """
        print(f"💾 Saving evaluation data: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Evaluation data saved successfully")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving JSON: {str(e)}")
            raise
    
    def run_complete_pipeline(
        self,
        question_pdf: str,
        answer_key_pdf: str,
        student_pdf: str,
        output_dir: str = "results",
        student_name: str = "Student",
        use_ocr: bool = True
    ) -> Dict:
        """
        Run the complete evaluation pipeline
        
        Args:
            question_pdf: Path to question paper PDF
            answer_key_pdf: Path to answer key PDF
            student_pdf: Path to student answer PDF
            output_dir: Directory to save outputs
            student_name: Student's name
            use_ocr: Whether to use OCR for student answers
            
        Returns:
            Complete evaluation results
        """
        print("\n" + "="*60)
        print("🚀 STARTING COMPLETE EVALUATION PIPELINE")
        print("="*60 + "\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Process Question Paper
            print("\n📌 STEP 1: Processing Question Paper")
            print("-" * 60)
            self.process_question_paper(question_pdf)
            
            # Step 2: Process Answer Key
            print("\n📌 STEP 2: Processing Answer Key")
            print("-" * 60)
            self.process_answer_key(answer_key_pdf)
            
            # Step 3: Process Student Answer
            print("\n📌 STEP 3: Processing Student Answer")
            print("-" * 60)
            text_pdf_path = os.path.join(output_dir, "student_text_answer.pdf")
            self.process_student_answer(
                pdf_path=student_pdf,
                output_text_pdf=text_pdf_path,
                use_ocr=use_ocr
            )
            
            # Step 4: Evaluate Answers
            print("\n📌 STEP 4: Evaluating Answers with AI")
            print("-" * 60)
            evaluation_result = self.evaluate_answers()
            
            # Step 5: Generate Result PDF
            print("\n📌 STEP 5: Generating Result PDF")
            print("-" * 60)
            result_pdf_path = os.path.join(output_dir, "student_result.pdf")
            self.generate_result_pdf(
                evaluation_result=evaluation_result,
                output_path=result_pdf_path,
                student_name=student_name
            )
            
            # Step 6: Save JSON data
            print("\n📌 STEP 6: Saving Evaluation Data")
            print("-" * 60)
            json_path = os.path.join(output_dir, "evaluation.json")
            self.save_evaluation_json(evaluation_result, json_path)
            
            # Final summary
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\n📁 Output Files:")
            print(f"   1. Text PDF: {text_pdf_path}")
            print(f"   2. Result PDF: {result_pdf_path}")
            print(f"   3. JSON Data: {json_path}")
            print(f"\n📊 Results:")
            print(f"   Total Marks: {evaluation_result['obtained_marks']}/{evaluation_result['total_marks']}")
            print(f"   Percentage: {evaluation_result['percentage']}%")
            print(f"   Grade: {evaluation_result['grade']}")
            print("\n" + "="*60 + "\n")
            
            return {
                "evaluation_result": evaluation_result,
                "output_files": {
                    "text_pdf": text_pdf_path,
                    "result_pdf": result_pdf_path,
                    "json_data": json_path
                }
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise
    
    def _is_image_based_pdf(self, pdf_path: str) -> bool:
        """
        Check if PDF is image-based (scanned) or text-based
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if image-based, False if text-based
        """
        try:
            # Try to extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # If very little text extracted, it's likely image-based
            if len(text.strip()) < 100:
                return True
            
            # Check ratio of text length to file size
            # Text PDFs have more text per KB
            file_size_kb = os.path.getsize(pdf_path) / 1024
            text_ratio = len(text) / file_size_kb
            
            # If ratio is very low, it's likely scanned
            return text_ratio < 50
            
        except:
            # If extraction fails, assume image-based
            return True


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AnswerEvaluationPipeline(ollama_model="llama3-gpu:latest")
    
    # Run complete pipeline
    result = pipeline.run_complete_pipeline(
        question_pdf="question_paper.pdf",
        answer_key_pdf="answer_key.pdf",
        student_pdf="student_answer.pdf",
        output_dir="results",
        student_name="John Doe",
        use_ocr=True
    )
    
    print("Evaluation complete!")
