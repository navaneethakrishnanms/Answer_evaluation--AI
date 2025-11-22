"""
Test script to extract text from uploaded PDFs
"""
import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from pdf_processor import PDFProcessor

print("="*60)
print("PDF Extraction Test")
print("="*60)

# Initialize processor without DeepSeek for quick test
processor = PDFProcessor(use_deepseek=False)

# Check uploaded files
upload_dir = './uploads'
files = os.listdir(upload_dir)

print(f"\nFiles in uploads directory:")
for f in files:
    if f.endswith('.pdf'):
        print(f"  - {f}")

# Test question paper
question_paper_files = [f for f in files if f.startswith('question_paper_') and f.endswith('.pdf')]
if question_paper_files:
    print("\n" + "="*60)
    print("TESTING QUESTION PAPER EXTRACTION")
    print("="*60)
    
    qp_file = os.path.join(upload_dir, question_paper_files[0])
    print(f"\nProcessing: {question_paper_files[0]}")
    
    # Extract raw text first
    raw_text = processor.extract_text_from_pdf(qp_file)
    print(f"\nRaw text extracted ({len(raw_text)} chars):")
    print("-"*60)
    print(raw_text[:500] if raw_text else "NO TEXT EXTRACTED")
    print("-"*60)
    
    # Extract questions with marks
    questions = processor.extract_questions_with_marks(qp_file)
    print(f"\n✅ Extracted {len(questions)} questions")
    
    if questions:
        print("\nQuestions found:")
        for q in questions:
            print(f"  Q{q['question_number']}: {q['question_text'][:80]}... ({q['max_marks']} marks)")
    else:
        print("⚠️ No questions extracted! This is the problem.")
        print("\nTrying OCR...")
        ocr_text = processor.ocr_pdf(qp_file)
        print(f"\nOCR text ({len(ocr_text)} chars):")
        print("-"*60)
        print(ocr_text[:500] if ocr_text else "NO OCR TEXT")
else:
    print("\n⚠️ No question paper found in uploads!")

# Test answer key
answer_key_files = [f for f in files if f.startswith('answer_key_') and f.endswith('.pdf')]
if answer_key_files:
    print("\n" + "="*60)
    print("TESTING ANSWER KEY EXTRACTION")
    print("="*60)
    
    ak_file = os.path.join(upload_dir, answer_key_files[0])
    print(f"\nProcessing: {answer_key_files[0]}")
    
    # Extract raw text first
    raw_text = processor.extract_text_from_pdf(ak_file)
    print(f"\nRaw text extracted ({len(raw_text)} chars):")
    print("-"*60)
    print(raw_text[:500] if raw_text else "NO TEXT EXTRACTED")
    print("-"*60)
    
    # Extract answers
    answers = processor.extract_answers(ak_file)
    print(f"\n✅ Extracted {len(answers)} answers")
    
    if answers:
        print("\nAnswers found:")
        for a in answers:
            print(f"  A{a['question_number']}: {a['answer_text'][:80]}...")
    else:
        print("⚠️ No answers extracted! This is the problem.")
        print("\nTrying OCR...")
        ocr_text = processor.ocr_pdf(ak_file)
        print(f"\nOCR text ({len(ocr_text)} chars):")
        print("-"*60)
        print(ocr_text[:500] if ocr_text else "NO OCR TEXT")
else:
    print("\n⚠️ No answer key found in uploads!")

print("\n" + "="*60)
print("END OF TEST")
print("="*60)
