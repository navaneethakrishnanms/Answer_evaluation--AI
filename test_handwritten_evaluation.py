"""
Test Script: Complete Handwritten Answer Evaluation Pipeline
Tests the entire flow from handwritten PDF to evaluated result
"""

import os
import sys
from datetime import datetime

def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def test_complete_pipeline():
    """
    Test the complete evaluation pipeline:
    1. Convert handwritten PDF to text PDF
    2. Extract questions, answer key, and student answers
    3. Evaluate using Ollama LLM
    4. Generate result PDF
    """
    
    print_header("🧪 Testing Complete Evaluation Pipeline")
    
    # Import required modules
    try:
        print("📦 Importing modules...")
        from ocr_to_text_pdf import HandwrittenToTextPDF
        from pdf_processor import PDFProcessor
        from ollama_evaluator import OllamaEvaluator
        from pdf_generator import ResultPDFGenerator
        print("✅ All modules imported successfully\n")
    except Exception as e:
        print(f"❌ Error importing modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if required files exist
    print("📁 Checking for test files...")
    
    uploads_dir = "uploads"
    results_dir = "results"
    
    # Find uploaded files
    question_paper = None
    answer_key = None
    student_paper = None
    
    if os.path.exists(uploads_dir):
        files = os.listdir(uploads_dir)
        for f in files:
            if f.startswith("question_paper_"):
                question_paper = os.path.join(uploads_dir, f)
            elif f.startswith("answer_key_"):
                answer_key = os.path.join(uploads_dir, f)
            elif f.startswith("student_") and not f.startswith("student_ocr_"):
                student_paper = os.path.join(uploads_dir, f)
    
    print(f"Question Paper: {question_paper if question_paper else 'Not found'}")
    print(f"Answer Key: {answer_key if answer_key else 'Not found'}")
    print(f"Student Paper: {student_paper if student_paper else 'Not found'}")
    
    if not all([question_paper, answer_key, student_paper]):
        print("\n⚠️  Missing required files. Please upload:")
        print("   1. Question paper PDF (starts with question_paper_)")
        print("   2. Answer key PDF (starts with answer_key_)")
        print("   3. Student handwritten PDF (starts with student_)")
        return False
    
    print("\n✅ All required files found!\n")
    
    # Initialize components
    print_header("🔧 Initializing Components")
    
    try:
        ocr_converter = HandwrittenToTextPDF()
        pdf_processor = PDFProcessor(use_deepseek=True)
        ollama_evaluator = OllamaEvaluator(model_name="llama3.1:latest")
        result_generator = ResultPDFGenerator()
        print("✅ All components initialized\n")
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return False
    
    # Step 1: Convert handwritten PDF to text PDF
    print_header("📝 Step 1: Converting Handwritten PDF to Text PDF")
    
    student_name = "Test Student"
    student_text_pdf = os.path.join(uploads_dir, f"student_ocr_{os.path.basename(student_paper)}")
    
    try:
        ocr_result = ocr_converter.convert_handwritten_to_text_pdf(
            input_pdf_path=student_paper,
            output_pdf_path=student_text_pdf,
            student_name=student_name
        )
        
        if not ocr_result['success']:
            print(f"❌ OCR conversion failed: {ocr_result.get('error')}")
            return False
        
        print(f"\n✅ Text PDF created with {ocr_result['answers_detected']} answers detected\n")
    
    except Exception as e:
        print(f"❌ Error in OCR conversion: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Extract questions with marks
    print_header("📋 Step 2: Extracting Questions from Question Paper")
    
    try:
        questions = pdf_processor.extract_questions_with_marks(question_paper)
        
        if not questions:
            print("❌ No questions found in question paper")
            return False
        
        print(f"✅ Extracted {len(questions)} questions")
        print(f"Total Marks: {sum(q['max_marks'] for q in questions)}\n")
        
        for q in questions[:3]:  # Show first 3
            print(f"  Q{q['question_number']}: {q['question_text'][:60]}... ({q['max_marks']} marks)")
        print()
    
    except Exception as e:
        print(f"❌ Error extracting questions: {e}")
        return False
    
    # Step 3: Extract answer key
    print_header("✅ Step 3: Extracting Answer Key")
    
    try:
        answer_key_data = pdf_processor.extract_answers(answer_key)
        
        if not answer_key_data:
            print("❌ No answers found in answer key")
            return False
        
        print(f"✅ Extracted {len(answer_key_data)} answers from answer key\n")
        
        for a in answer_key_data[:3]:  # Show first 3
            print(f"  A{a['question_number']}: {a['answer_text'][:60]}...")
        print()
    
    except Exception as e:
        print(f"❌ Error extracting answer key: {e}")
        return False
    
    # Step 4: Extract student answers
    print_header("📝 Step 4: Extracting Student Answers")
    
    try:
        student_answers = pdf_processor.extract_student_answers(student_text_pdf)
        
        if not student_answers:
            print("❌ No student answers found")
            return False
        
        print(f"✅ Extracted {len(student_answers)} student answers\n")
        
        for s in student_answers[:3]:  # Show first 3
            print(f"  Student A{s['question_number']}: {s['student_answer'][:60]}...")
        print()
    
    except Exception as e:
        print(f"❌ Error extracting student answers: {e}")
        return False
    
    # Step 5: Evaluate using Ollama LLM
    print_header("🧠 Step 5: Evaluating with Ollama LLM (Concept-Based)")
    
    try:
        evaluation = ollama_evaluator.evaluate_all_answers(
            questions=questions,
            answer_key=answer_key_data,
            student_answers=student_answers
        )
        
        print(f"\n✅ Evaluation complete!\n")
        print(f"📊 Results:")
        print(f"   Total Marks: {evaluation['obtained_marks']:.2f}/{evaluation['total_marks']}")
        print(f"   Percentage: {evaluation['percentage']:.2f}%")
        print(f"   Grade: {evaluation['grade']}")
        print(f"   Method: {evaluation['evaluation_method']}\n")
    
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Generate result PDF
    print_header("📄 Step 6: Generating Result PDF")
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        result_pdf = os.path.join(results_dir, f"result_{student_name.replace(' ', '_')}_test.pdf")
        
        result_generator.generate_result_pdf(
            evaluation_result=evaluation,
            student_name=student_name,
            output_path=result_pdf
        )
        
        print(f"\n✅ Result PDF generated: {result_pdf}\n")
    
    except Exception as e:
        print(f"❌ Error generating result PDF: {e}")
        return False
    
    # Summary
    print_header("🎉 Test Complete - Summary")
    
    print("✅ All steps completed successfully!\n")
    print("📁 Generated Files:")
    print(f"   1. Text PDF: {student_text_pdf}")
    print(f"   2. Result PDF: {result_pdf}\n")
    print("📊 Evaluation Results:")
    print(f"   - Student: {student_name}")
    print(f"   - Questions Evaluated: {len(evaluation['question_wise_results'])}")
    print(f"   - Marks: {evaluation['obtained_marks']:.2f}/{evaluation['total_marks']}")
    print(f"   - Percentage: {evaluation['percentage']:.2f}%")
    print(f"   - Grade: {evaluation['grade']}\n")
    
    print("🎓 Question-wise Breakdown:")
    for result in evaluation['question_wise_results'][:5]:  # Show first 5
        q_num = result['Question_Number']
        marks = result['Awarded_Marks']
        max_marks = result['Max_Marks']
        concept = result['Concept_Match_Score'] * 100
        print(f"   Q{q_num}: {marks:.2f}/{max_marks} (Concept: {concept:.0f}%)")
    
    print("\n" + "="*70)
    print("✨ Pipeline test completed successfully!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 Handwritten Answer Evaluation Pipeline - Test Suite")
    print("="*70)
    print("\nThis script tests the complete flow:")
    print("  1. Handwritten PDF → Text PDF (DeepSeek-OCR)")
    print("  2. Question/Answer extraction")
    print("  3. Concept-based evaluation (Ollama LLM)")
    print("  4. Result PDF generation")
    print("\n" + "="*70 + "\n")
    
    success = test_complete_pipeline()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed. Please check the errors above.")
        sys.exit(1)