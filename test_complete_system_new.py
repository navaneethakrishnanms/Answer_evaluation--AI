"""
Test Script for Complete Evaluation System
Tests: OCR → Text PDF → Extraction → Ollama Evaluation
"""

import os
import sys
from pdf_processor_ollama import PDFProcessorOllama
from ollama_evaluator import OllamaEvaluator
from ocr_to_text_pdf_converter import get_converter

def test_system():
    """Test the complete system pipeline"""
    
    print("\n" + "="*80)
    print("🧪 TESTING COMPLETE AI-POWERED ANSWER EVALUATION SYSTEM")
    print("="*80 + "\n")
    
    # Test 1: Check Ollama
    print("TEST 1: Checking Ollama Installation")
    print("-" * 80)
    try:
        evaluator = OllamaEvaluator(model_name="llama3.1:latest")
        print("✅ Ollama is installed and model is available\n")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        print("Please install Ollama from https://ollama.ai")
        print("Then run: ollama pull llama3.1:latest\n")
        return False
    
    # Test 2: Check PDF Processor
    print("TEST 2: Checking PDF Processor")
    print("-" * 80)
    try:
        processor = PDFProcessorOllama()
        print("✅ PDF Processor initialized successfully\n")
    except Exception as e:
        print(f"❌ PDF Processor initialization failed: {e}\n")
        return False
    
    # Test 3: Check OCR Converter
    print("TEST 3: Checking OCR to Text PDF Converter")
    print("-" * 80)
    try:
        converter = get_converter()
        print("✅ OCR Converter ready\n")
    except Exception as e:
        print(f"❌ OCR Converter initialization failed: {e}\n")
        return False
    
    # Test 4: Check if sample files exist
    print("TEST 4: Checking for Sample Files")
    print("-" * 80)
    
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        files = os.listdir(uploads_dir)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF files in uploads/")
            for f in pdf_files:
                print(f"   - {f}")
            print()
        else:
            print("⚠️  No PDF files found in uploads/")
            print("   Upload some sample PDFs to test the system\n")
    else:
        print("⚠️  uploads/ directory not found")
        print("   Creating it now...")
        os.makedirs(uploads_dir, exist_ok=True)
        print("   ✅ Created uploads/ directory\n")
    
    # Test 5: Test OCR initialization (without processing)
    print("TEST 5: Testing DeepSeek OCR Initialization")
    print("-" * 80)
    try:
        from deepseek_ocr import get_deepseek_ocr
        ocr = get_deepseek_ocr()
        # Don't initialize yet, just check if it's available
        print("✅ DeepSeek OCR module loaded successfully")
        print("   (Will initialize on first use to save memory)\n")
    except Exception as e:
        print(f"❌ DeepSeek OCR not available: {e}")
        print("   This is required for handwritten text extraction\n")
        return False
    
    # Test 6: Test FPDF2
    print("TEST 6: Testing FPDF2 (Text PDF Generation)")
    print("-" * 80)
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Test", ln=True)
        
        test_pdf_path = "test_fpdf.pdf"
        pdf.output(test_pdf_path)
        
        if os.path.exists(test_pdf_path):
            print("✅ FPDF2 is working correctly")
            os.remove(test_pdf_path)
            print("   Created and removed test PDF successfully\n")
        else:
            print("❌ FPDF2 test failed - could not create PDF\n")
            return False
    except Exception as e:
        print(f"❌ FPDF2 test failed: {e}\n")
        return False
    
    # Test 7: Test ReportLab (Result PDF)
    print("TEST 7: Testing ReportLab (Result PDF Generation)")
    print("-" * 80)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        test_result_path = "test_reportlab.pdf"
        doc = SimpleDocTemplate(test_result_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("Test Result PDF", styles['Title'])]
        doc.build(story)
        
        if os.path.exists(test_result_path):
            print("✅ ReportLab is working correctly")
            os.remove(test_result_path)
            print("   Created and removed test result PDF successfully\n")
        else:
            print("❌ ReportLab test failed - could not create PDF\n")
            return False
    except Exception as e:
        print(f"❌ ReportLab test failed: {e}\n")
        return False
    
    # Test 8: Test Ollama Evaluation
    print("TEST 8: Testing Ollama Evaluation (Quick Test)")
    print("-" * 80)
    try:
        print("Running a simple evaluation test...")
        
        test_evaluation = evaluator.evaluate_answer(
            question_number=1,
            question_text="What is the capital of France?",
            max_marks=2.0,
            correct_answer="The capital of France is Paris.",
            student_answer="Paris is the capital of France."
        )
        
        if 'Error' in test_evaluation and test_evaluation['Error']:
            print(f"❌ Evaluation test failed")
            print(f"   Response: {test_evaluation}\n")
            return False
        else:
            print("✅ Ollama evaluation is working!")
            print(f"   Test result: {test_evaluation['Awarded_Marks']}/{test_evaluation['Max_Marks']} marks")
            print(f"   Concept match: {test_evaluation['Concept_Match_Score']*100:.1f}%")
            print(f"   Feedback: {test_evaluation['Feedback']}\n")
    except Exception as e:
        print(f"❌ Ollama evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False
    
    # Summary
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n🎉 System is ready to use!")
    print("\nNext steps:")
    print("1. Run: start_complete_system.bat (Windows) or python app_ollama_complete.py")
    print("2. Open browser: http://localhost:5000")
    print("3. Upload question paper, answer key, and student paper")
    print("4. Get AI-powered concept-based evaluation results!\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
