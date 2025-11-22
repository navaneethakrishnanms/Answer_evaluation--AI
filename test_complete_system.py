"""
System Test Script for Ollama-Integrated Answer Evaluation
Tests all components: DeepSeek-OCR, Ollama, Vector DB, PDF Generation
"""

import sys
import os

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_imports():
    """Test if all required modules can be imported"""
    print_section("Testing Python Package Imports")
    
    required_packages = [
        ('flask', 'Flask'),
        ('PyPDF2', 'PyPDF2'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('chromadb', 'ChromaDB'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('sklearn', 'Scikit-learn'),
        ('fpdf', 'FPDF2'),
        ('reportlab', 'ReportLab'),
        ('pdf2image', 'pdf2image')
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name:<25} - OK")
        except ImportError as e:
            print(f"❌ {name:<25} - MISSING")
            all_ok = False
    
    return all_ok

def test_cuda():
    """Test CUDA/GPU availability"""
    print_section("Testing CUDA/GPU Availability")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA Available: Yes")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA Available: No")
            print("   DeepSeek-OCR will run on CPU (slower)")
        
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def test_ollama():
    """Test Ollama installation and model availability"""
    print_section("Testing Ollama Installation")
    
    import subprocess
    
    try:
        # Check if Ollama is installed
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Ollama is not installed")
            print("   Install from: https://ollama.ai")
            return False
        
        print("✅ Ollama is installed")
        print(f"   Version: {result.stdout.strip()}")
        
        # Check available models
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if "llama3.1" in result.stdout:
            print("✅ llama3.1 model available")
        else:
            print("⚠️  llama3.1 model not found")
            print("   Run: ollama pull llama3.1:latest")
        
        return True
        
    except FileNotFoundError:
        print("❌ Ollama command not found")
        print("   Install from: https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def test_poppler():
    """Test Poppler installation"""
    print_section("Testing Poppler Installation")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["pdfinfo", "-v"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("✅ Poppler is installed")
            return True
        else:
            print("❌ Poppler not found in PATH")
            print("   Windows: Run fix_poppler.bat")
            print("   Linux: sudo apt-get install poppler-utils")
            print("   Mac: brew install poppler")
            return False
            
    except FileNotFoundError:
        print("❌ Poppler not found")
        print("   Windows: Run fix_poppler.bat")
        print("   Linux: sudo apt-get install poppler-utils")
        print("   Mac: brew install poppler")
        return False

def test_deepseek_ocr():
    """Test DeepSeek-OCR initialization"""
    print_section("Testing DeepSeek-OCR")
    
    try:
        from deepseek_ocr import get_deepseek_ocr
        
        print("Initializing DeepSeek-OCR (this may download the model)...")
        ocr = get_deepseek_ocr()
        
        success = ocr.initialize()
        
        if success:
            print("✅ DeepSeek-OCR initialized successfully")
            return True
        else:
            print("❌ DeepSeek-OCR initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing DeepSeek-OCR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_db():
    """Test Vector Database"""
    print_section("Testing Vector Database (ChromaDB)")
    
    try:
        from vector_db_manager import VectorDBManager
        
        db = VectorDBManager(persist_directory='./test_vector_db')
        
        # Test storing and retrieving data
        test_questions = [
            {
                'question_number': 1,
                'question_text': 'Test question?',
                'max_marks': 5
            }
        ]
        
        db.store_question_paper(test_questions)
        retrieved = db.get_all_questions()
        
        if len(retrieved) > 0:
            print("✅ ChromaDB working correctly")
            
            # Cleanup
            db.clear_all_data()
            return True
        else:
            print("❌ ChromaDB not storing data correctly")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Vector DB: {e}")
        return False

def test_ollama_evaluator():
    """Test Ollama Evaluator"""
    print_section("Testing Ollama Evaluator")
    
    try:
        from ollama_evaluator import OllamaEvaluator
        
        evaluator = OllamaEvaluator(model_name="llama3.1:latest")
        
        # Test with simple evaluation
        print("Running test evaluation...")
        result = evaluator.evaluate_answer(
            question_number=1,
            question_text="What is 2+2?",
            max_marks=2,
            correct_answer="2+2 equals 4",
            student_answer="The answer is 4"
        )
        
        if result and not result.get('Error'):
            print("✅ Ollama Evaluator working correctly")
            print(f"   Test result: {result['Awarded_Marks']}/{result['Max_Marks']} marks")
            return True
        else:
            print("❌ Ollama Evaluator returned error")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama Evaluator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_generation():
    """Test PDF generation"""
    print_section("Testing PDF Generation")
    
    try:
        from pdf_generator import ResultPDFGenerator
        
        generator = ResultPDFGenerator()
        
        test_result = {
            'total_marks': 10,
            'obtained_marks': 8.5,
            'percentage': 85.0,
            'grade': 'A',
            'evaluation_method': 'Test Mode',
            'question_wise_results': [
                {
                    'Question_Number': 1,
                    'Max_Marks': 5,
                    'Awarded_Marks': 4.5,
                    'Concept_Match_Score': 0.9,
                    'Feedback': 'Very good answer'
                }
            ]
        }
        
        output_path = './test_result.pdf'
        generator.generate_result_pdf(
            evaluation_result=test_result,
            student_name="Test Student",
            output_path=output_path
        )
        
        if os.path.exists(output_path):
            print("✅ PDF generation working correctly")
            os.remove(output_path)  # Cleanup
            return True
        else:
            print("❌ PDF file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error testing PDF generation: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" 🧪 SYSTEM TEST - Ollama Integrated Answer Evaluation")
    print("="*60)
    
    results = {
        'Python Packages': test_imports(),
        'CUDA/GPU': test_cuda(),
        'Ollama': test_ollama(),
        'Poppler': test_poppler(),
        'DeepSeek-OCR': test_deepseek_ocr(),
        'Vector Database': test_vector_db(),
        'Ollama Evaluator': test_ollama_evaluator(),
        'PDF Generation': test_pdf_generation()
    }
    
    # Summary
    print("\n" + "="*60)
    print(" 📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("-"*60)
    
    if passed == total:
        print("\n✅ All tests passed! System is ready to use.")
        print("\nRun the application with:")
        print("  python app_ollama_integrated.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nRefer to COMPLETE_GUIDE_OLLAMA.md for troubleshooting.")
    
    print("\n")

if __name__ == "__main__":
    main()
