"""
Test the complete evaluation pipeline
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing Imports...")
    print("-" * 60)
    
    try:
        from evaluation_pipeline import AnswerEvaluationPipeline
        print("✅ evaluation_pipeline")
    except Exception as e:
        print(f"❌ evaluation_pipeline: {e}")
        return False
    
    try:
        from ollama_evaluator import OllamaEvaluator
        print("✅ ollama_evaluator")
    except Exception as e:
        print(f"❌ ollama_evaluator: {e}")
        return False
    
    try:
        from pdf_generator import OCRtoPDFConverter, ResultPDFGenerator
        print("✅ pdf_generator")
    except Exception as e:
        print(f"❌ pdf_generator: {e}")
        return False
    
    try:
        from pdf_processor_fast import PDFProcessor
        print("✅ pdf_processor_fast")
    except Exception as e:
        print(f"❌ pdf_processor_fast: {e}")
        return False
    
    return True

def test_ollama():
    """Test if Ollama is installed and working"""
    print("\n🤖 Testing Ollama...")
    print("-" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"\n📋 Available models:")
            print(result.stdout)
            return True
        else:
            print("❌ Ollama command failed")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama is NOT installed")
        print("\n💡 Install Ollama from: https://ollama.ai")
        print("   After installation, run: ollama pull llama3.1")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

def test_pipeline_basic():
    """Test basic pipeline functionality"""
    print("\n🔧 Testing Pipeline Initialization...")
    print("-" * 60)
    
    try:
        from evaluation_pipeline import AnswerEvaluationPipeline
        
        pipeline = AnswerEvaluationPipeline(ollama_model="llama3.1:latest")
        print("✅ Pipeline initialized successfully")
        
        # Check components
        print("\n📦 Components:")
        print(f"   ✅ PDF Processor: {type(pipeline.pdf_processor).__name__}")
        print(f"   ✅ Ollama Evaluator: {type(pipeline.ollama_evaluator).__name__}")
        print(f"   ✅ PDF Converter: {type(pipeline.pdf_converter).__name__}")
        print(f"   ✅ Result Generator: {type(pipeline.result_generator).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 TESTING COMPLETE EVALUATION SYSTEM")
    print("="*70)
    
    results = {
        "imports": test_imports(),
        "ollama": test_ollama(),
        "pipeline": test_pipeline_basic()
    }
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - System is ready!")
        print("="*70)
        print("\n🚀 Next steps:")
        print("   1. Start the server: python app_ollama.py")
        print("   2. Open browser: http://localhost:5000")
        print("   3. Upload your PDFs and start evaluating!")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues above")
        print("="*70)
        print("\n💡 Common fixes:")
        print("   - Install Ollama: https://ollama.ai")
        print("   - Install Python packages: pip install -r requirements_new.txt")
        print("   - Pull Ollama model: ollama pull llama3.1")
    
    print("\n")
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
