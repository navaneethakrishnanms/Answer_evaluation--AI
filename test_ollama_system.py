"""
Test script for Ollama LLM Evaluation System
"""

import sys


def test_ollama():
    """Test if Ollama is installed and working"""
    print("\n" + "="*60)
    print("🧪 Testing Ollama Installation")
    print("="*60 + "\n")
    
    try:
        import subprocess
        
        # Test if Ollama is installed
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ Ollama is not working properly")
            return False
        
        # List available models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("\n📦 Available models:")
            print(result.stdout)
            
            if 'llama3.2' in result.stdout:
                print("✅ llama3.2 model is available")
            else:
                print("⚠️  llama3.2 not found. Installing...")
                print("   Run: ollama pull llama3.2")
        
        return True
        
    except FileNotFoundError:
        print("❌ Ollama is not installed!")
        print("\n📥 Installation instructions:")
        print("   Windows: Download from https://ollama.ai")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   macOS: brew install ollama")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {str(e)}")
        return False


def test_dependencies():
    """Test if Python dependencies are installed"""
    print("\n" + "="*60)
    print("🧪 Testing Python Dependencies")
    print("="*60 + "\n")
    
    packages = [
        ('flask', 'Flask'),
        ('PyPDF2', 'PyPDF2'),
        ('pdf2image', 'pdf2image'),
        ('fpdf', 'fpdf2'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('chromadb', 'ChromaDB')
    ]
    
    all_installed = True
    
    for import_name, display_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def test_gpu():
    """Test if GPU is available"""
    print("\n" + "="*60)
    print("🧪 Testing GPU Availability")
    print("="*60 + "\n")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  No GPU detected - will use CPU")
            print("   OCR will be slower without GPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def test_evaluator():
    """Test Ollama evaluator with a simple example"""
    print("\n" + "="*60)
    print("🧪 Testing Ollama Evaluator")
    print("="*60 + "\n")
    
    try:
        from ollama_evaluator import OllamaEvaluator
        
        evaluator = OllamaEvaluator(model_name="llama3.2")
        
        print("📝 Running test evaluation...")
        result = evaluator.evaluate_answer(
            question_number=1,
            question_text="What is photosynthesis?",
            max_marks=5,
            answer_key="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            student_answer="Plants use sunlight to make food."
        )
        
        print(f"\n✅ Evaluation Result:")
        print(f"   Question: 1")
        print(f"   Marks: {result.get('awarded_marks', 0)}/{5}")
        print(f"   Score: {result.get('concept_match_score', 0):.2f}")
        print(f"   Feedback: {result.get('feedback', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing evaluator: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 Ollama LLM Evaluation System - Test Suite")
    print("="*60)
    
    tests = [
        ("Ollama Installation", test_ollama),
        ("Python Dependencies", test_dependencies),
        ("GPU Availability", test_gpu),
        ("Ollama Evaluator", test_evaluator)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60 + "\n")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! System is ready to use.")
        print("="*60 + "\n")
        print("🚀 Start the application:")
        print("   python app_ollama.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("="*60 + "\n")
        print("📚 Refer to README_OLLAMA.md for installation instructions")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
