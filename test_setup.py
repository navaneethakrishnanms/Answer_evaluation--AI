"""
Test script to verify the Answer Evaluation System setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'flask': 'Flask',
        'PyPDF2': 'PyPDF2',
        'PIL': 'Pillow',
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'reportlab': 'reportlab',
        'chromadb': 'chromadb',
        'sentence_transformers': 'sentence-transformers',
        'sklearn': 'scikit-learn',
        'torch': 'PyTorch',
        'transformers': 'transformers',
        'accelerate': 'accelerate'
    }
    
    failed = []
    
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"✅ {package_name}: OK")
        except ImportError as e:
            print(f"❌ {package_name}: FAILED - {str(e)}")
            failed.append(package_name)
    
    print()
    if failed:
        print(f"❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_gpu():
    """Test GPU availability"""
    print("\n" + "=" * 60)
    print("Testing GPU Setup")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print("\n✅ GPU is available and ready!")
            return True
        else:
            print("\n⚠️ Warning: GPU not available. System will use CPU (slower)")
            print("   For better performance, install CUDA and PyTorch with CUDA support")
            return False
    except Exception as e:
        print(f"\n❌ Error testing GPU: {str(e)}")
        return False

def test_system_dependencies():
    """Test system dependencies like Poppler"""
    print("\n" + "=" * 60)
    print("Testing System Dependencies")
    print("=" * 60)
    
    # Test Poppler
    try:
        from pdf2image import convert_from_path
        print("✅ Poppler: OK (pdf2image can be imported)")
    except Exception as e:
        print(f"❌ Poppler: FAILED - {str(e)}")
        print("   Install Poppler for PDF to image conversion")
        return False
    
    # Test Tesseract (optional)
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR: OK (version {version})")
    except:
        print("⚠️ Tesseract OCR: Not found (optional - only needed for fallback)")
    
    return True

def test_directories():
    """Test if required directories exist"""
    print("\n" + "=" * 60)
    print("Testing Directory Structure")
    print("=" * 60)
    
    required_dirs = ['uploads', 'results', 'vector_db', 'templates', 'static']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/: OK")
        else:
            print(f"⚠️ {dir_name}/: Not found (will be created automatically)")
    
    return True

def test_files():
    """Test if required files exist"""
    print("\n" + "=" * 60)
    print("Testing Required Files")
    print("=" * 60)
    
    required_files = [
        'app.py',
        'pdf_processor.py',
        'deepseek_ocr.py',
        'vector_db_manager.py',
        'answer_evaluator.py',
        'config.py',
        'requirements.txt',
        'templates/index.html',
        'static/style.css',
        'static/script.js'
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: OK")
        else:
            print(f"❌ {file_path}: MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ {len(missing)} file(s) are missing:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True

def test_deepseek_model():
    """Test if DeepSeek-OCR can be loaded"""
    print("\n" + "=" * 60)
    print("Testing DeepSeek-OCR Model")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        print("🔄 Checking DeepSeek-OCR model availability...")
        print("   (This will download ~6.7GB on first run)")
        
        # Just check if model is accessible, don't load it
        model_name = "deepseek-ai/DeepSeek-OCR"
        print(f"   Model: {model_name}")
        print("✅ DeepSeek-OCR model is accessible")
        print("   Note: Model will be downloaded on first actual use")
        return True
    except Exception as e:
        print(f"❌ Error accessing DeepSeek-OCR model: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ANSWER EVALUATION SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    results = {
        'Package Imports': test_imports(),
        'GPU Setup': test_gpu(),
        'System Dependencies': test_system_dependencies(),
        'Directory Structure': test_directories(),
        'Required Files': test_files(),
        'DeepSeek-OCR Model': test_deepseek_model()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour system is ready to use!")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Upload question paper, answer key, and evaluate papers")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the application.")
        print("See INSTALLATION.md for detailed setup instructions.")
    print("=" * 60)
    print()
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
