"""
Quick System Test Script
Tests all components of the fixed answer evaluation system
"""

import os
import sys

print("="*70)
print("🧪 SYSTEM COMPONENT TEST")
print("="*70)
print()

# Test 1: Check Python packages
print("[1/5] Testing Python dependencies...")
try:
    import flask
    print("  ✅ Flask installed:", flask.__version__)
except:
    print("  ❌ Flask not found - Run: pip install flask")
    sys.exit(1)

try:
    import torch
    print("  ✅ PyTorch installed:", torch.__version__)
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️ CUDA not available - DeepSeek-OCR will use CPU (slower)")
except:
    print("  ❌ PyTorch not found - Run: pip install torch")

try:
    import transformers
    print("  ✅ Transformers installed:", transformers.__version__)
except:
    print("  ❌ Transformers not found - Run: pip install transformers")

try:
    from pdf2image import convert_from_path
    print("  ✅ pdf2image installed")
except:
    print("  ⚠️ pdf2image not found - May need: pip install pdf2image")

print()

# Test 2: Check Ollama
print("[2/5] Testing Ollama...")
import subprocess
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0:
        print("  ✅ Ollama is installed and running")
        if "llama3-gpu" in result.stdout:
            print("  ✅ llama3-gpu:latest model found")
        else:
            print("  ⚠️ llama3-gpu:latest not found")
            print("     Run: ollama pull llama3-gpu:latest")
    else:
        print("  ❌ Ollama not running")
        print("     Run: ollama serve")
except FileNotFoundError:
    print("  ❌ Ollama not installed")
    print("     Install from: https://ollama.ai")
except Exception as e:
    print(f"  ❌ Error checking Ollama: {e}")

print()

# Test 3: Check folder structure
print("[3/5] Testing folder structure...")
folders = ['uploads', 'results', 'vector_db', 'templates', 'static']
for folder in folders:
    if os.path.exists(folder):
        print(f"  ✅ {folder}/ exists")
    else:
        print(f"  ⚠️ {folder}/ not found - will be created automatically")

print()

# Test 4: Check key files
print("[4/5] Testing key files...")
files = [
    'app_fixed.py',
    'deepseek_ocr.py',
    'ollama_evaluator.py',
    'pdf_processor.py',
    'vector_db_manager.py',
    'pdf_generator.py'
]
for file in files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING!")

print()

# Test 5: Try importing components
print("[5/5] Testing component imports...")
try:
    from deepseek_ocr import get_deepseek_ocr
    print("  ✅ deepseek_ocr module imports successfully")
except Exception as e:
    print(f"  ❌ deepseek_ocr import failed: {e}")

try:
    from ollama_evaluator import OllamaEvaluator
    print("  ✅ ollama_evaluator module imports successfully")
except Exception as e:
    print(f"  ❌ ollama_evaluator import failed: {e}")

try:
    from pdf_processor import PDFProcessor
    print("  ✅ pdf_processor module imports successfully")
except Exception as e:
    print(f"  ❌ pdf_processor import failed: {e}")

try:
    from vector_db_manager import VectorDBManager
    print("  ✅ vector_db_manager module imports successfully")
except Exception as e:
    print(f"  ❌ vector_db_manager import failed: {e}")

print()
print("="*70)
print("🎯 TEST SUMMARY")
print("="*70)
print()
print("If all tests passed with ✅, you can run:")
print("  python app_fixed.py")
print("  or")
print("  start_fixed.bat")
print()
print("If you see ⚠️ warnings, the system may still work but with limitations.")
print("If you see ❌ errors, install missing dependencies first.")
print()
print("="*70)
