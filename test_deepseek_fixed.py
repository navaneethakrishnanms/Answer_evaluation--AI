"""
Test script to verify DeepSeek-OCR works with the fixed device placement
"""

import os
import sys
from deepseek_ocr import get_deepseek_ocr

def test_image_ocr(image_path: str):
    """Test OCR on a single image"""
    print("="*70)
    print("🧪 TESTING DEEPSEEK-OCR WITH FIXED DEVICE PLACEMENT")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    print(f"\n📸 Testing with image: {image_path}")
    
    # Get DeepSeek-OCR instance
    ocr = get_deepseek_ocr()
    
    # Extract text
    result = ocr.extract_text_from_image(image_path)
    
    print("\n" + "="*70)
    print("📝 EXTRACTION RESULT:")
    print("="*70)
    print(result)
    print("="*70)
    
    if result and len(result) > 0:
        print(f"\n✅ SUCCESS: Extracted {len(result)} characters")
        return True
    else:
        print("\n❌ FAILED: No text extracted")
        return False

def test_pdf_ocr(pdf_path: str):
    """Test OCR on a PDF file"""
    print("\n" + "="*70)
    print("📄 TESTING PDF OCR")
    print("="*70)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF not found at {pdf_path}")
        return
    
    print(f"\n📄 Testing with PDF: {pdf_path}")
    
    # Get DeepSeek-OCR instance
    ocr = get_deepseek_ocr()
    
    # Extract text from PDF
    result = ocr.extract_text_from_pdf(pdf_path)
    
    print("\n" + "="*70)
    print("📝 EXTRACTION RESULT:")
    print("="*70)
    print(result[:500] + "..." if len(result) > 500 else result)
    print("="*70)
    
    if result and len(result) > 0:
        print(f"\n✅ SUCCESS: Extracted {len(result)} characters from PDF")
        return True
    else:
        print("\n❌ FAILED: No text extracted from PDF")
        return False

if __name__ == "__main__":
    # Test with image first
    test_image = "ghj.jpeg"  # Your test image
    if os.path.exists(test_image):
        print("Testing with image...")
        test_image_ocr(test_image)
    else:
        print(f"⚠️ Test image '{test_image}' not found, skipping image test")
    
    # Test with PDF if available
    test_pdf = "uploads/student_PDFGallery_20251031_101122.pdf"
    if os.path.exists(test_pdf):
        print("\n\nTesting with PDF...")
        test_pdf_ocr(test_pdf)
    else:
        print(f"\n⚠️ Test PDF not found at '{test_pdf}'")
        print("Please place a test PDF in the uploads folder or provide a path")
