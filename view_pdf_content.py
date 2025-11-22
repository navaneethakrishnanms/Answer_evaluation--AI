"""
Simple script to show what text is in the uploaded PDFs
"""
import PyPDF2
import os

upload_dir = './uploads'

print("="*70)
print("PDF CONTENT VIEWER")
print("="*70)

# List all PDF files
pdf_files = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]

if not pdf_files:
    print("\n⚠️ No PDF files found in uploads directory!")
    print(f"Directory: {os.path.abspath(upload_dir)}")
else:
    for pdf_file in pdf_files:
        filepath = os.path.join(upload_dir, pdf_file)
        
        print(f"\n{'='*70}")
        print(f"FILE: {pdf_file}")
        print('='*70)
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"Number of pages: {num_pages}")
                print("\n" + "-"*70)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    print(f"\nPAGE {page_num + 1}:")
                    print("-"*70)
                    
                    if text.strip():
                        print(text[:1000])  # Show first 1000 chars
                        if len(text) > 1000:
                            print(f"\n... (and {len(text) - 1000} more characters)")
                    else:
                        print("⚠️ NO TEXT FOUND - This is a scanned/image PDF")
                        print("   OCR will be needed to extract text")
                    
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nWhat this means:")
print("  • If you see text → The PDF has embedded text")
print("  • If you see 'NO TEXT FOUND' → The PDF is an image/scan")
print("  • For scanned PDFs, OCR (Tesseract or DeepSeek) is required")
print("="*70)
