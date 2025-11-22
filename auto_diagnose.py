"""
Comprehensive diagnostic and auto-fix tool
"""
import os
import sys
import PyPDF2

def print_section(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")

def check_pdf_content():
    """Check if PDFs have extractable text"""
    print_section("CHECKING PDF CONTENTS")
    
    upload_dir = './uploads'
    pdf_files = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory!")
        return False
    
    has_text = False
    
    for pdf_file in pdf_files:
        filepath = os.path.join(upload_dir, pdf_file)
        print(f"\n📄 Checking: {pdf_file}")
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if text.strip() and len(text) > 50:
                    print(f"   ✅ Has extractable text ({len(text)} characters)")
                    print(f"   Preview: {text[:100]}...")
                    has_text = True
                else:
                    print(f"   ⚠️  NO TEXT or very little text found")
                    print(f"   This is likely a SCANNED/IMAGE PDF")
                    print(f"   OCR is required!")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return has_text

def check_database():
    """Check ChromaDB contents"""
    print_section("CHECKING DATABASE")
    
    try:
        import chromadb
        
        client = chromadb.PersistentClient(path='./vector_db')
        
        # Check questions
        try:
            q_collection = client.get_collection(name="question_papers")
            q_data = q_collection.get()
            q_count = len(q_data['ids'])
            
            print(f"📝 Questions in database: {q_count}")
            if q_count == 0:
                print("   ⚠️  Database is EMPTY - No questions stored!")
            else:
                print(f"   ✅ Found questions: {q_data['ids']}")
        except:
            print("   ⚠️  Question collection not found")
            q_count = 0
        
        # Check answers  
        try:
            a_collection = client.get_collection(name="answer_keys")
            a_data = a_collection.get()
            a_count = len(a_data['ids'])
            
            print(f"✍️  Answers in database: {a_count}")
            if a_count == 0:
                print("   ⚠️  Database is EMPTY - No answers stored!")
            else:
                print(f"   ✅ Found answers: {a_data['ids']}")
        except:
            print("   ⚠️  Answer collection not found")
            a_count = 0
            
        return q_count > 0 or a_count > 0
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def test_extraction():
    """Test PDF extraction with current patterns"""
    print_section("TESTING EXTRACTION PATTERNS")
    
    try:
        from pdf_processor import PDFProcessor
        
        processor = PDFProcessor(use_deepseek=False)
        upload_dir = './uploads'
        
        # Test question paper
        q_files = [f for f in os.listdir(upload_dir) if f.startswith('question_paper_')]
        if q_files:
            q_file = os.path.join(upload_dir, q_files[0])
            print(f"\n📄 Testing: {q_files[0]}")
            
            questions = processor.extract_questions_with_marks(q_file)
            print(f"   Extracted: {len(questions)} questions")
            
            if questions:
                for q in questions[:3]:  # Show first 3
                    print(f"   ✅ Q{q['question_number']}: {q['question_text'][:60]}... ({q['max_marks']}m)")
            else:
                print("   ⚠️  No questions matched the extraction patterns!")
        
        # Test answer key
        a_files = [f for f in os.listdir(upload_dir) if f.startswith('answer_key_')]
        if a_files:
            a_file = os.path.join(upload_dir, a_files[0])
            print(f"\n📄 Testing: {a_files[0]}")
            
            answers = processor.extract_answers(a_file)
            print(f"   Extracted: {len(answers)} answers")
            
            if answers:
                for a in answers[:3]:  # Show first 3
                    print(f"   ✅ A{a['question_number']}: {a['answer_text'][:60]}...")
            else:
                print("   ⚠️  No answers matched the extraction patterns!")
                
    except Exception as e:
        print(f"❌ Error testing extraction: {e}")
        import traceback
        traceback.print_exc()

def provide_solutions():
    """Provide solutions based on findings"""
    print_section("RECOMMENDED SOLUTIONS")
    
    print("""
Based on the diagnosis above, here are the likely issues and solutions:

🔍 ISSUE 1: PDFs are SCANNED/IMAGES (No extractable text)
   SOLUTION:
   • Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
   • Or ensure DeepSeek-OCR is working (requires GPU)
   • Re-upload the PDFs after OCR is set up

🔍 ISSUE 2: PDFs don't match expected format
   SOLUTION:
   • Check TROUBLESHOOTING.md for correct format
   • Question format: Q1. Question text? (5 marks)
   • Answer format: Answer 1: Answer text...
   
🔍 ISSUE 3: Database is empty but extraction worked
   SOLUTION:
   • Re-upload the PDFs through the web interface
   • Check console logs for errors during storage
   
🔍 ISSUE 4: Extraction patterns don't match
   SOLUTION:
   • Share the first few lines of your PDF
   • Patterns can be customized in pdf_processor.py

📝 NEXT STEPS:
   1. Check the output above
   2. Identify which issue applies to you
   3. Follow the corresponding solution
   4. Re-test by uploading PDFs again
   
💡 TIP: Run 'python view_pdf_content.py' to see actual PDF text!
    """)

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║        AUTO-DIAGNOSTIC TOOL FOR ANSWER EVALUATION APP             ║
    ║                                                                   ║
    ║  This tool will check:                                           ║
    ║    • PDF content (text vs scanned)                               ║
    ║    • Database status                                             ║
    ║    • Extraction patterns                                         ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all checks
    has_text = check_pdf_content()
    has_data = check_database()
    test_extraction()
    provide_solutions()
    
    print_section("DIAGNOSIS COMPLETE")
    
    if not has_text:
        print("⚠️  PRIMARY ISSUE: Your PDFs appear to be scanned/images")
        print("    OCR is required to extract text from them.")
    elif not has_data:
        print("⚠️  PRIMARY ISSUE: Database is empty")
        print("    Extraction or storage is failing.")
    else:
        print("✅ PDFs have text and database has data")
        print("    The system should be working correctly.")

if __name__ == "__main__":
    main()
