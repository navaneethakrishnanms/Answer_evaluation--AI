"""
OCR to Text PDF Converter - Converts handwritten student answers to searchable text PDFs
Uses DeepSeek-OCR for extraction and FPDF2 for text-based PDF generation
"""

from fpdf import FPDF
from datetime import datetime
import os
from deepseek_ocr import get_deepseek_ocr
import re


class HandwrittenToTextPDF:
    """
    Converts handwritten PDFs (from phone camera images) to searchable text PDFs
    """
    
    def __init__(self):
        self.deepseek_ocr = get_deepseek_ocr()
        print("✅ HandwrittenToTextPDF initialized with DeepSeek-OCR")
    
    def convert_handwritten_to_text_pdf(self, input_pdf_path: str, output_pdf_path: str, 
                                        student_name: str = "Student") -> dict:
        """
        Main function: Convert handwritten PDF to searchable text PDF
        
        Args:
            input_pdf_path: Path to handwritten PDF (from camera images)
            output_pdf_path: Path for output text PDF
            student_name: Student name for header
            
        Returns:
            dict: Status information including extracted text and answer count
        """
        
        print(f"\n{'='*70}")
        print(f"📝 Converting Handwritten PDF to Searchable Text PDF")
        print(f"{'='*70}")
        print(f"Input: {os.path.basename(input_pdf_path)}")
        print(f"Output: {os.path.basename(output_pdf_path)}")
        print(f"Student: {student_name}")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Extract text using DeepSeek-OCR
            print("🔍 Step 1: Extracting handwritten text using DeepSeek-OCR...")
            extracted_text = self.deepseek_ocr.extract_text_from_pdf(input_pdf_path)
            
            if not extracted_text or len(extracted_text) < 50:
                raise ValueError("Failed to extract sufficient text from PDF. Please ensure the PDF contains clear handwritten content.")
            
            print(f"✅ Extracted {len(extracted_text)} characters of text")
            
            # Step 2: Parse and structure the answers
            print("\n📋 Step 2: Parsing student answers from extracted text...")
            structured_answers = self._parse_student_answers(extracted_text)
            
            print(f"✅ Identified {len(structured_answers)} student answers")
            
            # Step 3: Create text-based PDF
            print("\n📄 Step 3: Creating searchable text PDF...")
            self._create_searchable_pdf(
                structured_answers=structured_answers,
                raw_text=extracted_text,
                output_path=output_pdf_path,
                student_name=student_name
            )
            
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! Searchable PDF created")
            print(f"{'='*70}")
            print(f"📊 Summary:")
            print(f"  - Answers detected: {len(structured_answers)}")
            print(f"  - Text extracted: {len(extracted_text)} characters")
            print(f"  - Output: {output_pdf_path}")
            print(f"{'='*70}\n")
            
            return {
                'success': True,
                'output_path': output_pdf_path,
                'answers_detected': len(structured_answers),
                'text_length': len(extracted_text),
                'structured_answers': structured_answers,
                'raw_text': extracted_text
            }
            
        except Exception as e:
            print(f"\n❌ Error converting PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_student_answers(self, text: str) -> list:
        """
        Parse student answers from extracted text
        Detects question numbers and groups answer text
        """
        
        # Clean text
        clean_text = re.sub(r'<[^>]+>', ' ', text)  # Remove XML/HTML tags
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
        
        answers = []
        
        # Multiple patterns to detect answers
        patterns = [
            # Pattern 1: Q1, Q.1, Question 1
            r'[Qq](?:uestion)?[\s\.]*(\d+)[\.\):\s]+(.*?)(?=[Qq](?:uestion)?[\s\.]*\d+|$)',
            # Pattern 2: Answer 1, Ans 1, A1
            r'[Aa](?:ns(?:wer)?)?[\s\.]*(\d+)[\.\):\s]+(.*?)(?=[Aa](?:ns(?:wer)?)?[\s\.]*\d+|$)',
            # Pattern 3: Simple number with period: 1., 2., etc.
            r'(\d+)[\.\)][\s]+(.*?)(?=\d+[\.\)][\s]|$)',
            # Pattern 4: Roman numerals: (i), (ii), etc.
            r'\(([ivxIVX]+)\)[\s]*(.*?)(?=\([ivxIVX]+\)|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    # Extract question number
                    q_num_str = match.group(1)
                    
                    # Convert roman numerals if needed
                    if q_num_str.lower() in ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']:
                        roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 
                                   'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
                        q_num = roman_map.get(q_num_str.lower(), 0)
                    else:
                        q_num = int(q_num_str)
                    
                    # Extract answer text
                    answer_text = match.group(2).strip()
                    answer_text = re.sub(r'\s+', ' ', answer_text)  # Clean spacing
                    
                    # Only add if substantial text
                    if answer_text and len(answer_text) >= 20:
                        answers.append({
                            'question_number': q_num,
                            'answer_text': answer_text[:2000]  # Limit length
                        })
                        print(f"  ✓ Found Q{q_num}: {len(answer_text)} characters")
                
                except Exception as e:
                    continue
        
        # Remove duplicates - keep longest answer for each question
        unique_answers = {}
        for ans in answers:
            q_num = ans['question_number']
            if q_num not in unique_answers or len(ans['answer_text']) > len(unique_answers[q_num]['answer_text']):
                unique_answers[q_num] = ans
        
        # Sort by question number
        final_answers = sorted(unique_answers.values(), key=lambda x: x['question_number'])
        
        return final_answers
    
    def _create_searchable_pdf(self, structured_answers: list, raw_text: str, 
                               output_path: str, student_name: str):
        """
        Create a searchable text-based PDF using FPDF2
        """
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(0, 0, 139)  # Dark blue
        pdf.cell(0, 12, "Student Answer Sheet", 0, 1, "C")
        
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Student Name: {student_name}", 0, 1, "L")
        
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Extracted on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 0, 1, "L")
        pdf.cell(0, 6, "Method: DeepSeek-OCR Handwriting Recognition", 0, 1, "L")
        pdf.ln(8)
        
        # Add structured answers
        if structured_answers:
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, "Extracted Answers:", 0, 1, "L")
            pdf.ln(2)
            
            for answer in structured_answers:
                q_num = answer['question_number']
                text = answer['answer_text']
                
                # Question number header
                pdf.set_fill_color(240, 248, 255)  # Light blue background
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 7, f"Question {q_num}", 0, 1, "L", True)
                pdf.ln(2)
                
                # Answer text
                pdf.set_font("Arial", size=10)
                # Handle long text with word wrapping
                pdf.multi_cell(0, 5, text)
                pdf.ln(5)
        
        # Add raw extracted text as appendix
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Appendix: Full Extracted Text", 0, 1, "L")
        pdf.ln(3)
        
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(80, 80, 80)
        
        # Clean and add raw text
        cleaned_raw = self._clean_text_for_pdf(raw_text)
        pdf.multi_cell(0, 4, cleaned_raw)
        
        # Save PDF
        pdf.output(output_path)
        print(f"✅ PDF saved: {output_path}")
    
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF display"""
        if not text:
            return "No text extracted"
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        
        # Remove excessive blank lines
        lines = text.split('\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append('')
                prev_blank = True
        
        return '\n'.join(cleaned_lines)


def main():
    """
    Example usage and testing
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_to_text_pdf.py <input_pdf> [output_pdf] [student_name]")
        print("\nExample:")
        print("  python ocr_to_text_pdf.py student_handwritten.pdf student_text.pdf 'John Doe'")
        return
    
    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else "output_student_text.pdf"
    student_name = sys.argv[3] if len(sys.argv) > 3 else "Student"
    
    if not os.path.exists(input_pdf):
        print(f"❌ Error: Input file not found: {input_pdf}")
        return
    
    converter = HandwrittenToTextPDF()
    result = converter.convert_handwritten_to_text_pdf(
        input_pdf_path=input_pdf,
        output_pdf_path=output_pdf,
        student_name=student_name
    )
    
    if result['success']:
        print(f"\n✅ Conversion successful!")
        print(f"Output file: {result['output_path']}")
        print(f"Answers detected: {result['answers_detected']}")
    else:
        print(f"\n❌ Conversion failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
