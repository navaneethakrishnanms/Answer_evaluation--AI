import PyPDF2
import re
from PIL import Image
import io
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not available - DeepSeek-OCR will be used exclusively")

from pdf2image import convert_from_path

# Import poppler configuration
try:
    from poppler_config import POPPLER_PATH
except ImportError:
    POPPLER_PATH = None

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab not available - PDF generation will be limited")

from datetime import datetime
import os
from deepseek_ocr import get_deepseek_ocr

class PDFProcessor:
    def __init__(self, use_deepseek=True):
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
        else:
            self.styles = None
        
        self.use_deepseek = use_deepseek
        self.deepseek_ocr = None
        
        # Initialize DeepSeek OCR if enabled
        if self.use_deepseek:
            try:
                print("⏳ Initializing DeepSeek-OCR (this will happen once at startup)...")
                self.deepseek_ocr = get_deepseek_ocr(auto_init=True)
                if self.deepseek_ocr.initialized:
                    print("✅ DeepSeek-OCR initialized and ready for handwritten text extraction")
                else:
                    print("⚠️ DeepSeek-OCR created but not yet initialized (will initialize on first use)")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize DeepSeek-OCR: {e}")
                if TESSERACT_AVAILABLE:
                    print("   Falling back to Tesseract OCR")
                    self.use_deepseek = False
                else:
                    print("   ERROR: No OCR system available!")
                    raise RuntimeError("Neither DeepSeek-OCR nor Tesseract is available")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def ocr_pdf(self, pdf_path):
        """Extract text from scanned PDF using OCR (Tesseract fallback)"""
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract OCR not available")
            return ""
        
        try:
            # Use poppler path from config
            if POPPLER_PATH and os.path.exists(POPPLER_PATH):
                images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
            else:
                images = convert_from_path(pdf_path, dpi=300)
            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
            return text
        except Exception as e:
            print(f"Error performing OCR: {str(e)}")
            return ""
    
    def extract_text_with_deepseek(self, pdf_path):
        """Extract text using DeepSeek-OCR (for handwritten content)"""
        try:
            if self.deepseek_ocr is None:
                print("⚠️ DeepSeek-OCR not available, using fallback")
                return self.ocr_pdf(pdf_path)
            
            print(f"🧠 Using DeepSeek-OCR for text extraction: {os.path.basename(pdf_path)}")
            text = self.deepseek_ocr.extract_text_from_pdf(pdf_path)
            return text
        except Exception as e:
            print(f"❌ DeepSeek-OCR failed: {e}")
            print("   Falling back to Tesseract OCR")
            return self.ocr_pdf(pdf_path)
    
    def extract_questions_with_marks(self, pdf_path):
        """
        Extract questions from question paper with their marks
        Now handles DeepSeek-OCR table format output
        """
        # First try regular text extraction (for typed PDFs)
        text = self.extract_text_from_pdf(pdf_path)
        
        # If text extraction failed or returned very little, try OCR
        if not text.strip() or len(text) < 50:
            print("📄 Question paper appears to be scanned/handwritten, using OCR...")
            text = self.extract_text_with_deepseek(pdf_path)
        
        questions_data = []
        
        # Clean HTML/XML tags from OCR output
        clean_text = re.sub(r'<[^>]+>', ' ', text)  # Remove all tags
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
        
        print("🔍 Searching for questions with multiple pattern strategies...")
        
        # Strategy 1: Match questions in format like (i), (ii), A1 (i), etc with marks
        pattern1 = r'[A-C]?\d*\s*\(([ivxIVX]+)\)\s*(.*?)\((\d+)\s*[Mm]ark'
        matches1 = re.finditer(pattern1, clean_text, re.IGNORECASE | re.DOTALL)
        for idx, match in enumerate(matches1, 1):
            try:
                # Convert roman numerals to numbers
                roman = match.group(1).lower()
                roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
                q_num = roman_map.get(roman, idx)
                
                q_text = match.group(2).strip()
                marks = int(match.group(3))
                
                # Clean up question text
                q_text = re.sub(r'\s+', ' ', q_text)
                q_text = q_text[:1000]  # Allow longer text
                
                if q_text and len(q_text) > 10 and marks > 0:
                    questions_data.append({
                        'question_number': q_num,
                        'question_text': q_text,
                        'max_marks': marks
                    })
                    print(f"  ✓ Found Q{q_num}: {marks} marks")
            except Exception as e:
                print(f"  ✗ Error parsing match: {e}")
                continue
        
        # Strategy 2: Traditional patterns
        traditional_patterns = [
            r'Q[\s]*(\d+)[\.\):\s]+(.*?)\s*[\(\[](\d+)\s*[Mm]ark',
            r'(\d+)[\.\):\s]+(.*?)\s*[\(\[](\d+)\s*[Mm]ark',
            r'Question[\s]+(\d+)[\.\):\s]+(.*?)\s*[\(\[](\d+)\s*[Mm]ark',
        ]
        
        for pattern in traditional_patterns:
            matches = re.finditer(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    q_num = int(match.group(1))
                    q_text = match.group(2).strip()
                    marks = int(match.group(3))
                    
                    # Clean up question text
                    q_text = re.sub(r'\s+', ' ', q_text)
                    q_text = q_text[:1000]
                    
                    if q_text and len(q_text) > 10 and marks > 0:
                        questions_data.append({
                            'question_number': q_num,
                            'question_text': q_text,
                            'max_marks': marks
                        })
                        print(f"  ✓ Found Q{q_num}: {marks} marks")
                except:
                    continue
        
        # Sort by question number
        questions_data = sorted(questions_data, key=lambda x: x['question_number'])
        
        # Remove duplicates - keep the one with longer question text
        unique_questions = {}
        for q in questions_data:
            q_num = q['question_number']
            if q_num not in unique_questions or len(q['question_text']) > len(unique_questions[q_num]['question_text']):
                unique_questions[q_num] = q
        
        questions_data = list(unique_questions.values())
        
        print(f"✅ Extracted {len(questions_data)} questions from question paper")
        
        # Debug: Show sample if found
        if questions_data:
            print(f"📋 Sample question: Q{questions_data[0]['question_number']} - {questions_data[0]['question_text'][:100]}...")
        else:
            print("⚠️ No questions found! Showing first 500 chars of extracted text:")
            print(clean_text[:500])
        
        return questions_data
    
    def extract_answers(self, pdf_path):
        """Extract answers from answer key PDF"""
        # First try regular text extraction
        text = self.extract_text_from_pdf(pdf_path)
        
        # If text extraction failed or returned very little, try OCR
        if not text.strip() or len(text) < 50:
            print("📄 Answer key appears to be scanned/handwritten, using OCR...")
            text = self.extract_text_with_deepseek(pdf_path)
        
        # Clean HTML/XML tags
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        answers_data = []
        
        print("🔍 Searching for answers...")
        
        # Strategy 1: Match table format answers like (i) answer text
        pattern1 = r'\(([ivxIVX]+)\)\s*(.*?)(?=\([ivxIVX]+\)|Section|$)'
        matches1 = re.finditer(pattern1, clean_text, re.IGNORECASE | re.DOTALL)
        for idx, match in enumerate(matches1, 1):
            try:
                roman = match.group(1).lower()
                roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
                q_num = roman_map.get(roman, idx)
                
                answer_text = match.group(2).strip()
                answer_text = re.sub(r'\s+', ' ', answer_text)
                
                if answer_text and len(answer_text) > 15:
                    answers_data.append({
                        'question_number': q_num,
                        'answer_text': answer_text
                    })
                    print(f"  ✓ Found A{q_num}")
            except:
                continue
        
        # Strategy 2: Traditional patterns
        traditional_patterns = [
            r'(?:Answer|Ans|A)[\s]*(\d+)[\.\):\s]+(.*?)(?=(?:Answer|Ans|A)[\s]*\d+|Section|\Z)',
            r'Q[\s]*(\d+)[\.\):\s]+(?:Answer|Ans|A)?[\s]*(.*?)(?=Q[\s]*\d+|Section|\Z)',
            r'(\d+)[\.\):\s]+(.*?)(?=\d+[\.\)]|Section|\Z)',
        ]
        
        for pattern in traditional_patterns:
            matches = re.finditer(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    q_num = int(match.group(1))
                    answer_text = match.group(2).strip()
                    answer_text = re.sub(r'\s+', ' ', answer_text)
                    
                    if answer_text and len(answer_text) > 15:
                        answers_data.append({
                            'question_number': q_num,
                            'answer_text': answer_text
                        })
                        print(f"  ✓ Found A{q_num}")
                except:
                    continue
        
        # Sort by question number
        answers_data = sorted(answers_data, key=lambda x: x['question_number'])
        
        # Remove duplicates - keep the longest answer for each question
        unique_answers = {}
        for a in answers_data:
            q_num = a['question_number']
            if q_num not in unique_answers or len(a['answer_text']) > len(unique_answers[q_num]['answer_text']):
                unique_answers[q_num] = a
        
        answers_data = list(unique_answers.values())
        
        print(f"✅ Extracted {len(answers_data)} answers from answer key")
        
        # Debug: Show all extracted answers
        for ans in answers_data:
            print(f"   A{ans['question_number']}: {ans['answer_text'][:80]}...")
        
        if answers_data:
            print(f"📋 Sample answer: A{answers_data[0]['question_number']} - {answers_data[0]['answer_text'][:100]}...")
        else:
            print("⚠️ No answers found! Showing first 500 chars:")
            print(clean_text[:500])
        
        return answers_data
    
    def extract_student_answers(self, pdf_source):
        """
        Extract student's handwritten answers using DeepSeek-OCR
        Accepts either a PDF file path or pre-extracted text content
        """
        try:
            text = ""
            source_desc = "provided text"
            
            if isinstance(pdf_source, str) and os.path.exists(pdf_source):
                pdf_path = pdf_source
                source_desc = os.path.basename(pdf_path) or pdf_path
                print(f"📝 Extracting student answers from file: {source_desc}")
                
                # Use DeepSeek-OCR for handwritten content
                if self.use_deepseek and self.deepseek_ocr is not None:
                    print("🧠 Using DeepSeek-OCR for handwritten text extraction...")
                    text = self.deepseek_ocr.extract_text_from_pdf(pdf_path)
                else:
                    print("⚠️ Using Tesseract OCR (fallback)...")
                    text = self.ocr_pdf(pdf_path)
            elif isinstance(pdf_source, str):
                text = pdf_source
                print("📝 Extracting student answers from OCR text input")
            else:
                print("⚠️ Unsupported student answer source type. Expected file path or text string.")
                return []
            
            if not text or len(text.strip()) < 20:
                print("⚠️ Warning: Very little text extracted from student paper")
                print(f"📝 Extracted text preview: {text[:200] if text else ''}")
                return []
            
            print(f"📝 OCR extracted text preview (first 400 chars):\n{text[:400]}\n")
            
            # Clean HTML/XML tags but PRESERVE newlines for answer separation
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'-{2,}\s*Page\s*\d+\s*-{2,}', '\n', clean_text, flags=re.IGNORECASE)
            
            # Normalize spaces within lines but keep newlines
            lines = clean_text.split('\n')
            clean_text = '\n'.join(re.sub(r'[ \t]+', ' ', line) for line in lines)
            
            student_answers = []
            
            print(f"🔍 Searching for answers in cleaned text (first 400 chars):\n{clean_text[:400]}\n")
            
            # Pattern to match student answer format
            patterns = [
                r'(?:Answer|Ans|A)[\s]*(\d+)[\.\):\s]+(.*?)(?=(?:Answer|Ans|A)[\s]*\d+|\Z)',
                r'Q[\s]*(\d+)[\.\):\s]+(.*?)(?=Q[\s]*\d+|\Z)',
                r'(\d+)[\.\):\s]+(.*?)(?=\d+[\.\)]|\Z)',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, clean_text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    try:
                        q_num = int(match.group(1))
                        answer_text = match.group(2).strip()
                        # Normalize spaces but preserve structure
                        answer_text = re.sub(r'[ \t]+', ' ', answer_text)
                        answer_text = re.sub(r'\n+', '\n', answer_text)
                        
                        if answer_text and len(answer_text) > 10:
                            student_answers.append({
                                'question_number': q_num,
                                'student_answer': answer_text
                            })
                            print(f"  ✓ Found student A{q_num}: {answer_text[:60]}...")
                    except:
                        continue
            
            # Sort by question number
            student_answers = sorted(student_answers, key=lambda x: x['question_number'])
            
            # Remove duplicates - keep the longest answer
            unique_answers = {}
            for a in student_answers:
                q_num = a['question_number']
                if q_num not in unique_answers or len(a['student_answer']) > len(unique_answers[q_num]['student_answer']):
                    unique_answers[q_num] = a
            
            student_answers = list(unique_answers.values())
            
            print(f"✅ Extracted {len(student_answers)} student answers")
            return student_answers
        
        except Exception as e:
            print(f"❌ Error extracting student answers: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # Keep all other methods unchanged (generate_result_pdf, _calculate_grade, etc.)
    def generate_result_pdf(self, evaluation_result, output_path):
        """Generate a detailed result PDF"""
        if not REPORTLAB_AVAILABLE:
            print("❌ reportlab not available - cannot generate PDF")
            self._generate_text_report(evaluation_result, output_path.replace('.pdf', '.txt'))
            return
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                   rightMargin=0.5*inch, leftMargin=0.5*inch,
                                   topMargin=0.75*inch, bottomMargin=0.5*inch)
            
            story = []
            title_style = ParagraphStyle('CustomTitle', parent=self.styles['Heading1'],
                fontSize=20, textColor=colors.HexColor('#1a237e'), spaceAfter=20, alignment=1)
            
            title = Paragraph("📝 Answer Paper Evaluation Report", title_style)
            story.append(title)
            story.append(Spacer(1, 0.3*inch))
            
            summary_data = [
                ['Evaluation Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Marks', str(evaluation_result['total_marks'])],
                ['Obtained Marks', str(evaluation_result['obtained_marks'])],
                ['Percentage', f"{evaluation_result['percentage']:.2f}%"],
                ['Grade', self._calculate_grade(evaluation_result['percentage'])]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
                ('BACKGROUND', (1, 0), (1, -1), colors.white),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 0.5*inch))
            
            heading_style = ParagraphStyle('CustomHeading', parent=self.styles['Heading2'],
                fontSize=14, textColor=colors.HexColor('#1a237e'), spaceAfter=15)
            
            heading = Paragraph("Question-wise Evaluation", heading_style)
            story.append(heading)
            
            question_data = [['Q. No.', 'Max Marks', 'Obtained', 'Percentage', 'Remarks']]
            
            for q in evaluation_result['question_wise_marks']:
                q_num = q['question_number']
                max_marks = q['max_marks']
                obtained = q['marks_obtained']
                percentage = (obtained / max_marks * 100) if max_marks > 0 else 0
                remarks = self._get_remarks(percentage)
                
                question_data.append([
                    str(q_num), str(max_marks), f"{obtained:.2f}",
                    f"{percentage:.1f}%", remarks
                ])
            
            question_table = Table(question_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.3*inch, 2*inch])
            question_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            for i in range(1, len(question_data)):
                if i % 2 == 0:
                    question_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f5f5f5'))
                    ]))
            
            story.append(question_table)
            story.append(Spacer(1, 0.3*inch))
            
            footer_style = ParagraphStyle('Footer', parent=self.styles['Normal'],
                fontSize=9, textColor=colors.grey, alignment=1)
            footer_text = "Generated by AI-Powered Answer Evaluation System using DeepSeek-OCR"
            footer = Paragraph(footer_text, footer_style)
            story.append(footer)
            
            doc.build(story)
            print(f"✅ Result PDF generated successfully: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating result PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_grade(self, percentage):
        """Calculate grade based on percentage"""
        if percentage >= 90: return 'A+'
        elif percentage >= 80: return 'A'
        elif percentage >= 70: return 'B+'
        elif percentage >= 60: return 'B'
        elif percentage >= 50: return 'C'
        elif percentage >= 40: return 'D'
        else: return 'F'
    
    def _get_remarks(self, percentage):
        """Get remarks based on percentage"""
        if percentage >= 90: return 'Excellent'
        elif percentage >= 75: return 'Very Good'
        elif percentage >= 60: return 'Good'
        elif percentage >= 50: return 'Average'
        else: return 'Needs Improvement'
    
    def _generate_text_report(self, evaluation_result, output_path):
        """Generate a simple text report when reportlab is not available"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("ANSWER PAPER EVALUATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Marks: {evaluation_result['total_marks']}\n")
                f.write(f"Obtained Marks: {evaluation_result['obtained_marks']}\n")
                f.write(f"Percentage: {evaluation_result['percentage']:.2f}%\n")
                f.write(f"Grade: {self._calculate_grade(evaluation_result['percentage'])}\n")
                f.write("\n" + "-"*60 + "\n")
                f.write("QUESTION-WISE EVALUATION\n")
                f.write("-"*60 + "\n\n")
                
                for q in evaluation_result['question_wise_marks']:
                    q_num = q['question_number']
                    max_marks = q['max_marks']
                    obtained = q['marks_obtained']
                    percentage = (obtained / max_marks * 100) if max_marks > 0 else 0
                    
                    f.write(f"Question {q_num}:\n")
                    f.write(f"  Max Marks: {max_marks}\n")
                    f.write(f"  Obtained: {obtained:.2f}\n")
                    f.write(f"  Percentage: {percentage:.1f}%\n")
                    f.write(f"  Remarks: {self._get_remarks(percentage)}\n\n")
                
                f.write("="*60 + "\n")
                f.write("Generated by AI-Powered Answer Evaluation System (DeepSeek-OCR)\n")
            
            print(f"✅ Text report generated: {output_path}")
        except Exception as e:
            print(f"❌ Error generating text report: {e}")
