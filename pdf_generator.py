"""
OCR Text to PDF Converter and Result PDF Generator
"""

from fpdf import FPDF
from datetime import datetime
from typing import List, Dict


class OCRtoPDFConverter:
    def __init__(self):
        self.pdf = None
    
    def create_text_pdf(self, extracted_text: str, output_path: str, 
                       title: str = "Student Answer Sheet (OCR Extracted)") -> str:
        print(f"📄 Creating text-based PDF from OCR output...")
        
        try:
            self.pdf = FPDF()
            self.pdf.set_auto_page_break(auto=True, margin=15)
            self.pdf.add_page()
            self.pdf.set_font("Arial", size=12)
            
            # Title
            self.pdf.set_font("Arial", "B", 16)
            self.pdf.cell(0, 10, title, 0, 1, "C")
            self.pdf.ln(5)
            
            # Date
            self.pdf.set_font("Arial", "I", 10)
            gen_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.pdf.cell(0, 5, f"Generated: {gen_date}", 0, 1, "C")
            self.pdf.ln(10)
            
            # Text content
            self.pdf.set_font("Arial", size=11)
            cleaned_text = self._clean_text(extracted_text)
            
            for line in cleaned_text.split('\n'):
                if line.strip():
                    self.pdf.multi_cell(0, 6, line.strip())
                else:
                    self.pdf.ln(3)
            
            self.pdf.output(output_path)
            print(f"✅ Text PDF created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating text PDF: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return "No text extracted"
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
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


class ResultPDFGenerator:
    def __init__(self):
        self.pdf = None
    
    def generate_result_pdf(self, evaluation_result: Dict, student_name: str = "Student",
                           output_path: str = "student_result.pdf") -> str:
        print(f"📊 Generating result PDF...")
        
        try:
            self.pdf = FPDF()
            self.pdf.set_auto_page_break(auto=True, margin=15)
            self.pdf.add_page()
            
            self._add_header(student_name)
            self._add_summary(evaluation_result)
            self._add_question_wise_results(evaluation_result['question_wise_results'])
            self._add_footer(evaluation_result)
            
            self.pdf.output(output_path)
            print(f"✅ Result PDF generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating result PDF: {str(e)}")
            raise
    
    def _add_header(self, student_name: str):
        self.pdf.set_font("Arial", "B", 20)
        self.pdf.set_text_color(0, 0, 128)
        self.pdf.cell(0, 15, "EVALUATION REPORT", 0, 1, "C")
        
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 10, f"Student: {student_name}", 0, 1, "L")
        
        self.pdf.set_font("Arial", size=10)
        eval_date = datetime.now().strftime("%B %d, %Y at %H:%M")
        self.pdf.cell(0, 5, f"Evaluated on: {eval_date}", 0, 1, "L")
        self.pdf.ln(10)
    
    def _add_summary(self, result: Dict):
        self.pdf.set_fill_color(240, 248, 255)
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "SUMMARY", 0, 1, "L", True)
        self.pdf.ln(2)
        
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(60, 8, "Total Marks:", 0, 0)
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 8, f"{result['obtained_marks']:.2f} / {result['total_marks']}", 0, 1)
        
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(60, 8, "Percentage:", 0, 0)
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 8, f"{result['percentage']:.2f}%", 0, 1)
        
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(60, 8, "Grade:", 0, 0)
        self.pdf.set_font("Arial", "B", 14)
        
        grade = result['grade']
        if grade in ['A+', 'A']:
            self.pdf.set_text_color(0, 128, 0)
        elif grade in ['B+', 'B']:
            self.pdf.set_text_color(0, 0, 255)
        elif grade == 'C':
            self.pdf.set_text_color(255, 140, 0)
        else:
            self.pdf.set_text_color(255, 0, 0)
        
        self.pdf.cell(0, 8, grade, 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.set_font("Arial", "I", 9)
        self.pdf.cell(0, 6, f"Method: {result.get('evaluation_method', 'AI-Based')}", 0, 1)
        self.pdf.ln(10)
    
    def _add_question_wise_results(self, results: List[Dict]):
        self.pdf.set_fill_color(240, 248, 255)
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "QUESTION-WISE BREAKDOWN", 0, 1, "L", True)
        self.pdf.ln(5)
        
        for result in results:
            q_num = result['Question_Number']
            awarded = result['Awarded_Marks']
            max_marks = result['Max_Marks']
            concept_score = result.get('Concept_Match_Score', 0) * 100
            feedback = result['Feedback']
            
            self.pdf.set_font("Arial", "B", 12)
            self.pdf.set_fill_color(245, 245, 245)
            self.pdf.cell(0, 8, f"Question {q_num}", 0, 1, "L", True)
            
            self.pdf.set_font("Arial", size=11)
            self.pdf.cell(50, 6, "Marks Obtained:", 0, 0)
            
            percentage = (awarded / max_marks * 100) if max_marks > 0 else 0
            if percentage >= 80:
                self.pdf.set_text_color(0, 128, 0)
            elif percentage >= 60:
                self.pdf.set_text_color(0, 0, 255)
            elif percentage >= 40:
                self.pdf.set_text_color(255, 140, 0)
            else:
                self.pdf.set_text_color(255, 0, 0)
            
            self.pdf.set_font("Arial", "B", 11)
            self.pdf.cell(0, 6, f"{awarded:.2f} / {max_marks}", 0, 1)
            self.pdf.set_text_color(0, 0, 0)
            
            self.pdf.set_font("Arial", size=11)
            self.pdf.cell(50, 6, "Concept Match:", 0, 0)
            self.pdf.cell(0, 6, f"{concept_score:.1f}%", 0, 1)
            
            self.pdf.set_font("Arial", "I", 10)
            self.pdf.multi_cell(0, 5, f"Feedback: {feedback}")
            self.pdf.ln(5)
    
    def _add_footer(self, result: Dict):
        self.pdf.ln(10)
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Remarks:", 0, 1)
        
        self.pdf.set_font("Arial", size=11)
        percentage = result['percentage']
        
        if percentage >= 90:
            remark = "Outstanding! Excellent conceptual understanding."
        elif percentage >= 80:
            remark = "Very good work! Strong grasp of concepts."
        elif percentage >= 70:
            remark = "Good performance. Most concepts well understood."
        elif percentage >= 60:
            remark = "Satisfactory. Some concepts need more practice."
        elif percentage >= 50:
            remark = "Average. Review key concepts and practice more."
        else:
            remark = "Needs improvement. Focus on core concepts."
        
        self.pdf.multi_cell(0, 6, remark)
        
        self.pdf.ln(10)
        self.pdf.set_font("Arial", "I", 9)
        self.pdf.set_text_color(128, 128, 128)
        self.pdf.multi_cell(0, 5, 
            "Note: AI-based concept evaluation. Review recommended for final grading."
        )
