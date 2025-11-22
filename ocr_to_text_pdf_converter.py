"""
OCR to Text-Based PDF Converter
Extracts handwritten text using DeepSeek OCR and converts to searchable text PDF
"""

from fpdf import FPDF
import os
import tempfile
from deepseek_ocr import get_deepseek_ocr
import re


class OCRToTextPDFConverter:
    """Convert handwritten/scanned PDF to text-based PDF using DeepSeek OCR and FPDF2"""
    
    def __init__(self):
        self.deepseek_ocr = None
        
    def initialize_ocr(self):
        """Initialize DeepSeek OCR"""
        if self.deepseek_ocr is None:
            try:
                print("🔄 Initializing DeepSeek-OCR...")
                self.deepseek_ocr = get_deepseek_ocr()
                if not self.deepseek_ocr.initialize():
                    raise Exception("Failed to initialize DeepSeek-OCR")
                print("✅ DeepSeek-OCR ready")
            except Exception as e:
                print(f"❌ Error initializing DeepSeek-OCR: {e}")
                raise
    
    def extract_text_from_handwritten_pdf(self, pdf_path: str) -> str:
        """
        Extract text from handwritten/scanned PDF using DeepSeek OCR
        
        Args:
            pdf_path: Path to the handwritten PDF
            
        Returns:
            Extracted text content
        """
        self.initialize_ocr()
        
        print(f"\n{'='*60}")
        print(f"📝 Extracting handwritten text from: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        
        # Use DeepSeek OCR to extract text
        extracted_text = self.deepseek_ocr.extract_text_from_pdf(pdf_path)
        
        if not extracted_text or len(extracted_text.strip()) < 20:
            raise Exception("Very little or no text extracted from PDF. Please check if the PDF contains readable content.")
        
        # Clean up the extracted text
        cleaned_text = self._clean_extracted_text(extracted_text)
        
        print(f"✅ Successfully extracted {len(cleaned_text)} characters")
        print(f"📄 Preview: {cleaned_text[:200]}...\n")
        
        return cleaned_text
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive page markers
        text = re.sub(r'---\s*Page\s*\d+\s*---', '\n\n', text)
        
        # Clean up markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        
        return text.strip()
    
    def create_text_pdf(self, text_content: str, output_path: str):
        """
        Create a text-based PDF from extracted text using FPDF2
        
        Args:
            text_content: The text content to write to PDF
            output_path: Path where to save the text-based PDF
        """
        print(f"\n{'='*60}")
        print(f"📄 Creating text-based PDF...")
        print(f"{'='*60}\n")
        
        try:
            # Create PDF object
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Set font
            pdf.set_font("Arial", size=11)
            
            # Add title
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, 'Extracted Student Answers', ln=True, align='C')
            pdf.ln(5)
            
            # Reset to normal font
            pdf.set_font("Arial", size=11)
            
            # Split text into paragraphs and questions
            paragraphs = text_content.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if it's a question header
                if re.match(r'^(Question|Q\.?|Answer|Ans\.?)\s*\d+', para, re.IGNORECASE):
                    pdf.ln(3)
                    pdf.set_font("Arial", 'B', 12)
                    # Write question header
                    pdf.multi_cell(0, 8, para)
                    pdf.set_font("Arial", size=11)
                else:
                    # Regular text
                    pdf.multi_cell(0, 6, para)
                    pdf.ln(2)
            
            # Save the PDF
            pdf.output(output_path)
            
            print(f"✅ Text-based PDF created successfully!")
            print(f"📁 Saved to: {output_path}\n")
            
        except Exception as e:
            print(f"❌ Error creating text PDF: {e}")
            raise
    
    def convert_handwritten_to_text_pdf(self, input_pdf_path: str, output_pdf_path: str) -> str:
        """
        Complete conversion pipeline: Handwritten PDF -> OCR -> Text PDF
        
        Args:
            input_pdf_path: Path to handwritten/scanned PDF
            output_pdf_path: Path where to save the text-based PDF
            
        Returns:
            Path to the created text-based PDF
        """
        print(f"\n{'='*70}")
        print(f"🔄 HANDWRITTEN PDF TO TEXT PDF CONVERSION PIPELINE")
        print(f"{'='*70}\n")
        print(f"📥 Input:  {os.path.basename(input_pdf_path)}")
        print(f"📤 Output: {os.path.basename(output_pdf_path)}\n")
        
        try:
            # Step 1: Extract text using DeepSeek OCR
            print("STEP 1: Extracting handwritten text with DeepSeek OCR")
            print("-" * 70)
            extracted_text = self.extract_text_from_handwritten_pdf(input_pdf_path)
            
            if not extracted_text:
                raise Exception("No text could be extracted from the PDF")
            
            # Step 2: Create text-based PDF
            print("STEP 2: Creating searchable text-based PDF")
            print("-" * 70)
            self.create_text_pdf(extracted_text, output_pdf_path)
            
            print(f"\n{'='*70}")
            print(f"✅ CONVERSION COMPLETE!")
            print(f"{'='*70}")
            print(f"📄 Text-based PDF ready for evaluation")
            print(f"📁 Location: {output_pdf_path}\n")
            
            return output_pdf_path
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"❌ CONVERSION FAILED!")
            print(f"{'='*70}")
            print(f"Error: {str(e)}\n")
            raise


# Singleton instance
_converter_instance = None

def get_converter():
    """Get or create singleton converter instance"""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = OCRToTextPDFConverter()
    return _converter_instance
