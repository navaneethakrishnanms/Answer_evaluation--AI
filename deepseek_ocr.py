# --- START OF CORRECTED FILE: deepseek_ocr.py ---

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os
from pdf2image import convert_from_path
import tempfile
import re  # For cleaning the output

# Import poppler configuration
try:
    from poppler_config import POPPLER_PATH
except ImportError:
    POPPLER_PATH = None

class DeepSeekOCR:
    """
    DeepSeek-OCR wrapper for extracting text from handwritten documents
    """
    def __init__(self, auto_init=False):
        self.model = None
        self.tokenizer = None
        self.model_name = "deepseek-ai/DeepSeek-OCR"
        self.initialized = False
        
        if auto_init:
            self.initialize()
            
    def initialize(self):
        """Initialize the model (lazy loading) with better error handling"""
        if self.initialized:
            return True
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("⚠️ Warning: No GPU detected. DeepSeek-OCR requires CUDA for optimal performance.")
                print("   Falling back to CPU (will be significantly slower)")
            else:
                print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                
            print("🔄 Loading DeepSeek-OCR tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded")
            
            print("📦 Loading DeepSeek-OCR model (this may take a few minutes)...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            }
            if device == "cuda":
                model_kwargs["device_map"] = "auto"

            try:
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs).eval()
            except Exception as e:
                print(f"⚠️ Standard loading failed, trying with eager attention: {e}")
                model_kwargs['_attn_implementation'] = 'eager'
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs).eval()
            
            self.initialized = True
            print("✅ DeepSeek-OCR initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing DeepSeek-OCR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _parse_output(self, raw_text: str) -> str:
        """
        Cleans the raw output from the model to extract only human-readable text.
        The model returns text in format: <|ref|>text<|/ref|><|det|>[[coords]]<|/det|>
        """
        if not raw_text:
            return ""
        
        import re
        
        # Remove debug tensor output lines
        lines = raw_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip debug lines
            if 'torch.Size' in line or line.strip().startswith('===') or 'BASE:' in line or 'PATCHES:' in line or 'NO PATCHES' in line:
                continue
            cleaned_lines.append(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Remove <|det|>...</|det|> blocks (coordinates)
        cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', cleaned, flags=re.DOTALL)
        
        # Remove <|ref|> and <|/ref|> tags
        cleaned = re.sub(r'<\|/?ref\|>', '', cleaned)
        
        # Remove standalone "text" on its own line (artifact)
        cleaned = re.sub(r'^\s*text\s*$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up excessive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a single image using the correct .infer() method and parse the result.
        """
        if not self.initialized:
            if not self.initialize():
                return ""
        
        try:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
            print(f"🚀 Running OCR on image: {os.path.basename(image_path)}")
            
            with tempfile.TemporaryDirectory() as temp_output_dir:
                # Capture stdout to get the printed text from the model
                import sys
                from io import StringIO
                
                captured_output = StringIO()
                old_stdout = sys.stdout
                sys.stdout = captured_output
                
                try:
                    # <-- CHANGE 1: Model prints output to stdout, capture it
                    raw_result = self.model.infer(
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=temp_output_dir,
                        base_size=1024,
                        crop_mode=True,
                        save_results=True  # Changed to True to potentially save output
                    )
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                
                # Get the captured text
                captured_text = captured_output.getvalue()
                
                print(f"🔍 DEBUG: Captured {len(captured_text)} chars from stdout")
                print(f"🔍 DEBUG: Return value type: {type(raw_result)}")
                
                # The text is in the captured output, not the return value
                result_str = captured_text if captured_text else ""
                
                print(f"🔍 DEBUG: Text preview: {result_str[:300] if result_str else 'EMPTY'}")
                
                cleaned_result = self._parse_output(result_str)
                print(f"🔍 DEBUG: After cleanup: {len(cleaned_result)} chars")
                print(f"🔍 DEBUG: Final preview: {cleaned_result[:200] if cleaned_result else 'EMPTY'}")

                if cleaned_result and len(cleaned_result) > 20:
                    print(f"✅ Extracted {len(cleaned_result)} characters")
                    return cleaned_result
                else:
                    print(f"⚠️ Insufficient text extracted ({len(cleaned_result)} chars), retrying...")
                    raise Exception("Empty OCR result")

        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")
            # Fallback retry logic
            try:
                print("🔄 Retry #1: Using minimal configuration...")
                prompt = "<image>\nExtract text."
                with tempfile.TemporaryDirectory() as temp_output_dir:
                    # Capture stdout for retry as well
                    import sys
                    from io import StringIO
                    
                    captured_retry = StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = captured_retry
                    
                    try:
                        raw_result_retry = self.model.infer(
                            tokenizer=self.tokenizer,
                            prompt=prompt,
                            image_file=image_path,
                            output_path=temp_output_dir,
                            base_size=640,
                            crop_mode=False,
                            save_results=True
                        )
                    finally:
                        sys.stdout = old_stdout
                    
                    retry_text = captured_retry.getvalue()
                    cleaned_result_retry = self._parse_output(retry_text)

                    if cleaned_result_retry:
                        print(f"✅ Retry successful: Extracted {len(cleaned_result_retry)} characters")
                        return cleaned_result_retry
            except Exception as e2:
                print(f"❌ All retry attempts failed: {e2}")
                import traceback
                traceback.print_exc()
            
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF by converting to images and using DeepSeek-OCR
        """
        if not self.initialized:
            if not self.initialize():
                return ""
        
        try:
            print(f"📄 Converting PDF to images: {os.path.basename(pdf_path)}")
            
            if POPPLER_PATH and os.path.exists(POPPLER_PATH):
                images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
            else:
                images = convert_from_path(pdf_path, dpi=300)
            
            all_text = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, image in enumerate(images, 1):
                    print(f"📄 Processing page {i}/{len(images)}...")
                    
                    temp_image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    image.save(temp_image_path, 'JPEG')
                    
                    page_text = self.extract_text_from_image(temp_image_path)
                    
                    if page_text:
                        all_text.append(page_text)
            
            combined_text = "\n\n".join(all_text)
            print(f"✅ Successfully extracted text from {len(images)} pages")
            
            return combined_text
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.initialized = False
        print("🧹 DeepSeek-OCR resources cleaned up")

_deepseek_ocr_instance = None

def get_deepseek_ocr(auto_init=False):
    """Get or create singleton DeepSeek-OCR instance"""
    global _deepseek_ocr_instance
    if _deepseek_ocr_instance is None:
        _deepseek_ocr_instance = DeepSeekOCR(auto_init=auto_init)
    return _deepseek_ocr_instance

# --- END OF CORRECTED FILE ---