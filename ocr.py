import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os

def evaluate_paper(image_path: str):
    """
    Evaluates a document image using DeepSeek-OCR and prints the extracted text.
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at path: {image_path}")
        return

    # 1. GPU check
    if not torch.cuda.is_available():
        print("❌ No NVIDIA GPU detected. This model requires CUDA.")
        return
    
    print("✅ GPU detected. Proceeding with evaluation.")

    # 2. Load tokenizer and model
    model_name = "deepseek-ai/DeepSeek-OCR"

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("📦 Loading model (this will download ~6.7 GB on first run)...")
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔧 Trying alternative loading method...")
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='eager',  # Fallback to eager attention if flash_attention_2 fails
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

    # 3. Prepare prompt
    print("🧩 Preparing OCR task...")
    # You can choose different prompts based on your needs:
    # - "<image>\nFree OCR." - Basic OCR
    # - "<image>\n<|grounding|>Convert the document to markdown." - Document to markdown
    # - "<image>\n<|grounding|>OCR this image." - General OCR with grounding
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Run inference using the model's built-in infer method
    print("🚀 Running OCR inference...")
    try:
        # DeepSeek-OCR has a custom infer() method that handles everything
        # Parameters:
        # - base_size: Controls resolution (512/640/1024/1280)
        # - image_size: Target size for processing
        # - crop_mode: Whether to use tiling for large images
        # - save_results: Save visualization and results
        # - test_compress: Test different compression ratios
        
        result = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,      # Base: standard resolution
            image_size=640,      # Processing size
            crop_mode=True,      # Use Gundam mode (tiling)
            save_results=True,   # Save outputs
            test_compress=False  # Don't test compression
        )

        print("\n" + "="*60)
        print("🧠 DEEPSEEK-OCR RESULT:")
        print("="*60)
        print(result)
        print("="*60)
        print(f"\n📁 Results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try simpler approach without save_results
        print("\n🔧 Retrying with simpler configuration...")
        try:
            result = model.infer(
                tokenizer=tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=640,       # Smaller base size
                image_size=640,
                crop_mode=False,     # No tiling
                save_results=False,  # Don't save
                test_compress=False
            )
            print("\n" + "="*60)
            print("🧠 DEEPSEEK-OCR RESULT:")
            print("="*60)
            print(result)
            print("="*60)
        except Exception as inner_e:
            print(f"❌ Retry also failed: {inner_e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    image_to_evaluate = "im.jpeg"
    evaluate_paper(image_to_evaluate)
