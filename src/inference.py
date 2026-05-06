import torch
import os
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz
from PIL import Image
from parse_and_clean import process_and_save_results, JSONSaverWithImages, get_scaled_image_with_factor, scale_bboxes_to_original

# ============================================================= #
#                         PDF utils                             #
# ============================================================= #

def get_matrix(page, dpi_default=200, max_pixels=50000000):
    rect = page.rect
    if rect.width * rect.height > max_pixels:
        factor = (max_pixels / (rect.width * rect.height)) ** 0.5
    else:
        factor = dpi_default / 72
    return fitz.Matrix(factor, factor)

def is_page_safe_to_render(page, max_image_pixels=100_000_000):
    image_list = page.get_images(full=True)
    if not image_list:
        return True, "OK"

    for img in image_list:
        xref = img[0]
        if xref == 0:
            continue
        width = img[2]
        height = img[3]
        if width * height > max_image_pixels:
            return False, "Image too large"

    return True, "OK"

def fitz_doc_to_image(page, target_dpi=200):
    """
    Convert PDF page to image with scaling safeguard.
    Returns the image and the scale factor applied.
    """
    mat = get_matrix(page, target_dpi)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    if pm.width == 0 or pm.height == 0:
        return None, 1.0
    
    scale_factor = 1.0

    # Safeguard for very large pages (over 50 million pixels)
    """
    if pm.width > 4500 or pm.height > 4500:
        mat = fitz.Matrix(1, 1)  # Use default 72 DPI
        pm = page.get_pixmap(matrix=mat, alpha=False)
        scale_factor = 72.0 / target_dpi  # Calculate scaling factor
    """

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image, scale_factor

def load_images_from_pdf(pdf_file):
    images = []
    with fitz.open(pdf_file) as doc:
        for i in range(doc.page_count):
            page = doc[i]
            is_safe, _ = is_page_safe_to_render(page)
            if not is_safe:
                print(f"Skipping unsafe page {i}")
                continue
            img, scale_factor = fitz_doc_to_image(page)
            if img:
                images.append((i, img, scale_factor))
    return images

# ============================================================= #
#                     Model loading!                            #
# ============================================================= #

model_path = "./weights/DotsMOCR"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded successfully!")

# ============================================================= #
#                        Prompts!                            #
# ============================================================= #
prompt  = """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."""
prompt2 = """Extract the text content from this image."""
prompt3 = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# ============================================================= #
#                         Inference                             #
# ============================================================= #

def run_inference_on_image(image_input):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=24000)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print("\nOUTPUT TEXT\n")
    print(output_text[0])
    print()
    return output_text[0]  # Returns the string directly.

# ============================================================= #
#               Parsing and output cleaning.                    #
# ============================================================= #

def parse_model_output(output_text: str):
    """
    Parses the model output into a list of dictionaries.
    Uses the OutputCleaner from json_saver_with_images.
    """
    from json_saver_with_images import OutputCleaner
    
    cleaner = OutputCleaner()
    cleaned_data = cleaner.clean_model_output(output_text)
    
    return cleaned_data

# ============================================================= #
#                     Main processing                           #
# ============================================================= #

def run(file_path, output_dir="output"):
    """
    Main processing function that saves results as JSON and images.
    
    Args:
        file_path: Path to input PDF or image
        output_dir: Base output directory (default: "output")
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Result storing for each page.
    pages_data = []

    # ---- IMAGE CASE ----
    if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        print("Processing single image...")
        
        # Load and scale image (if needed). 
        original_image = Image.open(file_path)
        
        #scaled_image, scale_factor = get_scaled_image_with_factor(original_image)
        
        scaled_image = original_image
        scale_factor = 1.0

        #if scale_factor < 1.0:
        #    print(f"  Image downscaled by factor: {scale_factor:.3f}")
        
        # Inference.
        output_text = run_inference_on_image(scaled_image)
        
        # Parses and cleans the output.
        cleaned_output = parse_model_output(output_text)
        
        # Scales bounding boxes back to original image dimensions.
        #final_output = scale_bboxes_to_original(cleaned_output, scale_factor)
        
        final_output = cleaned_output

        # Stores page data with original image.
        pages_data.append({
            'image': original_image,
            'output': final_output
        })
        
        print(f"  Extracted {len(final_output)} layout elements")

    # ---- PDF CASE ----
    elif file_path.lower().endswith(".pdf"):
        print("Processing PDF...")
        pages = load_images_from_pdf(file_path)
        
        for page_idx, image, scale_factor in pages:
            print(f"\n--- Processing Page {page_idx} ---")
            
            if scale_factor < 1.0:
                print(f"  Image downscaled by factor: {scale_factor:.3f}")
            
            # Inference.
            output_text = run_inference_on_image(image)
            
            # Parses and cleans the output.
            cleaned_output = parse_model_output(output_text)
            
            # Scales bounding boxes back to original dimensions.
            final_output = scale_bboxes_to_original(cleaned_output, scale_factor)
            
            # Note: For PDF, 'image' is already the scaled version for inference
            # We need to get the original-sized image for visualization.
            if scale_factor < 1.0:
                # Re-render at original size for visualization.
                with fitz.open(file_path) as doc:
                    page = doc[page_idx]
                    mat_original = get_matrix(page, 200)  # Use original target DPI
                    pm_original = page.get_pixmap(matrix=mat_original, alpha=False)
                    original_image = Image.frombytes('RGB', (pm_original.width, pm_original.height), pm_original.samples)
            else:
                original_image = image
            
            # Store page data with original image.
            pages_data.append({
                'image': original_image,
                'output': final_output
            })
            
            print(f"  Page {page_idx}: Extracted {len(final_output)} layout elements")

    else:
        raise ValueError("Unsupported file type")

    # Saves all results!
    output_folder = process_and_save_results(file_path, pages_data, output_dir)
    
    return output_folder

# ============================================================= #
#                           Main!                               #
# ============================================================= #

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <file_path> [output_dir]")
        print("  file_path: Path to PDF or image file")
        print("  output_dir: Optional output directory (default: 'output')")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    try:
        result_folder = run(file_path, output_dir)
        print(f"\nProcessing complete!")
        print(f"Results saved in: {result_folder}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
