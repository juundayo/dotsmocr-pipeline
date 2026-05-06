# Script that saves model responses as clean JSON files with color-coded 
# bounding box images for visualization. Handles both PDF and image inputs
# and applies dynamic right padding to fix small model errors. Based on the 
# official dots mocr code at:
# https://github.com/rednote-hilab/dots.mocr/blob/main/dots_mocr/utils/layout_utils.py

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import fitz
from io import BytesIO
from datetime import datetime

# Colour map for different layout categories (RGBA format).
LAYOUT_TYPE_TO_COLOR = {
    "Text": (0, 128, 0, 256),            # Green
    "Picture": (255, 0, 255, 256),       # Magenta
    "Caption": (255, 165, 0, 256),       # Orange
    "Section-header": (0, 255, 255, 256),# Cyan
    "Footnote": (0, 128, 0, 256),        # Green
    "Formula": (128, 128, 128, 256),     # Gray
    "Table": (255, 192, 203, 256),       # Pink
    "Title": (255, 0, 0, 256),           # Red
    "List-item": (0, 0, 255, 256),       # Blue
    "Page-header": (0, 128, 0, 256),     # Green
    "Page-footer": (128, 0, 128, 256),   # Purple
    "Other": (165, 42, 42, 256),         # Brown
    "Unknown": (0, 0, 0, 0)
}


def get_scaled_image_with_factor(image: Image.Image, target_dpi: int = 200) -> Tuple[Image.Image, float]:
    """
    Scale down image if it exceeds 4500 pixels in either dimension.
    Returns the scaled image and the scaling factor used.
    
    Args:
        image: Input PIL Image
        target_dpi: Target DPI for rendering
        
    Returns:
        Tuple of (scaled_image, scale_factor)
    """
    # PIL to fitz page conversion for DPI-aware rendering.
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    pdf_bytes = fitz.open(stream=img_bytes).convert_to_pdf()
    doc = fitz.open('pdf', pdf_bytes)
    page = doc[0]
    
    # DPI scaling.
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    
    # Checks if image is too large and needs downscaling.
    scale_factor = 1.0
    if pm.width > 4500 or pm.height > 4500:
        # Default DPI (72).
        mat = fitz.Matrix(1.0, 1.0)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        # Calculates scaling factor between original and downscaled.
        scale_factor = 72.0 / target_dpi
    
    # Convert back to PIL Image.
    scaled_image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    
    return scaled_image, scale_factor


def scale_bboxes_to_original(bboxes: List[Dict], scale_factor: float) -> List[Dict]:
    """
    Scales bounding boxes from downscaled image coordinates back to original coordinates.
    
    Args:
        bboxes: List of bounding box dictionaries
        scale_factor: Scale factor to apply (original_size / scaled_size)
        
    Returns:
        List of scaled bounding boxes
    """
    if scale_factor == 1.0:
        return bboxes
    
    scaled_bboxes = []
    for bbox_dict in bboxes:
        bbox_dict_copy = bbox_dict.copy()
        if 'bbox' in bbox_dict_copy:
            original_bbox = bbox_dict_copy['bbox']
            # Scale up coordinates!
            scaled_bbox = [
                int(original_bbox[0] / scale_factor),
                int(original_bbox[1] / scale_factor),
                int(original_bbox[2] / scale_factor),
                int(original_bbox[3] / scale_factor)
            ]
            bbox_dict_copy['bbox'] = scaled_bbox
        scaled_bboxes.append(bbox_dict_copy)
    
    return scaled_bboxes

# ============================================================= #
#                         JSON cleaner.                         #
# ============================================================= #

class OutputCleaner:
    """Simplified version of the output cleaner for JSON validation."""
    
    def __init__(self):
        self.dict_pattern = re.compile(r'\{[^{}]*?"bbox"\s*:\s*\[[^\]]*?\][^{}]*?\}', re.DOTALL)
    
    def clean_model_output(self, model_output: str) -> List[Dict]:
        """Clean and validate model output"""
        try:
            # Try to parse as JSON directly.
            if isinstance(model_output, list):
                return self._clean_list_data(model_output)
            else:
                return self._clean_string_data(str(model_output))
        except Exception as e:
            print(f"⚠️ Cleaning failed: {e}")
            return []
    
    def _clean_list_data(self, data: List[Dict]) -> List[Dict]:
        """Clean list-type data"""
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Validates bbox if present.
            if 'bbox' in item:
                bbox = item['bbox']
                if isinstance(bbox, list) and len(bbox) == 4:
                    # Ensures coordinates are integers.
                    item['bbox'] = [int(x) for x in bbox]
                    cleaned.append(item)
                elif isinstance(bbox, list) and len(bbox) == 3:
                    # Removes invalid bbox but keep category and text.
                    new_item = {}
                    if 'category' in item:
                        new_item['category'] = item['category']
                    if 'text' in item:
                        new_item['text'] = item['text']
                    if new_item:
                        cleaned.append(new_item)
            else:
                # Keep items without bbox if they have category.
                if 'category' in item:
                    cleaned.append(item)
        
        return cleaned
    
    def _clean_string_data(self, data_str: str) -> List[Dict]:
        """Clean string-type data"""

        cleaned = []

        # -------------------------------
        # 1. Try FULL JSON parse first.
        # -------------------------------
        try:
            temp_str = data_str.strip()

            # Remove markdown code fences if present.
            if temp_str.startswith("```"):
                temp_str = re.sub(r"```[a-zA-Z]*", "", temp_str)
                temp_str = temp_str.replace("```", "").strip()

            parsed = json.loads(temp_str)

            # If it's a list, process normally.
            if isinstance(parsed, list):
                return self._clean_list_data(parsed)

            # If it's a single dict, wrap it.
            elif isinstance(parsed, dict):
                return self._clean_list_data([parsed])

        except Exception as e:
            print(f"⚠️ Full JSON parse failed, falling back to regex: {e}")

        # -------------------------------
        # 2. FALLBACK: original regex logic.
        # -------------------------------
        matches = self.dict_pattern.findall(data_str)

        for match in matches:
            try:
                # Try to fix common JSON issues.
                fixed = match.replace("'", '"')
                obj = json.loads(fixed)

                # Validate bbox.
                if 'bbox' in obj and isinstance(obj['bbox'], list):
                    if len(obj['bbox']) == 4:
                        obj['bbox'] = [int(x) for x in obj['bbox']]
                        cleaned.append(obj)
                    elif len(obj['bbox']) == 3:
                        # Remove invalid bbox.
                        new_obj = {}
                        if 'category' in obj:
                            new_obj['category'] = obj['category']
                        if 'text' in obj:
                            new_obj['text'] = obj['text']
                        if new_obj:
                            cleaned.append(new_obj)
                else:
                    cleaned.append(obj)

            except Exception as e:
                continue

        return cleaned

# ============================================================= #
#             Image drawing and JSON saving.                    #
# ============================================================= #

class JSONSaverWithImages:
    """Handles saving model output as clean JSON with visualization images."""
    
    def __init__(self, output_base_dir: str = "output"):
        self.output_base_dir = output_base_dir
        self.cleaner = OutputCleaner()
    
    def extract_base_name(self, file_path: str) -> str:
        """Extracts base name from file path (without extension)"""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def create_output_folder(self, base_name: str) -> str:
        """Creates output folder named after the input file"""
        folder_path = os.path.join(self.output_base_dir, base_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def get_category_number(self, category: str) -> int:
        """Convert category name to category number based on the encoding."""
        category_mapping = {
            "Text": 0,
            "Page-header": 1,
            "Section-header": 2,
            "Picture": 3,
            "Table": 4,
            "Caption": 5,
            "Page-footer": 6,
            # Additional mappings for other categories (default to 0 if not specified)
            "Title": 0,  # Treat as text
            "Footnote": 0,  # Treat as text
            "Formula": 0,  # Treat as text
            "List-item": 0,  # Treat as text
        }
        return category_mapping.get(category, 0)  # Default to 0 (text) if unknown

    def format_bounding_box(self, bbox: List[int]) -> str:
        """
        Convert bbox from [x1, y1, x2, y2] to polygon format: X1 Y1 X2 Y1 X2 Y2 X1 Y2
        """
        if len(bbox) != 4:
            return ""
        
        x1, y1, x2, y2 = bbox
        # Format: X1 Y1 X2 Y1 X2 Y2 X1 Y2
        return f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"

    def apply_right_padding(self, cells, image_width):
        """
        Expands bounding boxes slightly to the right using hybrid scaling:
        - proportional to box width
        - with a minimum based on image width
        """
        padded_cells = []

        for cell in cells:
            if 'bbox' not in cell:
                padded_cells.append(cell)
                continue

            x1, y1, x2, y2 = cell['bbox']

            box_width = x2 - x1
            delta = int(max(box_width * 0.02, image_width * 0.002))

            # Expand right side only, clamp to image boundary.
            new_x2 = min(image_width, x2 + delta)

            new_cell = cell.copy()
            new_cell['bbox'] = [x1, y1, new_x2, y2]

            padded_cells.append(new_cell)

        return padded_cells

    def save_json(self, data: List[Dict], folder_path: str, filename: str) -> str:
        """Saves data as JSON file (no re-cleaning if already processed)."""
        filepath = os.path.join(folder_path, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  💾 Saved JSON: {filename}")
        return filepath
    
    def draw_layout_on_image(self, image: Image.Image, cells: List[Dict]) -> Image.Image:
        """
        Draws color-coded bounding boxes on an image.
        
        Args:
            image: Source PIL Image
            cells: List of layout elements with bbox and category
            
        Returns:
            PIL Image with drawn boxes
        """
        if not cells:
            return image
        
        # Create a new PDF document for drawing.
        doc = fitz.open()
        
        # Convert PIL Image to bytes.
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create pixmap from image.
        pix = fitz.Pixmap(img_bytes)
        
        # Create page with image dimensions.
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
        
        # Draw each bounding box.
        for i, cell in enumerate(cells):
            if 'bbox' not in cell:
                continue
                
            bbox = cell['bbox']
            layout_type = cell.get('category', 'Unknown')
            
            # Get color for this layout type.
            color = LAYOUT_TYPE_TO_COLOR.get(layout_type, (0, 128, 0, 256))
            rgb_color = [c/255 for c in color[:3]]
            
            # Create rectangle.
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Draw filled rectangle with transparency.
            page.draw_rect(
                rect,
                color=None,
                fill=rgb_color,
                fill_opacity=0.3,
                width=0.5,
                overlay=True
            )
            
            # Draw border.
            page.draw_rect(
                rect,
                color=rgb_color,
                fill=None,
                width=1.5,
                overlay=True
            )
            
            # Add label with index and category.
            label = f"{i}_{layout_type}"
            text_x = bbox[0] + 5
            text_y = bbox[1] + 20
            
            # Draw text background for better visibility.
            text_rect = fitz.Rect(text_x - 2, text_y - 18, text_x + 200, text_y + 5)
            page.draw_rect(text_rect, color=None, fill=(1, 1, 1), fill_opacity=0.7)
            
            # Insert text.
            page.insert_text(
                (text_x, text_y),
                label,
                fontsize=12,
                color=rgb_color
            )
        
        # Convert back to PIL Image.
        mat = fitz.Matrix(1.0, 1.0)
        pix = page.get_pixmap(matrix=mat)
        
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    def save_image(self, image: Image.Image, cells: List[Dict], folder_path: str, filename: str) -> str:
        """Saves image with color-coded bounding boxes."""
        # Draw bounding boxes on image.
        annotated_image = self.draw_layout_on_image(image, cells)
        
        # Save the image.
        filepath = os.path.join(folder_path, filename)
        annotated_image.save(filepath, 'PNG')
        
        print(f"  🖼️ Saved image: {filename}")
        return filepath
    
    def save_single_page(self, 
                        page_data: Dict[str, Any], 
                        folder_path: str, 
                        page_num: int,
                        base_name: str) -> Tuple[str, str]:
        """
        Saves a single page's JSON and image.
        
        Args:
            page_data: Dictionary containing 'image' and 'output' (cleaned data)
            folder_path: Output folder path
            page_num: Page number (0-indexed)
            base_name: Base name for the file
            
        Returns:
            Tuple of (json_path, image_path)
        """
        # Create filename with leading zeros (e.g., 000, 001, etc.).
        filename_base = f"{base_name}_{page_num:03d}"
        json_filename = f"{filename_base}.json"
        image_filename = f"{filename_base}.png"
        
        # Apply dynamic right padding.
        image_width = page_data['image'].width
        padded_output = self.apply_right_padding(page_data['output'], image_width)

        # Save JSON with updated boxes.
        json_path = self.save_json(padded_output, folder_path, json_filename)

        # Save image with padded boxes.
        image_path = self.save_image(page_data['image'], padded_output, folder_path, image_filename)

        # Save the two txt files for OCR text and bounding box + category info.
        page_data_with_padded = page_data.copy()
        page_data_with_padded['output'] = padded_output
        self.save_txt_files(page_data_with_padded, folder_path, filename_base)
            
        return json_path, image_path
        
    def save_txt_files(self, page_data: Dict[str, Any], folder_path: str, filename_base: str):
        """
        Save two TXT files per page:
        1. Full OCR text (all text content concatenated)
        2. Bounding box info with category numbers in polygon format
        """
        # File 1: Full OCR text (filename_base_ocr.txt)
        ocr_text_path = os.path.join(folder_path, f"{filename_base}_ocr.txt")
        
        # Extract all text from the page output.
        all_texts = []
        for item in page_data['output']:
            if 'text' in item and item['text']:
                all_texts.append(item['text'])
        
        full_ocr_text = "\n\n".join(all_texts)  # Add double newline between regions.
        
        with open(ocr_text_path, 'w', encoding='utf-8') as f:
            f.write(full_ocr_text)
        
        print(f"  📝 Saved OCR text: {os.path.basename(ocr_text_path)}")
        
        # File 2: Bounding box info with categories (filename_base_boxes.txt)
        boxes_path = os.path.join(folder_path, f"{filename_base}_boxes.txt")
        
        with open(boxes_path, 'w', encoding='utf-8') as f:
            for item in page_data['output']:
                if 'bbox' in item:
                    category_num = self.get_category_number(item.get('category', 'Unknown'))
                    polygon_str = self.format_bounding_box(item['bbox'])
                    
                    if polygon_str:
                        f.write(f"{category_num} {polygon_str}\n")
        
        print(f"  📦 Saved bounding boxes: {os.path.basename(boxes_path)}")
    
    def save_metadata(self, folder_path: str, base_name: str, total_pages: int, file_type: str):
        """Saves metadata about the processing."""
        metadata = {
            'input_file': base_name,
            'file_type': file_type,
            'total_pages': total_pages,
            'processing_date': datetime.now().isoformat(),
            'output_folder': folder_path
        }
        
        metadata_path = os.path.join(folder_path, f"{base_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"  📋 Saved metadata: {os.path.basename(metadata_path)}")


def process_and_save_results(file_path: str, 
                            pages_data: List[Dict[str, Any]], 
                            output_dir: str = "output") -> str:
    """
    Main function to process and save results from the OCR pipeline.
    
    Args:
        file_path: Original input file path (PDF or image)
        pages_data: List of dictionaries containing 'image' and 'output' for each page
        output_dir: Base output directory
        
    Returns:
        Path to the output folder
    """
    saver = JSONSaverWithImages(output_dir)
    
    # Extract base name and create output folder.
    base_name = saver.extract_base_name(file_path)
    folder_path = saver.create_output_folder(base_name)
    
    # Determine file type.
    file_type = "PDF" if file_path.lower().endswith('.pdf') else "Image"
    
    print(f"\nSaving results to: {folder_path}")
    print(f"Processing {len(pages_data)} page(s) from {file_type}")
    
    # Save each page.
    for idx, page_data in enumerate(pages_data):
        print(f"\n  Page {idx + 1}/{len(pages_data)}:")
        saver.save_single_page(page_data, folder_path, idx, base_name)
    
    # Save metadata.
    saver.save_metadata(folder_path, base_name, len(pages_data), file_type)
    
    print(f"\nAll results saved successfully!")
    print(f"Output folder: {folder_path}")
    
    return folder_path

# ============================================================= #
#                            Main!                              #
# ============================================================= #

if __name__ == "__main__":
    print("JSON Saver with Images module loaded")
    print("Use process_and_save_results() to save OCR results")
