# PDF to PNG conversion script based on the official dots mocr code at:
# https://github.com/rednote-hilab/dots.mocr/blob/main/dots_mocr/utils/doc_utils.py

import fitz  # PyMuPDF

INPUT_PDF = r"C:\Users\anton\Desktop\パイソン２\0b60e7806c0e05d06263f3475b62d978a8fb2d42\0b60e7806c0e05d06263f3475b62d978a8fb2d42.pdf"
TARGET_DPI = 200

doc = fitz.open(INPUT_PDF)

print(f"Opened PDF: {INPUT_PDF}")
print(f"Total pages: {len(doc)}\n")

for i, page in enumerate(doc):
    print(f"--- Page {i} ---")

    # Step 1: OCR scaling.
    scale = TARGET_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    print(f"Initial render: {pix.width} x {pix.height}")

    # Step 2: Fallback logic for very large pages.
    if pix.width > 4500 or pix.height > 4500:
        print("⚠️ Fallback triggered (page too large)")
        mat = fitz.Matrix(1, 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)

    print(f"Final resolution: {pix.width} x {pix.height}")
    print(f"Total pixels: {pix.width * pix.height}")

    # Step 3: Save the PNG for each page!
    output_name = f"page_{i:03d}.png"
    pix.save(output_name)

    print(f"Saved: {output_name}\n")

doc.close()
