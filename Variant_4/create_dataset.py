import fitz
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np
import json
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
searching_words = {"name", "date", "signature", "address", "phone"}

def prepare_layoutlm_dataset(pdf_path, output_dir="layoutlm_data"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    dataset = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        img_np = np.array(img_pil)
        width, height = img_pil.size

        # Preprocess for line detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # OCR
        data = pytesseract.image_to_data(img_pil, output_type=Output.DICT)
        ocr_words = []
        ocr_boxes = []
        ocr_labels = []

        # Collect label/field and line boxes
        field_boxes = []  # red
        value_boxes = []  # green

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word == '':
                continue

            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if word.endswith(":") or word.lower() in searching_words:
                field_boxes.append((x, y, x + w, y + h))

                # Find longest line to the right
                candidates = []
                for lx, ly, lw, lh in line_boxes:
                    if lx > x + w and abs(ly - y) < h * 1.5 and lh <= h * 1.5:
                        candidates.append((lx, ly, lx + lw, ly + lh))
                if candidates:
                    longest = max(candidates, key=lambda b: b[2] - b[0])
                    value_boxes.append(longest)

        # Now label each OCR word with tag
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word == '' or int(data['conf'][i]) < 0:
                continue

            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            box = [int(1000 * x / width), int(1000 * y / height),
                   int(1000 * (x + w) / width), int(1000 * (y + h) / height)]
            abs_box = (x, y, x + w, y + h)

            label = "O"
            for fx1, fy1, fx2, fy2 in field_boxes:
                if overlaps(abs_box, (fx1, fy1, fx2, fy2)):
                    label = "B-FIELD"
                    break
            for vx1, vy1, vx2, vy2 in value_boxes:
                if overlaps(abs_box, (vx1, vy1, vx2, vy2)):
                    label = "B-VALUE"
                    break

            ocr_words.append(word)
            ocr_boxes.append(box)
            ocr_labels.append(label)

        # Save each page as one sample
        page_data = {
            "words": ocr_words,
            "boxes": ocr_boxes,
            "labels": ocr_labels,
            "page": page_num
        }

        dataset.append(page_data)

        img_pil.save(os.path.join(output_dir, f"page_{page_num+1}.png"))
        
        # Save to file per page (optional)
        with open(os.path.join(output_dir, f"page_{page_num+1}.json"), "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2)

    print(f"Dataset created with {len(dataset)} pages in '{output_dir}'")
    return dataset


def overlaps(box1, box2, threshold=0.5):
    """Check if two boxes overlap significantly"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    # Compute intersection
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)

    if area1 == 0:
        return False

    return inter_area / area1 > threshold




# Example usage
prepare_layoutlm_dataset("scanned_pdf.pdf")
