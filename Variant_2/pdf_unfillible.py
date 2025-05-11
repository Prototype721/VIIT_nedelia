import fitz
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


pdf_file = "pdf_form.pdf"


searching_words = {"name", "date", "signature", "address", "phone"}


def extract_and_show_fields_with_boxes(pdf_path):
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        img_np = np.array(img_pil)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store line boxes
        line_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # OCR for labels
        data = pytesseract.image_to_data(img_pil, output_type=Output.DICT)
        draw = ImageDraw.Draw(img_pil)

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word == '':
                continue
            
            #is_probably_label = pipeline.predict([[word, x, y, width, height]])[0]


            if word.endswith(":") or word.lower() in searching_words:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                label_box = (x, y, x + w, y + h)
                draw.rectangle(label_box, outline="red", width=2)
                draw.text((x, y - 12), word, fill="red")
                candidates = []
                # Try to find a nearby line to the right
                for lx, ly, lw, lh in line_boxes:
                    if (lx > x + w ):
                        candidates.append((lx, ly, lw, lh))
                if candidates:
                    longest = max(candidates, key=lambda b: b[2])
                    lx, ly, lw, lh = longest
                    draw.rectangle([lx, ly-h, lx + lw, ly + lh], outline="green", width=2)
                

        # Show image
        plt.figure(figsize=(10, 12))
        plt.imshow(img_pil)
        plt.axis('off')
        plt.title(f'Page {page_num + 1} - Label + Fill Field Detection')
        plt.show()





extract_and_show_fields_with_boxes(pdf_file)