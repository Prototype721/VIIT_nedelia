import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# Load the trained model (replace with your own model path)
pipeline = joblib.load("label_classifier_model.pkl")

def extract_and_show_fields_with_model(pdf_path):
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        img_np = np.array(img_pil)

        # Convert to grayscale for OpenCV
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Detect horizontal lines (fillable areas)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes for lines
        line_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # OCR for labels and detect fillable lines
        data = pytesseract.image_to_data(img_pil, output_type=Output.DICT)
        draw = ImageDraw.Draw(img_pil)

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word == '':
                continue

            # Replace the rule-based check with the trained model
            word_x, word_y, word_w, word_h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Use the trained model to predict if the word is a label
            is_label = pipeline.predict(pd.DataFrame([{
                "word": word, 
                "x": word_x, 
                "y": word_y, 
                "width": word_w, 
                "height": word_h
            }]))[0] == 1

            if is_label:  # If the model thinks it's a label
                label_box = (word_x, word_y, word_x + word_w, word_y + word_h)
                draw.rectangle(label_box, outline="red", width=2)
                draw.text((word_x, word_y - 12), word, fill="red")

                # Find the longest line to the right of the label
                candidates = []
                for lx, ly, lw, lh in line_boxes:
                    if (lx > word_x + word_w) and abs(ly - word_y) < word_h * 1.5 and lh <= word_h * 1.5:
                        candidates.append((lx, ly, lw, lh))

                if candidates:
                    # Select the longest (widest) line
                    longest = max(candidates, key=lambda b: b[2])  # b[2] is width
                    lx, ly, lw, lh = longest
                    draw.rectangle([lx, ly - word_h, lx + lw, ly + lh], outline="green", width=2)

        # Show the image with bounding boxes
        plt.figure(figsize=(10, 12))
        plt.imshow(img_pil)
        plt.axis('off')
        plt.title(f'Page {page_num + 1} - Detected Labels (red) & Fill Areas (green)')
        plt.show()

# Run on a scanned PDF
pdf_file = "scanned_pdf.pdf"
extract_and_show_fields_with_model(pdf_file)
