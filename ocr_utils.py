from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt
from collections import defaultdict


from paddleocr import PaddleOCR
from PIL import Image

# Load PaddleOCR once (do this at the top level)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def run_ocr_lines(image_path):
    image = Image.open(image_path).convert("RGB")
    result = ocr_model.ocr(str(image_path), cls=True)

    lines = []
    for line in result[0]:
        box, (text, confidence) = line
        if not text.strip():
            continue

        # Convert box format to [x0, y0, x1, y1]
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        lines.append({
            "text": text,
            "box": [
                int(1000 * x_min / image.width),
                int(1000 * y_min / image.height),
                int(1000 * x_max / image.width),
                int(1000 * y_max / image.height),
            ],
        })

    return image, lines


def prepare_lines_for_model(image, lines):
    words = []
    boxes = []

    for line in lines:
        words.append(line["text"])
        boxes.append(line["box"])

    return words, boxes

def merge_boxes(boxes):
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    return [x_min, y_min, x_max, y_max]






def visualize_ocr(image_path, words, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except:
        font = ImageFont.load_default()

    for word, box in zip(words, boxes):
        x0, y0, x1, y1 = [int(coord * image.width / 1000 if i % 2 == 0 else coord * image.height / 1000)
                          for i, coord in enumerate(box)]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0 + 1, y0 - 10), word, fill="blue", font=font)

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis("off")
    plt.show()