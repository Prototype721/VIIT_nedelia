from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def run_ocr(image_path):
#     image = Image.open(image_path).convert("RGB")
#     ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

#     words = []
#     boxes = []

#     for i in range(len(ocr_data["text"])):
#         word = ocr_data["text"][i]
#         #if word.strip() == "":       # TODO 
#            # continue
#         words.append(word)
#         x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        
#         box = [int(1000 * x / image.width),   # TODO
#                int(1000 * y / image.height),
#                int(1000 * (x + w) / image.width),
#                int(1000 * (y + h) / image.height)]
#         boxes.append(box)

#     return image, words, boxes



def run_ocr_lines(image_path):
    from PIL import Image
    import pytesseract

    image = Image.open(image_path).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    lines = []
    current_line = []
    last_line_num = -1
    line_boxes = []

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if not word:
            continue

        line_num = ocr_data["line_num"][i]
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        box = [int(1000 * x / image.width),
               int(1000 * y / image.height),
               int(1000 * (x + w) / image.width),
               int(1000 * (y + h) / image.height)]

        if line_num != last_line_num and current_line:
            lines.append((current_line, line_boxes))
            current_line = []
            line_boxes = []

        current_line.append(word)
        line_boxes.append(box)
        last_line_num = line_num

    if current_line:
        lines.append((current_line, line_boxes))

    for line_words, _ in lines:
        print("Line:", " ".join(line_words))

        
    return image, lines



def prepare_lines_for_model(image, lines):
    words = [' '.join(line_words) for line_words, _ in lines]
    boxes = [merge_boxes(line_boxes) for _, line_boxes in lines]
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