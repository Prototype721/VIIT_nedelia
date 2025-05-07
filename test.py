from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import pytesseract
import torch

# huggingface-cli delete-cache
# Or C:\Users\<YourUsername>\.cache\huggingface\hub\models
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


model = LayoutLMv3ForTokenClassification.from_pretrained(
    "nielsr/layoutlmv3-finetuned-funsd"
)

processor = LayoutLMv3Processor.from_pretrained(
    "nielsr/layoutlmv3-finetuned-funsd"#, apply_ocr=False
)



image = Image.open(r"E:/VVIT_SOBES/dataset/dataset/training_data/images/00040534.png")
if image.mode != "RGB":
    image = image.convert("RGB")

ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# Assuming ocr_result contains 'text', 'left', 'top', 'width', 'height' keys
words = ocr_result['text']
boxes = [
    [left, top, left + width, top + height]
    for left, top, width, height in zip(ocr_result['left'], ocr_result['top'], ocr_result['width'], ocr_result['height'])
]

encoding = processor(image, words, boxes=boxes, padding="max_length", truncation=True, return_tensors="pt")


outputs = model(**encoding)
logits = outputs.logits
predictions = logits.argmax(-1)


predicted_class_indices = torch.argmax(logits, dim=-1)  # shape: [batch_size, sequence_len]

predicted_classes = predicted_class_indices[0].tolist()
input_ids = encoding["input_ids"][0].tolist()



