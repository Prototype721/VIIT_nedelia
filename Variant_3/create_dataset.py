import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import io
import csv


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# Function to create the dataset (CSV) for form field labels
def extract_fields_with_boxes(pdf_path, output_csv="form_field_labels.csv"):
    doc = fitz.open(pdf_path)
    all_data = []

    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR for text data
        data = pytesseract.image_to_data(img_pil, output_type=Output.DICT)

        # Extracting bounding box data for each word
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word == '':
                continue

            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Now let's determine if this word is likely a label. We can do this manually or with a rule-based check.
            # You can later switch to a trained model for this part.

            # Here we'll use a basic rule for labels, you can update this with model-based predictions later.
            is_label = 0  # Default: not a label

            if word.endswith(":") or word.lower() in {"name", "date", "signature", "address", "phone"}:
                is_label = 1  # Mark as a label if it matches a rule

            # Save data to all_data list
            all_data.append([word, x, y, w, h, is_label])

        # Optionally show the page for manual confirmation
        # img_pil.show()  # Uncomment to visualize each page as you go through

    # Write all the collected data to CSV
    with open(output_csv, mode='w', newline='', encoding='UTF-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["word", "x", "y", "width", "height", "is_label"])
        # Write the data
        writer.writerows(all_data)

    print(f"CSV file '{output_csv}' has been created successfully.")

# Example usage
extract_fields_with_boxes("scanned_pdf.pdf")
