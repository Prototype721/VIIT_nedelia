import fitz  # PyMuPDF
import PyPDF2

def extract_form_fields(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    fields = reader.get_fields()
    return fields

def extract_bounding_boxes(pdf_path):
    doc = fitz.open(pdf_path)
    field_boxes = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        widgets = page.widgets()  # Extract form fields (widgets)

        for widget in widgets:
            rect = widget.rect
            field_info = {
                'page': page_num + 1,
                'field_name': widget.field_name,
                'rect': rect,  # Bounding box (x0, y0, x1, y1)
            }
            field_boxes.append(field_info)

    return field_boxes

# Example usage
pdf_file = "my_pdf.pdf"
fields = extract_form_fields(pdf_file)
bounding_boxes = extract_bounding_boxes(pdf_file)

print("Form fields found:")
for box in bounding_boxes:
    print(f"Page {box['page']}, Field: {box['field_name']}, Box: {box['rect']}")

