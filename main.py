# from ocr_utils import run_ocr, visualize_ocr
# from layoutlm_utils import run_model # , extract_fields_with_decoding, processor
# from layoutlm_utils import group_tokens_by_line, extract_fields_from_lines, processor



from layoutlm_utils import extract_fields_from_lines, processor, run_model, group_tokens_by_line
from ocr_utils import run_ocr_lines, prepare_lines_for_model


if __name__ == "__main__":
    
    image_path = "dataset/dataset/testing_data/images/82092117.png"

    # print("Running OCR...")
    # image, words, boxes = run_ocr(image_path)
    # #visualize_ocr(image_path, words, boxes)
    # print("Running LayoutLMv3 model...")
    # token_label_pairs, boxes = run_model(image, words, boxes)
    
    # print("Extracted fields:")
    # grouped_lines = group_tokens_by_line(token_label_pairs, boxes, processor.tokenizer)
    # fields = extract_fields_from_lines(grouped_lines, processor.tokenizer)
    
    # # tokens, labels = zip(*token_label_pairs)
    # # cleaned_text = clean_tokens(tokens, processor.tokenizer)
    # #fields = extract_fields_with_decoding(token_label_pairs, processor.tokenizer, boxes)

    # for k, v in fields.items():
    #     print(f"{k}: {v}")

    image, lines = run_ocr_lines(image_path)

    words, boxes = prepare_lines_for_model(image, lines)

    token_label_pairs, token_boxes = run_model(image, words, boxes)

    grouped_lines = group_tokens_by_line(token_label_pairs, token_boxes, processor.tokenizer)

    fields = extract_fields_from_lines(grouped_lines, processor.tokenizer)

    for k, v in fields.items():
        print(f"{k}: {v}")

