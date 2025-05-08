# from ocr_utils import run_ocr, visualize_ocr
# from layoutlm_utils import run_model # , extract_fields_with_decoding, processor
# from layoutlm_utils import group_tokens_by_line, extract_fields_from_lines, processor



from layoutlm_utils import extract_fields_from_lines, processor, run_model, group_tokens_by_line, extract_questions_with_boxes
from ocr_utils import run_ocr_lines, prepare_lines_for_model


if __name__ == "__main__":
    
    image_path = "dataset/dataset/testing_data/images/82092117.png"

    image, lines = run_ocr_lines(image_path)

    words, boxes = prepare_lines_for_model(image, lines)

    token_label_pairs, token_boxes = run_model(image, words, boxes)

    grouped_lines = group_tokens_by_line(token_label_pairs, token_boxes, processor.tokenizer)

    # fields = extract_fields_from_lines(grouped_lines, processor.tokenizer)

    # for k, v in fields.items():
    #     print(f"{k} : {v}")

    questions_with_boxes = extract_questions_with_boxes(grouped_lines, processor.tokenizer)

    for question, box in questions_with_boxes:
        print(f"QUESTION: {question} | BOX: {box}")



