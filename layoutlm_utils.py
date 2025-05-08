import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from collections import Counter



LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-HEADER", "I-HEADER"]
id2label = {i: label for i, label in enumerate(LABELS)}


processor = LayoutLMv3Processor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

# def run_model(image, words, boxes):
    
#     encoding = processor(
#         image,
#         words,
#         boxes=boxes,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)[0].tolist()

#     input_ids = encoding["input_ids"].squeeze().tolist()
#     tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
#     word_ids = encoding.word_ids(batch_index=0)

#     # token_label_pairs = []
#     # for token, word_id, label_id in zip(tokens, word_ids, predictions):
#     #     if word_id is None:
#     #         continue
#     #     token_label_pairs.append((token, label_id))


#     token_label_pairs = []
#     for input_id, word_id, label_id in zip(encoding["input_ids"][0], word_ids, predictions):
#         if word_id is None:
#             continue
#         token_label_pairs.append((input_id.item(), label_id))


#     # labels = [id2label[label_id] for _, label_id in token_label_pairs]
#     # print("label counter:", Counter(labels))

#     return token_label_pairs, input_ids


# def clean_tokens(tokens, tokenizer):
#     # Join input IDs and decode properly
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     return tokenizer.decode(input_ids, skip_special_tokens=True)



# def extract_fields_with_decoding(token_label_pairs, input_ids, tokenizer):
#     fields = {}
#     current_question_tokens = []
#     current_answer_tokens = []
#     in_question = False
#     in_answer = False
#     print("TOKENS__________________")
#     for input_id, label_id in token_label_pairs:

#         token_str = tokenizer.decode([input_id])
#         print(f"{token_str} — {id2label[label_id]}")
#         label = id2label[label_id]

#         if label == "B-QUESTION":
#             # flush
#             if current_question_tokens and current_answer_tokens:
#                 q = tokenizer.decode(current_question_tokens, skip_special_tokens=True).strip()
#                 a = tokenizer.decode(current_answer_tokens, skip_special_tokens=True).strip()
#                 if q and a:
#                     fields[q] = a
#             current_question_tokens = [input_id]
#             current_answer_tokens = []
#             in_question = True
#             in_answer = False

#         elif label == "I-QUESTION" and in_question:
#             current_question_tokens.append(input_id)

#         elif label == "B-ANSWER":
#             if in_answer and current_question_tokens and current_answer_tokens:
#                 q = tokenizer.decode(current_question_tokens, skip_special_tokens=True).strip()
#                 a = tokenizer.decode(current_answer_tokens, skip_special_tokens=True).strip()
#                 if q and a:
#                     fields[q] = a
#             current_answer_tokens = [input_id]
#             in_answer = True
#             in_question = False

#         elif label == "I-ANSWER" and in_answer:
#             current_answer_tokens.append(input_id)

#     # Final flush in case last pair didn't get saved
#     if current_question_tokens and current_answer_tokens:
#         q = tokenizer.decode(current_question_tokens, skip_special_tokens=True).strip()
#         a = tokenizer.decode(current_answer_tokens, skip_special_tokens=True).strip()
#         if q and a:
#             fields[q] = a
#     print("END TOKENS__________________")
#     return fields














def run_model(image, words, boxes):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()

    input_ids = encoding["input_ids"][0].tolist()
    word_ids = encoding.word_ids(batch_index=0)

    token_label_pairs = []
    token_boxes = []
    word_to_tokens = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_to_tokens:
            word_to_tokens[word_id] = {"input_ids": [], "label_ids": [], "box": boxes[word_id]}
        word_to_tokens[word_id]["input_ids"].append(input_ids[idx])
        word_to_tokens[word_id]["label_ids"].append(predictions[idx])

    for word_data in word_to_tokens.values():
        ids = word_data["input_ids"]
        labels = word_data["label_ids"]
        majority_label = max(set(labels), key=labels.count)
        token_label_pairs.append((ids, majority_label))
        token_boxes.append(word_data["box"])

    for (input_ids, label_id), box in zip(token_label_pairs, token_boxes):
        token_text = processor.tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"{id2label[label_id]:12} → {token_text} | Box: {box}")

    return token_label_pairs, token_boxes



from collections import defaultdict

def group_tokens_by_line(token_label_pairs, boxes, tokenizer, y_threshold=5):
    lines = defaultdict(list)

    for (input_ids, label_id), box in zip(token_label_pairs, boxes):
        y_center = (box[1] + box[3]) // 2
        matched_line = None

        for line_y in lines:
            if abs(line_y - y_center) <= y_threshold:
                matched_line = line_y
                break

        if matched_line is not None:
            lines[matched_line].append(((input_ids, label_id), box))
        else:
            lines[y_center].append(((input_ids, label_id), box))

    # Отсортировать токены в каждой строке по x
    sorted_lines = []
    for y in sorted(lines):
        line = lines[y]
        line.sort(key=lambda item: (item[1][0] + item[1][2]) // 2)  # sort by x_center
        sorted_lines.append(line)

    return sorted_lines


def extract_fields_from_lines(grouped_lines, tokenizer):
    fields = {}
    current_question = ""
    current_answer = ""
    current_label = None

    for line in grouped_lines:
        tokens = []
        labels = []

        for (input_ids, label_id), _ in line:
            tokens.append(input_ids)
            labels.append(label_id)

        text = tokenizer.decode([tid for ids in tokens for tid in ids], skip_special_tokens=True).strip()
        if not text:
            continue

        majority_label = max(set(labels), key=labels.count)
        label = id2label[majority_label]

        # Treat both B- and I- as possible starts
        if label.endswith("QUESTION"):
            if current_question and current_answer:
                fields[current_question] = current_answer
            current_question = text
            current_answer = ""

        elif label.endswith("ANSWER"):
            if current_answer:
                current_answer += " " + text
            else:
                current_answer = text

    if current_question and current_answer:
        fields[current_question] = current_answer

    return fields



def extract_questions_with_boxes(grouped_lines, tokenizer):
    questions = []

    for line in grouped_lines:
        tokens = []
        labels = []
        boxes = []

        for (input_ids, label_id), box in line:
            label = id2label[label_id]
            if label.endswith("QUESTION"):
                tokens.append(input_ids)
                labels.append(label_id)
                boxes.append(box)

        if not tokens:
            continue

        text = tokenizer.decode([tid for ids in tokens for tid in ids], skip_special_tokens=True).strip()
        if not text:
            continue

        # Merge all boxes into one that contains the full line
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        full_box = [x0, y0, x1, y1]

        questions.append((text, full_box))

    return questions
