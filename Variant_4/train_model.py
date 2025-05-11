import os
import glob
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3Tokenizer, LayoutLMv3ForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

# Initialize the tokenizer
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base", use_fast=True)

# Prepare the dataset from the layoutlm_data folder
def load_layoutlm_data(data_dir="layoutlm_data"):
    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    dataset = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            data = json.load(f)  # <-- Use json.load() to load the JSON data
            dataset.append(data)

    # Split into train and validation sets (80% train, 20% validation)
    split = int(0.8 * len(dataset))  # 80% for training
    train_data = dataset[:split]
    val_data = dataset[split:]

    return train_data, val_data

train_data, val_data = load_layoutlm_data()

def compute_metrics(p: Trainer, output: torch.Tensor):
    predictions, labels = output
    preds = predictions.argmax(axis=-1)
    true_labels = labels
    
    return classification_report(true_labels, preds, output_dict=True)

# Convert the data into the `Dataset` format required by Hugging Face
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenize and align labels
def tokenize_and_align_labels(example, tokenizer):
    # Tokenize the words
    
    # Create bounding boxes, rescale to the layout format (0 to 1000)
    boxes = example['boxes']
    normalized_boxes = []
    for box in boxes:
        box = box[0]
        if len(box) == 4:  # Only process valid boxes with 4 elements
            x_min, y_min, x_max, y_max = box
            normalized_boxes.append([
                int(1000 * x_min / example['width']), 
                int(1000 * y_min / example['height']), 
                int(1000 * (x_max - x_min) / example['width']), 
                int(1000 * (y_max - y_min) / example['height'])
            ])
        else:
            print(f"Skipping invalid box: {box}")
    encoding = tokenizer(example['words'], padding="max_length", truncation=True, return_tensors="pt")

    # Ensure the bounding boxes are added correctly
    encoding['bbox'] = normalized_boxes
    
    # Align the labels
    labels = example['labels']
    encoding['labels'] = labels
    
    return encoding

# Tokenize and prepare dataset
train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
val_dataset = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

# Define the model
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=3)  # Adjust num_labels based on your use case

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Output directory
    #evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    learning_rate=2e-5,               # Learning rate
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=16,    # Batch size for evaluation
    num_train_epochs=3,               # Number of epochs
    weight_decay=0.01,                # Weight decay for regularization
    logging_dir="./logs",             # Directory for logging
    logging_steps=10,                 # Log every 10 steps
    save_steps=500,                   # Save model every 500 steps
    save_total_limit=2,               # Limit the number of saved models
    evaluation_strategy="steps",      # Evaluation during training (optional)
    load_best_model_at_end=True       # Load the best model when training ends
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Define this function for evaluation metrics
)

# Start training
trainer.train()

# Save the trained model
model.save_pretrained("./trained_layoutlmv3_model")
tokenizer.save_pretrained("./trained_layoutlmv3_model")
