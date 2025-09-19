from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

# Load model & tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load your dataset (JSON format)
dataset = load_dataset("json", data_files="/home/zaman/Code/LLM/data_old.json")

# Improved preprocess function with proper label handling
def preprocess(examples):
    inputs = examples["question"]
    targets = examples["answer"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding=False  # Let data collator handle padding
    )
    
    # Tokenize targets/labels properly
    labels = tokenizer(
        text_target=targets,  # Use text_target parameter (new way)
        max_length=512, 
        truncation=True, 
        padding=False  # Let data collator handle padding
    )
    
    # Important: Replace padding token ids in labels with -100 
    # so they are ignored in loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
print("Preprocessing dataset...")
tokenized = dataset.map(preprocess, batched=True)

# Check a sample to debug
print("Sample input:", tokenized["train"][0]["input_ids"][:10])
print("Sample label:", tokenized["train"][0]["labels"][:10])

# Split dataset into train/eval (80/20 split)
train_test_split = tokenized["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Data collator for seq2seq models
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # Important: use -100 for ignored tokens in loss
    padding=True
)

# More conservative training arguments
training_args = TrainingArguments(
    output_dir="./legal-llm-fixed",
    eval_strategy="steps",
    eval_steps=8,  # Evaluate every 8 steps (more frequent)
    save_strategy="steps",
    save_steps=8,
    learning_rate=3e-4,  # Higher learning rate
    per_device_train_batch_size=4,  # Increased batch size
    per_device_eval_batch_size=4,
    num_train_epochs=20,  # More epochs
    weight_decay=0.01,
    warmup_steps=10,  # Add warmup
    fp16=False,  # Disable mixed precision to avoid NaN issues
    logging_dir="./logs",
    logging_steps=2,  # More frequent logging
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  # Keep only 3 checkpoints
    prediction_loss_only=True,
    dataloader_pin_memory=False,  # Avoid memory issues
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,  # Use tokenizer instead of processing_class
)

print("Starting training...")

# Train the model
trainer.train()

# Save the model
trainer.save_model("./legal-llm-fixed")
tokenizer.save_pretrained("./legal-llm-fixed")

print("Training completed successfully!")
print("Model saved to ./legal-llm-fixed")

# Test the model quickly
print("\n=== Quick Test ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_question = "I want to take divorce from my husband?"
inputs = tokenizer(test_question, return_tensors="pt", max_length=128, truncation=True)

# Move inputs to the same device as model
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {test_question}")
    print(f"Answer: {response}")