from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files="/home/zaman/Code/LLM/data.json")

def preprocess(examples):
    inputs = examples["question"]
    targets = examples["answer"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding=False  
    )
    
    labels = tokenizer(
        text_target=targets,  
        max_length=512, 
        truncation=True, 
        padding=False 
    )
    
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Preprocessing dataset...")
tokenized = dataset.map(preprocess, batched=True)

print("Sample input:", tokenized["train"][0]["input_ids"][:10])
print("Sample label:", tokenized["train"][0]["labels"][:10])

train_test_split = tokenized["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100, 
    padding=True
)

training_args = TrainingArguments(
    output_dir="./legal-llm-new",
    eval_strategy="steps",
    eval_steps=8,  
    save_strategy="steps",
    save_steps=8,
    learning_rate=3e-4,  
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    num_train_epochs=20,  
    weight_decay=0.01,
    warmup_steps=10,  
    fp16=False,  
    logging_dir="./logs",
    logging_steps=2,  
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  
    prediction_loss_only=True,
    dataloader_pin_memory=False,  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer, 
)

print("Starting training...")

trainer.train()

trainer.save_model("./legal-llm-new")
tokenizer.save_pretrained("./legal-llm-new")

print("Training completed successfully!")
print("Model saved to ./legal-llm-new")

print("\n=== Quick Test ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_question = "I want to take divorce from my husband?"
inputs = tokenizer(test_question, return_tensors="pt", max_length=128, truncation=True)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {test_question}")
    print(f"Answer: {response}")