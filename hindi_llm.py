import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import pandas as pd
from tqdm import tqdm
import os

class HindiDataset(Dataset):
    """Custom dataset class for Hindi text data"""
    
    def __init__(self, json_file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load your JSON dataset
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Process the data based on your JSON structure
        self.texts = self.process_json_data()
    
    def process_json_data(self):
        """
        Process JSON data specifically for your legal Q&A dataset
        Structure: {"question": "प्रश्न", "answer": "उत्तर"}
        """
        texts = []
        
        if isinstance(self.data, list):
            for item in self.data:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    # Format for conversational training
                    # This creates a natural Q&A flow for the model to learn
                    combined_text = f"प्रश्न: {item['question']}\nउत्तर: {item['answer']}<|endoftext|>"
                    texts.append(combined_text)
                    
                    # Also add just the answer part for better response generation
                    answer_text = f"{item['answer']}<|endoftext|>"
                    texts.append(answer_text)
        
        print(f"✅ Processed {len(texts)} training examples from {len(self.data)} Q&A pairs")
        return texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def load_hindi_model_and_tokenizer(model_name="gpt2"):
    """
    Load pre-trained model and tokenizer optimized for your Hindi legal dataset
    Best models for Hindi fine-tuning:
    - "gpt2" (lightweight, good for Hindi fine-tuning)
    - "microsoft/DialoGPT-medium" (conversational)
    - "ai4bharat/indic-bert" (Indian languages)
    """
    print(f"📥 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add Hindi-specific legal tokens for better performance
    hindi_legal_tokens = [
        '।', '॥', 'धारा', 'अनुच्छेद', 'आईपीसी', 'संविधान', 
        'अधिनियम', 'न्यायालय', 'मुकदमा', 'शिकायत', 'कानूनी'
    ]
    
    # Add tokens that aren't already in vocabulary
    new_tokens = [token for token in hindi_legal_tokens if token not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"🔤 Added {len(new_tokens)} new Hindi legal tokens")
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add end-of-text token for better training
    if "<|endoftext|>" not in tokenizer.vocab:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    
    print(f"✅ Model loaded with vocabulary size: {len(tokenizer)}")
    return model, tokenizer

def train_hindi_llm(json_file_path, output_dir="./hindi-llm-model", epochs=50):
    """Main training function"""
    
    print("🚀 Starting Hindi LLM Training...")
    
    # Load model and tokenizer
    print("📥 Loading pre-trained model...")
    model, tokenizer = load_hindi_model_and_tokenizer()
    
    # Create dataset
    print("📊 Processing dataset...")
    dataset = HindiDataset(json_file_path, tokenizer)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments optimized for your Hindi legal dataset
    # Compatible with both old and new transformers versions
    training_args_dict = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 2,  # Reduced for legal content
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,  # Effective batch size = 2*4 = 8
        "warmup_steps": 100,
        "logging_steps": 25,
        "save_steps": 250,
        "eval_steps": 250,
        "save_total_limit": 3,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "learning_rate": 5e-5,  # Good for fine-tuning
        "weight_decay": 0.01,
        "report_to": None,  # Disable wandb logging
        "seed": 42
    }
    
    # Handle version compatibility for evaluation_strategy vs eval_strategy
    try:
        training_args_dict["eval_strategy"] = "steps"
        training_args = TrainingArguments(**training_args_dict)
    except TypeError:
        # Older versions use eval_strategy instead
        training_args_dict["eval_strategy"] = "steps"
        training_args = TrainingArguments(**training_args_dict)
    
    # Handle fp16 compatibility
    try:
        training_args.fp16 = True
    except AttributeError:
        print("⚠️  FP16 not supported in this transformers version, using FP32")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    print("🎯 Starting training...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training complete! Model saved to {output_dir}")
    
    return model, tokenizer

def test_model(model_path, prompt="भारतीय संविधान का अनुच्छेद 21 क्या है?"):
    """Test the trained legal model with relevant prompts"""
    
    print(f"🧪 Testing model with prompt: {prompt}")
    
    # Load the trained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Format prompt like training data
    formatted_prompt = f"प्रश्न: {prompt}\nउत्तर: "
    
    # Generate response
    inputs = tokenizer.encode(formatted_prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 150,  # Allow longer legal responses
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    answer = response[len(formatted_prompt):].strip()
    
    # Clean up response
    if '<|endoftext|>' in answer:
        answer = answer.split('<|endoftext|>')[0].strip()
    
    return answer

# Example usage
if __name__ == "__main__":
    # Configuration
    JSON_FILE_PATH = "/home/zaman/Code/LLM/data_old_hi.json"  # Replace with your JSON file path
    OUTPUT_DIR = "./hindi-llm-finetuned"
    
    # Train the model
    try:
        model, tokenizer = train_hindi_llm(
            json_file_path=JSON_FILE_PATH,
            output_dir=OUTPUT_DIR,
            epochs=3
        )
        
        # Test the model with legal questions from your dataset
        print("\n🧪 Testing the trained Hindi legal model:")
        test_prompts = [
            "भारतीय संविधान का अनुच्छेद 21 क्या है?",
            "IPC की धारा 302 क्या है?",
            "मेरे पति मुझे मारते हैं, मुझे क्या करना चाहिए?",
            "कार्यालय में यौन उत्पीड़न के लिए कौन सा कानून है?",
            "संपत्ति विवाद के लिए कैसे मुकदमा दर्ज करें?"
        ]
        
        for prompt in test_prompts:
            response = test_model(OUTPUT_DIR, prompt)
            print(f"प्रश्न: {prompt}")
            print(f"उत्तर: {response}\n")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure your JSON file exists and has the correct format.")

# Additional utility functions

def validate_json_dataset(json_file_path):
    """Validate your JSON dataset format"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📋 Dataset validation:")
        print(f"   Total items: {len(data)}")
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            print(f"   Sample item keys: {list(sample.keys()) if isinstance(sample, dict) else 'String item'}")
            print(f"   Sample content: {str(sample)[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False

def prepare_dataset_for_training(raw_json_path, processed_json_path):
    """Preprocess your dataset if needed"""
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add preprocessing logic here based on your specific needs
    processed_data = []
    
    for item in data:
        # Example preprocessing
        if isinstance(item, dict):
            # Clean and format text
            if 'text' in item:
                cleaned_text = item['text'].strip()
                if len(cleaned_text) > 10:  # Filter out very short texts
                    processed_data.append({'text': cleaned_text})
    
    # Save processed dataset
    with open(processed_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Processed dataset saved to {processed_json_path}")
    print(f"   Original items: {len(data)}")
    print(f"   Processed items: {len(processed_data)}")