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
import pandas as pd
from tqdm import tqdm
import os
import random

class ImprovedHindiLegalDataset(Dataset):
    """Improved dataset class for Hindi legal training"""
    
    def __init__(self, json_file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and expand your JSON dataset
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.texts = self.create_expanded_training_data()
    
    def create_expanded_training_data(self):
        """Create expanded and varied training examples"""
        texts = []
        
        for item in self.data:
            if 'question' in item and 'answer' in item:
                question = item['question'].strip()
                answer = item['answer'].strip()
                
                # Format 1: Full Q&A with special tokens
                qa_text = f"<|startoftext|>प्रश्न: {question}\nउत्तर: {answer}<|endoftext|>"
                texts.append(qa_text)
                
                # Format 2: Question-only prompt for completion
                q_prompt = f"<|startoftext|>प्रश्न: {question}\nउत्तर:"
                texts.append(q_prompt)
                
                # Format 3: Answer-only for language modeling
                a_only = f"<|startoftext|>{answer}<|endoftext|>"
                texts.append(a_only)
                
                # Format 4: Variations with different question starters
                variations = [
                    f"<|startoftext|>प्रश्न: {question}\nजवाब: {answer}<|endoftext|>",
                    f"<|startoftext|>{question}\nउत्तर: {answer}<|endoftext|>",
                ]
                texts.extend(variations)
        
        print(f"✅ Created {len(texts)} training examples from {len(self.data)} Q&A pairs")
        return texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with consistent parameters
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

def setup_improved_tokenizer(model_name="gpt2"):
    """Setup tokenizer with proper special tokens"""
    print(f"🔧 Setting up improved tokenizer for {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
    }
    
    # Add Hindi legal vocabulary
    hindi_legal_vocab = [
        "धारा", "अनुच्छेद", "आईपीसी", "संविधान", "अधिनियम", 
        "न्यायालय", "मुकदमा", "शिकायत", "कानूनी", "अधिकार",
        "सजा", "जुर्माना", "कारावास", "भरण-पोषण", "तलाक",
        "संपत्ति", "विरासत", "उत्पीड़न", "घरेलू", "हिंसा"
    ]
    
    # Add new tokens
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(hindi_legal_vocab)
    
    print(f"✅ Added {num_added_tokens} special tokens and {len(hindi_legal_vocab)} Hindi legal terms")
    return tokenizer

def train_improved_hindi_legal_model(
    json_file_path, 
    output_dir="./improved-hindi-legal-model", 
    epochs=5,
    model_name="gpt2"
):
    """Improved training function with better parameters"""
    
    print("🚀 Starting Improved Hindi Legal LLM Training...")
    
    # Setup tokenizer first
    tokenizer = setup_improved_tokenizer(model_name)
    
    # Load model and resize embeddings
    print("📥 Loading and preparing model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create improved dataset
    print("📊 Creating expanded training dataset...")
    dataset = ImprovedHindiLegalDataset(json_file_path, tokenizer, max_length=256)
    
    # Split dataset
    train_size = int(0.85 * len(dataset))  # Use more data for training
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    # Improved training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "warmup_steps": 200,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "learning_rate": 1e-4,  # Higher learning rate for better Hindi learning
        "weight_decay": 0.01,
        "report_to": None,
        "seed": 42,
        "dataloader_num_workers": 0,
    }
    
    # Handle version compatibility
    try:
        training_args_dict["evaluation_strategy"] = "steps"
        training_args = TrainingArguments(**training_args_dict)
    except TypeError:
        training_args_dict["eval_strategy"] = "steps"
        training_args = TrainingArguments(**training_args_dict)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    print("🎯 Starting improved training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving improved model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Improved training complete! Model saved to {output_dir}")
    return model, tokenizer

def test_improved_model(model_path, prompt, max_length=150):
    """Test the improved model with better generation parameters"""
    
    print(f"🧪 Testing: {prompt}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Format prompt properly
    formatted_prompt = f"<|startoftext|>प्रश्न: {prompt}\nउत्तर:"
    
    # Encode input
    inputs = tokenizer.encode(formatted_prompt, return_tensors='pt')
    
    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    if "उत्तर:" in response:
        answer = response.split("उत्तर:")[-1].strip()
        return answer
    else:
        return response[len(formatted_prompt):].strip()

# Quick fix function for immediate testing
def quick_fix_model_test(model_path):
    """Quick test to see if the model can generate proper Hindi"""
    
    print("🔧 Quick Fix Testing...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Simple Hindi test
    test_cases = [
        "भारतीय संविधान",
        "आईपीसी की धारा", 
        "कानूनी सलाह"
    ]
    
    for test_input in test_cases:
        inputs = tokenizer.encode(test_input, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Output: {result}")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Use the improved training
    try:
        print("🔄 Starting improved training process...")
        model, tokenizer = train_improved_hindi_legal_model(
            json_file_path="legal_dataset.json",
            epochs=10,  # More epochs
            model_name="gpt2"
        )
        
        # Test improved model
        test_questions = [
            "भारतीय संविधान का अनुच्छेद 21 क्या है?",
            "IPC की धारा 302 क्या है?",
            "घरेलू हिंसा के लिए कौन सा कानून है?"
        ]
        
        for question in test_questions:
            answer = test_improved_model("./improved-hindi-legal-model", question)
            print(f"प्रश्न: {question}")
            print(f"उत्तर: {answer}")
            print("="*80)
            
    except Exception as e:
        print(f"❌ Error in improved training: {e}")
        print("🔧 Running quick fix test on existing model...")
        quick_fix_model_test("./hindi-llm-finetuned")