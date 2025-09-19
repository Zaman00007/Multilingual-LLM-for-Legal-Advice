import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model and tokenizer
model_dir = "./legal-llm-fixed"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.to(device)
model.eval()

# Function to get model response
def ask_question(question, max_length=128, num_beams=4):
    inputs = tokenizer(question, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test questions
test_questions = [
    "I want to take divorce from my husband?",
    "My husband is beating me, what should I do?",
    "My colleague is sexually harassing me at office, what can I do?",
    "Someone is encroaching on my land, what should I do?",
    "My employer is not paying my wages, what legal action can I take?",
    "My senior is sexually harrasing me?"
]

# Get answers
for q in test_questions:
    answer = ask_question(q)
    print(f"\nQuestion: {q}\nAnswer: {answer}")
