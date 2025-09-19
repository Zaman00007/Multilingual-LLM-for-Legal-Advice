import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_dir = "./legal-llm-fixed"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.to(device)
model.eval()

translator = Translator()

def ask_question_english(question_en, max_length=128, num_beams=4):
    inputs = tokenizer(question_en, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    response_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_en

def ask_question_hindi(question_hi):
    # Translate Hindi input to English
    question_en = translator.translate(question_hi, src='hi', dest='en').text
    answer_en = ask_question_english(question_en)
    answer_hi = translator.translate(answer_en, src='en', dest='hi').text
    return answer_hi

if __name__ == "__main__":
    while True:
        question_hi = input("\nप्रश्न (Hindi में, 'exit' से बाहर निकलें): ")
        if question_hi.lower() in ["exit", "quit"]:
            break
        answer_hi = ask_question_hindi(question_hi)
        print(f"उत्तर (Hindi में): {answer_hi}")
