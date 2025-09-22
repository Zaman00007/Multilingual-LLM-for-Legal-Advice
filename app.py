import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_dir = "./legal-llm-new"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.to(device)
model.eval()

translator = Translator()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def ask_question(text):
    detected_lang = translator.detect(text).lang  
    
    if detected_lang == "en":
        answer_en = ask_question_english(text)
        return {"language": "en", "answer": answer_en}
    elif detected_lang == "hi":
        question_en = translator.translate(text, src='hi', dest='en').text
        answer_en = ask_question_english(question_en)
        answer_hi = translator.translate(answer_en, src='en', dest='hi').text
        return {"language": "hi", "answer": answer_hi}
    else:
        return {"language": detected_lang, "answer": "Unsupported language"}

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    return ask_question(query.text)
