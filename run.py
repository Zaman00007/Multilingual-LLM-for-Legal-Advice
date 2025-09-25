import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

english_model_dir = "/home/zaman/Code/LLM/legal-llm-fixed"
hindi_model_dir = "/home/zaman/Code/LLM/legal-llm-new"

tokenizer_en = AutoTokenizer.from_pretrained(english_model_dir)
model_en = AutoModelForSeq2SeqLM.from_pretrained(english_model_dir)
model_en.to(device)
model_en.eval()

tokenizer_hi = AutoTokenizer.from_pretrained(hindi_model_dir)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(hindi_model_dir)
model_hi.to(device)
model_hi.eval()

translator = Translator()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

def ask_model(text, lang, max_length=128, num_beams=4):
    if lang == "en":
        tokenizer = tokenizer_en
        model = model_en
    elif lang == "hi":
        tokenizer = tokenizer_hi
        model = model_hi
    else:
        return "Unsupported language"

    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
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

def ask_question(text):
    detected_lang = translator.detect(text).lang

    if detected_lang in ["en", "hi"]:
        response = ask_model(text, detected_lang)
        return {"language": detected_lang, "answer": response}
    else:
        return {"language": detected_lang, "answer": "Unsupported language"}

@app.post("/ask")
async def ask(query: Query):
    return ask_question(query.text)
