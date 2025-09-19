import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_dir = "./legal-llm-fixed"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.to(device)
model.eval()

translator = Translator()
app = FastAPI()

# -------------------------
# CORS Setup
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helper functions
# -------------------------
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
    detected_lang = translator.detect(text).lang  # 'en' or 'hi'
    
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

# -------------------------
# API Route
# -------------------------
class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    return ask_question(query.text)
