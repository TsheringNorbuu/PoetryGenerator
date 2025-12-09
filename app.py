from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import os

# ==========================
# APP SETUP
# ==========================
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================
# LOAD HUGGING FACE MODEL
# ==========================
MODEL_NAME = "Tshering6/poetry-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",      # automatically chooses GPU/CPU
    load_in_8bit=True       # reduces memory usage drastically
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# ==========================
# REQUEST SCHEMA
# ==========================
class Prompt(BaseModel):
    text: str

# ==========================
# POEM FORMATTER
# ==========================
def format_poem(text: str) -> str:
    """
    Convert raw model output into clean free-verse poetry.
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^[\s:–—\-]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bi\s*e\s*,", "i.e.", text, flags=re.I)

    words = text.split()
    lines = []
    current_line = []
    max_words_per_line = 6

    for word in words:
        current_line.append(word)
        if len(current_line) >= max_words_per_line or word.endswith((".", ",", "!", "?")):
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    # Build poem without stanza breaks
    poem = "\n".join(lines)
    return poem

# ==========================
# ROUTES
# ==========================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
def generate_poetry(prompt: Prompt):
    if not prompt.text.strip():
        return {"poem": "Please enter a prompt to generate a poem."}

    clean_prompt = (
        "Write a joyful poem with fresh imagery and short poetic lines.\n\n"
        f"Theme: {prompt.text}\n\nPoem:\n"
    )

    output = generator(
        clean_prompt,
        max_length=180,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        do_sample=True
    )

    full_text = output[0]["generated_text"]
    poem = full_text.replace(clean_prompt, "").strip()
    poem = poem.lstrip(":—–- \n")
    poem = format_poem(poem)

    return {"poem": poem}

# ==========================
# RUN ON RENDER
# ==========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
