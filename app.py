from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# ==========================
# APP SETUP
# ==========================
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================
# LOAD LOCAL MODEL
# ==========================
MODEL_PATH = "./poetry_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU only
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
    Convert raw model output into clean free-verse poetry without stanzas.
    """

    if not text:
        return ""

    # ------------------------
    # 1. Basic cleanup
    # ------------------------
    text = text.strip()

    # Remove leading junk like ":" or dashes
    text = re.sub(r"^[\s:–—\-]+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix broken i.e.
    text = re.sub(r"\bi\s*e\s*,", "i.e.", text, flags=re.I)

    # ------------------------
    # 2. Intelligent line breaks
    # ------------------------
    words = text.split()
    lines = []
    current_line = []
    max_words_per_line = 6  # keeps lines poetic

    for word in words:
        current_line.append(word)

        # End line naturally
        if len(current_line) >= max_words_per_line or word.endswith((".", ",", "!", "?")):
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    # ------------------------
    # 3. Combine lines into poem (no blank lines)
    # ------------------------
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

    # Structured prompt (better poetic behavior)
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

    # Remove prompt text and stray punctuation
    poem = full_text.replace(clean_prompt, "").strip()
    poem = poem.lstrip(":—–- \n")

    # Format as poem
    poem = format_poem(poem)

    return {"poem": poem}
