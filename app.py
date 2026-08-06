from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initilize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model and tokenizer

model = T5ForConditionalGeneration.from_pretrained("./saved_summary")
tokenizers = T5Tokenizer.from_pretrained("./saved_summary")

# define device

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device: ", device)

model.to(device)

#templeting 

template = Jinja2Templates(directory=".")

# inpute schema for dialogue -- > string

class DialogueInput(BaseModel):
    dialogue : str


def data_clean(text):
    text = re.sub(r"\r\n", " ", text) # removes lines
    text = re.sub(r"\s+", " ", text) # Extra spaces
    text = text.strip().lower() # remove leading and trailing space and lowercase

    return text



def summarization_text(dialogue : str) -> str:
    dialogue = data_clean(dialogue)
    inputs = tokenizers(dialogue,
                       padding="max_length",
                       max_length = 512,
                       truncation= True,
                       return_tensors = "pt").to(device)

    model.to(device)
    model.eval()

    targets = model.generate(input_ids =inputs["input_ids"],
                            attention_mask = inputs["attention_mask"],
                            max_length=150,
                            min_length=10,
                            num_beams=4,
                            early_stopping=True)

    summary = tokenizers.decode(targets[0], 
                               skip_special_tokens=True)


    return summary


# API ENDPOINT

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput) :
    summary = summarization_text(dialogue_input.dialogue)
    return {"summary":summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) :
    return template.TemplateResponse(request=request, name="index.html")