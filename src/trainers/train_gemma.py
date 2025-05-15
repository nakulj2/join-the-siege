from transformers import pipeline
import torch
from PIL import Image
from utils.extract_text import extract_text

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "text2text-generation",
    model="google/gemma-3-4b-it",
    device=device,
    torch_dtype=torch.bfloat16 if device == 0 else torch.float32
)

prompt = extract_text("/Users/nakuljain/Desktop/HERON DATA/join-the-siege/data/drivers_license/drivers_license_1.jpg")
system_prompt = "What type of document is this? Only respond with one of the classifications: bank_statement, drivers_license, invoice."

# Construct messages in the proper format
messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]}
]

# Run the pipeline with the message structure
output = pipe(text=messages, max_new_tokens=100)

print(output)
