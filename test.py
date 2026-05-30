import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.getenv("MODEL_NAME", "eugenieee/synthdiary")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Today, I"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=1.5,
        top_p=0.95,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)