#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script runs after downloading the Gemma3 model files from 
https://huggingface.co/google/gemma-3-1b-it 

They should be kept together in a directory and model_path should
be updated accordingly
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your local model directory
model_path = "/home/aiden/Downloads/gemma3"

print(f"Loading Gemma 3 from {model_path} ...")

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,   # safest for CPU
    device_map="cpu"
)

# Input prompt
prompt = "Explain why the sky appears blue during the day."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

# Decode and print
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Model Output ---")
print(result)
