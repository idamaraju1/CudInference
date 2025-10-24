import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import GPT2Tokenizer
from src.transformer import create_model
from src.trainer import Trainer

def load_model(model_path, vocab_size, model_size='small', device='cuda'):
    """Load a trained model from checkpoint"""
    model = create_model(vocab_size, model_size)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        print(f"Epoch: {checkpoint['epoch']}, Eval Loss: {checkpoint['eval_loss']:.4f}")
    else:
        print(f"No checkpoint found at {model_path}, using randomly initialized model")
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_path = "../models/best_model.pt"
    model = load_model(model_path, tokenizer.vocab_size, 'small', device)
    
    # Create trainer for generation
    trainer = Trainer(model, tokenizer, device)
    
    print("\nInteractive text generation. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        try:
            # Generate with different settings
            print("\n--- Standard Generation ---")
            generated = trainer.generate(prompt, max_length=100, temperature=0.8, top_k=50)
            print(generated)
            
            print("\n--- Creative Generation (higher temperature) ---")
            generated = trainer.generate(prompt, max_length=100, temperature=1.2, top_k=40)
            print(generated)
            
            print("\n--- Conservative Generation (lower temperature) ---")
            generated = trainer.generate(prompt, max_length=100, temperature=0.5, top_k=30)
            print(generated)
            
        except Exception as e:
            print(f"Error generating text: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()