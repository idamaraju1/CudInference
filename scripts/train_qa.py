import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Tokenizer
from datasets import load_dataset
from src.transformer import create_model
from src.trainer import Trainer, TextDataset

def create_qa_examples():
    """Create fact-based Q&A examples for better training"""
    qa_examples = [
        "Question: What is machine learning?\nAnswer: Machine learning is a subset of artificial intelligence that uses algorithms and statistical models to enable computers to learn and make decisions from data without being explicitly programmed.",
        
        "Question: What is photosynthesis?\nAnswer: Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. This process occurs in chloroplasts and is essential for life on Earth.",
        
        "Question: Who was Albert Einstein?\nAnswer: Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He is widely regarded as one of the greatest and most influential physicists of all time.",
        
        "Question: What is DNA?\nAnswer: DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, and reproduction of all known living organisms.",
        
        "Question: What is the speed of light?\nAnswer: The speed of light in a vacuum is approximately 299,792,458 meters per second, commonly denoted as 'c' in physics equations.",
        
        "Question: What is gravity?\nAnswer: Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, it gives weight to physical objects and causes them to fall.",
        
        "Question: What is the periodic table?\nAnswer: The periodic table is a tabular arrangement of chemical elements organized by their atomic number, electron configuration, and recurring chemical properties.",
        
        "Question: What is evolution?\nAnswer: Evolution is the process by which species of organisms change over time through genetic variation and natural selection, leading to the development of new species.",
        
        "Question: What is the solar system?\nAnswer: The solar system consists of the Sun and all celestial objects bound to it by gravity, including eight planets, their moons, asteroids, comets, and other objects.",
        
        "Question: What is water?\nAnswer: Water is a transparent, tasteless, odorless chemical compound with the molecular formula H2O. It consists of two hydrogen atoms bonded to one oxygen atom."
    ] * 50  # Repeat for more training examples
    
    return qa_examples

def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create Q&A training data
    print("Creating Q&A training data...")
    qa_texts = create_qa_examples()
    
    # Try to load additional Wikipedia data
    try:
        print("Loading additional Wikipedia data...")
        wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
        
        for item in wiki_dataset:
            text = item.get('text', '').strip()
            if text and len(text) > 100:
                # Convert to Q&A format
                sentences = text.split('.')[:2]
                if len(sentences) >= 2:
                    topic = sentences[0].strip()
                    info = sentences[1].strip()
                    qa_text = f"Question: What is {topic.split()[-1] if topic.split() else 'this'}?\nAnswer: {topic}. {info}."
                    qa_texts.append(qa_text)
                    
    except Exception as e:
        print(f"Could not load Wikipedia data: {e}")
        print("Using only basic Q&A examples...")
    
    print(f"Total training examples: {len(qa_texts)}")
    
    # Create dataset
    dataset = TextDataset(qa_texts, tokenizer, max_length=256)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create Q&A optimized model
    model = create_model(vocab_size=tokenizer.vocab_size, model_size='qa')
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, tokenizer, device)
    
    print("\nStarting Q&A training...")
    
    # Train the model
    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        epochs=30,  # More epochs for better learning
        save_dir="models"
    )
    
    print("Training completed! Model saved as 'models/best_model.pt'")
    print("\nTo test the model, run:")
    print("python scripts/chat.py --model models/best_model.pt")

if __name__ == "__main__":
    main()