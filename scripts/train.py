import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Tokenizer
from datasets import load_dataset
from src.transformer import create_model
from src.trainer import Trainer, TextDataset

def load_dataset_texts(dataset_name="wikipedia", subset=None, split="train", max_samples=20000):
    """Load a proper dataset for training with focus on factual content"""
    try:
        print(f"Loading {dataset_name} dataset...")
        
        # Load dataset from Hugging Face
        if dataset_name == "wikipedia":
            # Load Wikipedia dataset with multiple languages if available
            try:
                dataset = load_dataset("wikipedia", "20220301.en", split=f"{split}[:{max_samples}]")
            except:
                dataset = load_dataset("wikipedia", "20220301.simple", split=f"{split}[:{max_samples}]") 
        elif dataset_name == "wikitext":
            dataset = load_dataset(dataset_name, subset or "wikitext-103-raw-v1", split=split)
        elif dataset_name == "openwebtext":
            dataset = load_dataset("openwebtext", split=f"{split}[:{max_samples}]")
        else:
            dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
        
        # Extract and format text content for Q&A
        texts = []
        for item in dataset:
            text = item.get('text', '').strip()
            
            # Filter and clean text
            if text and len(text) > 100:  # Longer texts for better factual content
                # Clean Wikipedia markup
                text = clean_wikipedia_text(text)
                
                # Format as Q&A style if it contains factual information
                formatted_text = format_as_qa(text)
                texts.append(formatted_text)
                
                if len(texts) >= max_samples:
                    break
        
        print(f"Loaded {len(texts)} text samples")
        return texts
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to WikiText...")
        try:
            return load_wikitext_fallback(max_samples)
        except:
            print("Falling back to sample data...")
            return load_sample_data()

def clean_wikipedia_text(text):
    """Clean Wikipedia text for better training"""
    import re
    
    # Remove wiki markup
    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
    text = re.sub(r'\[\[([^|]*)\|([^]]*)\]\]', r'\2', text)  # Links with alt text
    text = re.sub(r'\[\[([^]]*)\]\]', r'\1', text)  # Simple links
    text = re.sub(r'<[^>]*>', '', text)  # HTML tags
    text = re.sub(r'=+ .* =+', '', text)  # Section headers
    text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines
    
    return text.strip()

def format_as_qa(text):
    """Format text to encourage factual Q&A responses"""
    # Add some Q&A formatting to encourage better responses
    sentences = text.split('.')[:3]  # Take first few sentences
    if len(sentences) >= 2:
        # Create implicit Q&A format
        topic = sentences[0].strip()
        facts = '. '.join(sentences[1:]).strip()
        
        return f"Question: What is {topic.split()[0] if topic.split() else 'this'}?\nAnswer: {topic}. {facts}."
    
    return text

def load_wikitext_fallback(max_samples):
    """Fallback to WikiText dataset"""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{max_samples}]")
    texts = []
    
    for item in dataset:
        text = item.get('text', '').strip()
        if text and len(text) > 100:
            texts.append(clean_wikipedia_text(text))
            
    return texts

def load_sample_data():
    """Fallback sample data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "Transformers have become the dominant architecture in NLP.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Large language models can generate human-like text.",
        "Training neural networks requires careful optimization.",
        "Backpropagation is the algorithm used to train neural networks.",
        "GPUs accelerate deep learning computations significantly.",
    ] * 100
    
    return sample_texts

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    print("Loading data...")
    # Try different datasets in order of preference for Q&A
    datasets_to_try = [
        ("wikipedia", None),
        # ("wikitext", "wikitext-103-raw-v1"), 
        # ("wikitext", "wikitext-2-raw-v1")
    ]
    
    texts = None
    for dataset_name, subset in datasets_to_try:
        try:
            texts = load_dataset_texts(dataset_name, subset, max_samples=15000)
            break
        except:
            continue
    
    if texts is None:
        texts = load_sample_data()
    
    # Create dataset with longer sequences for better context
    dataset = TextDataset(texts, tokenizer, max_length=256)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model optimized for Q&A
    model = create_model(vocab_size=tokenizer.vocab_size, model_size='qa')
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, tokenizer, device)
    
    # Train the model
    print("\nStarting training...")
    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        epochs=20,
        save_dir="../models"
    )
    
    # Test generation
    print("\nTesting generation...")
    prompts = [
        "The quick brown",
        "Machine learning is",
        "Neural networks",
    ]
    
    for prompt in prompts:
        generated = trainer.generate(prompt, max_length=50, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {generated}")
        print("-" * 50)

if __name__ == "__main__":
    main()