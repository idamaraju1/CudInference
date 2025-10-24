import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from src.transformer import create_model
import argparse

class ChatBot:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            # Create model architecture optimized for Q&A
            model = create_model(vocab_size=self.tokenizer.vocab_size, model_size='qa')
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            print(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating fresh model (untrained)")
            model = create_model(vocab_size=self.tokenizer.vocab_size, model_size='qa')
            model.to(self.device)
            return model
    
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text continuation from prompt"""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_ids)
                logits = outputs[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits[logits < values[:, -1, None]] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check if we hit end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Stop if sequence gets too long
                if input_ids.size(1) > 512:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def format_qa_prompt(self, question):
        """Format user input as a Q&A prompt"""
        if not question.strip().endswith('?'):
            # If not a question, make it one
            return f"Question: {question}?\nAnswer:"
        return f"Question: {question}\nAnswer:"
    
    def chat(self):
        """Interactive chat loop"""
        print("=== DTL Wikipedia Q&A Bot ===")
        print("Ask me factual questions about topics in Wikipedia!")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help - Show this help")
                    print("  quit/exit/q - End conversation")
                    print("  clear - Clear conversation history")
                    print("\nExample questions:")
                    print("  What is machine learning?")
                    print("  Tell me about Albert Einstein")
                    print("  What is photosynthesis?")
                    continue
                
                if user_input.lower() == 'clear':
                    print("Conversation cleared!")
                    continue
                    
                if not user_input:
                    continue
                
                # Format as Q&A and generate response
                qa_prompt = self.format_qa_prompt(user_input)
                print("\nBot: ", end="", flush=True)
                response = self.generate_text(
                    qa_prompt, 
                    max_length=80,
                    temperature=0.7,  # Lower temperature for more factual responses
                    top_k=40,
                    top_p=0.8
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

def main():
    parser = argparse.ArgumentParser(description='Chat with your DTL language model')
    parser.add_argument('--model', '-m', type=str, default='models/best_model.pt',
                       help='Path to trained model file')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Warning: Model file {args.model} not found. Will use untrained model.")
    
    # Create chatbot
    chatbot = ChatBot(args.model, args.device)
    
    # Start chat
    chatbot.chat()

if __name__ == "__main__":
    main()