import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            encoded = tokenizer.encode(text, truncation=True, max_length=max_length)
            if len(encoded) > 1:
                self.examples.append(encoded)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1])
        target_ids = torch.tensor(tokens[1:])
        
        return input_ids, target_ids

class Trainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.scaler = GradScaler()
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        self.train_losses = []
        self.eval_losses = []
        
    def train_epoch(self, dataloader, accumulation_steps=4):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            with autocast(device_type='cuda'):
                logits = self.model(input_ids)
                loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                with autocast(device_type='cuda'):
                    logits = self.model(input_ids)
                    loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                        logits.view(-1, logits.size(-1)), target_ids.view(-1)
                    )
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_dataloader, eval_dataloader, epochs=10, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
        best_eval_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch(train_dataloader)
            eval_loss = self.evaluate(eval_dataloader)
            
            self.train_losses.append(train_loss)
            self.eval_losses.append(eval_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print("Saved best model!")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
        
        self.plot_losses(save_dir)
    
    def plot_losses(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.eval_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50):
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                with autocast(device_type='cuda'):
                    logits = self.model(input_ids)
                
                # Get logits for last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        return generated_text