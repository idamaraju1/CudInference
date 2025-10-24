import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Use dtype-safe minimum value to avoid fp16 overflow and boolean mask
            min_val = torch.finfo(scores.dtype).min if scores.dtype.is_floating_point else -1e9
            mask_bool = mask if mask.dtype == torch.bool else mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask_bool, min_val)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.to(input_ids.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            causal_mask = causal_mask * attention_mask
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

def create_model(vocab_size, model_size='small'):
    configs = {
        'small': {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
        'medium': {'d_model': 512, 'n_heads': 8, 'n_layers': 8, 'd_ff': 2048},
        'large': {'d_model': 768, 'n_heads': 12, 'n_layers': 12, 'd_ff': 3072},
        'qa': {'d_model': 512, 'n_heads': 8, 'n_layers': 10, 'd_ff': 2048, 'max_seq_length': 1024}
    }
    
    config = configs[model_size]
    return TransformerLM(vocab_size, **config)