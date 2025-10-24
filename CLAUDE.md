# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DTL (Deep Learning Transformer) is a lightweight PyTorch implementation of a transformer-based language model optimized for training on consumer GPUs (RTX 4060). The project includes both general language modeling and Q&A-specific training capabilities with an interactive chatbot interface.

## Architecture Overview

### Core Components

The transformer implementation (`src/transformer.py`) consists of:

1. **MultiHeadAttention** (src/transformer.py:6-39): Self-attention mechanism with Q, K, V projections. Includes dtype-safe masking for fp16 compatibility.

2. **TransformerBlock** (src/transformer.py:50-66): Single transformer layer combining multi-head attention + feed-forward network with layer normalization and residual connections.

3. **TransformerLM** (src/transformer.py:86-134): Complete language model with:
   - Token embedding + positional encoding
   - Stack of transformer blocks
   - Causal masking (autoregressive)
   - Linear head for logits

4. **Model Factory** (src/transformer.py:136-145): `create_model()` provides preconfigured models:
   - `small`: 256 dim, 8 heads, 6 layers (memory-efficient)
   - `medium`: 512 dim, 8 heads, 8 layers
   - `large`: 768 dim, 12 heads, 12 layers
   - `qa`: 512 dim, 8 heads, 10 layers, max_seq_length=1024 (Q&A optimized)

### Training Infrastructure (`src/trainer.py`)

- **TextDataset**: Tokenizes and pads text to fixed length, creates input/target pairs for next-token prediction
- **Trainer**: Handles:
  - Mixed precision training (autocast + GradScaler)
  - Gradient accumulation for effective larger batch sizes
  - Cosine annealing learning rate scheduler
  - Best model and checkpoint checkpointing
  - Generation with temperature and top-k sampling

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training

**General language model training (Wikipedia-based):**
```bash
python scripts/train.py
```
- Uses Wikipedia dataset (50k samples max)
- Trains on 256-token sequences
- Batch size: 8, Epochs: 20
- Saves to `models/best_model.pt` and checkpoints every 5 epochs

**Q&A-specific training:**
```bash
python scripts/train_qa.py
```
- Uses handcrafted Q&A examples + Wikipedia data
- Trains Q&A-optimized model config (1024 max_seq_length)
- Batch size: 4, Epochs: 30
- Better suited for question-answering tasks

### Inference & Chat

**Interactive text generation:**
```bash
python scripts/generate.py
```
- Tests different temperature/top-k settings (standard, creative, conservative)
- Load from `../models/best_model.pt`

**Q&A Chatbot:**
```bash
python scripts/chat.py --model models/best_model.pt --device cuda
```
- Interactive Q&A interface
- Auto-formats user input as "Question: {input}\nAnswer:"
- Commands: `help`, `quit`, `exit`, `clear`
- Optional args: `--model` (model path), `--device` (cuda/cpu)

### Diagnostics
```bash
python test.py
```
- Quick CUDA availability check and version info

## Key Design Decisions

### Memory Optimization
- Gradient accumulation (4 steps) enables larger effective batch sizes on limited VRAM
- Mixed precision (fp16) reduces memory footprint and speeds training
- Configurable model sizes for different hardware constraints
- Causal masking uses dtype-safe minimum values to prevent fp16 overflow

### Data Processing
- WikiText/Wikipedia datasets with text cleaning (removes markup, templates, links)
- Padding to fixed sequence length for efficient batching
- Q&A training uses explicit "Question:\nAnswer:" formatting to encourage factual responses

### Training Strategy
- AdamW optimizer with weight decay (L2 regularization)
- Cosine annealing reduces overfitting and improves generalization
- Separate validation set for early stopping signals
- Top-k and top-p sampling for diverse generation

## Development Workflow

1. **Modify architecture**: Edit `src/transformer.py` (model configs, attention, blocks)
2. **Modify training**: Edit `src/trainer.py` (loss, optimization, generation)
3. **Create new training task**: Add script to `scripts/` using `TextDataset` and `Trainer`
4. **Test changes**: Run appropriate script with small dataset sample first
5. **Checkpoints**: Automatically saved to `models/` directory with epoch number

## Important Notes

- Models assume GPT2Tokenizer (50257 vocab size)
- Max sequence length: 512 (default) or 1024 (Q&A config)
- Device defaults to CUDA if available, falls back to CPU
- Model checkpoints include optimizer state for resuming training
- Generation uses multinomial sampling (stochastic) not beam search
