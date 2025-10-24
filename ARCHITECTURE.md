# DTL Architecture Documentation

This document provides visual and textual explanations of how the DTL transformer model works.

## High-Level Model Architecture

The TransformerLM is a decoder-only transformer designed for language modeling. Here's how data flows through the model:

```mermaid
graph TD
    Input["Input IDs<br/>(batch_size, seq_length)"]
    Embed["Embedding Layer<br/>(vocab_size → d_model)"]
    PosEnc["Positional Encoding<br/>(add position info)"]

    Block1["Transformer Block 1<br/>(Multi-Head Attention<br/>+ Feed-Forward)"]
    Block2["Transformer Block 2"]
    BlockN["Transformer Block N"]

    LayerNorm["Layer Normalization"]
    Head["Linear Head<br/>(d_model → vocab_size)"]
    Output["Logits<br/>(batch_size, seq_length, vocab_size)"]

    Input --> Embed
    Embed --> PosEnc
    PosEnc --> Block1
    Block1 --> Block2
    Block2 --> BlockN
    BlockN --> LayerNorm
    LayerNorm --> Head
    Head --> Output
```

## Transformer Block Deep Dive

Each transformer block consists of two main components: multi-head attention and a feed-forward network, both with residual connections and layer normalization.

```mermaid
graph TD
    X["Input: x<br/>(batch, seq_len, d_model)"]

    Attn["Multi-Head Attention<br/>(self-attention)"]
    AttnOut["Attention Output<br/>(batch, seq_len, d_model)"]

    DropAttn["Dropout"]
    Add1["Add & Normalize<br/>(Residual + LayerNorm)"]

    FF["Feed-Forward Network<br/>(Linear → ReLU → Linear)"]
    FFOut["FF Output<br/>(batch, seq_len, d_model)"]

    DropFF["Dropout"]
    Add2["Add & Normalize<br/>(Residual + LayerNorm)"]

    Output["Output: x'<br/>(batch, seq_len, d_model)"]

    X --> Attn
    Attn --> AttnOut
    AttnOut --> DropAttn
    X -.->|Residual| Add1
    DropAttn --> Add1

    Add1 --> FF
    FF --> FFOut
    FFOut --> DropFF
    Add1 -.->|Residual| Add2
    DropFF --> Add2

    Add2 --> Output
```

## Multi-Head Attention Mechanism

Multi-head attention allows the model to attend to different representation subspaces. Here's how it works:

```mermaid
graph TD
    Q["Query: Q<br/>(batch, seq_len, d_model)"]
    K["Key: K<br/>(batch, seq_len, d_model)"]
    V["Value: V<br/>(batch, seq_len, d_model)"]

    WQ["Linear Projection<br/>(W_Q)"]
    WK["Linear Projection<br/>(W_K)"]
    WV["Linear Projection<br/>(W_V)"]

    Reshape["Reshape for<br/>Multi-Head<br/>(batch, n_heads, seq_len, d_k)"]

    Score["Attention Scores<br/>Q·K^T / √d_k"]
    Mask["Apply Causal Mask<br/>(prevent future tokens)"]
    Softmax["Softmax<br/>(normalize weights)"]

    Context["Weighted Sum<br/>(attention × V)"]

    Concat["Concatenate Heads"]
    WO["Output Projection<br/>(W_O)"]

    Output["Attention Output<br/>(batch, seq_len, d_model)"]

    Q --> WQ
    K --> WK
    V --> WV

    WQ --> Reshape
    WK --> Reshape
    WV --> Reshape

    Reshape --> Score
    Score --> Mask
    Mask --> Softmax
    Softmax --> Context

    Context --> Concat
    Concat --> WO
    WO --> Output
```

## Feed-Forward Network

Simple but effective network that processes each position independently:

```mermaid
graph LR
    X["Input<br/>(batch, seq_len, d_model)"]

    Linear1["Linear Layer<br/>(d_model → d_ff)<br/>d_ff = 4 × d_model"]
    ReLU["ReLU Activation<br/>max(0, x)"]
    Linear2["Linear Layer<br/>(d_ff → d_model)"]

    Output["Output<br/>(batch, seq_len, d_model)"]

    X --> Linear1
    Linear1 --> ReLU
    ReLU --> Linear2
    Linear2 --> Output
```

## Positional Encoding

Since transformers have no built-in notion of sequence position, we add positional information using sinusoidal functions:

```mermaid
graph TD
    Embed["Token Embeddings<br/>(batch, seq_len, d_model)"]

    PE["Positional Encodings<br/>PE(pos, 2i) = sin(pos/10000^(2i/d_model))<br/>PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))"]

    Add["Element-wise Addition"]

    Output["Position-Aware Embeddings<br/>(batch, seq_len, d_model)"]

    Embed --> Add
    PE --> Add
    Add --> Output
```

## Complete Forward Pass Example

Let's trace a single example through the entire model:

```mermaid
sequenceDiagram
    participant Input as Input IDs
    participant Embed as Embedding
    participant PosEnc as Positional<br/>Encoding
    participant Block as Transformer<br/>Block
    participant Head as Output<br/>Head
    participant Logits as Logits

    Input->>Embed: [1, 45, 234, 512]
    Embed->>Embed: Convert to embeddings<br/>(4, 256)
    Embed->>PosEnc: Add position info
    PosEnc->>Block: (4, 256)
    Block->>Block: Self-attention<br/>(4, 256)
    Block->>Block: Feed-forward<br/>(4, 256)
    Block->>Head: (4, 256)
    Head->>Head: Project to vocab<br/>(4, 50257)
    Head->>Logits: (4, 50257)
    Logits->>Logits: Softmax + sample<br/>next token
```

## Causal Masking (Autoregressive)

For language modeling, we prevent the model from seeing future tokens. This is done with causal masking in attention:

```mermaid
graph TD
    Seq["Sequence of length 4<br/>Positions: 0, 1, 2, 3"]

    Mask["Causal Mask Matrix<br/>(4x4)"]

    Attention["Attention Weights<br/>at Position 2"]

    Seq --> Mask
    Mask --> Attention

    subgraph mask["Causal Mask (✓ = allowed, ✗ = blocked)"]
        M["Position 0: ✓ [0]<br/>Position 1: ✓ [0,1]<br/>Position 2: ✓ [0,1,2]<br/>Position 3: ✓ [0,1,2,3]"]
    end

    Attention --> M
```

## Model Configuration Sizes

Different model configurations for different hardware constraints:

```mermaid
graph LR
    Small["Small<br/>d_model: 256<br/>n_heads: 8<br/>n_layers: 6<br/>d_ff: 1024<br/>~27M params"]

    Medium["Medium<br/>d_model: 512<br/>n_heads: 8<br/>n_layers: 8<br/>d_ff: 2048<br/>~85M params"]

    Large["Large<br/>d_model: 768<br/>n_heads: 12<br/>n_layers: 12<br/>d_ff: 3072<br/>~355M params"]

    QA["QA Optimized<br/>d_model: 512<br/>n_heads: 8<br/>n_layers: 10<br/>d_ff: 2048<br/>max_seq: 1024<br/>~110M params"]

    Small -.-> Medium
    Medium -.-> Large
    Small -.-> QA
```

## Training Data Flow

How data flows through training:

```mermaid
graph TD
    Raw["Raw Text Data<br/>Wikipedia, WikiText"]

    Clean["Text Cleaning<br/>Remove markup, templates"]

    Format["Format as Q&A<br/>Question: ...<br/>Answer: ..."]

    Tokenize["Tokenization<br/>GPT2Tokenizer<br/>(50257 vocab)"]

    Pad["Padding to<br/>max_length: 512<br/>or 1024 for Q&A"]

    Pair["Create Input/Target Pairs<br/>input: [token_0, ..., token_n-1]<br/>target: [token_1, ..., token_n]"]

    Batch["Batching<br/>batch_size: 4-8"]

    Model["Forward Pass<br/>Compute Logits"]

    Loss["CrossEntropyLoss<br/>Compare predictions<br/>vs targets"]

    Backward["Backward Pass<br/>Compute Gradients"]

    Update["Update Weights<br/>AdamW Optimizer<br/>with LR Scheduling"]

    Raw --> Clean
    Clean --> Format
    Format --> Tokenize
    Tokenize --> Pad
    Pad --> Pair
    Pair --> Batch
    Batch --> Model
    Model --> Loss
    Loss --> Backward
    Backward --> Update
```

## Generation Process

How the model generates new text token-by-token:

```mermaid
sequenceDiagram
    participant Prompt as Input Prompt
    participant Encode as Tokenizer
    participant Model as Model
    participant Sample as Sampling
    participant Output as Generated Text
  
    Prompt->>Encode: "What is machine"
    Encode->>Model: [token_ids]
  
    loop Generate Tokens (max_length times)
        Model->>Model: Forward pass<br/>get logits for last token
        Model->>Sample: logits
        Sample->>Sample: Apply temperature
        Sample->>Sample: Apply top-k filtering
        Sample->>Sample: Multinomial sampling
        Sample->>Model: next_token_id
        Model->>Model: Append to sequence
  
        alt Stop condition met
            Model->>Output: Generated text
        end
    end
```

## Key Concepts

### Self-Attention
Each token can "attend to" all previous tokens in the sequence. The attention mechanism learns which tokens are most relevant for predicting the next token.

### Residual Connections
`x' = LayerNorm(x + f(x))` helps with training deep networks by allowing gradients to flow directly through skip connections.

### Layer Normalization
Normalizes activations across the d_model dimension, stabilizing training and improving convergence.

### Gradient Accumulation
Accumulates gradients over multiple batches before updating weights, effectively increasing batch size without requiring more GPU memory.

### Mixed Precision Training
Uses float16 for forward/backward passes (faster, less memory) but float32 for weight updates (more stable).

## Parameter Count Example

For the 'small' model with vocab_size=50257:

```
Embedding: 50257 × 256 = ~12.9M
Position Encoding: 5000 × 256 = ~1.3M (not counted in parameters)

Per Transformer Block:
  - Multi-Head Attention:
    - W_Q, W_K, W_V, W_O: 4 × (256 × 256) = ~262K
  - Feed-Forward:
    - Linear1: 256 × 1024 = ~262K
    - Linear2: 1024 × 256 = ~262K
  Total per block: ~786K

6 blocks: ~4.7M

Final LayerNorm: 256
Output Head: 256 × 50257 = ~12.9M

Total: ~12.9M + 4.7M + 12.9M ≈ 30.5M parameters
```

## Memory Requirements

During training with a batch of 8 sequences (256 tokens each):

```
Activations: 8 × 256 × d_model × num_blocks × 2 (forward + backward)
Parameters: model_params × (forward + backward + optimizer states)
KV Cache (inference): seq_len × 2 × num_layers × d_model (for generation)

Example (small model, batch_size=8):
Forward: ~2.5 GB
Backward: ~2.5 GB
Optimizer states: ~1.5 GB
Total: ~6.5 GB (RTX 4060 has 8GB)
```

