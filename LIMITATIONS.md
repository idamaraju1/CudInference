# ONNX Engine Limitations

**TL;DR:** This engine can only run **very simple neural networks**. It cannot run language models, image models (CNNs), transformers, or most real-world models.

## What CAN Run

✅ **Simple feedforward networks** with:
- Linear/Dense layers (via Gemm/MatMul)
- ReLU activation
- Element-wise addition (with scalar broadcasting)
- Residual/skip connections

✅ **Example models that work:**
- Simple MLPs (multi-layer perceptrons)
- Small fully-connected networks
- Basic autoencoders (if only using supported ops)

## Critical Limitations

### 1. Very Limited Operations (Only 4 Implemented!)

**Implemented:**
- `MatMul` - Matrix multiplication
- `Gemm` - General matrix multiply (with bias, alpha=1, beta=1 only)
- `ReLU` - Rectified linear activation
- `Add` - Element-wise addition (scalar broadcasting only)

**Recognized but NOT implemented:**
- `Conv`, `MaxPool`, `Flatten`, `Reshape`, `Softmax`, `BatchNorm`
- And hundreds of other ONNX operators...

**What this means:**
- ❌ No CNNs (needs Conv, MaxPool, BatchNorm)
- ❌ No Transformers (needs LayerNorm, Softmax, multi-head attention)
- ❌ No LSTMs/GRNs (needs special RNN ops)
- ❌ No modern activations (GELU, Swish, etc.)
- ❌ No dropout, normalization layers

### 2. Data Type Support

**Supported:**
- ✅ `FLOAT32` only

**Not supported:**
- ❌ `FLOAT16` / `BFLOAT16` (common in modern models for speed)
- ❌ `INT8` / `UINT8` (quantized models)
- ❌ `INT32` / `INT64` (shape/index tensors)

**What this means:**
- ❌ Cannot run quantized models
- ❌ Cannot run mixed-precision models
- ❌ Cannot run models optimized for inference

### 3. No Dynamic Shapes

- All tensor shapes must be known at parse time
- Cannot handle:
  - Variable-length sequences (like text)
  - Dynamic batch sizes
  - Attention masks with varying lengths

**What this means:**
- ❌ No language models (they use dynamic sequence lengths)
- ❌ No batch processing with varying sizes

### 4. Limited Broadcasting

- Only scalar broadcasting in Add operation
- No general tensor broadcasting

**What this means:**
- ❌ Cannot handle many common operations like bias addition across dimensions

### 5. No External Data Support

- Weights must be embedded in the .onnx file
- Large models (>2GB) typically use external data files

**What this means:**
- ❌ Cannot load models larger than ~2GB
- ❌ Most real-world models won't load

### 6. GEMM Constraints

- Only supports `alpha=1.0` and `beta=1.0`
- PyTorch Linear layers work because they use these defaults
- Custom GEMM operations with different alpha/beta will fail

### 7. No Graph Optimizations

- No operator fusion
- No memory optimization
- No subgraph execution
- Every operation runs independently

**What this means:**
- Slower than production ONNX runtimes (ONNX Runtime, TensorRT)

## Real-World Model Compatibility

| Model Type | Can Run? | Why Not? |
|------------|----------|----------|
| **Simple MLP** | ✅ YES | Uses only Linear + ReLU + Add |
| **LeNet-5** | ❌ NO | Needs Conv2D, MaxPool |
| **ResNet** | ❌ NO | Needs Conv2D, BatchNorm, MaxPool |
| **VGG** | ❌ NO | Needs Conv2D, MaxPool |
| **BERT** | ❌ NO | Needs LayerNorm, Softmax, Attention, GELU |
| **GPT** | ❌ NO | Needs LayerNorm, Softmax, Attention, GELU |
| **YOLO** | ❌ NO | Needs Conv2D, many other ops |
| **Stable Diffusion** | ❌ NO | Needs Conv2D, GroupNorm, Attention, etc. |
| **Whisper** | ❌ NO | Needs Conv1D, Attention, LayerNorm |

## Test It Yourself

Try loading a real model:

```python
# Download a simple CNN model
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")
```

Then run it:
```bash
./build/onnx_gpu_engine resnet18.onnx
```

**Result:** Will fail with "Unsupported operation: Conv" after parsing.

## What This Engine Is Good For

This is a **learning/educational project** for:
- ✅ Understanding ONNX format structure
- ✅ Learning CUDA kernel development
- ✅ Experimenting with custom operators
- ✅ Testing simple feedforward networks
- ✅ Prototyping custom inference engines

## To Support Real Models

You would need to implement:

**Minimum for CNNs (100+ hours):**
- Conv2D, Conv2DTranspose
- MaxPool2D, AvgPool2D
- BatchNorm, InstanceNorm
- Softmax, LogSoftmax
- Flatten, Reshape, Transpose
- Concat, Split
- Dropout (inference mode)

**Minimum for Transformers (200+ hours):**
- LayerNorm, GroupNorm
- Softmax
- MatMul with broadcasting
- Reshape, Transpose, Slice, Gather
- GELU, Tanh activation
- Multi-dimensional broadcasting
- Attention mechanisms
- Dynamic shapes

**Minimum for Production (500+ hours):**
- All of the above
- INT8/FP16 support
- Graph optimizations
- Operator fusion
- Memory pool management
- Multi-GPU support
- Dynamic batching
- 100+ more ONNX operators

## Recommended Alternatives

For running real models, use:
- **ONNX Runtime** - Official runtime, supports all ops
- **TensorRT** - NVIDIA's optimized runtime
- **OpenVINO** - Intel's runtime
- **TVM** - Compiler-based approach

This engine is great for learning but not for production use.
