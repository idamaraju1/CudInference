# Quick Start Guide

Get OnnxRunner running in 5 minutes!

## Prerequisites Check

Run the dependency checker:
```bash
./scripts/check_dependencies.sh
```

If any dependencies are missing, install them following the instructions.

## Build Steps

### 1. Setup ONNX Protobuf Files

```bash
./scripts/setup_onnx_proto.sh
```

This downloads and compiles the ONNX schema definitions.

### 2. Configure GPU Architecture

Edit `CMakeLists.txt` line ~10:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
```

**Find your GPU's compute capability:**
- RTX 20 series (Turing): 75
- RTX 30 series (Ampere): 86
- RTX 40 series (Ada): 89
- Full list: https://developer.nvidia.com/cuda-gpus

### 3. Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Build time: ~2-5 minutes depending on your system.

## Create a Test Model

Using Python (requires PyTorch):
```bash
cd ..  # Back to project root
python3 scripts/create_test_model.py
```

This creates three test models:
- `simple_linear.onnx` - Linear layer + ReLU
- `two_layer.onnx` - Two-layer network
- `residual.onnx` - Residual block with Add operation

## Run Your First Model

```bash
./build/onnx_gpu_engine simple_linear.onnx
```

Expected output:
```
[HH:MM:SS] [INFO ] === OnnxRunner GPU Engine ===
[HH:MM:SS] [INFO ] Model: simple_linear.onnx
[HH:MM:SS] [INFO ] Device: GPU
[HH:MM:SS] [INFO ] Parsing ONNX model: simple_linear.onnx
...
[HH:MM:SS] [INFO ] === Execution Complete ===
```

## Try Different Options

**Verbose mode (show timing for each operation):**
```bash
./build/onnx_gpu_engine two_layer.onnx --verbose
```

**Debug mode (detailed logging):**
```bash
./build/onnx_gpu_engine residual.onnx --debug
```

**CPU fallback (for testing/debugging):**
```bash
./build/onnx_gpu_engine simple_linear.onnx --cpu
```

**Benchmark mode (compare CPU multi-threading vs GPU):**
```bash
./build/onnx_gpu_engine two_layer.onnx --benchmark
```

This will:
- Run the model on CPU with 1, 2, 3, ..., N threads (auto-detected)
- Run the model on GPU
- Show live progress with timing comparisons
- Save results to `results.json`

**Visualize benchmark results:**
```bash
# Open in your browser
firefox visualization/benchmark_viewer.html
```
Then click "Load File" and select `results.json` to see:
- Interactive statistics dashboard
- Animated performance race
- Side-by-side bar chart comparison

## Using Your Own ONNX Model

Place your `.onnx` file in the project directory and run:
```bash
./build/onnx_gpu_engine your_model.onnx
```

**Note:** Currently supported operations:
- MatMul / Gemm
- ReLU
- Add (element-wise)

Models using other operations will show an "Unsupported operation" error.

## Troubleshooting

**"CUDA error: no CUDA-capable device is detected"**
- Run with `--cpu` flag to use CPU fallback
- Check `nvidia-smi` to verify GPU is detected

**"Unsupported operation: Conv"**
- The model uses operations not yet implemented
- See README.md "Supported Operations" section

**Build fails with "cuda_runtime.h not found"**
- Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`
- Add to PATH: `export PATH=$CUDA_HOME/bin:$PATH`

**"Failed to parse ONNX model"**
- Ensure protobuf files are generated (Step 1)
- Check ONNX model is valid: `python3 -c "import onnx; onnx.checker.check_model('model.onnx')"`

## Next Steps

1. **Explore the code**: Start with `src/main.cpp` to understand the flow
2. **Add new operations**: See README.md "Extending the Engine" section
3. **Optimize kernels**: Experiment with different CUDA optimization techniques
4. **Profile performance**: Use NVIDIA Nsight Systems for detailed profiling

## Example: Export Your Own Model

```python
import torch
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.net(x)

# Export to ONNX
model = MyModel()
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "my_model.onnx", opset_version=13)
```

Then run:
```bash
./build/onnx_gpu_engine my_model.onnx --verbose
```

Happy hacking! 🚀
