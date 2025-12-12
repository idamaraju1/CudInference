# ONNX GPU Execution Engine

A custom ONNX inference engine built with C++17 and CUDA. This project parses ONNX model files using Protocol Buffers and executes computation graphs using custom CUDA kernels for GPU acceleration.

## Features

- Custom CUDA kernels for GPU-accelerated inference
- Support for multiple NVIDIA GPU architectures (Turing, Ampere, Ada)
- CPU fallback mode with OpenMP multi-threading support
- Comprehensive benchmarking: CPU (1-N threads) vs GPU performance comparison
- Interactive HTML visualization for benchmark results
- Performance profiling with CUDA events
- Supports common operations: MatMul, Gemm, ReLU, Add

## Prerequisites

### System Requirements

- **CUDA Toolkit**: 11.0 or later (tested with 12.4.131)
- **CMake**: 3.18 or later
- **C++ Compiler**: GCC 13 (gcc-13/g++-13) with OpenMP support
- **Protocol Buffers**: Development libraries and compiler
- **OpenMP**: For multi-threaded CPU execution (included with GCC)
- **Python 3**: For creating test models and running validation scripts (optional)

### Installing Dependencies

#### Ubuntu/Debian

```bash
# Install build tools and CUDA (if not already installed)
sudo apt-get update
sudo apt-get install cmake build-essential

# Install GCC 13
sudo apt-get install gcc-13 g++-13

# Install Protocol Buffers
sudo apt-get install protobuf-compiler libprotobuf-dev

# Install Python (for test model creation)
sudo apt-get install python3 python3-pip

# Install Python requirements
pip install -r scripts/requirements.txt
```

## Build Instructions

### Quick Setup (Automated)

For a complete automated setup that downloads everything and builds the project:

```bash
./scripts/setup/full_setup.sh
```

This will:
1. Check all dependencies
2. Download SmolLM2-135M model and tokenizer from HuggingFace
3. Setup ONNX protobuf definitions
4. Configure and build the project

**Note**: The model download is ~500MB and may take a few minutes depending on your internet connection.

### Manual Setup (Step-by-Step)

If you prefer manual control or want to use your own models:

#### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd OnnxRunner
```

#### 2. Download and Compile ONNX Proto Files

Run the setup script to download ONNX proto definitions and compile them:

```bash
./scripts/setup/setup_onnx_proto.sh
```

This will create the `third_party/onnx/` directory with compiled protobuf files.

#### 3. (Optional) Download Test Model

To download the SmolLM2-135M model for text generation:

```bash
python3 scripts/setup/download_model.py
```

This downloads:
- `model.onnx` - SmolLM2-135M language model (~500MB)
- `tokenizer.json` - HuggingFace tokenizer

#### 4. Configure GPU Architecture (Optional)

Edit `CMakeLists.txt` line 10 to match your GPU compute capability:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
```

- **75**: Turing (RTX 20 series, GTX 16 series)
- **86**: Ampere (RTX 30 series, A100)
- **89**: Ada Lovelace (RTX 40 series)

You can specify multiple architectures separated by semicolons, or just one for faster compilation.

#### 5. Build the Project

```bash
mkdir build
cd build

# Configure with GCC 13
cmake -D CMAKE_C_COMPILER=/usr/bin/gcc-13 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
      -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 ..

# Build
make -j$(nproc)
```

The executable `onnx_gpu_engine` will be created in the `build/` directory.

## Usage

### Basic Usage

```bash
./build/onnx_gpu_engine model.onnx
```

### Command-Line Options

```bash
./build/onnx_gpu_engine <model.onnx> [options]

Options:
  --cpu             Use CPU fallback instead of GPU
  --cpu-threads N   Max CPU threads for benchmark mode (default: auto-detect)
                    Benchmark will test 1 to N threads
  --verbose         Print detailed timing information
  --quiet           Suppress logs; stream generated text only
  --debug           Enable debug logging
  --benchmark       Run multi-configuration benchmark (CPU 1-N threads + GPU)
  --output FILE     Save benchmark results to JSON file (default: results.json)
  --help            Show this help message
```

### Benchmarking Mode

Run comprehensive performance benchmarks comparing CPU (single-threaded, multi-threaded) and GPU execution:

```bash
# Benchmark with auto-detected thread count (default: hardware_concurrency)
./build/onnx_gpu_engine model.onnx --benchmark

# Benchmark with specific maximum thread count
./build/onnx_gpu_engine model.onnx --benchmark --cpu-threads 8

# Save results to custom file
./build/onnx_gpu_engine model.onnx --benchmark --output my_results.json
```

The benchmark will:
1. Run the model on CPU with 1, 2, 3, ..., N threads
2. Run the model on GPU
3. Display live progress with timing comparisons
4. Save detailed results to `results.json` (or specified file)

### Visualizing Benchmark Results

Open the interactive visualization in your browser:

```bash
# Open the HTML file
firefox visualization/benchmark_viewer.html
# or
google-chrome visualization/benchmark_viewer.html
```

The viewer supports **two types of benchmarks**:

**1. Standard Benchmark** (`results.json`):
- Load using the first input section (Standard Benchmark)
- Shows execution time comparison across CPU thread counts and GPU
- Click "Load JSON" and select `results.json`, or copy/paste the JSON content

**2. Generation Benchmark** (`generation_results.json`):
- Load using the second input section (Generation Benchmark)
- Shows token generation performance (tokens/sec) across all configurations
- Click "Load JSON" and select `generation_results.json`, or copy/paste the JSON content

**Visualization Features:**
- **Statistics Dashboard**: Key metrics including speedups and execution times
- **Live Performance Race**: Animated comparison showing relative speeds (standard benchmark)
- **Execution Time Chart**: Side-by-side comparison of all configurations (standard benchmark)
- **Token Generation Chart**: Tokens per second comparison with speedup indicators (generation benchmark)
- **Interactive Controls**: Adjustable animation speed for the performance race
- **Dual View**: Load both benchmark types simultaneously to see comprehensive performance analysis

### Text Generation Mode (for LLM models)

Run language models in auto-regressive generation mode.

**Using the downloaded SmolLM2-135M model:**

If you ran `full_setup.sh` or `download_model.py`, you can use the downloaded model:

```bash
./build/onnx_gpu_engine model.onnx \
  --input "The sky is blue because" \
  --tokenizer tokenizer.json \
  --generate \
  --max-tokens 50 \
  --temperature 0.0
```

**Using a custom model:**

```bash
./build/onnx_gpu_engine /path/to/your/model.onnx \
  --input "Your input prompt" \
  --tokenizer /path/to/tokenizer.json \
  --generate \
  --max-tokens 50 \
  --temperature 0.0
```

**Required flags for generation mode:**
- `--input TEXT`: Input text prompt to generate from
- `--tokenizer FILE`: Path to tokenizer.json file (HuggingFace format)
- `--generate`: Enable auto-regressive text generation mode

**Optional flags:**
- `--max-tokens N`: Maximum number of tokens to generate (default: 50)
- `--temperature F`: Sampling temperature (0.0 = greedy/deterministic, higher = more random, default: 1.0)
- `--cpu`: Use CPU instead of GPU for generation
- `--verbose`: Print detailed timing per token

### Generation Benchmarking Mode

Compare text generation performance across different CPU thread counts and GPU:

```bash
# Benchmark generation with auto-detected thread count
./build/onnx_gpu_engine model.onnx \
  --benchmark-generation \
  --input "The sky is blue because" \
  --tokenizer tokenizer.json \
  --max-tokens 50
```

**Specify maximum thread count:**
```bash
./build/onnx_gpu_engine model.onnx \
  --benchmark-generation \
  --input "Your prompt here" \
  --tokenizer tokenizer.json \
  --max-tokens 50 \
  --cpu-threads 8
```

**Save results to custom file:**
```bash
./build/onnx_gpu_engine model.onnx \
  --benchmark-generation \
  --input "Your prompt here" \
  --tokenizer tokenizer.json \
  --max-tokens 50 \
  --output my_generation_results.json
```

**Required flags:**
- `--benchmark-generation`: Enable generation benchmark mode
- `--input TEXT`: Input text prompt to generate from
- `--tokenizer FILE`: Path to tokenizer.json file

**Optional flags:**
- `--max-tokens N`: Maximum tokens to generate (default: 50)
- `--temperature F`: Sampling temperature (default: 1.0)
- `--cpu-threads N`: Maximum CPU threads to test (default: auto-detect)
- `--output FILE`: JSON output file (default: generation_results.json)

**What it does:**
- Runs text generation on CPU with 1, 2, 3, ..., N threads
- Runs text generation on GPU
- Measures tokens per second for each configuration
- Reports prefill time (first token) and decode latency (subsequent tokens)
- Saves detailed timing results to JSON file

**Example output:**
```
=== Generation Benchmark Summary ===

Prompt: "The sky is blue because"
Prompt tokens: 6
Target tokens: 50

Configuration   Tokens Gen   Total (ms)     Tokens/sec     Prefill (ms)   Decode (ms)    Avg Decode (ms)
-------------------------------------------------------------------------------------------------
CPU-1T          50           15234.56       3.28           1523.46        13711.10       279.82
CPU-2T          50           8456.23        5.91           845.62         7610.61        155.32
CPU-4T          50           5123.45        9.76           512.35         4611.10        94.10
CPU-8T          50           3456.78        14.47          345.68         3111.10        63.49
GPU             50           234.56         213.15         23.46          211.10         4.31
-------------------------------------------------------------------------------------------------

Best Throughput: GPU with 213.15 tokens/sec
Speedup vs CPU-1T: 64.98x
```

### Creating Test Models

Use the provided Python script to create test ONNX models:

```bash
python3 scripts/export_models.py
```

This will generate simple ONNX models for testing the engine (simple_linear.onnx, two_layer.onnx, residual.onnx).

## Supported Operations

### Arithmetic / Linear Algebra
- **MatMul**: Matrix multiplication (uses cuBLAS for large matrices)
- **Gemm**: General matrix multiply with bias (alpha=1, beta=1 only)
- **Add**: Element-wise addition with scalar broadcasting
- **Sub**: Element-wise subtraction
- **Mul**: Element-wise multiplication

### Activations
- **ReLU**: Rectified Linear Unit (vectorized with float4 optimization)
- **Sigmoid**: Sigmoid activation function

### Tensor Manipulation
- **Transpose**: Matrix/tensor transposition
- **Gather**: Gather elements along an axis using indices
- **Shape**: Get shape of a tensor (metadata operation)
- **Cast**: Type conversion between data types

### Reductions
- **ReduceSum**: Sum reduction along specified axes

### Advanced Operations (LLM Support)
- **RotaryEmbedding**: Rotary position embeddings for transformers
- **GroupQueryAttention**: Multi-head attention with grouped queries and KV cache
- **SimplifiedLayerNormalization**: Layer normalization (epsilon=1e-5)
- **SkipSimplifiedLayerNormalization**: Layer normalization with skip/residual connection

**Note**: All operations support both CPU (with OpenMP multi-threading) and GPU execution where applicable.

## Architecture Overview

```
ONNX File → ModelParser → Graph → GpuExecutor → CUDA Kernels → Output
```

### Core Components

- **ModelParser** (`src/core/model_parser.*`): Parses ONNX files using Protocol Buffers
- **Graph** (`src/core/graph.*`): Computation graph container with topological sorting
- **Node** (`src/core/node.*`): Individual operation representation
- **Tensor** (`src/utils/tensor.*`): RAII-managed tensor with GPU/CPU memory
- **GpuExecutor** (`src/gpu/gpu_executor.*`): Graph execution orchestrator
- **CUDA Kernels** (`src/gpu/kernels/*.cu`): Custom CUDA kernel implementations

## Troubleshooting

### Build Errors

**Error: "too many arguments on command line"**
- Solution: Make sure `CMAKE_CUDA_ARCHITECTURES` uses semicolons: `"75;86;89"` not `75 86 89`

**Error: "undefined reference to `__cxa_call_terminate@CXXABI_1.3.15`"**
- Solution: Use GCC 13 as shown in build instructions above

**Error: "identifier 'uintptr_t' is undefined" in CUDA files**
- Solution: Already fixed in the codebase with `#include <cstdint>`

**Error: "'memcpy' is not a member of 'std'"**
- Solution: Already fixed in the codebase with `#include <cstring>`

### Runtime Issues

**CUDA out of memory errors**
- Try using smaller models or enable CPU fallback mode with `--cpu`

**No GPU detected**
- Verify CUDA installation: `nvidia-smi`
- Check CUDA Toolkit: `nvcc --version`

## Development

### Adding New Operations

1. Add enum value to `OpType` in `src/core/node.hpp`
2. Update `stringToOpType()` in `src/core/node.cpp`
3. Create CUDA kernel in `src/gpu/kernels/your_op.cu`
4. Declare kernel in `src/gpu/kernels/kernels.cuh`
5. Add executor method in `src/gpu/gpu_executor.{hpp,cpp}`
6. Update `CMakeLists.txt` to include new `.cu` file

### Code Style

- C++17 standard
- All code in `onnx_runner` namespace
- RAII for resource management
- Use `CUDA_CHECK()` macro for all CUDA API calls
- Logging via `LOG_INFO`, `LOG_DEBUG`, `LOG_ERROR` macros

## Limitations

- Only FLOAT32 data type supported
- No dynamic shapes (shapes must be known at parse time)
- Limited broadcasting support (scalar broadcasting only)
- No graph optimizations or operator fusion
- GEMM only supports alpha=1.0 and beta=1.0

## Project Structure

```
ONNX-GPU-Execution-Engine/
├── src/
│   ├── main.cpp                 # Entry point and CLI argument parsing
│   ├── core/                    # ONNX parsing and graph representation
│   │   ├── model_parser.*
│   │   ├── graph.*
│   │   └── node.*
│   ├── gpu/                     # GPU execution and benchmarking
│   │   ├── gpu_executor.*       # Graph executor with CPU/GPU support
│   │   ├── benchmark.*          # Multi-configuration benchmark system
│   │   ├── ops/                 # Operation implementations (.inl files)
│   │   └── kernels/             # CUDA kernels
│   │       ├── kernels.cuh
│   │       ├── matmul.cu
│   │       ├── relu.cu
│   │       ├── add.cu
│   │       ├── gather.cu
│   │       ├── sigmoid.cu
│   │       ├── layernorm.cu
│   │       ├── rotary_embedding.cu
│   │       ├── group_query_attention.cu
│   │       └── ... (other kernels)
│   └── utils/                   # Utilities
│       ├── tensor.*
│       └── logger.*
├── scripts/                     # Build and setup scripts
│   ├── setup/                   # Setup scripts subdirectory
│   │   ├── setup_onnx_proto.sh
│   │   └── full_setup.sh
│   ├── export_models.py
│   ├── validate_onnx.py
│   └── hf_tokenizer.py
├── visualization/               # Benchmark visualization
│   └── benchmark_viewer.html   # Interactive HTML dashboard
├── third_party/                 # Generated files (not in git)
│   └── onnx/
├── CMakeLists.txt
├── CLAUDE.md                    # AI assistant guidance
└── README.md
```

## License

[Your license here]

## Contributing

[Contributing guidelines here]

## Acknowledgments

- ONNX project for model format specification
- NVIDIA CUDA team for GPU computing platform
