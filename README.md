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

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ONNX-GPU-Execution-Engine
```

### 2. Download and Compile ONNX Proto Files

Run the setup script to download ONNX proto definitions and compile them:

```bash
./scripts/setup_onnx_proto.sh
```

This will create the `third_party/onnx/` directory with compiled protobuf files.

### 3. Configure GPU Architecture (Optional)

Edit `CMakeLists.txt` line 10 to match your GPU compute capability:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
```

- **75**: Turing (RTX 20 series, GTX 16 series)
- **86**: Ampere (RTX 30 series, A100)
- **89**: Ada Lovelace (RTX 40 series)

You can specify multiple architectures separated by semicolons, or just one for faster compilation.

### 4. Build the Project

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

Then either:
- Click "Load File" and select `results.json`
- Copy/paste the JSON content directly

The visualization provides:
- **Statistics Dashboard**: Key metrics including speedups and execution times
- **Live Performance Race**: Animated comparison showing relative speeds
- **Bar Chart**: Side-by-side comparison of all configurations
- **Interactive Controls**: Adjustable animation speed

### Creating Test Models

Use the provided Python script to create test ONNX models:

```bash
python3 scripts/create_test_model.py
```

This will generate simple ONNX models for testing the engine.

## Supported Operations

Currently implemented operations:

- **MatMul**: Matrix multiplication
- **Gemm**: General matrix multiply with bias (alpha=1, beta=1)
- **ReLU**: Rectified Linear Unit activation
- **Add**: Element-wise addition with scalar broadcasting

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
│   │   └── kernels/             # CUDA kernels
│   │       ├── kernels.cuh
│   │       ├── matmul.cu
│   │       ├── relu.cu
│   │       └── add.cu
│   └── utils/                   # Utilities
│       ├── tensor.*
│       └── logger.*
├── scripts/                     # Build and setup scripts
│   ├── setup_onnx_proto.sh
│   └── create_test_model.py
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
