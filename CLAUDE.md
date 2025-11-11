# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OnnxRunner is a custom ONNX GPU execution engine built with C++17 and CUDA. It parses ONNX model files using Protocol Buffers and executes computation graphs using custom CUDA kernels. The project is designed for learning, experimentation, and as a foundation for custom neural network inference.

## Build System

The project uses CMake (minimum 3.18) with both C++ and CUDA compilation.

**Initial setup:**
```bash
./scripts/setup_onnx_proto.sh  # Downloads and compiles ONNX protobuf definitions
```

**Configure GPU architecture in CMakeLists.txt:**
Edit line 10 to match your GPU compute capability (75=Turing RTX 20, 86=Ampere RTX 30, 89=Ada RTX 40).

**Build commands:**
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

**Run the engine:**
```bash
./build/onnx_gpu_engine model.onnx [--cpu] [--verbose] [--debug] [--benchmark] [--cpu-threads N] [--output FILE]
```

**Benchmark mode:**
```bash
# Run comprehensive CPU (1-N threads) vs GPU benchmark
./build/onnx_gpu_engine model.onnx --benchmark

# Specify max thread count (default: auto-detect hardware_concurrency)
./build/onnx_gpu_engine model.onnx --benchmark --cpu-threads 8

# Results saved to results.json by default (or specify with --output)
```

**Visualize benchmark results:**
```bash
# Open in browser (Firefox, Chrome, etc.)
firefox visualization/benchmark_viewer.html
# Then load results.json via the UI
```

**Create test models:**
```bash
python3 scripts/create_test_model.py
```

## Architecture

The codebase follows a clean separation between ONNX parsing, graph representation, and GPU execution:

**Data Flow:**
```
ONNX File → ModelParser → Graph → GpuExecutor → CUDA Kernels → Output
```

**Core Components:**

1. **ModelParser** (`src/core/model_parser.{hpp,cpp}`): Parses `.onnx` files using Protocol Buffers, extracts the computation graph, initializers (weights), and metadata. Entry point is `parse()` which returns a `Graph` object.

2. **Graph** (`src/core/graph.{hpp,cpp}`): Container for the entire computation graph, holding nodes, initializers (constant tensors like weights), input/output names. Provides `topologicalSort()` to determine execution order based on data dependencies.

3. **Node** (`src/core/node.{hpp,cpp}`): Represents a single operation (MatMul, ReLU, Add, etc.). Contains:
   - `OpType` enum defining supported operations
   - Input/output tensor names
   - Operation attributes (e.g., alpha/beta for Gemm, transA/transB for MatMul)
   - Helper methods to access attributes with defaults

4. **Tensor** (`src/utils/tensor.{hpp,cpp}`): RAII-managed tensor class supporting both CPU and GPU memory:
   - Uses smart pointers with custom `CudaDeleter` for automatic GPU memory cleanup
   - Supports shape manipulation, device transfers (`toGPU()`, `toCPU()`), and in-place operations
   - Currently only supports FLOAT32, though enum includes FLOAT16, INT32, INT64, UINT8
   - Use `CUDA_CHECK()` macro for all CUDA API calls

5. **GpuExecutor** (`src/gpu/gpu_executor.{hpp,cpp}`): Orchestrates graph execution:
   - Performs topological sort to determine operation execution order
   - Maintains tensor storage (named map) during execution
   - Dispatches to operation-specific `execute*()` methods
   - Includes `GPUTimer` class for performance benchmarking with CUDA events
   - Supports CPU fallback mode with OpenMP multi-threading (controlled by `num_cpu_threads_` parameter)

6. **BenchmarkExecutor** (`src/gpu/benchmark.{hpp,cpp}`): Multi-configuration benchmark system:
   - Runs the model on CPU with 1, 2, 3, ..., max_threads configurations
   - Runs the model on GPU for comparison
   - Collects detailed timing data for each configuration
   - Exports results to JSON format with per-operation and total timing data
   - Displays live progress visualization with color-coded output
   - Supports adjustable speed control for visualization

7. **CUDA Kernels** (`src/gpu/kernels/`):
   - `matmul.cu`: Tiled matrix multiplication for small matrices, falls back to cuBLAS for large ones
     - Includes multi-threaded CPU implementation using OpenMP
   - `relu.cu`: Vectorized with float4 optimization
     - Includes multi-threaded CPU implementation using OpenMP
   - `add.cu`: Element-wise addition with scalar broadcasting support
   - All kernels declared in `kernels.cuh`

**Key Design Patterns:**
- All code is in the `onnx_runner` namespace
- RAII for GPU memory management (no manual cudaFree calls)
- CPU fallback implementations available for all operations
- Logging via `LOG_INFO`, `LOG_DEBUG`, `LOG_ERROR` macros from `utils/logger.hpp`

## Supported Operations

Currently implemented:
- **MatMul**: Matrix multiplication
- **Gemm**: General matrix multiply with bias (alpha=1, beta=1 only)
- **ReLU**: Rectified Linear Unit activation
- **Add**: Element-wise addition with scalar broadcasting

The `OpType` enum in `src/core/node.hpp` lists additional operations (CONV, MAXPOOL, SOFTMAX, etc.) that are recognized but not yet implemented.

## Adding New Operations

To add a new CUDA operation:

1. Add enum value to `OpType` in `src/core/node.hpp`
2. Update `stringToOpType()` in `src/core/node.cpp` to parse the ONNX op name
3. Create CUDA kernel file in `src/gpu/kernels/your_op.cu`:
   - Implement `__global__` kernel function
   - Implement launcher function (called from C++ code)
   - Declare launcher in `kernels.cuh`
4. Add `execute<OpName>()` method to `GpuExecutor` in `src/gpu/gpu_executor.{hpp,cpp}`
5. Add dispatch case in `GpuExecutor::executeNode()`
6. Add `.cu` file to `CUDA_SOURCES` in `CMakeLists.txt`

## Important Constraints

- Only FLOAT32 data type is currently supported (despite enum values)
- No dynamic shapes - shapes must be known at parse time
- Limited broadcasting support (only scalar broadcasting in Add)
- No graph optimizations or operator fusion
- GEMM only supports alpha=1.0 and beta=1.0

## Project Dependencies

Required:
- CUDA Toolkit (11.0+)
- CMake (3.18+)
- Protocol Buffers (protobuf compiler and dev libraries)
- C++17 compiler (GCC 7+ or Clang 5+) with OpenMP support
- OpenMP (for multi-threaded CPU execution)

The project links against:
- `protobuf::libprotobuf`
- `CUDA::cudart`
- `CUDA::cublas`
- `OpenMP::OpenMP_CXX`

**Important:** OpenMP must be enabled for CUDA files. The CMakeLists.txt includes `-Xcompiler -fopenmp` flags for CUDA compilation to enable multi-threading in CPU fallback implementations.

## File Organization

```
src/
├── main.cpp                 # Entry point, CLI argument parsing, test input creation
├── core/                    # Graph representation and ONNX parsing
│   ├── model_parser.*       # Protocol buffer parsing
│   ├── graph.*              # Graph container and topological sort
│   └── node.*               # Operation node and OpType definitions
├── gpu/                     # Execution engine and benchmarking
│   ├── gpu_executor.*       # Graph executor with CPU/GPU support
│   ├── benchmark.*          # Multi-configuration benchmark system
│   └── kernels/             # CUDA kernel implementations
│       ├── kernels.cuh      # Kernel declarations
│       └── *.cu             # Individual operation kernels (GPU + CPU fallback)
└── utils/                   # Utilities
    ├── tensor.*             # Tensor class with GPU memory management
    └── logger.*             # Logging macros

visualization/               # Benchmark visualization
└── benchmark_viewer.html    # Interactive HTML dashboard for results.json

third_party/onnx/            # Generated protobuf files (created by setup script)
scripts/                     # Build and setup scripts
```

## Testing Strategy

The project includes Python scripts for testing and validation:

**Create test models:**
```bash
python3 scripts/export_models.py
```
This creates `simple_linear.onnx`, `two_layer.onnx`, and `residual.onnx` with all weights embedded inline.

**Validate C++ output against ONNX Runtime:**
```bash
python3 scripts/validate_onnx.py
```
This compares the C++ engine output against the reference ONNX Runtime implementation for all test models.

**Manual testing:**
1. Run with `--verbose` to see per-operation timing
2. Run with `--debug` for detailed logging
3. Use `--cpu` flag to compare GPU vs CPU results
4. Use `--benchmark` flag for comprehensive CPU (1-N threads) vs GPU comparison

**Benchmark testing and visualization:**
```bash
# Run comprehensive benchmark
./build/onnx_gpu_engine model.onnx --benchmark

# Visualize results
firefox visualization/benchmark_viewer.html  # Load results.json via UI
```

The benchmark generates a JSON file with:
- `operations[]`: Per-operation timing for each configuration
  - `cpu_1_thread_ms`, `cpu_2_threads_ms`, ..., `cpu_N_threads_ms`
  - `gpu_ms`
- `total_cpu_*_thread*_ms`: Total execution time for each CPU configuration
- `total_gpu_ms`: Total GPU execution time
- `max_threads`: Maximum number of threads tested

**Important:** The C++ parser requires weights to be embedded in the .onnx file (no external .data files). The export script handles this automatically.
