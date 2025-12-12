#!/bin/bash

# Script to check if all dependencies are installed

echo "=== OnnxRunner Dependency Checker ==="
echo

all_good=true

# Check CMake
echo -n "Checking CMake... "
if command -v cmake &> /dev/null; then
    version=$(cmake --version | head -n1 | awk '{print $3}')
    echo "✓ Found (version $version)"
else
    echo "✗ Not found"
    echo "  Install: sudo apt-get install cmake (Ubuntu) or brew install cmake (macOS)"
    all_good=false
fi

# Check protoc
echo -n "Checking Protocol Buffers... "
if command -v protoc &> /dev/null; then
    version=$(protoc --version | awk '{print $2}')
    echo "✓ Found (version $version)"
else
    echo "✗ Not found"
    echo "  Install: sudo apt-get install protobuf-compiler libprotobuf-dev (Ubuntu)"
    echo "          brew install protobuf (macOS)"
    all_good=false
fi

# Check NVCC (CUDA)
echo -n "Checking CUDA (nvcc)... "
if command -v nvcc &> /dev/null; then
    version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ Found (version $version)"
else
    echo "✗ Not found"
    echo "  Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    all_good=false
fi

# Check for CUDA libraries
if [ -n "$CUDA_HOME" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    echo "CUDA installation found at: /usr/local/cuda"
elif command -v nvcc &> /dev/null; then
    cuda_path=$(which nvcc | sed 's/\/bin\/nvcc//')
    echo "CUDA installation found at: $cuda_path"
fi

# Check C++ compiler
echo -n "Checking C++ compiler... "
if command -v g++ &> /dev/null; then
    version=$(g++ --version | head -n1 | awk '{print $NF}')
    echo "✓ Found g++ (version $version)"
elif command -v clang++ &> /dev/null; then
    version=$(clang++ --version | head -n1 | awk '{print $4}')
    echo "✓ Found clang++ (version $version)"
else
    echo "✗ Not found"
    echo "  Install: sudo apt-get install build-essential (Ubuntu)"
    all_good=false
fi

# Check for GPU
echo -n "Checking for NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    echo "✓ Found: $gpu_name (compute capability $compute_cap)"
else
    echo "⚠ nvidia-smi not found (GPU might not be available)"
fi

echo
echo "=== Python Dependencies (for test model generation) ==="

echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    version=$(python3 --version | awk '{print $2}')
    echo "✓ Found (version $version)"

    # Check for PyTorch
    echo -n "Checking PyTorch... "
    if python3 -c "import torch" 2>/dev/null; then
        torch_version=$(python3 -c "import torch; print(torch.__version__)")
        echo "✓ Found (version $torch_version)"
    else
        echo "⚠ Not found (optional, needed for creating test models)"
        echo "  Install: pip install torch"
    fi
else
    echo "⚠ Not found (optional)"
fi

echo
echo "=== Summary ==="
if $all_good; then
    echo "✓ All required dependencies are installed!"
    echo
    echo "Next steps:"
    echo "  1. Run: ./scripts/setup_onnx_proto.sh"
    echo "  2. Edit CMakeLists.txt to set your GPU architecture"
    echo "  3. Run: mkdir build && cd build && cmake .. && make"
else
    echo "✗ Some dependencies are missing. Please install them first."
    exit 1
fi
