#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "OnnxRunner Google Colab Setup"
echo "=========================================="

# Detect if running in Colab
if [ -d "/content" ]; then
    echo "✓ Detected Google Colab environment"
    IN_COLAB=true
else
    echo "⚠ Not in Colab, but proceeding anyway..."
    IN_COLAB=false
fi

# Install system dependencies
echo ""
echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq cmake build-essential protobuf-compiler libprotobuf-dev > /dev/null 2>&1
echo "✓ System dependencies installed"

# Check for CUDA and GPU
echo ""
echo "Step 2: Checking GPU and CUDA compatibility..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

    # Get compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    echo "✓ GPU detected with compute capability: ${COMPUTE_CAP:0:1}.${COMPUTE_CAP:1}"

    # Get CUDA driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "  CUDA Driver Version: $DRIVER_VERSION"

    # Check CUDA runtime version
    if command -v nvcc &> /dev/null; then
        RUNTIME_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')
        echo "  CUDA Runtime Version: $RUNTIME_VERSION"

        # Extract major versions for comparison
        DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
        RUNTIME_MAJOR=$(echo $RUNTIME_VERSION | cut -d. -f1)

        # Warn if runtime > driver (common cause of the error)
        if [ "$RUNTIME_MAJOR" -gt "$DRIVER_MAJOR" ]; then
            echo ""
            echo "⚠ WARNING: CUDA runtime ($RUNTIME_VERSION) is newer than driver ($DRIVER_VERSION)"
            echo "  This may cause 'driver version insufficient' errors."
            echo "  Attempting to use compatible CUDA version..."

            # Try to find and use older CUDA version
            for cuda_ver in 11.8 11.7 11.6 11.5 11.4 11.3 11.2 11.1 11.0; do
                if [ -d "/usr/local/cuda-${cuda_ver}" ]; then
                    export CUDA_HOME="/usr/local/cuda-${cuda_ver}"
                    export PATH="$CUDA_HOME/bin:$PATH"
                    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
                    echo "  ✓ Using CUDA $cuda_ver at $CUDA_HOME"
                    break
                fi
            done
        fi
    else
        echo "⚠ nvcc not found, installing CUDA toolkit..."
        apt-get install -y -qq nvidia-cuda-toolkit > /dev/null 2>&1
    fi

    # Map to common architectures
    if [ "$COMPUTE_CAP" == "75" ]; then
        echo "  Architecture: Turing (Tesla T4)"
    elif [ "$COMPUTE_CAP" == "80" ] || [ "$COMPUTE_CAP" == "86" ]; then
        echo "  Architecture: Ampere (A100/RTX 30 series)"
    elif [ "$COMPUTE_CAP" == "89" ] || [ "$COMPUTE_CAP" == "90" ]; then
        echo "  Architecture: Ada/Hopper (RTX 40 series/H100)"
    fi
else
    echo "✗ No GPU detected! CUDA operations will fail."
    COMPUTE_CAP="75"  # Default fallback
fi

# Update CMakeLists.txt with detected compute capability
echo ""
echo "Step 3: Configuring CMake for GPU architecture..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    sed -i "s/set(CMAKE_CUDA_ARCHITECTURES [0-9]\+)/set(CMAKE_CUDA_ARCHITECTURES $COMPUTE_CAP)/" "$PROJECT_ROOT/CMakeLists.txt"
    echo "✓ CMakeLists.txt configured for compute capability $COMPUTE_CAP"
else
    echo "✗ CMakeLists.txt not found at $PROJECT_ROOT"
    exit 1
fi

# Setup ONNX protobuf definitions
echo ""
echo "Step 4: Setting up ONNX protobuf definitions..."
cd "$PROJECT_ROOT"
if [ -f "scripts/setup_onnx_proto.sh" ]; then
    chmod +x scripts/setup_onnx_proto.sh
    ./scripts/setup_onnx_proto.sh
    echo "✓ ONNX protobuf setup complete"
else
    echo "✗ setup_onnx_proto.sh not found"
    exit 1
fi

# Build the project
echo ""
echo "Step 5: Building OnnxRunner..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "✓ Build complete"

# Install Python dependencies for testing
echo ""
echo "Step 6: Installing Python dependencies..."
pip install -q onnx onnxruntime numpy
echo "✓ Python dependencies installed"

# Create test models
echo ""
echo "Step 7: Creating test models..."
cd "$PROJECT_ROOT"
if [ -f "scripts/create_test_model.py" ]; then
    python3 scripts/create_test_model.py
    echo "✓ Test models created"
else
    echo "⚠ create_test_model.py not found, skipping test model creation"
fi

# Quick CUDA verification test
echo ""
echo "Step 8: Verifying CUDA compatibility..."
cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);

    printf("CUDA Runtime: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Driver:  %d.%d\n", driverVersion/1000, (driverVersion%100)/10);

    if (runtimeVersion > driverVersion) {
        printf("\n⚠ WARNING: Runtime version exceeds driver version!\n");
        printf("This will cause 'driver version insufficient' errors.\n");
        return 1;
    }

    float *d_test;
    cudaError_t err = cudaMalloc(&d_test, sizeof(float));
    if (err != cudaSuccess) {
        printf("\n✗ CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaFree(d_test);

    printf("\n✓ CUDA is working correctly!\n");
    return 0;
}
EOF

nvcc /tmp/cuda_test.cu -o /tmp/cuda_test 2>/dev/null
if [ $? -eq 0 ]; then
    /tmp/cuda_test
    if [ $? -ne 0 ]; then
        echo ""
        echo "════════════════════════════════════════"
        echo "⚠ CUDA COMPATIBILITY ISSUE DETECTED"
        echo "════════════════════════════════════════"
        echo "The application may fail with 'driver version insufficient' error."
        echo ""
        echo "Possible solutions:"
        echo "1. Restart Colab runtime (Runtime → Restart runtime)"
        echo "2. Try CUDA 11.x manually:"
        echo "   export CUDA_HOME=/usr/local/cuda-11.8"
        echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
        echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
        echo "   cd build && cmake .. && make -j\$(nproc)"
        echo ""
        echo "3. Run in CPU mode: ./build/onnx_gpu_engine model.onnx --cpu"
        echo "════════════════════════════════════════"
    fi
else
    echo "⚠ Could not compile CUDA test, skipping verification"
fi
rm -f /tmp/cuda_test.cu /tmp/cuda_test

# Summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  ./build/onnx_gpu_engine model.onnx --verbose"
echo ""
echo "Or validate against ONNX Runtime:"
echo "  python3 scripts/validate_onnx.py"
echo ""
echo "Available test models:"
if [ -f "simple_linear.onnx" ]; then echo "  - simple_linear.onnx"; fi
if [ -f "two_layer.onnx" ]; then echo "  - two_layer.onnx"; fi
if [ -f "residual.onnx" ]; then echo "  - residual.onnx"; fi
echo ""
echo "If you get 'driver version insufficient' errors:"
echo "  - Use CPU mode: ./build/onnx_gpu_engine model.onnx --cpu"
echo "  - Or restart Colab runtime and rebuild"
echo ""
