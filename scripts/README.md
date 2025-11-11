# Testing Scripts

This directory contains Python scripts for creating and validating ONNX models.

## Scripts

### `export_models.py`
Creates test ONNX models for the C++ engine.

**Usage:**
```bash
python3 scripts/export_models.py
```

**What it does:**
- Exports three test models: `simple_linear`, `two_layer`, and `residual`
- Uses the same test input as the C++ code ([0, 0.01, 0.02, ..., 0.09])
- Embeds all weights inline (no external .data files)
- Uses ONNX opset 18

**Output:**
- `simple_linear.onnx` - Linear layer + ReLU
- `two_layer.onnx` - Two linear layers with ReLU
- `residual.onnx` - Residual block with skip connection

### `validate_onnx.py`
Validates C++ engine output against ONNX Runtime.

**Usage:**
```bash
python3 scripts/validate_onnx.py
```

**What it does:**
- Runs each test model through ONNX Runtime (reference implementation)
- Runs the same models through the C++ engine
- Compares outputs numerically
- Reports pass/fail for each model

**Success criteria:**
- Maximum absolute difference < 1e-4 (0.0001)
- This accounts for floating-point precision differences

## Requirements

Install with pip:
```bash
pip install torch onnx onnxruntime numpy
```

Or use the project's virtual environment:
```bash
source .venv/bin/activate
```

## Other Scripts

### `setup_onnx_proto.sh`
Downloads and compiles ONNX Protocol Buffer definitions. Run this once during initial project setup:
```bash
./scripts/setup_onnx_proto.sh
```

## Workflow

1. **Initial setup** (one time):
   ```bash
   ./scripts/setup_onnx_proto.sh
   mkdir build && cd build
   cmake ..
   make
   cd ..
   ```

2. **Create test models**:
   ```bash
   python3 scripts/export_models.py
   ```

3. **Run validation**:
   ```bash
   python3 scripts/validate_onnx.py
   ```

4. **Test manually**:
   ```bash
   ./build/onnx_gpu_engine simple_linear.onnx
   ./build/onnx_gpu_engine two_layer.onnx --verbose
   ./build/onnx_gpu_engine residual.onnx --debug
   ```
