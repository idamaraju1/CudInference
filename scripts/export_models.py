#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format for testing the C++ engine.
All weights are embedded inline (no external .data files).
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import os
import sys


class SimpleLinear(nn.Module):
    """Simple model: Linear -> ReLU"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class TwoLayerNet(nn.Module):
    """Two layer network: Linear -> ReLU -> Linear"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Simple residual connection"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual
        out = self.relu(out)
        return out


def create_test_input(shape):
    """Create consistent test input matching what the C++ code uses"""
    total_size = np.prod(shape)
    data = np.arange(total_size, dtype=np.float32) * 0.01
    return data.reshape(shape)


def export_model(model, model_name, input_shape):
    """Export a PyTorch model to ONNX with embedded weights"""
    print(f"\nExporting: {model_name}")

    model.eval()
    input_tensor = torch.from_numpy(create_test_input(input_shape))
    onnx_file = f"{model_name}.onnx"

    # Export to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        verbose=False
    )

    # Force inline data (no external .data files)
    model_onnx = onnx.load(onnx_file)
    onnx.save(model_onnx, onnx_file)

    # Clean up any external data files
    external_data_file = f"{onnx_file}.data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)

    print(f"  ✓ Created {onnx_file}")
    print(f"  Input shape: {list(input_shape)}")

    # Show sample output
    with torch.no_grad():
        output = model(input_tensor)
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Sample output: {output.numpy().flatten()[:5]}")


def main():
    print("="*60)
    print("ONNX Model Export")
    print("="*60)
    print("\nExporting test models for C++ ONNX engine...")

    # Export all test models
    export_model(SimpleLinear(), "simple_linear", (1, 10))
    export_model(TwoLayerNet(), "two_layer", (1, 10))
    export_model(ResidualBlock(), "residual", (1, 10))

    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print("\nRun models with:")
    print("  ./build/onnx_gpu_engine simple_linear.onnx")
    print("  ./build/onnx_gpu_engine two_layer.onnx")
    print("  ./build/onnx_gpu_engine residual.onnx")
    print("\nValidate against ONNX Runtime:")
    print("  python3 scripts/validate_onnx.py")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch onnx numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
