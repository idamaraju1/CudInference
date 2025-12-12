#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format for testing the C++ engine.
All weights are embedded inline (no external .data files).
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, TensorProto
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


def create_sparse_linear_model():
    """Create an ONNX model with sparse weights manually

    Model structure: MatMul(input, sparse_weight) -> ReLU -> output
    Input shape: [1, 10]
    Sparse weight: [10, 5] with ~80% sparsity
    Output shape: [1, 5]
    """
    print("\nCreating sparse linear model...")

    # Define the sparse weight matrix [10, 5] with ~80% sparsity
    # We'll have 10 non-zero values out of 50 total (20% density)
    weight_shape = [10, 5]
    total_elements = weight_shape[0] * weight_shape[1]

    # Create sparse weight in COO format
    # Non-zero indices (using 2D format: [NNZ, 2])
    sparse_indices = np.array([
        [0, 0],  # weight[0, 0] = 0.5
        [1, 1],  # weight[1, 1] = 0.6
        [2, 2],  # weight[2, 2] = 0.7
        [3, 3],  # weight[3, 3] = 0.8
        [4, 4],  # weight[4, 4] = 0.9
        [5, 0],  # weight[5, 0] = 0.4
        [6, 1],  # weight[6, 1] = 0.3
        [7, 2],  # weight[7, 2] = 0.2
        [8, 3],  # weight[8, 3] = 0.1
        [9, 4],  # weight[9, 4] = 0.15
    ], dtype=np.int64)

    # Non-zero values
    sparse_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.15], dtype=np.float32)

    nnz = len(sparse_values)
    print(f"  Sparse weight: {weight_shape}, NNZ: {nnz}/{total_elements} ({100*nnz/total_elements:.1f}% density)")

    # Create the sparse weight tensor proto
    values_tensor = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[nnz],
        vals=sparse_values.flatten().tolist()
    )

    indices_tensor = helper.make_tensor(
        name='weight_indices',
        data_type=TensorProto.INT64,
        dims=[nnz, 2],
        vals=sparse_indices.flatten().tolist()
    )

    sparse_weight = helper.make_sparse_tensor(
        values_tensor,
        indices_tensor,
        weight_shape
    )

    # Create graph nodes
    # Node 1: MatMul
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weight'],
        outputs=['matmul_out'],
        name='matmul'
    )

    # Node 2: ReLU
    relu_node = helper.make_node(
        'Relu',
        inputs=['matmul_out'],
        outputs=['output'],
        name='relu'
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[matmul_node, relu_node],
        name='sparse_linear',
        inputs=[
            helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5])
        ],
        initializer=[],  # No dense initializers
        sparse_initializer=[sparse_weight]  # Sparse weight in sparse_initializer
    )

    # Create model
    model = helper.make_model(graph, producer_name='onnx-sparse-test')
    model.opset_import[0].version = 18
    model.ir_version = 9  # Use IR version 9 for compatibility with older ONNX Runtime

    # Save model
    onnx_file = "sparse_linear.onnx"
    onnx.save(model, onnx_file)

    print(f"  ✓ Created {onnx_file}")
    print(f"  Input shape: [1, 10]")
    print(f"  Output shape: [1, 5]")

    # Compute expected output with the test input
    test_input = create_test_input((1, 10))

    # Build dense weight matrix from sparse representation
    dense_weight = np.zeros(weight_shape, dtype=np.float32)
    for i in range(nnz):
        row, col = sparse_indices[i]
        dense_weight[row, col] = sparse_values[i]

    # Compute: (input @ weight) -> relu
    matmul_result = test_input @ dense_weight
    expected_output = np.maximum(0, matmul_result)

    print(f"  Expected output: {expected_output.flatten()}")

    return expected_output


def main():
    print("="*60)
    print("ONNX Model Export")
    print("="*60)
    print("\nExporting test models for C++ ONNX engine...")

    # Export all test models
    export_model(SimpleLinear(), "simple_linear", (1, 10))
    export_model(TwoLayerNet(), "two_layer", (1, 10))
    export_model(ResidualBlock(), "residual", (1, 10))

    # Create sparse model
    create_sparse_linear_model()

    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print("\nRun models with:")
    print("  ./build/onnx_gpu_engine simple_linear.onnx")
    print("  ./build/onnx_gpu_engine two_layer.onnx")
    print("  ./build/onnx_gpu_engine residual.onnx")
    print("  ./build/onnx_gpu_engine sparse_linear.onnx")
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
