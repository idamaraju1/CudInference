
#!/usr/bin/env python3
"""
Create large ONNX models for CPU vs GPU performance benchmarking.
All models use only supported operations: Linear (Gemm) and ReLU.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import os


class DeepMLP(nn.Module):
    """Deep Multi-Layer Perceptron with configurable depth and width"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_test_input(shape):
    """Create consistent test input matching C++ code"""
    total_size = np.prod(shape)
    data = np.arange(total_size, dtype=np.float32) * 0.01
    return data.reshape(shape)


def export_model(model, model_name, input_shape):
    """Export PyTorch model to ONNX with embedded weights"""
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

    # Force inline data (no external files)
    model_onnx = onnx.load(onnx_file)
    onnx.save(model_onnx, onnx_file)

    # Clean up external data files
    external_data_file = f"{onnx_file}.data"
    if os.path.exists(external_data_file):
        os.remove(external_data_file)

    # Calculate model size
    file_size_mb = os.path.getsize(onnx_file) / (1024 * 1024)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    print(f"  ✓ Created {onnx_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Parameters: {num_params:,}")
    print(f"  Input shape: {list(input_shape)}")

    # Show sample output
    with torch.no_grad():
        output = model(input_tensor)
        print(f"  Output shape: {list(output.shape)}")

    return onnx_file


def benchmark_model(onnx_file):
    """Run the model on both CPU and GPU and show timing"""
    import subprocess

    print(f"\n{'─'*60}")
    print(f"Benchmarking: {onnx_file}")
    print(f"{'─'*60}")

    # Run on GPU
    print("\n🚀 Running on GPU...")
    result_gpu = subprocess.run(
        ["./build/onnx_gpu_engine", onnx_file, "--verbose"],
        capture_output=True,
        text=True
    )

    # Extract GPU time
    gpu_time = None
    for line in result_gpu.stdout.split('\n'):
        if "Graph execution took" in line:
            gpu_time = line.split("took")[1].strip().split()[0]
            print(f"  GPU execution time: {gpu_time} ms")

    # Run on CPU
    print("\n🖥️  Running on CPU...")
    result_cpu = subprocess.run(
        ["./build/onnx_gpu_engine", onnx_file, "--cpu", "--verbose"],
        capture_output=True,
        text=True
    )

    # Extract CPU time
    cpu_time = None
    for line in result_cpu.stdout.split('\n'):
        if "Graph execution took" in line:
            cpu_time = line.split("took")[1].strip().split()[0]
            print(f"  CPU execution time: {cpu_time} ms")

    # Calculate speedup
    if gpu_time and cpu_time:
        try:
            speedup = float(cpu_time) / float(gpu_time)
            print(f"\n  ⚡ Speedup: {speedup:.2f}x (GPU is {speedup:.2f}x faster)")
        except:
            pass


def main():
    print("="*60)
    print("Large Model Generator for CPU vs GPU Benchmarking")
    print("="*60)
    print("\nCreating progressively larger models...")
    print("(All models use only: Linear layers + ReLU)")

    models_to_create = [
        # (name, input_size, hidden_size, num_layers, output_size, batch_size)
        ("small_mlp",      512,    1024,      10,   10, 64),
        ("medium_mlp",     1024,   2048,      15,   10, 64),
        ("large_mlp",      2048,   4096,      20,   10, 64),
        ("xlarge_mlp",     2048,   4096,      25,   10, 128),
        ("xxlarge_mlp",    2048,   4096,     30,   10, 128),
    ]

    created_models = []

    for name, input_size, hidden_size, num_layers, output_size, batch_size in models_to_create:
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"  Architecture: {num_layers} layers, {hidden_size} hidden units")
        print(f"  Input: [{batch_size}, {input_size}]")
        print(f"{'='*60}")

        model = DeepMLP(input_size, hidden_size, num_layers, output_size)
        input_shape = (batch_size, input_size)

        onnx_file = export_model(model, name, input_shape)
        created_models.append((onnx_file, input_shape))

    print("\n" + "="*60)
    print("All models created!")
    print("="*60)

    # Ask if user wants to benchmark
    print("\n" + "="*60)
    print("Running benchmarks...")
    print("="*60)

    for onnx_file, _ in created_models:
        try:
            benchmark_model(onnx_file)
        except Exception as e:
            print(f"  ❌ Error benchmarking {onnx_file}: {e}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nCreated models (smallest to largest):")
    for onnx_file, _ in created_models:
        size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
        print(f"  {onnx_file:20} - {size_mb:6.2f} MB")

    print("\nTo manually test:")
    print("  ./build/onnx_gpu_engine <model.onnx>         # GPU mode")
    print("  ./build/onnx_gpu_engine <model.onnx> --cpu   # CPU mode")
    print("  ./build/onnx_gpu_engine <model.onnx> --verbose  # Show timing")


if __name__ == "__main__":
    import sys
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
