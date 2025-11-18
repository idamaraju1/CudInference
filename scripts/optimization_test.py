"""
Author: Chris V
Benchmark GPU baseline vs GPU optimized (multi-threaded) ONNX parsing/execution.

This script:
  - Creates several MLP ONNX models (uses code from `create_large_models.py`)
  - Runs each model on:
      1) Baseline GPU engine (single-threaded parser)
      2) Optimized GPU engine (multi-threaded parser)
  - Extracts timing lines from the C++ binary output:
        "Graph execution took __ ms"
  - Prints per-model speedup of optimized vs baseline.

Feel free to extend this if you add any other optimized section of the pipeline,
I only changed the parser for now so that is what this tests. -Chris
"""

#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import onnx
import os
import subprocess


# config
BASELINE_GPU_BIN = "./build/onnx_gpu_engine"       # baseline parser
OPTIMIZED_GPU_BIN = "./build/onnx_gpu_engine"      # multi-threaded parser (same binary, different flag)


# ---------------------------------------------------------------------
# Model definition and export
# ---------------------------------------------------------------------
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

    # Clean up external data files (if any)
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


# ---------------------------------------------------------------------
# benchmark helpers
# ---------------------------------------------------------------------
def extract_parse_time_ms(process_output):
    """
    Extract parsing time in milliseconds from lines like:
        "Model parsing took XXX ms"
    Returns float or None.
    """
    for line in process_output.splitlines():
        if "Model parsing took" in line:
            # e.g., "Model parsing took 12.34 ms"
            parts = line.split("took", 1)[1].strip().split()
            # parts[0] should be the numeric value
            try:
                return float(parts[0])
            except (ValueError, IndexError):
                continue
    return None


def run_engine(binary, onnx_file, extra_args=None):
    """Run the given engine binary on the ONNX file and return (stdout, stderr)."""
    if extra_args is None:
        extra_args = []

    cmd = [binary, onnx_file] + extra_args
    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  ❌ Engine returned non-zero exit code: {result.returncode}")
        print("  Stdout:\n", result.stdout)
        print("  Stderr:\n", result.stderr)

    return result.stdout, result.stderr


def benchmark_parsers(onnx_file):
    """
    Benchmark baseline vs optimized parser for a single model file.
    Returns (baseline_parse_ms, optimized_parse_ms).
    """
    print(f"\n{'─'*60}")
    print(f"Benchmarking parsers for: {onnx_file}")
    print(f"{'─'*60}")

    # Baseline parser
    print("\n🧱 Baseline (single-threaded parser)...")
    stdout_base, _ = run_engine(BASELINE_GPU_BIN, onnx_file, extra_args=["--verbose"])
    baseline_ms = extract_parse_time_ms(stdout_base)
    if baseline_ms is not None:
        print(f"  Baseline parse time: {baseline_ms:.3f} ms")
    else:
        print("  ⚠️ Could not find parsing time for baseline.")

    # Optimized parser
    print("\n⚙️  Optimized (multi-threaded parser)...")
    stdout_opt, _ = run_engine(OPTIMIZED_GPU_BIN, onnx_file,
                               extra_args=["--verbose", "--mt-parser"])
    optimized_ms = extract_parse_time_ms(stdout_opt)
    if optimized_ms is not None:
        print(f"  Optimized parse time: {optimized_ms:.3f} ms")
    else:
        print("  ⚠️ Could not find parsing time for optimized.")

    # Speedup (parser-only)
    if baseline_ms is not None and optimized_ms is not None:
        if optimized_ms > 0:
            speedup = baseline_ms / optimized_ms
            print(f"\n  ⚡ Parser speedup: {speedup:.2f}x "
                  f"(optimized parser is {speedup:.2f}x faster)")
        else:
            print("\n  ⚠️ Optimized parse time is zero or invalid; cannot compute speedup.")

    return baseline_ms, optimized_ms


# ---------------------------------------------------------------------
# driver: create models and benchmark parsers only
# ---------------------------------------------------------------------
def main():
    print("="*60)
    print("Baseline vs Optimized ONNX Parser Benchmark (parse time only)")
    print("="*60)
    print("\nCreating progressively larger models (MLP: Linear + ReLU)...")

    models_to_create = [
        # (name, input_size, hidden_size, num_layers, output_size, batch_size)
        ("small_mlp",      512,    1024,      10,   10, 64),
        ("medium_mlp",     1024,   2048,      15,   10, 64),
        ("large_mlp",      2048,   4096,      20,   10, 64),
        ("xlarge_mlp",     2048,   4096,      25,   10, 128),
        ("xxlarge_mlp",    2048,   4096,      30,   10, 128),
    ]

    created_models = []

    # Create ONNX models
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

    # Run parser benchmarks
    print("\n" + "="*60)
    print("Running baseline vs optimized parser benchmarks (parse time only)...")
    print("="*60)

    results = []

    for onnx_file, _ in created_models:
        try:
            baseline_ms, optimized_ms = benchmark_parsers(onnx_file)
            results.append((onnx_file, baseline_ms, optimized_ms))
        except Exception as e:
            print(f"  ❌ Error benchmarking {onnx_file}: {e}")

    # Summary
    print("\n" + "="*60)
    print("Summary: Baseline vs Optimized Parser (parse time)")
    print("="*60)

    for onnx_file, base, opt in results:
        size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
        if base is not None and opt is not None and opt > 0:
            speedup = base / opt
            print(f"  {onnx_file:20} ({size_mb:6.2f} MB)  "
                  f"baseline_parse={base:8.3f} ms  optimized_parse={opt:8.3f} ms  "
                  f"speedup={speedup:5.2f}x")
        else:
            print(f"  {onnx_file:20} ({size_mb:6.2f} MB)  parsing timing unavailable")

    print("\nTo run manually:")
    print(f"  {BASELINE_GPU_BIN} <model.onnx> --verbose              # baseline parser")
    print(f"  {OPTIMIZED_GPU_BIN} <model.onnx> --verbose --mt-parser  # optimized parser")


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