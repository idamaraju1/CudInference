#!/usr/bin/env python3
"""
Validate C++ ONNX engine output against ONNX Runtime reference implementation.
"""

import numpy as np
import onnxruntime as ort
import subprocess
import sys
import os


def create_test_input(shape):
    """Create consistent test input matching what the C++ code uses"""
    total_size = np.prod(shape)
    data = np.arange(total_size, dtype=np.float32) * 0.01
    return data.reshape(shape)


def validate_model(onnx_file, input_shape, tolerance=1e-4):
    """Compare C++ engine output against ONNX Runtime"""

    if not os.path.exists(onnx_file):
        print(f"❌ Error: {onnx_file} not found!")
        print("Run 'python3 scripts/export_models.py' first to create test models.")
        return False

    model_name = os.path.splitext(os.path.basename(onnx_file))[0]

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Create test input
    input_data = create_test_input(input_shape)

    print(f"Input shape: {input_shape}")
    print(f"Input: {input_data.flatten()}")

    # Run ONNX Runtime (reference implementation)
    try:
        session = ort.InferenceSession(onnx_file)
        ort_output = session.run(None, {'input': input_data})[0]
    except Exception as e:
        print(f"❌ Error running ONNX Runtime: {e}")
        return False

    print(f"\nONNX Runtime output: {ort_output.flatten()}")

    # Run C++ engine
    cpp_executable = "./build/onnx_gpu_engine"
    if not os.path.exists(cpp_executable):
        print(f"\n❌ Error: {cpp_executable} not found!")
        print("Please build the project first:")
        print("  cd build && cmake .. && make")
        return False

    try:
        result = subprocess.run(
            [cpp_executable, onnx_file],
            capture_output=True,
            text=True,
            timeout=10
        )
    except subprocess.TimeoutExpired:
        print("❌ Error: C++ engine timed out")
        return False

    if result.returncode != 0:
        print(f"\n❌ Error running C++ engine:")
        print(result.stderr)
        return False

    # Parse C++ output
    cpp_output = None
    for line in result.stdout.split('\n'):
        if line.startswith('Output output'):
            if ':' in line:
                array_str = line.split(':', 1)[1].strip()
                array_str = array_str.strip('[]')
                cpp_output = np.array([float(x) for x in array_str.split(',')])
                break

    if cpp_output is None:
        print("\n❌ Error: Could not parse C++ output")
        print("Output was:")
        print(result.stdout)
        return False

    print(f"C++ Engine output:   {cpp_output.flatten()}")

    # Compare outputs
    max_diff = np.abs(ort_output.flatten() - cpp_output.flatten()).max()
    mean_diff = np.abs(ort_output.flatten() - cpp_output.flatten()).mean()

    print(f"\n{'─'*60}")
    print("Validation Results:")
    print(f"{'─'*60}")
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Tolerance threshold:      {tolerance:.2e}")

    if max_diff < tolerance:
        print(f"✅ PASSED - Outputs match within tolerance")
        return True
    else:
        print(f"❌ FAILED - Outputs differ by more than tolerance")
        print("\nDetailed comparison:")
        print(f"{'Index':<8} {'ONNX RT':<15} {'C++':<15} {'Diff':<15}")
        print('─' * 60)
        for i, (ort_val, cpp_val) in enumerate(zip(ort_output.flatten(), cpp_output.flatten())):
            diff = abs(ort_val - cpp_val)
            status = "⚠️" if diff > tolerance else " "
            print(f"{status} {i:<6} {ort_val:<15.6f} {cpp_val:<15.6f} {diff:<15.2e}")
        return False


def main():
    print("="*60)
    print("ONNX Runtime Validation")
    print("="*60)
    print("\nComparing C++ engine output against ONNX Runtime...")

    # Test models with their input shapes
    test_cases = [
        ("simple_linear.onnx", (1, 10)),
        ("two_layer.onnx", (1, 10)),
        ("residual.onnx", (1, 10)),
        ("sparse_linear.onnx", (1, 10)),
    ]

    results = {}

    for onnx_file, input_shape in test_cases:
        model_name = os.path.splitext(onnx_file)[0]
        results[model_name] = validate_model(onnx_file, input_shape)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
