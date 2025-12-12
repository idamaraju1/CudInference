#!/usr/bin/env python3
"""
Comprehensive validation of all ONNX operations against ONNX Runtime.
Tests each operation individually and in combination.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ENGINE = os.path.join(REPO_ROOT, "build", "onnx_gpu_engine")


class OperationTester:
    """Test individual ONNX operations"""

    def __init__(self, engine_path: str, tolerance: float = 1e-4):
        self.engine_path = engine_path
        self.tolerance = tolerance
        self.results = {}

    def create_test_input(self, shape):
        """Create consistent test input"""
        total_size = np.prod(shape)
        data = (np.arange(total_size, dtype=np.float32) % 100) * 0.01
        return data.reshape(shape)

    def run_onnx_runtime(self, model_bytes: bytes, input_data: np.ndarray, input_name: str = "input") -> Dict[str, np.ndarray]:
        """Run ONNX Runtime on model"""
        session = ort.InferenceSession(model_bytes)
        outputs = [o.name for o in session.get_outputs()]
        ort_outs = session.run(outputs, {input_name: input_data})
        return dict(zip(outputs, ort_outs))

    def parse_engine_output(self, stdout: str) -> Dict[str, np.ndarray]:
        """Parse engine output"""
        outputs = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("Output "):
                continue
            if ":" not in line:
                continue

            # Parse "Output name [shape]: [data]"
            header, data_str = line.split(":", 1)
            parts = header.split()
            name = parts[1] if len(parts) >= 2 else f"out_{len(outputs)}"

            # Extract array data
            data_str = data_str.strip()
            if data_str.startswith("[") and data_str.endswith("]"):
                data_str = data_str[1:-1]

            data_tokens = [tok.strip() for tok in data_str.split(",") if tok.strip()]
            if not data_tokens:
                continue

            outputs[name] = np.array([float(tok) for tok in data_tokens], dtype=np.float32)

        return outputs

    def run_engine(self, model_path: str, mode: str = "cpu") -> Tuple[bool, str, str]:
        """Run the engine on a model"""
        args = ["--cpu"] if mode == "cpu" else []
        cmd = [self.engine_path, model_path] + args

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"

    def compare_outputs(self, ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """Compare reference and test outputs"""
        issues = []

        for name, ref_val in ref.items():
            if name not in test:
                issues.append(f"Missing output: {name}")
                continue

            test_val = test[name]

            if ref_val.shape != test_val.shape:
                # Try to reshape if sizes match
                if ref_val.size == test_val.size:
                    test_val = test_val.reshape(ref_val.shape)
                else:
                    issues.append(f"{name}: shape mismatch {ref_val.shape} vs {test_val.shape}")
                    continue

            max_diff = np.abs(ref_val.flatten() - test_val.flatten()).max()
            mean_diff = np.abs(ref_val.flatten() - test_val.flatten()).mean()

            if max_diff > self.tolerance:
                issues.append(f"{name}: max_diff={max_diff:.2e} (mean={mean_diff:.2e}) > tolerance={self.tolerance:.2e}")

        return len(issues) == 0, "; ".join(issues) if issues else "OK"

    def test_operation(self, op_name: str, model_builder, mode: str = "cpu") -> Tuple[bool, str]:
        """Test a single operation"""
        try:
            # Build model
            model, input_data, input_name = model_builder()

            # Save to temp file
            temp_model = f"/tmp/test_{op_name}_{mode}.onnx"
            onnx.save(model, temp_model)

            # Run ONNX Runtime
            ref_outputs = self.run_onnx_runtime(model.SerializeToString(), input_data, input_name)

            # Run engine
            success, stdout, stderr = self.run_engine(temp_model, mode)
            if not success:
                return False, f"Engine failed: {stderr[:200]}"

            # Parse engine output
            test_outputs = self.parse_engine_output(stdout)

            # Compare
            match, msg = self.compare_outputs(ref_outputs, test_outputs)

            # Cleanup
            if os.path.exists(temp_model):
                os.remove(temp_model)

            return match, msg

        except Exception as e:
            return False, f"Exception: {str(e)}"

    # ==== Operation Test Builders ====

    def build_matmul(self):
        """MatMul operation"""
        input_shape = [2, 3]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        W = numpy_helper.from_array(
            np.random.randn(3, 4).astype(np.float32) * 0.5,
            name="W"
        )

        node = helper.make_node("MatMul", ["input", "W"], ["output"])

        graph = helper.make_graph(
            [node], "matmul_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
            [W]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_gemm(self):
        """Gemm operation (General Matrix Multiply with bias)"""
        input_shape = [2, 3]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        W = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32) * 0.5, name="W")
        B = numpy_helper.from_array(np.random.randn(4).astype(np.float32) * 0.1, name="B")

        node = helper.make_node("Gemm", ["input", "W", "B"], ["output"], alpha=1.0, beta=1.0)

        graph = helper.make_graph(
            [node], "gemm_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
            [W, B]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_add(self):
        """Add operation"""
        input_shape = [2, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        B = numpy_helper.from_array(np.random.randn(2, 4).astype(np.float32) * 0.5, name="B")

        node = helper.make_node("Add", ["input", "B"], ["output"])

        graph = helper.make_graph(
            [node], "add_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            [B]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_sub(self):
        """Sub operation"""
        input_shape = [2, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        B = numpy_helper.from_array(np.random.randn(2, 4).astype(np.float32) * 0.5, name="B")

        node = helper.make_node("Sub", ["input", "B"], ["output"])

        graph = helper.make_graph(
            [node], "sub_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            [B]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_mul(self):
        """Mul operation"""
        input_shape = [2, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        S = numpy_helper.from_array(np.random.randn(2, 4).astype(np.float32) * 0.5, name="S")

        node = helper.make_node("Mul", ["input", "S"], ["output"])

        graph = helper.make_graph(
            [node], "mul_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            [S]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_relu(self):
        """ReLU operation"""
        input_shape = [2, 4]
        # Use consistent test input that matches the C++ engine
        input_data = self.create_test_input(input_shape)

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        node = helper.make_node("Relu", ["input"], ["output"])

        graph = helper.make_graph(
            [node], "relu_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            []
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, input_data, "input"

    def build_sigmoid(self):
        """Sigmoid operation"""
        input_shape = [2, 4]
        # Use consistent test input that matches the C++ engine
        input_data = self.create_test_input(input_shape)

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        node = helper.make_node("Sigmoid", ["input"], ["output"])

        graph = helper.make_graph(
            [node], "sigmoid_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            []
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, input_data, "input"

    def build_transpose(self):
        """Transpose operation"""
        input_shape = [2, 3, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        node = helper.make_node("Transpose", ["input"], ["output"], perm=[2, 0, 1])

        graph = helper.make_graph(
            [node], "transpose_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 2, 3])],
            []
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_gather(self):
        """Gather operation"""
        input_shape = [3, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        indices = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name="indices")

        node = helper.make_node("Gather", ["input", "indices"], ["output"], axis=0)

        graph = helper.make_graph(
            [node], "gather_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
            [indices]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_shape(self):
        """Shape operation"""
        input_shape = [2, 3, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        node = helper.make_node("Shape", ["input"], ["output"])

        graph = helper.make_graph(
            [node], "shape_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.INT64, [3])],
            []
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_cast(self):
        """Cast operation"""
        input_shape = [2, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        node = helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)

        graph = helper.make_graph(
            [node], "cast_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)],
            []
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def build_reducesum(self):
        """ReduceSum operation"""
        input_shape = [2, 3, 4]
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")

        node = helper.make_node("ReduceSum", ["input", "axes"], ["output"], keepdims=1)

        graph = helper.make_graph(
            [node], "reducesum_test", [X],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1, 4])],
            [axes]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        return model, self.create_test_input(input_shape), "input"

    def run_all_tests(self, modes: List[str] = ["cpu"]):
        """Run all operation tests"""
        operations = {
            "MatMul": self.build_matmul,
            "Gemm": self.build_gemm,
            "Add": self.build_add,
            "Sub": self.build_sub,
            "Mul": self.build_mul,
            "ReLU": self.build_relu,
            "Sigmoid": self.build_sigmoid,
            "Transpose": self.build_transpose,
            "Gather": self.build_gather,
            "Shape": self.build_shape,
            "Cast": self.build_cast,
            "ReduceSum": self.build_reducesum,
        }

        print("=" * 70)
        print("COMPREHENSIVE OPERATION VALIDATION")
        print("=" * 70)
        print(f"Testing {len(operations)} operations in {len(modes)} mode(s)")
        print(f"Tolerance: {self.tolerance:.2e}\n")

        for mode in modes:
            print(f"\n{'='*70}")
            print(f"Mode: {mode.upper()}")
            print(f"{'='*70}\n")

            mode_results = {}

            for op_name, builder in operations.items():
                print(f"Testing {op_name:20s} ... ", end="", flush=True)
                success, msg = self.test_operation(op_name, builder, mode)
                mode_results[op_name] = success

                if success:
                    print(f"✅ PASS")
                else:
                    print(f"❌ FAIL: {msg}")

            self.results[mode] = mode_results

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        for mode, mode_results in self.results.items():
            passed = sum(1 for v in mode_results.values() if v)
            total = len(mode_results)

            print(f"\n{mode.upper()} Mode: {passed}/{total} tests passed")
            print("-" * 70)

            for op_name, success in mode_results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {op_name:20s} {status}")

        # Overall
        all_passed = all(all(results.values()) for results in self.results.values())

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("=" * 70)
            return 0
        else:
            print("⚠️  SOME TESTS FAILED")
            print("=" * 70)
            return 1


def main():
    parser = argparse.ArgumentParser(description="Validate all ONNX operations")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Path to engine binary")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for comparison")
    parser.add_argument("--cpu", action="store_true", help="Test CPU mode")
    parser.add_argument("--gpu", action="store_true", help="Test GPU mode")
    parser.add_argument("--both", action="store_true", help="Test both CPU and GPU modes")
    args = parser.parse_args()

    if not os.path.exists(args.engine):
        print(f"❌ Engine not found at {args.engine}")
        print("Build it first: cd build && cmake .. && make")
        return 1

    # Determine modes to test
    modes = []
    if args.both:
        modes = ["cpu", "gpu"]
    elif args.gpu:
        modes = ["gpu"]
    else:
        modes = ["cpu"]  # Default to CPU

    tester = OperationTester(args.engine, args.tolerance)
    tester.run_all_tests(modes)

    return tester.print_summary()


if __name__ == "__main__":
    sys.exit(main())
