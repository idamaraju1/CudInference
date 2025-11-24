#!/usr/bin/env python3
"""
Validate the custom engine against ONNX Runtime on a comprehensive ops model.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "all_ops_test.onnx")
DEFAULT_ENGINE = os.path.join(REPO_ROOT, "build", "onnx_gpu_engine")


def create_input(shape=(2, 3)) -> np.ndarray:
    # Mirror the engine's createTestInput: sequential values with step 0.01
    data = (np.arange(np.prod(shape), dtype=np.float32) % 100) * 0.01
    return data.reshape(shape)


def ensure_model(model_path: str):
    if os.path.exists(model_path):
        return
    generator = os.path.join(REPO_ROOT, "scripts", "create_all_ops_model.py")
    print(f"Model not found at {model_path}, generating with {generator}")
    subprocess.check_call([sys.executable, generator, "--output", model_path])


def run_onnx_runtime(model_path: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
    session = ort.InferenceSession(model_path)
    outputs = [o.name for o in session.get_outputs()]
    ort_outs = session.run(outputs, {"input": input_data})
    return dict(zip(outputs, ort_outs))


def parse_engine_outputs(stdout: str) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Output "):
            continue
        if ":" not in line:
            continue
        header, data_str = line.split(":", 1)
        parts = header.split()
        name = parts[1] if len(parts) >= 2 else f"out_{len(outputs)}"
        data_tokens = [tok.strip() for tok in data_str.replace("[", "").replace("]", "").split(",") if tok.strip()]
        if not data_tokens:
            continue
        outputs[name] = np.array([float(tok) for tok in data_tokens], dtype=np.float64)
    return outputs


def run_engine(engine_path: str, model_path: str, extra_args=None) -> Tuple[int, str, str]:
    cmd = [engine_path, model_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def compare_outputs(ort_outs: Dict[str, np.ndarray],
                    engine_outs: Dict[str, np.ndarray],
                    tol: float) -> bool:
    ok = True
    for name, ref in ort_outs.items():
        if name not in engine_outs:
            print(f"❌ Missing engine output: {name}")
            ok = False
            continue
        raw = engine_outs[name]
        if raw.size != ref.size:
            print(f"❌ Shape mismatch for {name}: engine {raw.size} vals vs ref {ref.size}")
            ok = False
            continue
        cast = raw.astype(ref.dtype).reshape(ref.shape)
        if np.issubdtype(ref.dtype, np.integer):
            equal = np.array_equal(cast, ref)
            diff = (cast != ref).sum()
            if not equal:
                print(f"❌ Integer mismatch for {name}: {diff} differing elements")
                ok = False
        else:
            max_diff = np.max(np.abs(cast - ref))
            mean_diff = np.mean(np.abs(cast - ref))
            if max_diff > tol:
                print(f"❌ Float mismatch for {name}: max diff {max_diff:.3e} (mean {mean_diff:.3e}) exceeds tol {tol}")
                ok = False
            else:
                print(f"✅ {name}: max diff {max_diff:.3e}, mean {mean_diff:.3e}")
    extra = set(engine_outs.keys()) - set(ort_outs.keys())
    for name in extra:
        print(f"ℹ️ Engine produced extra output '{name}' (ignored)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate custom engine against ONNX Runtime on all-ops model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to ONNX model")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Path to onnx_gpu_engine binary")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Absolute tolerance for float comparisons")
    parser.add_argument("--engine-args", nargs="*", default=None, help="Extra args passed to engine (default: --cpu)")
    args = parser.parse_args()

    ensure_model(args.model)
    input_data = create_input()

    print(f"Running ONNX Runtime on {args.model}")
    ort_outs = run_onnx_runtime(args.model, input_data)

    if not os.path.exists(args.engine):
        print(f"❌ Engine binary not found at {args.engine}. Build it first.")
        return 1

    engine_args = args.engine_args if args.engine_args is not None else ["--cpu"]
    engine_cmd = [args.engine, args.model] + (engine_args or [])
    print(f"\nRunning custom engine: {' '.join(engine_cmd)}")
    retcode, stdout, stderr = run_engine(args.engine, args.model, engine_args)
    if retcode != 0:
        print("❌ Engine run failed")
        print(stderr)
        return 1

    engine_outs = parse_engine_outputs(stdout)
    print("\nComparing outputs...")
    success = compare_outputs(ort_outs, engine_outs, args.tolerance)

    if success:
        print("\n🎉 Validation passed")
        return 0
    else:
        print("\n⚠️ Validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
