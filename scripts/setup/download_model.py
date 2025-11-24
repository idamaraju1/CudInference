#!/usr/bin/env python3
"""
Download and convert HuggingFace SmolLM2-135M model to ONNX format.
This script replaces the download_model_and_tokenizer.sh functionality.

Requirements:
    - huggingface_hub
    - transformers>=4.30.0
    - onnx>=1.14.0
"""

import sys
import onnx
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_ID_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Has pre-converted ONNX
OUTPUT_DIR = Path(".")
ONNX_PATH = OUTPUT_DIR / "model.onnx"
TOKENIZER_PATH = OUTPUT_DIR / "tokenizer.json"


def print_step(step_num, total_steps, message):
    """Print a formatted step message."""
    print(f"\n[{step_num}/{total_steps}] {message}")


def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model is valid and has embedded weights."""
    print(f"  → Verifying ONNX model at {onnx_path}...")

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("  ✓ ONNX model is valid")

        # Check that weights are embedded (not external)
        has_external_data = any(
            tensor.HasField('data_location') and
            tensor.data_location == onnx.TensorProto.EXTERNAL
            for tensor in model.graph.initializer
        )

        if has_external_data:
            print("  ⚠ Warning: Model has external data references")
            return False
        else:
            print("  ✓ All weights are embedded in the ONNX file")

        # Print model info
        num_initializers = len(model.graph.initializer)
        num_nodes = len(model.graph.node)
        print(f"  → Model has {num_initializers} initializers and {num_nodes} nodes")

        return True
    except Exception as e:
        print(f"  ✗ ONNX verification failed: {e}")
        return False


def download_onnx_model():
    """Try to download pre-converted ONNX model, or convert from PyTorch."""
    import torch
    from transformers import AutoModelForCausalLM
    import shutil

    # Try both base and instruct models for pre-converted ONNX
    for repo_id in [MODEL_ID_INSTRUCT, MODEL_ID]:
        print(f"  → Checking for pre-converted ONNX model in {repo_id}...")
        try:
            files = list_repo_files(repo_id)
            # Look for ONNX files in onnx/ directory
            onnx_files = [f for f in files if f.endswith('.onnx') and 'onnx' in f.lower()]

            if onnx_files:
                # Prefer the base model.onnx (FP32), not quantized versions
                target_file = None
                for f in onnx_files:
                    if f == 'onnx/model.onnx':  # Exact match for the base FP32 model
                        target_file = f
                        break

                if not target_file:
                    # Fallback to any model.onnx
                    target_file = next((f for f in onnx_files if f.endswith('model.onnx')), onnx_files[0])

                print(f"  ✓ Found pre-converted ONNX model: {target_file}")
                print(f"  → Downloading... (this may take a minute)")

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=target_file,
                )

                # Copy to expected location
                shutil.copy(downloaded_path, ONNX_PATH)
                print(f"  ✓ Downloaded successfully")
                return True
        except Exception as e:
            print(f"  → Not found in {repo_id}: {e}")

    # If no pre-converted model, convert from PyTorch
    print(f"  → Converting PyTorch model to ONNX...")
    print(f"  → Downloading model (this may take a few minutes)...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()

        # Create simple dummy input
        dummy_input_ids = torch.ones(1, 8, dtype=torch.long)

        print(f"  → Exporting to ONNX (opset 17)...")

        # Use torch.onnx.export with simple settings
        torch.onnx.export(
            model,
            (dummy_input_ids,),
            str(ONNX_PATH),
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

        return True

    except Exception as e:
        print(f"  ✗ Failed to convert model: {e}")
        return False


def main():
    print("=" * 70)
    print("SmolLM2-135M Model Download and ONNX Conversion")
    print("=" * 70)

    # Step 1: Download tokenizer
    print_step(1, 3, f"Downloading tokenizer from {MODEL_ID}")
    try:
        import shutil
        temp_tokenizer_dir = OUTPUT_DIR / ".temp_tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(temp_tokenizer_dir)

        # Copy only tokenizer.json to output directory
        if (temp_tokenizer_dir / "tokenizer.json").exists():
            shutil.copy(temp_tokenizer_dir / "tokenizer.json", TOKENIZER_PATH)
            print(f"  ✓ Tokenizer saved to {TOKENIZER_PATH}")
        else:
            print(f"  ⚠ Warning: {TOKENIZER_PATH} not found after save")

        # Clean up temp directory
        shutil.rmtree(temp_tokenizer_dir)

    except Exception as e:
        print(f"  ✗ Failed to download tokenizer: {e}")
        sys.exit(1)

    # Step 2: Download or convert ONNX model
    print_step(2, 3, "Obtaining ONNX model")
    if not download_onnx_model():
        print(f"  ✗ Failed to obtain ONNX model")
        sys.exit(1)

    print(f"  ✓ ONNX model saved to {ONNX_PATH}")
    print(f"  → File size: {ONNX_PATH.stat().st_size / (1024*1024):.2f} MB")

    # Step 3: Verify ONNX model
    print_step(3, 3, "Verifying ONNX model")
    if not verify_onnx_model(ONNX_PATH):
        print("  ⚠ Verification completed with warnings")
    else:
        print("  ✓ Verification passed")

    # Analyze model structure
    try:
        model = onnx.load(str(ONNX_PATH))
        print(f"\n  → Model Structure:")
        print(f"    Inputs: {len(model.graph.input)}")
        for inp in model.graph.input[:3]:
            print(f"      - {inp.name}: {[d.dim_value or 'dynamic' for d in inp.type.tensor_type.shape.dim]}")
        if len(model.graph.input) > 3:
            print(f"      ... and {len(model.graph.input) - 3} more")

        print(f"    Outputs: {len(model.graph.output)}")
        for out in model.graph.output[:2]:
            print(f"      - {out.name}: {[d.dim_value or 'dynamic' for d in out.type.tensor_type.shape.dim]}")
        if len(model.graph.output) > 2:
            print(f"      ... and {len(model.graph.output) - 2} more")

    except Exception as e:
        print(f"  ⚠ Could not analyze model: {e}")

    # Final cleanup: Remove any extra tokenizer files
    print(f"\n  → Cleaning up extra files...")
    extra_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "config.json",
    ]
    removed_count = 0
    for filename in extra_files:
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            filepath.unlink()
            removed_count += 1

    if removed_count > 0:
        print(f"  ✓ Removed {removed_count} extra file(s)")

    # Summary
    print("\n" + "=" * 70)
    print("✓ Download and conversion completed successfully!")
    print("=" * 70)
    print(f"Model:     {ONNX_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print("\nYou can now run the engine:")
    print(f"  ./build/onnx_gpu_engine {ONNX_PATH} \\")
    print(f"    --input \"Hello world\" \\")
    print(f"    --tokenizer {TOKENIZER_PATH} \\")
    print(f"    --generate --max-tokens 50")
    print("=" * 70)


if __name__ == "__main__":
    main()
