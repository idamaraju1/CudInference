#!/usr/bin/env python3
"""
Validate GPU_PERSISTENT mode against ONNX Runtime reference implementation.

This script validates both:
1. Single-shot inference with GPU_PERSISTENT mode
2. Autoregressive text generation with KV caching

Usage:
    # Validate simple models (single-shot inference)
    python3 scripts/validate_gpu_persist.py --mode single-shot

    # Validate text generation with SmolLM2
    python3 scripts/validate_gpu_persist.py --mode generation \
        --model path/to/SmolLM2.onnx \
        --tokenizer path/to/tokenizer.json \
        --input "The sky is blue because" \
        --max-tokens 10

    # Run all tests
    python3 scripts/validate_gpu_persist.py --mode all
"""

import argparse
import numpy as np
import onnxruntime as ort
import subprocess
import sys
import os
from typing import Dict, List, Tuple
import re
import json


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_subheader(text: str):
    """Print a formatted subheader"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'─'*70}{Colors.RESET}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ{Colors.RESET} {text}")


class Tokenizer:
    """Simple tokenizer wrapper"""

    def __init__(self, tokenizer_path: str):
        try:
            from tokenizers import Tokenizer as HFTokenizer
            self.tokenizer = HFTokenizer.from_file(tokenizer_path)
        except ImportError:
            print_warning("tokenizers package not installed. Install with: pip install tokenizers")
            print_info("Will try to parse C++ output for token IDs")
            self.tokenizer = None
        except Exception as e:
            print_warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        return self.tokenizer.decode(token_ids)

    def is_available(self) -> bool:
        """Check if tokenizer is available"""
        return self.tokenizer is not None


class ONNXRuntimeReference:
    """ONNX Runtime reference implementation for validation"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run_single_shot(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run a single inference"""
        # Run inference
        outputs = self.session.run(None, {name: inputs[name] for name in self.input_names if name in inputs})

        # Convert output list to dict
        return {name: output for name, output in zip(self.output_names, outputs)}

    def run_autoregressive(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float = 0.0
    ) -> Tuple[List[int], List[Dict[str, np.ndarray]]]:
        """
        Run autoregressive generation with ONNX Runtime.

        Returns:
            - List of generated token IDs
            - List of step outputs (for validation)
        """
        generated_tokens = input_ids.copy()
        step_outputs = []

        # Determine number of KV cache layers from model inputs
        num_layers = len([name for name in self.input_names if name.startswith('past_key_values.') and name.endswith('.key')])

        # Initialize past_kv with empty tensors for prefill step
        # Shape: (batch_size=1, num_heads=3, past_seq_len=0, head_dim=64)
        past_kv = {}
        for layer_idx in range(num_layers):
            past_kv[f'past_key_values.{layer_idx}.key'] = np.zeros((1, 3, 0, 64), dtype=np.float32)
            past_kv[f'past_key_values.{layer_idx}.value'] = np.zeros((1, 3, 0, 64), dtype=np.float32)

        for step in range(max_tokens):
            # Prepare inputs for this step
            if step == 0:
                # Prefill: process all tokens
                step_input_ids = np.array([input_ids], dtype=np.int64)
                seq_len = len(input_ids)
            else:
                # Decode: process only last token
                step_input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                seq_len = 1

            # Create attention mask for total sequence length (including past)
            past_seq_len = 0 if step == 0 else len(generated_tokens) - seq_len
            total_seq_len = past_seq_len + seq_len
            attention_mask = np.ones((1, total_seq_len), dtype=np.int64)

            # Create position IDs for current tokens
            if step == 0:
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                position_ids = np.array([[past_seq_len]], dtype=np.int64)

            # Build input dict
            inputs = {
                'input_ids': step_input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }

            # Add past KV cache
            inputs.update(past_kv)

            # Run inference
            outputs = self.run_single_shot(inputs)
            step_outputs.append(outputs)

            # Extract logits
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Sample next token
            if temperature == 0.0:
                next_token = int(np.argmax(next_token_logits))
            else:
                # Apply temperature and sample
                next_token_logits = next_token_logits / temperature
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token = int(np.random.choice(len(probs), p=probs))

            generated_tokens.append(next_token)

            # Update past KV cache from present outputs
            past_kv = {}
            for name, tensor in outputs.items():
                if name.startswith('present.'):
                    past_name = 'past_key_values.' + name[8:]  # Remove 'present.' prefix
                    past_kv[past_name] = tensor

        return generated_tokens, step_outputs


class CppEngineRunner:
    """Runner for C++ ONNX engine"""

    def __init__(self, executable_path: str = "./build/onnx_gpu_engine"):
        self.executable_path = executable_path
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ engine not found at {executable_path}")

    def run_single_shot(self, model_path: str, timeout: int = 10) -> Dict[str, np.ndarray]:
        """Run single-shot inference"""
        try:
            result = subprocess.run(
                [self.executable_path, model_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("C++ engine timed out")

        if result.returncode != 0:
            error_msg = f"C++ engine failed with return code {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}\n"
            raise RuntimeError(error_msg)

        # Parse output
        # Format: "Output output_name [shape]: [values]"
        outputs = {}
        for line in result.stdout.split('\n'):
            if line.startswith('Output '):
                if ':' in line:
                    parts = line.split(':', 1)
                    # Extract name: "Output output_name [1, 5]" -> "output_name"
                    name_part = parts[0].replace('Output ', '').strip()
                    # Remove shape if present: "output_name [1, 5]" -> "output_name"
                    if '[' in name_part:
                        name = name_part[:name_part.index('[')].strip()
                    else:
                        name = name_part
                    # Parse values
                    array_str = parts[1].strip().strip('[]')
                    output = np.array([float(x) for x in array_str.split(',')])
                    outputs[name] = output

        return outputs

    def run_generation(
        self,
        model_path: str,
        tokenizer_path: str,
        input_text: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout: int = 120
    ) -> Tuple[List[int], str]:
        """
        Run text generation.

        Returns:
            - List of generated token IDs (parsed from debug output)
            - Generated text
        """
        cmd = [
            self.executable_path,
            model_path,
            '--input', input_text,
            '--tokenizer', tokenizer_path,
            '--generate',
            '--max-tokens', str(max_tokens),
            '--temperature', str(temperature),
            '--debug'  # Enable debug output to see token IDs
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("C++ engine timed out during generation")

        if result.returncode != 0:
            error_msg = f"C++ engine failed with return code {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}\n"
            raise RuntimeError(error_msg)

        # Parse output
        token_ids = []
        generated_text = ""

        # Extract token IDs and generated text from output
        for line in result.stdout.split('\n'):
            # Look for: "Full sequence token IDs: [1, 2, 3, ...]"
            if 'Full sequence token IDs:' in line:
                match = re.search(r'Full sequence token IDs:\s*\[([\d,\s]+)\]', line)
                if match:
                    ids_str = match.group(1)
                    token_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip()]

            # Extract generated text
            if '=== Generated Text ===' in line:
                # Get next non-empty line
                lines = result.stdout.split('\n')
                idx = lines.index(line)
                for i in range(idx + 1, len(lines)):
                    if lines[i].strip() and '===' not in lines[i] and '[INFO]' not in lines[i]:
                        generated_text = lines[i].strip()
                        break

        return token_ids, generated_text


class Validator:
    """Main validation class"""

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.cpp_runner = CppEngineRunner()

    def compare_tensors(
        self,
        reference: np.ndarray,
        actual: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare two tensors and return whether they match.

        Returns:
            - bool: Whether tensors match within tolerance
            - dict: Statistics (max_diff, mean_diff, etc.)
        """
        # Flatten both for comparison (handles shape differences like (1, 5) vs (5,))
        ref_flat = reference.flatten()
        actual_flat = actual.flatten()

        if ref_flat.shape != actual_flat.shape:
            return False, {
                'max_diff': float('inf'),
                'mean_diff': float('inf'),
                'error': f'Size mismatch: {reference.shape} ({ref_flat.shape[0]} elements) vs {actual.shape} ({actual_flat.shape[0]} elements)'
            }

        diff = np.abs(ref_flat - actual_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < self.tolerance

        stats = {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'tolerance': self.tolerance,
            'passed': passed
        }

        return passed, stats

    def validate_single_shot(self, model_path: str, input_shape: Tuple[int, ...]) -> bool:
        """Validate single-shot inference"""
        model_name = os.path.basename(model_path)
        print_subheader(f"Validating: {model_name}")

        # Create test input
        total_size = np.prod(input_shape)
        input_data = (np.arange(total_size, dtype=np.float32) * 0.01).reshape(input_shape)

        print_info(f"Input shape: {input_shape}")
        print_info(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")

        # Run ONNX Runtime
        try:
            onnx_ref = ONNXRuntimeReference(model_path)
            ref_outputs = onnx_ref.run_single_shot({'input': input_data})
            print_success(f"ONNX Runtime completed ({len(ref_outputs)} outputs)")
        except Exception as e:
            print_error(f"ONNX Runtime failed: {e}")
            return False

        # Run C++ engine
        try:
            cpp_outputs = self.cpp_runner.run_single_shot(model_path)
            print_success(f"C++ engine completed ({len(cpp_outputs)} outputs)")
        except Exception as e:
            print_error(f"C++ engine failed: {e}")
            return False

        # Compare outputs
        all_passed = True
        print_info("Comparing outputs...")

        for name, ref_output in ref_outputs.items():
            if name not in cpp_outputs:
                print_error(f"Output '{name}' missing from C++ engine")
                all_passed = False
                continue

            # Debug: show shapes
            print_info(f"  ONNX RT shape: {ref_output.shape}, C++ shape: {cpp_outputs[name].shape}")

            passed, stats = self.compare_tensors(ref_output, cpp_outputs[name])

            if passed:
                print_success(f"Output '{name}': max_diff={stats['max_diff']:.2e}, mean_diff={stats['mean_diff']:.2e}")
            else:
                if 'error' in stats:
                    print_error(f"Output '{name}': {stats['error']}")
                else:
                    print_error(f"Output '{name}': max_diff={stats['max_diff']:.2e}, mean_diff={stats['mean_diff']:.2e} (threshold: {self.tolerance:.2e})")
                all_passed = False

        return all_passed

    def validate_generation(
        self,
        model_path: str,
        tokenizer_path: str,
        input_text: str,
        max_tokens: int,
        temperature: float = 0.0
    ) -> bool:
        """Validate autoregressive text generation with token-by-token comparison"""
        print_subheader(f"Validating Generation: {os.path.basename(model_path)}")

        print_info(f"Input: '{input_text}'")
        print_info(f"Max tokens: {max_tokens}")
        print_info(f"Temperature: {temperature}")

        # Load tokenizer
        tokenizer = Tokenizer(tokenizer_path)

        # Run C++ engine
        print_info("\nRunning C++ engine...")
        try:
            cpp_tokens, cpp_text = self.cpp_runner.run_generation(
                model_path, tokenizer_path, input_text, max_tokens, temperature
            )
            print_success(f"C++ engine completed")
            print_info(f"Generated {len(cpp_tokens)} tokens")
            print_info(f"Generated text: '{cpp_text}'")
        except Exception as e:
            print_error(f"C++ engine failed: {e}")
            return False

        # Basic validation
        if not cpp_tokens:
            print_error("No tokens generated")
            return False

        if not cpp_text:
            print_error("No text generated")
            return False

        # Try to run ONNX Runtime comparison if tokenizer is available
        if tokenizer.is_available():
            print_info("\nRunning ONNX Runtime reference...")
            try:
                # Tokenize input
                input_token_ids = tokenizer.encode(input_text)
                print_info(f"Input token IDs: {input_token_ids}")

                # Run ONNX Runtime
                onnx_ref = ONNXRuntimeReference(model_path)
                ort_tokens, _ = onnx_ref.run_autoregressive(
                    input_token_ids, max_tokens, temperature
                )
                ort_text = tokenizer.decode(ort_tokens)

                print_success(f"ONNX Runtime completed")
                print_info(f"Generated {len(ort_tokens)} tokens")
                print_info(f"Generated text: '{ort_text}'")

                # Compare token-by-token
                print_info("\nToken-by-token comparison:")
                all_match = True
                min_len = min(len(cpp_tokens), len(ort_tokens))

                # Show prompt tokens first
                prompt_len = len(input_token_ids)
                print_info(f"Prompt tokens ({prompt_len}): {input_token_ids}")

                # Compare generated tokens
                for i in range(min_len):
                    cpp_tok = cpp_tokens[i] if i < len(cpp_tokens) else None
                    ort_tok = ort_tokens[i] if i < len(ort_tokens) else None

                    # Skip prompt tokens (already known to match)
                    if i < prompt_len:
                        continue

                    gen_step = i - prompt_len
                    if cpp_tok == ort_tok:
                        print_success(f"  Step {gen_step}: {ort_tok}")
                    else:
                        print_error(f"  Step {gen_step}: C++={cpp_tok}, ONNX RT={ort_tok}")
                        all_match = False

                # Check length mismatch
                if len(cpp_tokens) != len(ort_tokens):
                    print_warning(f"Token count mismatch: C++={len(cpp_tokens)}, ONNX RT={len(ort_tokens)}")
                    all_match = False

                # Compare final text
                print_info("\nText comparison:")
                print_info(f"  C++:      '{cpp_text}'")
                print_info(f"  ONNX RT:  '{ort_text}'")

                if cpp_text.strip() == ort_text.strip():
                    print_success("Generated text matches!")
                else:
                    print_error("Generated text differs!")
                    all_match = False

                return all_match

            except Exception as e:
                print_error(f"ONNX Runtime comparison failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        # If tokenizer is not available, we can't do a proper comparison
        print_warning("Tokenizer not available - cannot compare with ONNX Runtime")
        print_info("Install tokenizers: pip install tokenizers")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate GPU_PERSISTENT mode against ONNX Runtime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode',
        choices=['single-shot', 'generation', 'all'],
        default='all',
        help='Validation mode (default: all)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to ONNX model (for generation mode)'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer.json (for generation mode)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='The sky is blue because',
        help='Input text for generation (default: "The sky is blue because")'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=10,
        help='Maximum tokens to generate (default: 10)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 = greedy)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Numerical comparison tolerance (default: 1e-4)'
    )

    args = parser.parse_args()

    print_header("GPU_PERSISTENT Mode Validation")

    validator = Validator(tolerance=args.tolerance)
    results = {}

    # Single-shot tests
    if args.mode in ['single-shot', 'all']:
        print_header("Single-Shot Inference Tests")

        test_cases = [
            ("simple_linear.onnx", (1, 10)),
            ("two_layer.onnx", (1, 10)),
            ("residual.onnx", (1, 10)),
            ("sparse_linear.onnx", (1, 10)),
        ]

        for model_file, input_shape in test_cases:
            if not os.path.exists(model_file):
                print_warning(f"Model {model_file} not found, skipping")
                continue

            model_name = os.path.splitext(model_file)[0]
            try:
                results[f"single_shot_{model_name}"] = validator.validate_single_shot(
                    model_file, input_shape
                )
            except Exception as e:
                print_error(f"Validation failed: {e}")
                results[f"single_shot_{model_name}"] = False

    # Generation tests
    if args.mode in ['generation', 'all']:
        print_header("Text Generation Tests")

        if args.model and args.tokenizer:
            # User-specified model
            test_name = f"generation_{os.path.splitext(os.path.basename(args.model))[0]}"
            try:
                results[test_name] = validator.validate_generation(
                    args.model,
                    args.tokenizer,
                    args.input,
                    args.max_tokens,
                    args.temperature
                )
            except Exception as e:
                print_error(f"Validation failed: {e}")
                results[test_name] = False
        elif args.mode == 'generation':
            print_error("Generation mode requires --model and --tokenizer arguments")
            return 1
        else:
            print_info("Skipping generation tests (no model specified)")
            print_info("Use --model and --tokenizer to test generation")

    # Summary
    if results:
        print_header("Validation Summary")

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for name, result in results.items():
            if result:
                print_success(f"{name:40} PASSED")
            else:
                print_error(f"{name:40} FAILED")

        print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")

        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.RESET}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️  {total - passed} test(s) failed{Colors.RESET}")
            return 1
    else:
        print_error("No tests were run")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
