#!/usr/bin/env python3
"""
Helper script for tokenization using tokenizer.json
Called from C++ code to encode/decode text with BPE tokenizer
"""
import sys
import json
import argparse

def load_tokenizer(tokenizer_path):
    """Load tokenizer from tokenizer.json"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path.endswith('/') else str(tokenizer_path).rsplit('/', 1)[0],
            local_files_only=True
        )
        return tokenizer
    except ImportError:
        # Fallback: manually parse tokenizer.json for basic functionality
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)

        # Build simple vocab lookup
        vocab = data.get('model', {}).get('vocab', {})
        return {'vocab': vocab, 'vocab_size': len(vocab)}

def encode(text, tokenizer_path):
    """Encode text to token IDs"""
    try:
        from tokenizers import Tokenizer
        # Load directly from tokenizer.json file
        if not tokenizer_path.endswith('tokenizer.json'):
            tokenizer_path = tokenizer_path.rstrip('/') + '/tokenizer.json'

        tokenizer = Tokenizer.from_file(tokenizer_path)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        token_ids = encoding.ids

        # Output: space-separated token IDs
        print(' '.join(map(str, token_ids)))
        return 0
    except ImportError:
        print("ERROR: tokenizers library not installed. Install with: pip install tokenizers", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

def decode(token_ids, tokenizer_path):
    """Decode token IDs to text"""
    try:
        from tokenizers import Tokenizer
        # Load directly from tokenizer.json file
        if not tokenizer_path.endswith('tokenizer.json'):
            tokenizer_path = tokenizer_path.rstrip('/') + '/tokenizer.json'

        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Convert token_ids string to list of ints
        ids = [int(x) for x in token_ids.split()]
        text = tokenizer.decode(ids, skip_special_tokens=True)

        # Output: decoded text
        print(text)
        return 0
    except ImportError:
        print("ERROR: tokenizers library not installed. Install with: pip install tokenizers", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

def main():
    parser = argparse.ArgumentParser(description='Tokenize or decode text using tokenizer.json')
    parser.add_argument('--encode', type=str, help='Text to encode')
    parser.add_argument('--decode', type=str, help='Space-separated token IDs to decode')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json or model directory')

    args = parser.parse_args()

    if args.encode:
        return encode(args.encode, args.tokenizer)
    elif args.decode:
        return decode(args.decode, args.tokenizer)
    else:
        print("ERROR: Must specify --encode or --decode", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
