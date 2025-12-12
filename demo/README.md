# ONNX Text Autocomplete Demo

A minimal web demo that uses the custom ONNX GPU execution engine to autocomplete text as you type.

## Features

- Real-time text generation as you type
- Automatic triggering on space or 300ms pause
- Live token streaming (no buffering)
- Request cancellation when new input arrives
- Clean, responsive UI

## Prerequisites

1. **Built ONNX GPU Engine**: The engine must be compiled at `../build/onnx_gpu_engine`
2. **Model Files**: You need:
   - `model.onnx` (your ONNX model file)
   - `tokenizer.json`

   Both should be in the project root directory (one level up from `demo/`)

3. **Python 3.8+**

## Setup

### 1. Install Python Dependencies

```bash
cd demo
pip install -r requirements.txt
```

Or use a virtual environment (recommended):

```bash
cd demo
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify File Paths

Make sure your directory structure looks like this:

```
ONNX-GPU-Execution-Engine/
├── build/
│   └── onnx_gpu_engine      # Compiled engine
├── model.onnx                # Your ONNX model
├── tokenizer.json            # Tokenizer file
└── demo/
    ├── backend.py            # FastAPI server
    ├── index.html            # Frontend
    ├── requirements.txt
    └── README.md
```

If your model or tokenizer are in different locations, edit the paths in `backend.py` (lines 52-54):

```python
model_path = os.path.join(os.path.dirname(__file__), "..", "model.onnx")
tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer.json")
```

## Running the Demo

### 1. Start the Backend Server

```bash
cd demo
python3 backend.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Open the Frontend

Simply open `index.html` in your web browser:

```bash
# On Linux
firefox index.html
# or
google-chrome index.html

# On macOS
open index.html

# On Windows
start index.html
```

Or manually navigate to: `file:///path/to/demo/index.html`

### 3. Use the Demo

1. Type some text in the input box (e.g., "Once upon a time")
2. Press space or pause for 300ms
3. Watch the generated continuation appear below in real-time
4. Continue typing to trigger new generations (previous ones auto-cancel)

## How It Works

### Backend (`backend.py`)

- FastAPI server with a single `/generate` endpoint
- Accepts POST requests with `{ "prompt": "text" }`
- Spawns subprocess running the ONNX GPU engine with `--quiet` flag
- Streams output character-by-character as Server-Sent Events (SSE)
- Automatically cancels previous generation when new request arrives
- Handles client disconnection gracefully

### Frontend (`index.html`)

- Vanilla JavaScript (no frameworks)
- Debounced input handling (300ms)
- Immediate trigger on space character
- Uses Fetch API with ReadableStream for token streaming
- AbortController for request cancellation
- Live status updates

### Engine CLI Interface

The backend calls the engine like this:

```bash
./onnx_gpu_engine model.onnx \
  --input "user prompt text" \
  --tokenizer tokenizer.json \
  --generate \
  --max-tokens 50 \
  --temperature 0.8 \
  --quiet
```

The `--quiet` flag makes the engine output only generated tokens (no logs or metadata).

## Configuration

### Generation Parameters

Edit `backend.py` line 78-80 to adjust:

```python
"--max-tokens", "50",      # Max tokens to generate
"--temperature", "0.8",    # Sampling temperature (0.0 = deterministic)
```

### Debounce Delay

Edit `index.html` line 87 to adjust typing delay:

```javascript
const DEBOUNCE_DELAY = 300;  // milliseconds
```

### Server Port

Edit `backend.py` line 133 to change port:

```python
uvicorn.run(app, host="127.0.0.1", port=8000)
```

Also update `index.html` line 86:

```javascript
const API_URL = 'http://127.0.0.1:8000/generate';
```

## Troubleshooting

### "Engine not found" error

Make sure the engine is built:
```bash
cd ..
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### "Model not found" error

Download or place your ONNX model in the project root, or update the path in `backend.py`.

### Generation not starting

1. Check backend logs for errors
2. Open browser console (F12) for frontend errors
3. Test the backend directly:
   ```bash
   curl -X POST http://127.0.0.1:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello"}'
   ```

### CORS errors

If you're serving the HTML from a different origin, the CORS middleware in `backend.py` should handle it. Check browser console for specific errors.

### Process not cancelling

The backend uses process IDs to track and kill previous subprocesses. If you see zombie processes, check that:
- The engine responds to SIGKILL signals properly
- You're not running multiple backend instances

## Limitations

- No authentication or rate limiting
- Single-user design (global process tracking)
- No persistent sessions
- No error recovery beyond cancellation
- Assumes engine outputs UTF-8 text

## License

Same as parent project.
