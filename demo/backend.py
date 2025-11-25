#!/usr/bin/env python3
"""
FastAPI backend for streaming text generation from ONNX GPU engine.
"""
import asyncio
import os
import signal
import sys
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track active subprocess globally so we can cancel it
active_process: Optional[subprocess.Popen] = None
process_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str


async def stream_tokens(prompt: str, request: Request):
    """
    Stream tokens from the ONNX GPU engine subprocess.
    Yields tokens as they arrive, one at a time.
    """
    global active_process

    # Cancel any existing process
    async with process_lock:
        if active_process and active_process.poll() is None:
            try:
                active_process.kill()
                active_process.wait(timeout=1)
            except Exception:
                pass

    # Path to the engine executable (configurable for Docker)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if running in Docker (paths in /app) or locally (paths relative)
    if os.path.exists("/app/onnx_gpu_engine"):
        engine_path = "/app/onnx_gpu_engine"
        model_path = "/app/model.onnx"
        tokenizer_path = "/app/tokenizer.json"
    else:
        engine_path = os.path.join(script_dir, "..", "build", "onnx_gpu_engine")
        model_path = os.path.join(script_dir, "..", "model.onnx")
        tokenizer_path = os.path.join(script_dir, "..", "tokenizer.json")

    # Verify files exist
    if not os.path.exists(engine_path):
        yield f"data: ERROR: Engine not found at {engine_path}\n\n"
        return
    if not os.path.exists(model_path):
        yield f"data: ERROR: Model not found at {model_path}\n\n"
        return
    if not os.path.exists(tokenizer_path):
        yield f"data: ERROR: Tokenizer not found at {tokenizer_path}\n\n"
        return

    # Build command
    cmd = [
        engine_path,
        model_path,
        "--input", prompt,
        "--tokenizer", tokenizer_path,
        "--generate",
        "--max-tokens", "50",
        "--temperature", "0.8",
        "--quiet"
    ]

    try:
        # Start subprocess
        async with process_lock:
            active_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            current_process = active_process

        # Stream output character by character, skipping the prompt
        prompt_index = 0
        prompt_skipped = False

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                if current_process and current_process.poll() is None:
                    current_process.kill()
                break

            # Read one character at a time for immediate streaming
            char = current_process.stdout.read(1)

            if not char:
                # Process finished - check for errors
                current_process.wait(timeout=1)
                if current_process.returncode != 0:
                    stderr_output = current_process.stderr.read()
                    if stderr_output:
                        yield f"data: ERROR: Engine failed: {stderr_output[:200]}\n\n"
                    else:
                        yield f"data: ERROR: Engine exited with code {current_process.returncode}\n\n"
                break

            # Skip characters that match the original prompt
            if not prompt_skipped:
                if prompt_index < len(prompt) and char == prompt[prompt_index]:
                    prompt_index += 1
                    if prompt_index >= len(prompt):
                        prompt_skipped = True
                    continue
                else:
                    # Prompt doesn't match, start streaming from here
                    prompt_skipped = True

            # Send character as SSE event (only after prompt is skipped)
            yield f"data: {char}\n\n"

    except Exception as e:
        yield f"data: ERROR: {str(e)}\n\n"
    finally:
        # Clean up
        async with process_lock:
            if current_process and current_process.poll() is None:
                try:
                    current_process.kill()
                    current_process.wait(timeout=1)
                except Exception:
                    pass
            if active_process == current_process:
                active_process = None


@app.post("/generate")
async def generate(body: GenerateRequest, request: Request):
    """
    Generate text completion endpoint.
    Streams tokens as Server-Sent Events.
    """
    return StreamingResponse(
        stream_tokens(body.prompt, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, "index.html")

    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Text Autocomplete Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
