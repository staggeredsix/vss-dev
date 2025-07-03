#!/usr/bin/env bash
# Simple helper to run Ollama and the pipeline on an uncommon port.
set -euo pipefail

PORT=51234

# Install PyTorch for GPUs using CUDA 12.8 and Whisper from source
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install -q git+https://github.com/openai/whisper.git


# Start Ollama on the chosen port in the background
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Please install it first." >&2
  exit 1
fi

# run ollama serve on specified port using environment variable
export OLLAMA_HOST="localhost:${PORT}"
ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for server
until curl -sf "http://localhost:${PORT}/api/tags" >/dev/null; do
  sleep 1
done

echo "Ollama running on port ${PORT}"

# Pull required models

ollama pull llava-llama3:8b
ollama pull dengcao/Qwen3-Reranker-8B:Q5_K_M

# Launch the Gradio frontend using the same port
python src/vss_engine/gradio_frontend.py --ollama-url "http://localhost:${PORT}"


# Stop Ollama
kill $OLLAMA_PID
