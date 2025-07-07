#!/usr/bin/env bash
# Simple helper to run Ollama and the pipeline on an uncommon port.
# Pass -p to make the Gradio interface public.
set -euo pipefail

PORT=51234
PUBLIC=0

while getopts "p" opt; do
  case $opt in
    p)
      PUBLIC=1
      ;;
    *)
      echo "Usage: $0 [-p]" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

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
SHARE_ARG=""
if [ "$PUBLIC" -eq 1 ]; then
  SHARE_ARG="--share"
fi
python src/vss_engine/gradio_frontend.py --ollama-url "http://localhost:${PORT}" $SHARE_ARG


# Stop Ollama
kill $OLLAMA_PID
