FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn requests opencv-python-headless numpy
COPY src/spec_server.py .
ENV OLLAMA_URL=http://ollama:11434
ENV DRAFT_MODEL=llava:7b-v1.6
CMD ["uvicorn", "spec_server:app", "--host", "0.0.0.0", "--port", "8002"]
