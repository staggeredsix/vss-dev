FROM nvcr.io/nvidia/pytorch:24.05-py3
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/vss_engine/gradio_frontend.py"]
