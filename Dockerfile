FROM nvcr.io/nvidia/pytorch:24.05-py3
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["jupyter", "lab", "--allow-root", "--ip", "0.0.0.0", "--port", "8888", "--no-browser"]
