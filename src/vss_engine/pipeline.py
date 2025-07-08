import requests
import whisper
import base64
import time
import re
import gradio as gr
import threading
import queue
from typing import Iterable, List, Dict, Optional, Union
import cv2
import numpy as np
import torch

import os
from .rag_db import RAGDatabase


class LocalPipeline:
    """Simple pipeline using local models via Ollama and Whisper."""

    def __init__(self, ollama_url: str = "http://localhost:11434", device: str | None = None,
                 rag_db_dir: str = "data/db") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.asr_model = whisper.load_model("small", device=self.device)
        self.rag_db_dir = rag_db_dir
        os.makedirs(self.rag_db_dir, exist_ok=True)

    def _db_for(self, source_id: str) -> RAGDatabase:
        """Return the RAG database for a given source."""
        path = os.path.join(self.rag_db_dir, f"{source_id}.pkl")
        return RAGDatabase(path)

    def transcribe(self, audio: Union[str, bytes, np.ndarray, None], source_id: str | None = None) -> str:
        """Transcribe audio from a path or raw bytes and cache the result."""
        if source_id:
            db = self._db_for(source_id)
            cached = db.get_transcript()
            if cached:
                return cached
        if audio is None:
            return ""
        if isinstance(audio, (str, bytes)):
            if isinstance(audio, str):
                input_data = audio
            else:
                # raw 16-bit PCM mono at 16 kHz
                array = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0
                input_data = array
        else:
            input_data = audio
        result = self.asr_model.transcribe(input_data)
        text = result.get("text", "")
        if source_id:
            db.add_transcript(text)
        return text

    def caption(self, image: Union[str, bytes, np.ndarray]) -> str:
        """Generate an image caption using the local VLM."""
        if isinstance(image, str):
            with open(image, "rb") as img:
                img_b64 = base64.b64encode(img.read()).decode()
        elif isinstance(image, bytes):
            img_b64 = base64.b64encode(image).decode()
        else:
            success, buf = cv2.imencode(".jpg", image)
            if not success:
                return ""
            img_b64 = base64.b64encode(buf.tobytes()).decode()

        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llava-llama3:8b",
                "prompt": "Describe this image.",
                "images": [img_b64],
                "stream": False,

            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def caption_frames(
        self,
        image_paths: list[str],
        fps: float | None = None,
        progress: gr.Progress | None = None,
        source_id: str | None = None,
    ) -> list[dict]:
        """Caption images in batches of five frames and report progress.

        Returns a list of dicts ``{"time": float, "caption": str}``.
        """
        if source_id:
            db = self._db_for(source_id)
            cached = db.get_captions()
            if cached:
                return cached

        captions: list[dict] = []
        total = len(image_paths)
        times: list[float] = []
        for i in range(0, total, 5):
            batch = image_paths[i:i + 5]
            images_b64 = []
            for p in batch:
                with open(p, "rb") as img:
                    images_b64.append(base64.b64encode(img.read()).decode())
            t0 = time.time()
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llava-llama3:8b",
                    "prompt": (
                        "Describe each image in one sentence. "
                        "Return one sentence per image separated by newline."
                    ),
                    "images": images_b64,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            elapsed = time.time() - t0
            times.append(elapsed)
            text = resp.json().get("response", "")
            batch_caps = [line.strip() for line in text.splitlines() if line.strip()]
            if len(batch_caps) < len(batch):
                batch_caps.extend([""] * (len(batch) - len(batch_caps)))
            for idx, cap in enumerate(batch_caps[: len(batch)]):
                frame_index = i + idx
                if fps:
                    timestamp = frame_index / fps
                else:
                    timestamp = 0.0
                captions.append({"time": timestamp, "caption": cap})
            if progress:
                processed = min(i + len(batch), total)
                avg = sum(times) / len(times)
                remaining = total - processed
                eta = int(avg * remaining)
                progress((processed, total), desc=f"{processed}/{total} ETA {eta}s")
        if progress:
            progress((total, total), desc="Done")
        if source_id:
            db.add_captions(captions)
        return captions

    def caption_realtime(
        self,
        video_path: str,
        target_fps: float = 4.0,
        min_fps: float = 2.0,
        source_id: str | None = None,
    ) -> list[dict]:
        """Caption video frames in near real time using two threads.

        Frames are sampled according to ``target_fps`` and dropped if inference
        is slower than ``min_fps``. Batch size is always one for lowest latency.
        """
        if source_id:
            db = self._db_for(source_id)
            cached = db.get_captions()
            if cached:
                return cached

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        step = max(1, int(round(orig_fps / target_fps)))
        frame_interval = 1.0 / target_fps
        q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=1)
        captions: list[dict] = []
        start = time.time()

        def capture_loop():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    success, buf = cv2.imencode(".jpg", frame)
                    if success:
                        try:
                            q.put(buf.tobytes(), block=False)
                        except queue.Full:
                            pass  # drop frame
                    time.sleep(frame_interval)
                idx += 1
            q.put(None)
            cap.release()

        def inference_loop():
            while True:
                item = q.get()
                if item is None:
                    break
                caption = self.caption(item)
                ts = time.time() - start
                captions.append({"time": ts, "caption": caption})

        t1 = threading.Thread(target=capture_loop)
        t2 = threading.Thread(target=inference_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if source_id:
            db.add_captions(captions)
        return captions

    def rerank(self, query: str, docs: list[str]):
        results = []
        for doc in docs:
            prompt = f"Query: {query}\nDocument: {doc}\nScore 0-1:"
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "dengcao/Qwen3-Reranker-8B:Q5_K_M",
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            score = float(resp.json().get("response", "0").strip())
            results.append((doc, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def answer(
        self,
        question: str,
        transcript: str | None = None,
        captions: list[dict] | None = None,
        source_id: str | None = None,
    ) -> str:
        """Generate an answer using RAG over the transcript and captions."""

        if source_id:
            db = self._db_for(source_id)
            if transcript is None:
                transcript = db.get_transcript()
            if captions is None:
                captions = db.get_captions()

        if transcript is None:
            transcript = ""

        docs: list[str] = []
        # Split transcript into sentences for retrieval
        for sent in re.split(r"(?<=[.!?])\s+", transcript):
            s = sent.strip()
            if s:
                docs.append(s)

        if captions:
            for c in captions:
                mm = int(c["time"] // 60)
                ss = int(c["time"] % 60)
                ts = f"{mm:02d}:{ss:02d}"
                docs.append(f"[{ts}] {c['caption']}")

        ranked = self.rerank(question, docs)
        context = "\n".join(doc for doc, _ in ranked[:5])

        prompt = (
            f"Context from video:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer the question and include the timestamp in mm:ss if relevant."
        )
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llava-llama3:8b",
                "prompt": prompt,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run local VSS pipeline")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for Ollama server")
    parser.add_argument("--audio", default="audio.wav", help="Path to audio file")
    parser.add_argument("--image", default="frame.jpg", help="Path to image file")
    args = parser.parse_args()

    pipe = LocalPipeline(args.ollama_url)
    transcript = pipe.transcribe(args.audio)
    print("Transcript:", transcript)

    captions = pipe.caption_frames([args.image], fps=1.0)
    print("Caption:", captions[0]["caption"] if captions else "")

    docs = ["doc one", "another document"]
    ranked = pipe.rerank("example query", docs)
    print("Reranked docs:", ranked)
