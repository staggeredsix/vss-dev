import requests
import whisper
import base64
import time
import re
import json
import gradio as gr
import threading
import queue
from typing import Iterable, List, Dict, Optional, Union
from telemetry import Telemetry
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import logging
from sentence_transformers import CrossEncoder
from jsonschema import validate, ValidationError


import os


from .rag_db import RAGDatabase


@dataclass
class FrameCaption:
    """Description of a single video frame."""

    frame: int
    time: float
    caption: str


class LocalPipeline:
    """Simple pipeline using local models via Ollama and Whisper."""

    def __init__(
        self,
        ollama_url: str = "http://ollama:11434",
        device: str | None = None,
        rag_db_dir: str = "data/db",
        telemetry: Telemetry | None = None,
        asr_url: str | None = None,
    ) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)
        self.telemetry = telemetry
        self.ollama_url = ollama_url.rstrip("/")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.asr_url = asr_url or os.environ.get("ASR_URL")
        self.asr_model = None
        if not self.asr_url:
            self.asr_model = whisper.load_model("small", device=self.device)
        # Lightweight cross-encoder for reranking
        self.rerank_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device
        )

        self.rag_db_dir = rag_db_dir
        os.makedirs(self.rag_db_dir, exist_ok=True)
        schema_path = os.path.join(os.path.dirname(__file__), "frame_schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.frame_schema = json.load(f)

    def _post_generate(self, payload: dict) -> requests.Response:
        """Helper to send a generate request to Ollama with logging."""
        model = payload.get("model", "")
        prompt = payload.get("prompt", "")
        self.logger.debug(
            "Sending generate request to %s with model=%s", self.ollama_url, model
        )
        if self.telemetry:
            self.telemetry.record("generate_start", model=model)
        start = time.time()
        resp = requests.post(f"{self.ollama_url}/api/generate", json=payload)
        elapsed = time.time() - start
        self.logger.info("Model %s finished in %.2fs", model, elapsed)
        if self.telemetry:
            self.telemetry.record("generate_end", model=model, elapsed=elapsed)
        resp.raise_for_status()
        return resp

    def _db_for(self, source_id: str) -> RAGDatabase:
        """Return the RAG database for a given source."""
        path = os.path.join(self.rag_db_dir, source_id)
        return RAGDatabase(path)

    def _validate_frame(self, frame: dict) -> bool:
        """Validate a frame dictionary against the JSON schema."""
        try:
            validate(frame, self.frame_schema)
        except ValidationError as e:
            self.logger.error("Invalid frame data: %s", e)
            return False
        return True

    def transcribe(
        self, audio: Union[str, bytes, np.ndarray, None], source_id: str | None = None
    ) -> str:
        """Transcribe audio from a path or raw bytes and cache the result."""
        if self.telemetry:
            self.telemetry.record("transcribe_start", source=source_id)
        self.logger.info(
            "Transcribing audio%s", f" for {source_id}" if source_id else ""
        )
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
                raw_bytes = open(audio, "rb").read()
            else:
                # raw 16-bit PCM mono at 16 kHz
                array = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0
                input_data = array
                raw_bytes = audio
        else:
            input_data = audio
            raw_bytes = audio.tobytes() if isinstance(audio, np.ndarray) else audio

        if self.asr_url:
            files = {"file": ("audio.pcm", raw_bytes)}
            t0 = time.time()
            resp = requests.post(f"{self.asr_url}/transcribe", files=files)
            elapsed = time.time() - t0
            resp.raise_for_status()
            result = resp.json()
            self.logger.info("Remote ASR finished in %.2fs", elapsed)
        else:
            t0 = time.time()
            result = self.asr_model.transcribe(input_data)
            elapsed = time.time() - t0
            self.logger.info("Whisper transcription finished in %.2fs", elapsed)

        text = result.get("text", "")
        segments = result.get("segments", [])
        if self.telemetry:
            self.telemetry.record("transcribe_end", source=source_id, elapsed=elapsed)
        if source_id:

            db.add_transcript_segments(text, segments)

        return text

    def caption(self, image: Union[str, bytes, np.ndarray], context: str = "") -> str:
        """Generate a frame caption using the local VLM with optional context.

        Regardless of the input format, the image is resized to 244x244 before
        being sent to the model."""
        self.logger.debug("Captioning image")

        frame: np.ndarray | None = None
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                with open(image, "rb") as img:
                    data = img.read()
                img_b64 = base64.b64encode(data).decode()
        elif isinstance(image, bytes):
            arr = np.frombuffer(image, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                img_b64 = base64.b64encode(image).decode()
        else:
            frame = image

        if frame is not None:
            frame = cv2.resize(frame, (244, 244))
            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                return ""
            img_b64 = base64.b64encode(buf.tobytes()).decode()

        prompt = (
            "You are a vision-language model describing frames from a video. "
            "The provided context comes from prior frames captured by the same vision model.\n"
            f"Previous captions: {context}\n"
            "Describe the current frame in detail."
        )
        resp = self._post_generate(
            {
                "model": "llava:34b-v1.6",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }
        )
        return resp.json().get("response", "")

    def caption_frames(
        self,
        image_paths: list[str],
        fps: float | None = None,
        progress: gr.Progress | None = None,
        source_id: str | None = None,
    ) -> list[dict]:
        """Caption images in batches of five frames and report progress.

        Returns a list of ``FrameCaption`` dictionaries with ``frame``, ``time``
        and ``caption`` fields.
        """
        if self.telemetry:
            self.telemetry.record("caption_start", count=len(image_paths))
        self.logger.debug("Captioning %d frames", len(image_paths))
        if source_id:

            db = self._db_for(source_id)
            cached = db.get_captions()

            if cached:
                return cached

        captions: list[dict] = []
        total = len(image_paths)
        times: list[float] = []
        context_caps: list[str] = []
        for i in range(0, total, 5):
            batch = image_paths[i : i + 5]
            images_b64 = []
            for p in batch:
                frame = cv2.imread(p)
                if frame is not None:
                    frame = cv2.resize(frame, (244, 244))
                    success, buf = cv2.imencode(".jpg", frame)
                    if not success:
                        continue
                    images_b64.append(base64.b64encode(buf.tobytes()).decode())
                else:
                    with open(p, "rb") as img:
                        images_b64.append(base64.b64encode(img.read()).decode())
            t0 = time.time()
            context = " ".join(context_caps[-5:])
            prompt = (
                "You are a vision-language model describing frames from a video. "
                "The provided context comes from prior frames generated by the same vision model.\n"
                f"Previous captions: {context}\n"
                "Describe each frame in detail."
            )
            resp = self._post_generate(
                {
                    "model": "llava:34b-v1.6",
                    "prompt": prompt,
                    "images": images_b64,
                    "stream": False,
                }
            )
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
                frame_dict = {"frame": frame_index, "time": timestamp, "caption": cap}
                if self._validate_frame(frame_dict):
                    captions.append(frame_dict)
                    context_caps.append(cap)
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
        if self.telemetry:
            self.telemetry.record("caption_end", count=len(image_paths))

        return captions

    def caption_realtime(
        self,
        video_path: str,
        progress: gr.Progress | None = None,
        source_id: str | None = None,
    ) -> list[dict]:
        """Caption video frames without dropping frames.

        A frame is processed every four frames from the input video. Progress is
        reported via ``progress`` if provided. Returns a list of ``FrameCaption``
        dictionaries validated against the schema.
        """
        if self.telemetry:
            self.telemetry.record("realtime_caption_start", video=video_path)
        self.logger.info("Realtime captioning %s", video_path)
        if source_id:

            db = self._db_for(source_id)
            cached = db.get_captions()

            if cached:
                return cached

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = 4
        total_to_process = (total_frames + step - 1) // step if total_frames else 0
        self.logger.info("Total frames to process: %d", total_to_process)
        q: queue.Queue[Optional[tuple[int, float, bytes]]] = queue.Queue()
        captions: list[dict] = []
        times: list[float] = []
        processed = 0
        context_caps: list[str] = []

        def capture_loop():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_ts = idx / orig_fps if orig_fps else 0.0
                    frame = cv2.resize(frame, (244, 244))
                    success, buf = cv2.imencode(".jpg", frame)
                    if success:
                        q.put((idx, frame_ts, buf.tobytes()))
                idx += 1
            q.put(None)
            cap.release()

        def inference_loop():
            nonlocal processed
            while True:
                item = q.get()
                if item is None:
                    break
                frame_idx, frame_ts, data = item
                context = " ".join(context_caps[-5:])
                t0 = time.time()
                caption = self.caption(data, context)
                elapsed = time.time() - t0
                times.append(elapsed)
                processed += 1
                frame_dict = {"frame": frame_idx, "time": frame_ts, "caption": caption}
                if self._validate_frame(frame_dict):
                    captions.append(frame_dict)
                    context_caps.append(caption)
                avg = sum(times) / len(times)
                remaining = max(total_to_process - processed, 0)
                eta = int(avg * remaining)
                if progress:
                    progress(
                        (processed, total_to_process),
                        desc=f"{processed}/{total_to_process} ETA {eta}s",
                    )
                self.logger.info(
                    "Processed %d/%d frames (ETA %ds)",
                    processed,
                    total_to_process,
                    eta,
                )

        t1 = threading.Thread(target=capture_loop)
        t2 = threading.Thread(target=inference_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if progress:
            progress((total_to_process, total_to_process), desc="Done")
        if source_id:

            db.add_captions(captions)
        if self.telemetry:
            self.telemetry.record("realtime_caption_end", frames=processed)

        return captions

    def rerank(self, query: str, docs: list[str]):
        """Return documents sorted by relevance using a local reranker."""
        self.logger.debug("Reranking %d documents", len(docs))
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self.rerank_model.predict(pairs)
        results = list(zip(docs, [float(s) for s in scores]))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def answer(
        self,
        question: str,
        transcript: str | None = None,
        captions: list[dict] | None = None,
        source_id: str | None = None,
        *,
        use_rerank: bool = True,
        top_n: int = 3,
    ) -> tuple[str, list[str]]:
        """Generate an answer using RAG over the transcript and captions."""
        if self.telemetry:
            self.telemetry.record("answer_start")
        self.logger.info("Answering question: %s", question)

        if source_id:

            db = self._db_for(source_id)
            if transcript is None:
                transcript = db.get_transcript()
            if captions is None:
                captions = db.get_captions()

        if transcript is None:
            transcript = ""

        docs: list[str] = []
        if source_id:
            search_k = max(top_n, 10) if use_rerank else top_n
            docs = [doc for doc, _ in db.search(question, top_k=search_k)]
        else:
            for sent in re.split(r"(?<=[.!?])\s+", transcript):
                s = sent.strip()
                if s:
                    docs.append(s)
            if captions:
                for c in captions:
                    if self._validate_frame(c):
                        mm = int(c["time"] // 60)
                        ss = int(c["time"] % 60)
                        ts = f"{mm:02d}:{ss:02d}"
                        docs.append(f"[{ts}] {c['caption']}")

        if use_rerank:
            ranked = self.rerank(question, docs)
            context_docs = [doc for doc, _ in ranked[:top_n]]
        else:
            context_docs = docs[:top_n]
        context = "\n".join(context_docs)

        prompt = (
            "You are answering a question about a video. "
            "The following context was generated by a vision model processing the video frames.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Provide a detailed answer and include the timestamp in mm:ss if relevant."
        )
        resp = self._post_generate(
            {
                "model": "llava:34b-v1.6",
                "prompt": prompt,
                "stream": False,
            }
        )
        result = resp.json().get("response", "")
        if self.telemetry:
            self.telemetry.record("answer_end")
        return result, context_docs


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run local VSS pipeline")
    parser.add_argument(
        "--ollama-url",
        default="http://ollama:11434",
        help="Base URL for Ollama server",
    )
    parser.add_argument("--audio", default="audio.wav", help="Path to audio file")
    parser.add_argument("--image", default="frame.jpg", help="Path to image file")
    args = parser.parse_args()

    telem = Telemetry()
    pipe = LocalPipeline(args.ollama_url, telemetry=telem)
    transcript = pipe.transcribe(args.audio)
    print("Transcript:", transcript)

    captions = pipe.caption_frames([args.image], fps=1.0)
    print("Caption:", captions[0]["caption"] if captions else "")

    docs = ["doc one", "another document"]
    ranked = pipe.rerank("example query", docs)
    print("Reranked docs:", ranked)
