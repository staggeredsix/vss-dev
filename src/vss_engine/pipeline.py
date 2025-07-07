import requests
import whisper
import base64
import time
import re
import gradio as gr


class LocalPipeline:
    """Simple pipeline using local models via Ollama and Whisper."""

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.asr_model = whisper.load_model("small")

    def transcribe(self, audio_path: str | None) -> str:
        if not audio_path:
            return ""
        result = self.asr_model.transcribe(audio_path)
        return result.get("text", "")

    def caption(self, image_path: str) -> str:
        """Generate an image caption using the local VLM."""
        with open(image_path, "rb") as img:
            img_b64 = base64.b64encode(img.read()).decode()

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
    ) -> list[dict]:
        """Caption images in batches of five frames and report progress.

        Returns a list of dicts ``{"time": float, "caption": str}``.
        """
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
        transcript: str,
        captions: list[dict] | None = None,
    ) -> str:
        """Generate an answer using RAG over the transcript and captions."""

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
