import requests
import whisper
import base64


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


    def caption_frames(self, image_paths: list[str]) -> list[str]:
        """Caption images in batches of five frames."""
        captions: list[str] = []
        for i in range(0, len(image_paths), 5):
            batch = image_paths[i : i + 5]
            images_b64 = []
            for p in batch:
                with open(p, "rb") as img:
                    images_b64.append(base64.b64encode(img.read()).decode())
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
            text = resp.json().get("response", "")
            batch_caps = [line.strip() for line in text.splitlines() if line.strip()]
            if len(batch_caps) < len(batch):
                batch_caps.extend([""] * (len(batch) - len(batch_caps)))
            captions.extend(batch_caps[: len(batch)])
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
        self, question: str, transcript: str, captions: list[str] | None = None
    ) -> str:
        """Generate an answer using the transcript and frame captions."""
        caption_text = "\n".join(captions or [])
        prompt = (
            f"Video transcript:\n{transcript}\n"
            f"Frame captions:\n{caption_text}\n\n"
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

    captions = pipe.caption_frames([args.image])
    print("Caption:", captions[0] if captions else "")

    docs = ["doc one", "another document"]
    ranked = pipe.rerank("example query", docs)
    print("Reranked docs:", ranked)
