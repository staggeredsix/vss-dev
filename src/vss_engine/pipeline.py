import requests
import whisper


class LocalPipeline:
    """Simple pipeline using local models via Ollama and Whisper."""

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.asr_model = whisper.load_model("small")

    def transcribe(self, audio_path: str) -> str:
        result = self.asr_model.transcribe(audio_path)
        return result.get("text", "")

    def caption(self, image_path: str) -> str:
        with open(image_path, "rb") as img:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llava-llama3:instruct",
                    "prompt": "Describe this image.",
                    "images": [image_path],
                },
            )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def rerank(self, query: str, docs: list[str]):
        results = []
        for doc in docs:
            prompt = f"Query: {query}\nDocument: {doc}\nScore 0-1:"
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "dengcao/Qwen3-Reranker-0.6B", "prompt": prompt},
            )
            resp.raise_for_status()
            score = float(resp.json().get("response", "0").strip())
            results.append((doc, score))
        return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    pipe = LocalPipeline()
    # Example usage; replace with your own paths and documents
    transcript = pipe.transcribe("audio.wav")
    print("Transcript:", transcript)

    caption = pipe.caption("frame.jpg")
    print("Caption:", caption)

    docs = ["doc one", "another document"]
    ranked = pipe.rerank("example query", docs)
    print("Reranked docs:", ranked)
