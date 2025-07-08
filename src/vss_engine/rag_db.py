import os
import pickle
from typing import List, Dict, Any

class RAGDatabase:

    """Simple on-disk store for a single video's transcript and captions."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Any] = {"transcript": "", "captions": []}

        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)


    def add_transcript(self, transcript: str) -> None:
        self.data["transcript"] = transcript
        self.save()

    def add_captions(self, captions: List[Dict[str, Any]]) -> None:
        self.data["captions"] = captions
        self.save()

    def get_transcript(self) -> str:
        return self.data.get("transcript", "")

    def get_captions(self) -> List[Dict[str, Any]]:
        return self.data.get("captions", [])

