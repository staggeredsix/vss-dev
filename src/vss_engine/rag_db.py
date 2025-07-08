import os
import pickle
from typing import List, Dict, Any

class RAGDatabase:
    """Simple on-disk store for video transcripts and captions."""

    def __init__(self, path: str = "rag_db.pkl") -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def add_transcript(self, video: str, transcript: str) -> None:
        video = os.path.abspath(video)
        entry = self.data.setdefault(video, {})
        entry["transcript"] = transcript
        self.save()

    def add_captions(self, video: str, captions: List[Dict[str, Any]]) -> None:
        video = os.path.abspath(video)
        entry = self.data.setdefault(video, {})
        entry["captions"] = captions
        self.save()

    def get_transcript(self, video: str) -> str:
        video = os.path.abspath(video)
        entry = self.data.get(video, {})
        return entry.get("transcript", "")

    def get_captions(self, video: str) -> List[Dict[str, Any]]:
        video = os.path.abspath(video)
        entry = self.data.get(video, {})
        return entry.get("captions", [])
