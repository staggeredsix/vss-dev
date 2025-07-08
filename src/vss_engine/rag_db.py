import os
import pickle
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class RAGDatabase:
    """Per-video RAG store with a simple vector index."""

    def __init__(self, path: str) -> None:
        self.dir = path
        os.makedirs(self.dir, exist_ok=True)
        self.meta_path = os.path.join(self.dir, "meta.pkl")
        self.vec_path = os.path.join(self.dir, "vectors.pkl")

        self.data: Dict[str, Any] = {"transcript": "", "captions": [], "docs": []}
        self.embeddings: np.ndarray | None = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.data = pickle.load(f)
        if os.path.exists(self.vec_path):
            with open(self.vec_path, "rb") as f:
                obj = pickle.load(f)
                self.embeddings = obj.get("embeddings")
                self.data["docs"] = obj.get("docs", [])

    def _save(self) -> None:
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "transcript": self.data.get("transcript", ""),
                    "captions": self.data.get("captions", []),
                },
                f,
            )
        with open(self.vec_path, "wb") as f:
            pickle.dump(
                {"docs": self.data.get("docs", []), "embeddings": self.embeddings}, f
            )

    def add_transcript(self, transcript: str) -> None:
        self.add_transcript_segments(transcript, [])

    def add_transcript_segments(
        self, transcript: str, segments: List[Dict[str, Any]] | None = None
    ) -> None:
        self.data["transcript"] = transcript
        docs = []
        if segments:
            for seg in segments:
                ts = self._format_ts(seg.get("start", 0.0))
                docs.append(f"[{ts}] {seg.get('text', '').strip()}")
        else:
            for sent in re.split(r"(?<=[.!?])\s+", transcript):
                s = sent.strip()
                if s:
                    docs.append(s)
        self._add_docs(docs)
        self._save()

    def add_captions(self, captions: List[Dict[str, Any]]) -> None:
        self.data["captions"] = captions
        docs = []
        for c in captions:
            ts = self._format_ts(c.get("time", 0.0))
            docs.append(f"[{ts}] {c.get('caption', '').strip()}")
        self._add_docs(docs)
        self._save()

    def get_transcript(self) -> str:
        return self.data.get("transcript", "")

    def get_captions(self) -> List[Dict[str, Any]]:
        return self.data.get("captions", [])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings is None or not self.data.get("docs"):
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        emb = self.embeddings
        scores = (
            emb @ q_emb / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        )
        idx = np.argsort(-scores)[:top_k]
        return [(self.data["docs"][i], float(scores[i])) for i in idx]

    def _add_docs(self, docs: List[str]) -> None:
        new_docs = [d for d in docs if d and d not in self.data["docs"]]
        if not new_docs:
            return
        embs = self.model.encode(new_docs, convert_to_numpy=True)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        self.data["docs"].extend(new_docs)

    @staticmethod
    def _format_ts(sec: float) -> str:
        mm = int(sec // 60)
        ss = int(sec % 60)
        return f"{mm:02d}:{ss:02d}"
