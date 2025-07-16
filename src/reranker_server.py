from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch

app = FastAPI()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=_device)

class RerankRequest(BaseModel):
    query: str
    docs: list[str]

@app.post("/rerank")
def rerank(req: RerankRequest):
    if not req.docs:
        return {"results": []}
    pairs = [(req.query, doc) for doc in req.docs]
    scores = _model.predict(pairs)
    results = [
        {"doc": doc, "score": float(score)} for doc, score in zip(req.docs, scores)
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
