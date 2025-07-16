from fastapi import FastAPI, UploadFile, File, Form
import requests
import base64
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "llava:7b-v1.6")

app = FastAPI()


def _ollama_generate(payload: dict) -> dict:
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    resp.raise_for_status()
    return resp.json()


def caption_image(image_bytes: bytes, context: str = "") -> str:
    img_b64 = base64.b64encode(image_bytes).decode()
    draft_payload = {
        "model": DRAFT_MODEL,
        "prompt": "Describe the frame in detail.",
        "images": [img_b64],
        "stream": False,
    }
    draft = _ollama_generate(draft_payload).get("response", "")
    prompt = (
        "You are a vision-language model refining a draft caption from a smaller model.\n"
        f"Draft: {draft}\n"
        f"Previous captions: {context}\n"
        "Provide a final detailed caption for the current frame."
    )
    final_payload = {
        "model": "llava:34b-v1.6",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }
    return _ollama_generate(final_payload).get("response", "")


@app.post("/caption")
async def caption(file: UploadFile = File(...), context: str = Form("")):
    data = await file.read()
    if not data:
        return {"caption": ""}
    cap = caption_image(data, context)
    return {"caption": cap}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
