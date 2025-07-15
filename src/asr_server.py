from fastapi import FastAPI, UploadFile, File
import numpy as np
import whisper
import torch

app = FastAPI()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = whisper.load_model("small", device=_device)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return {"text": "", "segments": []}
    audio = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
    result = _model.transcribe(audio)
    return {"text": result.get("text", ""), "segments": result.get("segments", [])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
