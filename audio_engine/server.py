from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import tempfile
from audio_engine.inference import predict

app = FastAPI(title="audio-engine")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        return predict(path)
    finally:
        Path(path).unlink(missing_ok=True)