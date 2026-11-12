import os

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from guardrails import check_input, check_output

# Adresses des autres services (dans Docker : le nom du service)
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://localhost:8001")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8002")

app = FastAPI(title="api-gateway")


@app.get("/")
def home():
    return FileResponse("static/index.html")   # la page web


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile):
    # 1. Guardrail d'entrée : vérifier le fichier
    audio = await file.read()
    error = check_input(file.filename, audio)
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        # 2. Envoyer le son à l'audio-engine -> probabilité normal / anormal
        r = httpx.post(f"{AUDIO_ENGINE_URL}/predict", files={"file": (file.filename, audio)}, timeout=60)
        prediction = r.json()

        # 3. Envoyer la prédiction aux agents -> rapport médical
        r = httpx.post(f"{AGENT_URL}/analyze", json=prediction, timeout=300)
        report = r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Un service interne ne répond pas")

    # 4. Guardrail de sortie : valider et filtrer la réponse du LLM
    try:
        report = check_output(report)
    except ValidationError:
        raise HTTPException(status_code=502, detail="Réponse du LLM invalide")

    # 5. Renvoyer le résultat au client
    return {"prediction": prediction, "report": report}
