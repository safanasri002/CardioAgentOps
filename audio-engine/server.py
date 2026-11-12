"""ÉTAPE 4 — Serveur de l'audio-engine : reçoit un .wav, renvoie normal / abnormal.

    uvicorn server:app --port 8001
"""
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile

from prepare_data import preprocess   # le MÊME traitement que pour l'entraînement

app = FastAPI(title="audio-engine")
session = ort.InferenceSession("data/cardio_model.onnx")   # chargé une seule fois au démarrage


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    # 1. .wav -> 40 MFCC (2000 Hz, 5 s, normalisé)
    try:
        features = preprocess(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire ce fichier audio")

    # 2. Prédiction ONNX : le modèle renvoie [labels, probabilités]
    x = features.astype(np.float32).reshape(1, -1)
    labels, probas = session.run(None, {session.get_inputs()[0].name: x})
    prob_abnormal = float(probas[0][1])

    # 3. Réponse
    return {
        "label": "abnormal" if labels[0] == 1 else "normal",
        "prob_abnormal": round(prob_abnormal, 3),
    }
