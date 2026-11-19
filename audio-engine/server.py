"""ÉTAPE 4 — Serveur de l'audio-engine : reçoit un .wav, renvoie normal / abnormal.

Les 3 modèles du K-Fold votent : au moins 2 votes « anormal » sur 3 -> abnormal.

    uvicorn server:app --port 8001
"""
import glob
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile

from prepare_data import preprocess   # le MÊME traitement que pour l'entraînement

app = FastAPI(title="audio-engine")
# Les modèles sont chargés une seule fois au démarrage (cardio_model_1.onnx, _2, _3)
sessions = [ort.InferenceSession(path) for path in sorted(glob.glob("data/cardio_model_*.onnx"))]
if not sessions:
    raise RuntimeError("Aucun modèle data/cardio_model_*.onnx : lancer python export_onnx.py")


@app.get("/health")
def health():
    return {"status": "ok", "n_models": len(sessions)}


@app.post("/predict")
async def predict(file: UploadFile):
    # 1. .wav -> 40 MFCC (2000 Hz, 5 s, normalisé)
    try:
        features = preprocess(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire ce fichier audio")

    # 2. Chaque modèle ONNX prédit : il renvoie [labels, probabilités]
    x = features.astype(np.float32).reshape(1, -1)
    votes, probs = [], []
    for session in sessions:
        labels, probas = session.run(None, {session.get_inputs()[0].name: x})
        votes.append(int(labels[0]))          # 1 = anormal, 0 = normal
        probs.append(float(probas[0][1]))     # probabilité d'être anormal

    # 3. Vote majoritaire (2 sur 3)
    n_abnormal = sum(votes)
    is_abnormal = n_abnormal > len(sessions) / 2

    # 4. Réponse
    return {
        "label": "abnormal" if is_abnormal else "normal",
        "votes_abnormal": f"{n_abnormal}/{len(sessions)}",
        "prob_abnormal": round(float(np.mean(probs)), 3),   # moyenne des 3 modèles (à titre indicatif)
    }
