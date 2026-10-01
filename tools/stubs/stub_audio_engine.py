"""Stub minimal du Service 3 (audio-engine), pour tester l'api-gateway
avant que le vrai moteur audio ne soit développé.

Lancer :
    uvicorn tools.stubs.stub_audio_engine:app --port 8003
"""

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Stub audio-engine")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    # Résultat simulé : le flag "abnormal" dépend de la taille du fichier,
    # juste pour avoir un peu de variation pendant les tests manuels.
    abnormal = len(data) % 2 == 0
    return {
        "probabilities": {"normal": 0.18, "abnormal": 0.82} if abnormal else {"normal": 0.91, "abnormal": 0.09},
        "predicted_class": "abnormal" if abnormal else "normal",
        "features": {
            "mfcc_mean": 12.4,
            "zcr": 0.07,
            "duration_s": round(len(data) / 32000, 2),
            "n_bytes_recus": len(data),
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "stub-audio-engine"}
