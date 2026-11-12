"""ÉTAPE 3 — Convertir le meilleur modèle (svm.joblib) au format ONNX.

ONNX = format universel : le serveur n'a plus besoin de scikit-learn, seulement d'onnxruntime (léger).

    python export_onnx.py
"""
import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx

# 1. Charger le modèle entraîné
model = joblib.load("data/svm.joblib")
X = np.load("data/features.npz")["X"].astype(np.float32)

# 2. Convertir en ONNX (zipmap=False -> les probabilités sortent sous forme de tableau simple)
onnx_model = to_onnx(model, X[:1], options={"zipmap": False})
with open("data/cardio_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 3. Vérifier : ONNX doit donner les mêmes prédictions que scikit-learn
session = ort.InferenceSession("data/cardio_model.onnx")
labels, probas = session.run(None, {session.get_inputs()[0].name: X[:100]})
same = (labels == model.predict(X[:100])).mean()
print(f"cardio_model.onnx créé | prédictions identiques à scikit-learn : {same:.0%}")
