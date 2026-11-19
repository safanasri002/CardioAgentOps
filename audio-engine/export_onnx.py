"""ÉTAPE 3 — Convertir les 3 modèles du K-Fold (svm_folds.joblib) au format ONNX.

ONNX = format universel : le serveur n'a plus besoin de scikit-learn, seulement d'onnxruntime (léger).
Un fichier ONNX par modèle -> le serveur les fait voter (2 sur 3).

    python export_onnx.py
"""
import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx

from kfold import vote

# 1. Charger les modèles entraînés (un par fold)
models = joblib.load("data/svm_folds.joblib")
X = np.load("data/features.npz")["X"].astype(np.float32)

sessions = []
for i, model in enumerate(models, start=1):
    # 2. Convertir en ONNX (zipmap=False -> les probabilités sortent sous forme de tableau simple)
    path = f"data/cardio_model_{i}.onnx"
    onnx_model = to_onnx(model, X[:1], options={"zipmap": False})
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 3. Vérifier : chaque ONNX doit donner les mêmes prédictions que scikit-learn
    session = ort.InferenceSession(path)
    labels, _ = session.run(None, {session.get_inputs()[0].name: X[:100]})
    same = (labels == model.predict(X[:100])).mean()
    print(f"{path} créé | prédictions identiques à scikit-learn : {same:.0%}")
    sessions.append(session)

# 4. Vérifier le VOTE : ONNX (comme dans server.py) vs scikit-learn (comme dans kfold.py)
onnx_votes = np.array([s.run(None, {s.get_inputs()[0].name: X[:100]})[0] for s in sessions])
onnx_vote = (onnx_votes.sum(axis=0) > len(sessions) / 2).astype(int)
same = (onnx_vote == vote(models, X[:100])).mean()
print(f"Vote des {len(sessions)} modèles ONNX identique au vote scikit-learn : {same:.0%}")
