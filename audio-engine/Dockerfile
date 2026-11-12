FROM python:3.11-slim
WORKDIR /app

# 1. Installer les bibliothèques (seulement ce qu'il faut pour PRÉDIRE, pas pour entraîner)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copier le code + le modèle ONNX (pas besoin de features.npz ni svm.joblib)
COPY server.py prepare_data.py ./
COPY data/cardio_model.onnx data/

# 3. Lancer le serveur
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
