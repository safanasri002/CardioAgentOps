# Service 6 — mlflow-server : image officielle MLflow (même version que sur le PC : 3.18)
FROM ghcr.io/mlflow/mlflow:v3.18.0

# Dossier où tout est stocké (on y branchera un volume Docker à l'étape 2)
RUN mkdir -p /mlflow/artifacts
EXPOSE 5000

# --backend-store-uri     : base SQLite -> paramètres, métriques, registre des modèles
# --artifacts-destination : dossier -> fichiers des modèles
# --allowed-hosts "*"     : accepter les appels des autres conteneurs (http://mlflow-server:5000)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "*"]
