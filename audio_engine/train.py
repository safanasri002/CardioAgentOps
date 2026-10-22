from pathlib import Path

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# ====================== Configuration ======================
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("audio_engine/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment("CardioAgentOps_audio")

# ====================== Chargement des données ======================
print("Chargement des données...", flush=True)
X = np.load(DATA_DIR / "X_mel_features.npy")
y = np.load(DATA_DIR / "y_labels.npy")

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Normal   : {np.sum(y == 0)}")
print(f"Abnormal : {np.sum(y == 1)}", flush=True)

# ====================== Split (avant toute normalisation) ======================
# Le scaler est dans le pipeline : il est fitté uniquement sur le train,
# ce qui évite le data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== Pipeline : scaler + SVM ======================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", random_state=42)),
])

# ====================== Entraînement + évaluation ======================
print("\n=== Entraînement SVM ===", flush=True)

with mlflow.start_run(run_name="SVM_RBF"):
    pipe.fit(X_train, y_train)
    print("Fit terminé, évaluation...", flush=True)

    y_pred = pipe.predict(X_test)
    y_score = pipe.decision_function(X_test)  # score continu

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_abnormal": recall_score(y_test, y_pred, pos_label=1),
        "precision_abnormal": precision_score(y_test, y_pred, pos_label=1),
        "f1_abnormal": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
    }

    mlflow.log_params({
        "model": "SVM", "kernel": "rbf", "C": 10, "gamma": "scale",
        "class_weight": "balanced", "test_size": 0.2, "random_state": 42,
        "n_samples": len(y), "n_features": X.shape[1],
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, name="model")

    print("\n" + classification_report(
        y_test, y_pred, target_names=["normal", "abnormal"]))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print(confusion_matrix(y_test, y_pred))
    print()
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    joblib.dump(pipe, MODELS_DIR / "svm_pipeline.joblib")
    print("\nPipeline sauvegardé :", MODELS_DIR / "svm_pipeline.joblib")

print("\nTerminé.")