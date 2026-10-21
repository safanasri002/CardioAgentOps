import pickle
from pathlib import Path


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib

# ====================== Configuration ======================
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("audio_engine/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment("CardioAgent-Audio-Benchmark")

# ====================== Chargement des données ======================
print("Chargement des données...")
X = np.load(DATA_DIR / "X_mel_features.npy")
y = np.load(DATA_DIR / "y_labels.npy")

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Normal   : {np.sum(y == 0)}")
print(f"Abnormal : {np.sum(y == 1)}")

# Normalisation Generalement on place le split avant la normalisation pour éviter le data leakage, mais ici on normalise avant pour garder la même distribution sur les deux sets
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=True, random_state=42)),
])

# Sauvegarde du scaler
# joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
# print("Scaler sauvegardé")

# ====================== 1. Decision Tree ======================
# print("\n=== Entraînement Decision Tree ===")
# with mlflow.start_run(run_name="DecisionTree"):
#     model_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
#     model_dt.fit(X_train, y_train)

#     y_pred = model_dt.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     recall_abnormal = recall_score(y_test, y_pred, pos_label=1)

#     mlflow.log_params({"model": "DecisionTree", "max_depth": 10})
#     mlflow.log_metrics({
#         "accuracy": acc,
#         "recall_abnormal": recall_abnormal
#     })

#     # Correction du problème skops
#     mlflow.sklearn.log_model(
#         model_dt,
#         name="model",
#         skops_trusted_types=["sklearn.tree._tree.Tree"]
#     )

#     print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
#     print(f"Accuracy : {acc:.4f} | Recall Abnormal : {recall_abnormal:.4f}")

#     joblib.dump(model_dt, MODELS_DIR / "decision_tree.joblib")
#     print("Decision Tree sauvegardé")

# ====================== 2. SVM ======================
print("\n=== Entraînement SVM ===")
# with mlflow.start_run(run_name="SVM_RBF"):
#     model_svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
#     model_svm.fit(X_train, y_train)

#     y_pred = model_svm.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     recall_abnormal = recall_score(y_test, y_pred, pos_label=1)

#     mlflow.log_params({"model": "SVM", "kernel": "rbf", "C": 10})
#     mlflow.log_metrics({
#         "accuracy": acc,
#         "recall_abnormal": recall_abnormal
#     })

#     mlflow.sklearn.log_model(model_svm, name="model")

#     print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
#     print(f"Accuracy : {acc:.4f} | Recall Abnormal : {recall_abnormal:.4f}")

#     joblib.dump(model_svm, MODELS_DIR / "svm.joblib")
#     print("SVM sauvegardé")

# print("\nEntraînement terminé avec succès.")
# print("Modèles sauvegardés dans : audio_engine/models/")
with mlflow.start_run(run_name="SVM_RBF"):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_abnormal": recall_score(y_test, y_pred, pos_label=1),
        "precision_abnormal": precision_score(y_test, y_pred, pos_label=1),
        "f1_abnormal": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    mlflow.log_params({"model": "SVM", "kernel": "rbf", "C": 10,
                       "gamma": "scale", "class_weight": "balanced"})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, name="model")
    joblib.dump(pipe, MODELS_DIR / "svm_pipeline.joblib")