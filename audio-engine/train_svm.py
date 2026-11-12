"""Entraîner le SVM avec plusieurs hyperparamètres -> chaque essai est sauvegardé dans MLflow.

    python train_svm.py
    mlflow ui            # puis ouvrir http://localhost:5000
"""
import joblib
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Charger les caractéristiques (calculées une seule fois sur Kaggle)
data = np.load("data/features.npz")
X, y = data["X"], data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Les hyperparamètres à tester
C_values = [0.1, 1, 10]          # C petit = modèle simple, C grand = colle plus aux données
gamma_values = ["scale", 0.01]   # largeur du noyau RBF

mlflow.set_experiment("cardio-audio")

best_recall, best_model = 0, None
for C in C_values:
    for gamma in gamma_values:
        # 3. Un "run" MLflow = un essai
        with mlflow.start_run(run_name=f"SVM_C{C}_g{gamma}"):
            model = make_pipeline(StandardScaler(),
                                  SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred)   # % de pathologiques détectés (critère du PDF)
            mlflow.log_params({"model": "SVM", "C": C, "gamma": gamma})
            mlflow.log_metrics({"recall_abnormal": recall,
                                "accuracy": accuracy_score(y_test, y_pred),
                                "f1": f1_score(y_test, y_pred)})
            print(f"C={C:<4} gamma={gamma:<6} recall={recall:.3f}")

            # 4. Garder le meilleur
            if recall > best_recall:
                best_recall, best_model = recall, model

joblib.dump(best_model, "data/svm.joblib")
print(f"\nMeilleur recall = {best_recall:.3f} -> sauvegardé dans data/svm.joblib")
