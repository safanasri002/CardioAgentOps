"""Entraîner le SVM en K-Fold (3 folds) avec plusieurs hyperparamètres -> chaque essai est sauvegardé dans MLflow.

    python train_svm.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # puis ouvrir http://localhost:5000
"""
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kfold import train_with_kfold

# Les hyperparamètres à tester
C_values = [0.1, 1, 10]          # C petit = modèle simple, C grand = colle plus aux données
gamma_values = ["scale", 0.01]   # largeur du noyau RBF

# Le scaler est DANS le pipeline -> il est ré-appris sur chaque fold (pas de fuite de données)
candidates = [
    (f"SVM_C{C}_g{gamma}", {"C": C, "gamma": gamma},
     make_pipeline(StandardScaler(),
                   SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True)))
    for C in C_values for gamma in gamma_values
]

# On récupère les 3 modèles du meilleur candidat (un par fold) -> ils voteront en inférence
fold_models = train_with_kfold("SVM", candidates)
joblib.dump(fold_models, "data/svm_folds.joblib")
print(f"-> {len(fold_models)} modèles sauvegardés dans data/svm_folds.joblib (puis : python export_onnx.py)")
