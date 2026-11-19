"""Entraîner un arbre de décision en K-Fold (3 folds) avec plusieurs hyperparamètres -> MLflow.

Mêmes folds et même expérience que train_svm.py -> les deux modèles se comparent directement.

    python train_tree.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # puis ouvrir http://localhost:5000
"""
import joblib
from sklearn.tree import DecisionTreeClassifier

from kfold import train_with_kfold

# Les hyperparamètres à tester
#    Pas de StandardScaler : un arbre compare chaque feature à un seuil, l'échelle ne change rien.
depth_values = [3, 5, 10, None]   # profondeur max : petit = règles simples, None = l'arbre pousse jusqu'au bout
leaf_values = [1, 10]             # nb min de sons par feuille : grand = évite d'apprendre par cœur

candidates = [
    (f"Tree_d{depth}_leaf{leaf}", {"max_depth": depth, "min_samples_leaf": leaf},
     DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=42))
    for depth in depth_values for leaf in leaf_values
]

# On récupère les 3 modèles du meilleur candidat (un par fold) -> ils voteront en inférence
fold_models = train_with_kfold("DecisionTree", candidates)
joblib.dump(fold_models, "data/tree_folds.joblib")
print(f"-> {len(fold_models)} modèles sauvegardés dans data/tree_folds.joblib")
