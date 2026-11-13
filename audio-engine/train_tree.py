"""Entraîner un arbre de décision avec plusieurs hyperparamètres -> chaque essai est sauvegardé dans MLflow.

Même découpage et même expérience que train_svm.py -> les deux modèles se comparent directement.

    python train_tree.py
    mlflow ui            # puis ouvrir http://localhost:5000
"""
import joblib
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Charger les caractéristiques (les mêmes 40 MFCC que pour le SVM)
data = np.load("data/features.npz")
X, y = data["X"], data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Les hyperparamètres à tester
#    Pas de StandardScaler : un arbre compare chaque feature à un seuil, l'échelle ne change rien.
depth_values = [3, 5, 10, None]   # profondeur max : petit = règles simples, None = l'arbre pousse jusqu'au bout
leaf_values = [1, 10]             # nb min de sons par feuille : grand = évite d'apprendre par cœur

mlflow.set_experiment("cardio-audio")

best_recall, best_model = 0, None
for depth in depth_values:
    for leaf in leaf_values:
        # 3. Un "run" MLflow = un essai
        with mlflow.start_run(run_name=f"Tree_d{depth}_leaf{leaf}"):
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,class_weight="balanced", random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred)   # % de pathologiques détectés 
            mlflow.log_params({"model": "DecisionTree", "max_depth": depth, "min_samples_leaf": leaf})
            mlflow.log_metrics({"recall_abnormal": recall,
                                "accuracy": accuracy_score(y_test, y_pred),
                                "f1": f1_score(y_test, y_pred),
                                "train_accuracy": model.score(X_train, y_train)})   # pour voir le surapprentissage
            print(f"depth={str(depth):<4} leaf={leaf:<3} recall={recall:.3f} "
                  f"acc test={accuracy_score(y_test, y_pred):.3f} / train={model.score(X_train, y_train):.3f}")

            # 4. Garder le meilleur
            if recall > best_recall:
                best_recall, best_model = recall, model

joblib.dump(best_model, "data/tree.joblib")
print(f"\nMeilleur recall = {best_recall:.3f} -> sauvegardé dans data/tree.joblib")
