"""Outils communs à TOUS les modèles : données + validation croisée K-Fold + suivi MLflow.

Utilisé par train_svm.py et train_tree.py -> tous les modèles sont évalués sur les MÊMES folds.

Principe (doc scikit-learn "Cross-validation" + vote majoritaire) :
    features.npz ─┬─ 80 % TRAIN ──► K-Fold (3 folds) ──► choix des hyperparamètres
                  │                                       └─► on GARDE les 3 modèles du meilleur candidat
                  └─ 20 % TEST  ──► modèle 1 ─┐
                                    modèle 2 ─┼─► VOTE (2 sur 3) ──► note finale (une seule fois)
                                    modèle 3 ─┘

Courbes d'apprentissage (artefacts MLflow, dossier "courbes/") :
    - chaque run (candidat)  : learning_curves_folds.png -> une courbe train/validation par fold
    - le meilleur run        : learning_curve_final.png  -> moyenne ± écart-type des folds + score TEST du vote
"""
import os

import matplotlib

matplotlib.use("Agg")                  # pas de fenêtre : on génère juste des images pour MLflow
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

N_FOLDS = 3                            # 3 modèles -> vote sans égalité possible (3-0 ou 2-1)
EXPERIMENT = "cardio-audio-kfold"
TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]   # parts du train utilisées pour tracer les courbes d'apprentissage
CURVE_METRICS = ["f2", "recall_abnormal", "accuracy"]

# Couleurs des courbes : train = bleu, validation = orange, test = vert
COLOR_TRAIN, COLOR_VAL, COLOR_TEST, COLOR_INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def load_split():
    """features.npz -> 80 % train (pour le K-Fold) / 20 % test (mis de côté pour la note finale)."""
    data = np.load("data/features.npz")
    X, y = data["X"], data["y"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def vote(models, X):
    """Vote majoritaire : chaque modèle prédit 0 (normal) ou 1 (anormal), la majorité gagne.

    Avec 3 modèles : au moins 2 votes « anormal » -> anormal, sinon normal.
    """
    preds = np.array([m.predict(X) for m in models])      # forme (nb_modèles, nb_sons)
    return (preds.sum(axis=0) > len(models) / 2).astype(int)


def compute_metrics(y_true, y_pred):
    return {"f2": fbeta_score(y_true, y_pred, beta=2),         # CRITÈRE DE CHOIX : recall compte 4x plus que la précision
            "recall_abnormal": recall_score(y_true, y_pred),   # % de pathologiques détectés
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)}


# ─────────────────────────────── Courbes d'apprentissage ───────────────────────────────

def learning_curve_points(model, X_tr, y_tr, X_eval, y_eval):
    """Entraîne sur 10 %, 25 %, ... 100 % de X_tr et mesure le score sur ces données d'entraînement
    ET sur des données jamais vues (X_eval).

    Lecture :
    - les deux courbes montent et se rapprochent -> le modèle apprend bien
    - grand écart train >> eval                  -> surapprentissage
    - les deux restent basses                    -> sous-apprentissage
    """
    curve = {"n_samples": []}
    for split in ("train", "eval"):
        for name in CURVE_METRICS:
            curve[f"{split}_{name}"] = []

    for frac in TRAIN_SIZES:
        if frac < 1.0:   # sous-échantillon stratifié -> garde ~23 % d'anormaux même à 10 %
            X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=frac, stratify=y_tr, random_state=42)
        else:
            X_sub, y_sub = X_tr, y_tr
        m = clone(model).fit(X_sub, y_sub)
        curve["n_samples"].append(len(y_sub))
        for split, (Xs, ys) in {"train": (X_sub, y_sub), "eval": (X_eval, y_eval)}.items():
            metrics = compute_metrics(ys, m.predict(Xs))
            for name in CURVE_METRICS:
                curve[f"{split}_{name}"].append(float(metrics[name]))
    return curve


def _style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=10, color=COLOR_INK)
    ax.set_xlabel("nb de sons d'entraînement", fontsize=8, color=COLOR_INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLOR_INK)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8, colors=COLOR_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_fold_curves(fold_curves, title):
    """Une colonne par fold, une ligne par métrique : train (bleu) vs validation du fold (orange)."""
    n_rows, n_cols = len(CURVE_METRICS), len(fold_curves)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3 * n_rows), sharey=True, squeeze=False)
    for col, curve in enumerate(fold_curves):
        for row, name in enumerate(CURVE_METRICS):
            ax = axes[row][col]
            ax.plot(curve["n_samples"], curve[f"train_{name}"], "o-", color=COLOR_TRAIN, lw=2, ms=5, label="train")
            ax.plot(curve["n_samples"], curve[f"eval_{name}"], "o-", color=COLOR_VAL, lw=2, ms=5, label="validation")
            _style(ax, f"Fold {col + 1}" if row == 0 else "", name if col == 0 else None)
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=12, color=COLOR_INK)
    fig.tight_layout()
    return fig


def plot_final_curve(fold_curves, test_scores, title):
    """Meilleurs hyperparamètres : moyenne ± écart-type des folds + score du VOTE des modèles sur le TEST."""
    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(5.5 * len(CURVE_METRICS), 4), sharey=True)
    n_cv = np.array(fold_curves[0]["n_samples"])
    for ax, name in zip(axes, CURVE_METRICS):
        for split, color, label in (("train", COLOR_TRAIN, "train (moyenne K-Fold)"),
                                    ("eval", COLOR_VAL, "validation (moyenne K-Fold)")):
            values = np.array([c[f"{split}_{name}"] for c in fold_curves])
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(n_cv, mean, "o-", color=color, lw=2, ms=5, label=label)
            ax.fill_between(n_cv, mean - std, mean + std, color=color, alpha=0.15, lw=0)   # bande ± écart-type
        ax.axhline(test_scores[name], ls="--", color=COLOR_TEST, lw=2,
                   label=f"TEST (vote des {len(fold_curves)} modèles) = {test_scores[name]:.2f}")
        _style(ax, name, "score")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=12, color=COLOR_INK)
    fig.tight_layout()
    return fig


# ─────────────────────────────── K-Fold + MLflow ───────────────────────────────

def kfold_scores(model, X, y):
    """Entraîne et évalue le modèle sur N_FOLDS découpages -> renvoie les scores, la courbe ET le modèle de chaque fold."""
    # Stratifié = chaque fold garde la même proportion normal / anormal (~23 % d'anormaux)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    scores = {"f2": [], "recall_abnormal": [], "accuracy": [], "f1": [], "train_accuracy": []}
    curves, fold_models = [], []
    for train_idx, val_idx in skf.split(X, y):
        m = clone(model)                       # un modèle neuf à chaque fold (sinon il garde l'ancien entraînement)
        m.fit(X[train_idx], y[train_idx])
        fold_models.append(m)                  # on GARDE le modèle : il votera en inférence
        for name, value in compute_metrics(y[val_idx], m.predict(X[val_idx])).items():
            scores[name].append(value)
        scores["train_accuracy"].append(m.score(X[train_idx], y[train_idx]))   # pour voir le surapprentissage
        curves.append(learning_curve_points(model, X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    return {name: np.array(values) for name, values in scores.items()}, curves, fold_models


def train_with_kfold(family, candidates):
    """Teste chaque candidat en K-Fold, garde les N_FOLDS modèles du meilleur, note leur VOTE sur le TEST.

    family     : nom du modèle ("SVM", "DecisionTree", ...)
    candidates : liste de (nom_du_run, hyperparamètres, modèle_non_entraîné)
    Renvoie la liste des N_FOLDS modèles entraînés (un par fold) -> à faire voter avec vote().
    """
    X_train, X_test, y_train, y_test = load_split()
    # Par défaut : base locale mlflow.db. Sur Kaggle : le serveur MLflow lancé par le notebook (variable d'environnement)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    # 1. K-Fold sur chaque candidat -> un run MLflow par candidat
    best = None
    for run_name, params, model in candidates:
        with mlflow.start_run(run_name=run_name) as run:
            scores, curves, fold_models = kfold_scores(model, X_train, y_train)

            mlflow.log_params({"model": family, "n_folds": N_FOLDS, **params})
            for name, values in scores.items():
                mlflow.log_metric(f"cv_{name}_mean", values.mean())
                mlflow.log_metric(f"cv_{name}_std", values.std())
                for fold, value in enumerate(values, start=1):    # courbe fold par fold dans MLflow
                    mlflow.log_metric(f"cv_{name}", value, step=fold)

            # Courbes d'apprentissage de chaque fold -> onglet "Artifacts" du run
            fig = plot_fold_curves(curves, f"{run_name} : courbes d'apprentissage par fold")
            mlflow.log_figure(fig, "courbes/learning_curves_folds.png")
            plt.close(fig)
            mlflow.log_dict({"train_sizes": TRAIN_SIZES, "folds": curves}, "courbes/learning_curves_folds.json")

            f2 = scores["f2"]
            print(f"{run_name:<22} F2 = {f2.mean():.3f} ± {f2.std():.3f}   "
                  f"recall = {scores['recall_abnormal'].mean():.3f}   acc = {scores['accuracy'].mean():.3f}   "
                  f"train acc = {scores['train_accuracy'].mean():.3f}")

            # 2. Garder le meilleur F2 MOYEN : priorité au recall (ne pas rater de malades)
            #    sans ignorer les fausses alarmes (un modèle qui dit toujours « anormal » a un F2 bas)
            if best is None or f2.mean() > best["f2"]:
                best = {"f2": f2.mean(), "name": run_name, "run_id": run.info.run_id,
                        "curves": curves, "fold_models": fold_models}

    # 3. Pas de ré-entraînement : les N_FOLDS modèles du meilleur candidat VOTENT, notés UNE fois sur le TEST
    models = best["fold_models"]
    test = compute_metrics(y_test, vote(models, X_test))
    # Pour comparer : score de chaque modèle seul sur le TEST (le vote doit être au moins aussi bon)
    alone = [compute_metrics(y_test, m.predict(X_test)) for m in models]

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_vote_{name}": value for name, value in test.items()})
        for i, metrics in enumerate(alone, start=1):
            mlflow.log_metrics({f"test_model{i}_{name}": value for name, value in metrics.items()})
        mlflow.set_tag("best", "true")
        mlflow.set_tag("inference", f"vote majoritaire de {len(models)} modèles")
        fig = plot_final_curve(best["curves"], test, f"Meilleur {family} : {best['name']}")
        mlflow.log_figure(fig, "courbes/learning_curve_final.png")
        plt.close(fig)

    print(f"\nMeilleur {family} : {best['name']} (F2 K-Fold = {best['f2']:.3f})")
    for i, metrics in enumerate(alone, start=1):
        print(f"  modèle {i} seul sur le TEST : " + "   ".join(f"{k} = {v:.3f}" for k, v in metrics.items()))
    print(f"Note finale du VOTE ({len(models)} modèles) sur le TEST : "
          + "   ".join(f"{k} = {v:.3f}" for k, v in test.items()))
    return models
