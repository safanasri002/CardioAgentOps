"""CNN du TP7 (getMyModel) sur les images mel-spectrogramme, en K-Fold (3 folds) + hard vote -> MLflow + ONNX.

Traduction PyTorch du TP7 (Keras) — PyTorch + ONNX sont imposés par l'énoncé :
    TP7 (Keras)                                   ici (PyTorch)
    ImageDataGenerator.flow_from_directory   ->   load_folder() : lit train_kfoldK/ et test_kfoldK/
    getMyModel() : Conv 64/32/16 + Dense 64  ->   CNN_TP7 : la même architecture
    Dense(1) + sigmoid + binary_crossentropy ->   Linear(64, 1) + BCEWithLogitsLoss (= sigmoid + BCE)
    ModelCheckpoint(monitor='val_acc')       ->   on garde l'epoch au meilleur F2 de validation
    plot_acc_loss + confusion matrix         ->   mêmes courbes / rapports, enregistrés dans MLflow

Ajouts par rapport au TP7 (mêmes règles que le SVM, voir kfold.py) :
    - poids des classes (~23 % d'anormaux), sinon le réseau dit toujours « normal »
    - choix sur le F2 (recall prioritaire), les 3 réseaux du meilleur candidat VOTENT sur test/
    - MLflow nested=True : 1 run parent par candidat, 1 run enfant par fold (courbes epoch par epoch)

    python prepare_images.py   (une fois)
    python train_cnn.py        (GPU conseillé : Kaggle)
"""
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from kfold import COLOR_INK, COLOR_TEST, COLOR_TRAIN, COLOR_VAL, EXPERIMENT, N_FOLDS, compute_metrics

IMAGES = Path("data/images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (240, 320)   # TP7 : 480 x 640. Divisé par 2 : 4x plus rapide, l'image d'origine ne fait que 40 x 157
EPOCHS = int(os.environ.get("EPOCHS", 30))   # comme le TP7
PATIENCE = 10           # arrêt si le F2 de validation ne progresse plus pendant 10 epochs
BATCH_SIZE = 32         # TP7 : 5 (trop lent sur 2500 images)

# Les hyperparamètres à tester : l'optimiseur du TP7 (SGD, lr = 0.0001) et Adam
CANDIDATES = [
    ("CNN_TP7_SGD_lr0.0001", {"optimizer": "SGD", "lr": 1e-4}),
    ("CNN_TP7_Adam_lr0.0001", {"optimizer": "Adam", "lr": 1e-4}),
]


# ─────────────────────────────── Données ───────────────────────────────

def load_folder(folder):
    """Comme flow_from_directory : lit folder/normal/*.png et folder/abnormal/*.png.

    Attention, bug du TP7 : flow_from_directory range les classes par ordre ALPHABÉTIQUE
    (abnormal = 0, normal = 1) alors que target_names = ['normal', 'abnormal']. Ici les labels sont explicites.
    """
    X, y = [], []
    for label, cls in ((0, "normal"), (1, "abnormal")):
        for png in sorted((IMAGES / folder / cls).glob("*.png")):
            X.append(np.asarray(Image.open(png).convert("RGB")).transpose(2, 0, 1))   # (3, 40, 157)
            y.append(label)
    return np.array(X, dtype=np.uint8), np.array(y)


# ─────────────────────────────── Le réseau (getMyModel du TP7) ───────────────────────────────

class CNN_TP7(nn.Module):
    """3 x [Conv 3x3 -> ReLU -> MaxPool 2x2] (64, 32 puis 16 filtres) -> Flatten -> Dense 64 -> Dropout 0.5 -> 1 sortie."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        )
        n_flat = self.features(torch.zeros(1, 3, *IMG_SIZE)).numel()   # taille après Flatten
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x):
        """x : images (B, 3, 40, 157), pixels entre 0 et 255 -> un score (logit) par image."""
        x = F.interpolate(x / 255.0, size=IMG_SIZE, mode="bilinear", align_corners=False)   # = target_size du TP7
        return self.classifier(self.features(x)).squeeze(1)


class ExportWrapper(nn.Module):
    """Pour l'ONNX : mêmes sorties que le SVM exporté -> [label, probabilités (normal, anormal)]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        p = torch.sigmoid(self.model(x))
        return (p >= 0.5).long(), torch.stack([1 - p, p], dim=1)


# ─────────────────────────────── Entraînement ───────────────────────────────

def predict_proba(model, X):
    """Probabilité d'être anormal pour chaque image (par paquets, sans gradient)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.from_numpy(X[i:i + 128]).float().to(DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def predict(model, X):
    return (predict_proba(model, X) >= 0.5).astype(int)   # comme le TP7 : probabilities > 0.5


def vote(models, X):
    """Hard vote : au moins 2 réseaux sur 3 disent « anormal » -> anormal (comme kfold.vote)."""
    preds = np.array([predict(m, X) for m in models])
    return (preds.sum(axis=0) > len(models) / 2).astype(int)


def bce_loss(model, X, y, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.from_numpy(X[i:i + 128]).float().to(DEVICE)
            yb = torch.from_numpy(y[i:i + 128]).float().to(DEVICE)
            total += loss_fn(model(xb), yb).item() * len(xb)
    return total / len(X)


def train_one_fold(params, X_tr, y_tr, X_val, y_val):
    """Entraîne un réseau neuf (EPOCHS epochs max), garde l'epoch au meilleur F2 de validation."""
    torch.manual_seed(42)
    model = CNN_TP7().to(DEVICE)
    if params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])        # comme le TP7
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Poids des classes : une erreur sur un anormal coûte (nb normaux / nb anormaux) ≈ 3x plus cher
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {k: [] for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy",
                               "train_f2", "val_f2", "val_recall_abnormal")}
    best_f2, best_state, best_epoch, waited = -1, None, 0, 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # 1. Un passage sur les images d'apprentissage, par paquets mélangés
        model.train()
        order = np.random.permutation(len(y_tr))
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X_tr[idx]).float().to(DEVICE)
            yb = torch.from_numpy(y_tr[idx]).float().to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        # 2. Mesures de l'epoch (comme history.history du TP7, + F2 et recall)
        train_m = compute_metrics(y_tr, predict(model, X_tr))
        val_m = compute_metrics(y_val, predict(model, X_val))
        m = {"train_loss": bce_loss(model, X_tr, y_tr, loss_fn), "val_loss": bce_loss(model, X_val, y_val, loss_fn),
             "train_accuracy": train_m["accuracy"], "val_accuracy": val_m["accuracy"],
             "train_f2": train_m["f2"], "val_f2": val_m["f2"], "val_recall_abnormal": val_m["recall_abnormal"]}
        for k, v in m.items():
            history[k].append(float(v))
        mlflow.log_metrics(m, step=epoch)
        print(f"    epoch {epoch:2d}  loss {m['train_loss']:.3f}/{m['val_loss']:.3f}  "
              f"acc {m['train_accuracy']:.3f}/{m['val_accuracy']:.3f}  F2 val {m['val_f2']:.3f}")

        # 3. Checkpoint (ModelCheckpoint du TP7) : on garde la meilleure epoch
        if val_m["f2"] > best_f2:
            best_f2, best_epoch, waited = val_m["f2"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    duration = time.time() - start
    model.load_state_dict(best_state)
    mlflow.log_metrics({"best_epoch": best_epoch, "training_time_s": duration})

    # 4. Matrice de confusion + rapport de classification (comme le TP7), sur la validation du fold
    y_pred = predict(model, X_val)
    report = (f"Meilleure epoch : {best_epoch}   temps d'entraînement : {duration:.0f} s\n\n"
              f"Confusion Matrix (lignes = vrai, colonnes = prédit ; normal, abnormal)\n"
              f"{confusion_matrix(y_val, y_pred)}\n\nClassification Report\n"
              f"{classification_report(y_val, y_pred, target_names=['normal', 'abnormal'])}")
    print(report)
    mlflow.log_text(report, "rapport_validation.txt")
    return model, history


# ─────────────────────────────── Courbes (plot_acc_loss du TP7) ───────────────────────────────

def plot_acc_loss(histories, title):
    """Une colonne par fold ; lignes : accuracy, loss, F2 — train (bleu) vs validation (orange)."""
    rows = (("accuracy", "accuracy"), ("loss", "loss"), ("f2", "F2"))
    fig, axes = plt.subplots(len(rows), len(histories), figsize=(3.6 * len(histories), 8), squeeze=False)
    for col, h in enumerate(histories):
        epochs = range(1, len(h["train_loss"]) + 1)
        for row, (key, label) in enumerate(rows):
            ax = axes[row][col]
            ax.plot(epochs, h[f"train_{key}"], "-", color=COLOR_TRAIN, lw=2, label="train")
            ax.plot(epochs, h[f"val_{key}"], "-", color=COLOR_VAL, lw=2, label="validation")
            if row == 0:
                ax.set_title(f"Fold {col + 1}", fontsize=10, color=COLOR_INK)
            if key != "loss":
                ax.set_ylim(0, 1.05)
            if row == len(rows) - 1:
                ax.set_xlabel("epoch", fontsize=8, color=COLOR_INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=COLOR_INK)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8, colors=COLOR_INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12, color=COLOR_INK)
    fig.tight_layout()
    return fig


def plot_test_summary(alone, test, title):
    """Barres : chaque réseau seul sur le TEST vs le VOTE des 3."""
    names = ["f2", "recall_abnormal", "accuracy"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.8 / (len(alone) + 1)
    x = np.arange(len(names))
    for i, m in enumerate(alone):
        ax.bar(x + i * width, [m[n] for n in names], width, color=COLOR_TRAIN, alpha=0.35 + 0.2 * i,
               label=f"réseau {i + 1} seul")
    ax.bar(x + len(alone) * width, [test[n] for n in names], width, color=COLOR_TEST, label="VOTE")
    ax.set_xticks(x + width * len(alone) / 2, names)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center")
    ax.set_title(title, fontsize=11, color=COLOR_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────── Export ONNX ───────────────────────────────

def export_onnx(models, X_check):
    """Un fichier ONNX par réseau (data/cnn_model_1.onnx, _2, _3) + vérification avec onnxruntime.
    Entrée : image (B, 3, 40, 157) en float, pixels 0-255 (le redimensionnement est DANS le modèle)."""
    import onnxruntime as ort

    X_check = X_check.astype(np.float32)
    sessions = []
    for i, model in enumerate(models, start=1):
        path = f"data/cnn_model_{i}.onnx"
        torch.onnx.export(ExportWrapper(model.cpu()).eval(), torch.zeros(1, *X_check.shape[1:]), path,
                          input_names=["image"], output_names=["label", "probabilities"],
                          dynamic_axes={"image": {0: "batch"}, "label": {0: "batch"}, "probabilities": {0: "batch"}},
                          dynamo=False)
        model.to(DEVICE)
        session = ort.InferenceSession(path)
        labels, _ = session.run(None, {"image": X_check})
        print(f"{path} créé | prédictions identiques à PyTorch : {(labels == predict(model, X_check)).mean():.0%}")
        sessions.append(session)

    onnx_votes = np.array([s.run(None, {"image": X_check})[0] for s in sessions])
    onnx_vote = (onnx_votes.sum(axis=0) > len(sessions) / 2).astype(int)
    print(f"Vote ONNX identique au vote PyTorch : {(onnx_vote == vote(models, X_check)).mean():.0%}")


# ─────────────────────────────── K-Fold + MLflow ───────────────────────────────

if __name__ == "__main__":
    print("Appareil :", DEVICE)
    folds = [(load_folder(f"train_kfold{k}"), load_folder(f"test_kfold{k}")) for k in range(1, N_FOLDS + 1)]
    X_test, y_test = load_folder("test")
    print(f"{N_FOLDS} folds chargés, test : {len(y_test)} images")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    # 1. K-Fold sur chaque candidat -> 1 run parent + 3 runs enfants (un par fold)
    best = None
    for run_name, params in CANDIDATES:
        print(f"\n===== {run_name} =====")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({"model": "CNN", "n_folds": N_FOLDS, "input": "image mel 40x157 -> 240x320", **params,
                               "batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE})
            scores = {"f2": [], "recall_abnormal": [], "accuracy": [], "f1": [], "train_accuracy": []}
            models, histories = [], []
            for k, ((X_tr, y_tr), (X_val, y_val)) in enumerate(folds, start=1):
                print(f"-- fold {k}")
                with mlflow.start_run(run_name=f"{run_name}_fold{k}", nested=True):
                    model, history = train_one_fold(params, X_tr, y_tr, X_val, y_val)
                models.append(model)
                histories.append(history)
                for name, value in compute_metrics(y_val, predict(model, X_val)).items():
                    scores[name].append(value)
                scores["train_accuracy"].append(compute_metrics(y_tr, predict(model, X_tr))["accuracy"])

            for name, values in scores.items():
                mlflow.log_metric(f"cv_{name}_mean", np.mean(values))
                mlflow.log_metric(f"cv_{name}_std", np.std(values))
                for k, value in enumerate(values, start=1):
                    mlflow.log_metric(f"cv_{name}", value, step=k)
            fig = plot_acc_loss(histories, f"{run_name} : courbes d'apprentissage par fold")
            mlflow.log_figure(fig, "courbes/learning_curves_folds.png")
            plt.close(fig)

            f2 = np.array(scores["f2"])
            print(f"{run_name}  F2 = {f2.mean():.3f} ± {f2.std():.3f}   recall = {np.mean(scores['recall_abnormal']):.3f}"
                  f"   acc = {np.mean(scores['accuracy']):.3f}   train acc = {np.mean(scores['train_accuracy']):.3f}")
            if best is None or f2.mean() > best["f2"]:
                best = {"f2": f2.mean(), "name": run_name, "run_id": run.info.run_id, "models": models}

    # 2. Les 3 réseaux du meilleur candidat VOTENT, notés UNE fois sur test/
    models = best["models"]
    test = compute_metrics(y_test, vote(models, X_test))
    alone = [compute_metrics(y_test, predict(m, X_test)) for m in models]
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_vote_{k}": v for k, v in test.items()})
        for i, metrics in enumerate(alone, start=1):
            mlflow.log_metrics({f"test_model{i}_{k}": v for k, v in metrics.items()})
        mlflow.set_tag("best", "true")
        mlflow.set_tag("inference", f"vote majoritaire de {len(models)} modèles")
        fig = plot_test_summary(alone, test, f"Meilleur CNN : {best['name']} — TEST")
        mlflow.log_figure(fig, "courbes/test_vote.png")
        plt.close(fig)
        mlflow.log_text(f"Confusion Matrix (vote, TEST)\n{confusion_matrix(y_test, vote(models, X_test))}",
                        "rapport_test_vote.txt")

    print(f"\nMeilleur CNN : {best['name']} (F2 K-Fold = {best['f2']:.3f})")
    for i, metrics in enumerate(alone, start=1):
        print(f"  réseau {i} seul sur le TEST : " + "   ".join(f"{k} = {v:.3f}" for k, v in metrics.items()))
    print(f"Note finale du VOTE ({len(models)} réseaux) sur le TEST : "
          + "   ".join(f"{k} = {v:.3f}" for k, v in test.items()))
    print("Confusion Matrix (vote, TEST) :\n", confusion_matrix(y_test, vote(models, X_test)))

    # 3. Sauvegarde : poids PyTorch + 3 fichiers ONNX
    torch.save([m.cpu().state_dict() for m in models], "data/cnn_folds.pt")
    for m in models:
        m.to(DEVICE)
    export_onnx(models, X_test)
