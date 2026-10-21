from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
import mlflow

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("audio_engine/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 12
BATCH = 32
LR = 1e-3

X = np.load(DATA_DIR / "X_mel_2d.npy")  # (N, n_mels, time)
y = np.load(DATA_DIR / "y_labels_2d.npy")
print("X", X.shape, "y", y.shape, "device", DEVICE)

# normalisation par échantillon
X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def loader(xa, ya, shuffle):
    xt = torch.tensor(xa, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    yt = torch.tensor(ya, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=BATCH, shuffle=shuffle)

train_loader = loader(X_train, y_train, True)
test_loader = loader(X_test, y_test, False)


class CNNCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


class TransformerCNN(nn.Module):
    """CNN puis petit Transformer sur l'axe temps."""

    def __init__(self, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.cnn(x)                  # (B, C, F, T)
        h = h.mean(dim=2).transpose(1, 2)  # (B, T, C)
        h = self.tr(h).mean(dim=1)
        return self.fc(h)


def make_resnet():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def run_epoch(model, data, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total, preds, labels = 0.0, [], []
    for xb, yb in data:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(yb)
        preds.append(logits.argmax(1).detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return total / len(y_true), y_true, y_pred


def fit(name, model):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_recall, best_state = -1.0, None
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        for epoch in range(1, EPOCHS + 1):
            tr_loss, _, _ = run_epoch(model, train_loader, opt)
            _, y_true, y_pred = run_epoch(model, test_loader)
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, pos_label=1)
            print(f"{name} epoch {epoch:02d} loss={tr_loss:.4f} acc={acc:.3f} recall_abn={rec:.3f}")
            if rec > best_recall:
                best_recall = rec
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, y_true, y_pred = run_epoch(model, test_loader)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, pos_label=1)
        mlflow.log_metrics({"accuracy": acc, "recall_abnormal": rec})
        print(classification_report(y_true, y_pred, target_names=["normal", "abnormal"]))
        path = MODELS_DIR / f"{name}.pt"
        torch.save(model.state_dict(), path)
        print(name, "sauvé", path, "recall", round(rec, 4))


if __name__ == "__main__":
    mlflow.set_experiment("CardioAgent-Audio-Benchmark")
    fit("CNNCustom", CNNCustom())
    fit("ResNet18", make_resnet())
    fit("TransformerCNN", TransformerCNN())