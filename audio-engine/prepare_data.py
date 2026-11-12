"""ÉTAPE 1 — Data mining : à lancer UNE SEULE FOIS.

Lit tous les .wav, les nettoie, extrait les MFCC et sauvegarde le résultat
dans data/features.npz. Ensuite, les modèles chargent directement ce fichier.

    python prepare_data.py
"""
import csv
from pathlib import Path

import librosa
import numpy as np

RAW = Path("data/raw")              # où tu mets PASCAL et PhysioNet
OUTPUT = Path("data/features.npz")  # le résultat (petit fichier)


def preprocess(path):
    """Un fichier .wav -> un vecteur de 40 nombres (MFCC)."""
    y, sr = librosa.load(path, sr=2000)            # 1. rééchantillonnage à 2000 Hz
    y = librosa.util.fix_length(y, size=2000 * 5)  # 2. durée fixe : 5 secondes
    y = librosa.util.normalize(y)                  # 3. volume normalisé
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=256)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 4. moyenne + écart-type


def list_files():
    """Renvoie (chemin, label) avec label 0 = normal, 1 = pathologique."""
    files = []
    # PASCAL : le label est le nom du dossier (normal / extra / murmur)
    for folder, label in [("normal", 0), ("extra", 1), ("murmur", 1)]:
        for wav in (RAW / "pascal" / folder).glob("*.wav"):
            files.append((wav, label))

    # PhysioNet : le label est dans REFERENCE.csv  (-1 = normal, 1 = anormal)
    for ref in (RAW / "physionet").rglob("REFERENCE.csv"):
        for name, label in csv.reader(open(ref)):
            files.append((ref.parent / f"{name}.wav", 1 if label == "1" else 0))
    return files


if __name__ == "__main__":
    if OUTPUT.exists():
        print("Déjà fait :", OUTPUT, "(supprime-le pour recommencer)")
        exit()

    X, y = [], []
    for path, label in list_files():
        try:
            X.append(preprocess(path))
            y.append(label)
        except Exception:
            print("Fichier ignoré (corrompu) :", path)

    np.savez(OUTPUT, X=np.array(X), y=np.array(y))
    print(f"{len(y)} sons sauvegardés -> normal: {y.count(0)}, pathologique: {y.count(1)}")
