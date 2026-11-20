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


# Image mel-spectrogramme pour les CNN — réglages du TP7 (Wav_To_melSpect_imgs) adaptés à 2000 Hz :
#   TP7 : sr=44100, n_fft=1024, hop=256, n_mels=40, power=1.0, fmin=20, fmax=8000, amplitude_to_db
#   ici : sr=2000 (imposé par l'énoncé) -> fmax=1000 (moitié de 2000 Hz), n_fft=256, hop=64
N_MELS = 40       # 40 bandes Mel = 40 lignes de l'image (comme le TP7)
N_FFT = 256       # fenêtre de 128 ms à 2000 Hz
HOP = 64          # pas de 32 ms -> 5 s = 157 colonnes
F_MIN, F_MAX = 20, 1000


def mel_image(path):
    """Un fichier .wav -> image couleur (40 x 157 x 3, uint8) du mel-spectrogramme, comme les PNG du TP7.

    Différence avec le TP7 : on dessine directement les pixels (pas de plt.savefig avec marges ni axes)
    -> le serveur peut refaire EXACTEMENT la même image à partir d'un .wav.
    """
    import matplotlib                              # import ici : seuls les CNN en ont besoin

    y, sr = librosa.load(path, sr=2000)            # 1-3 : MÊME nettoyage que preprocess()
    y = librosa.util.fix_length(y, size=2000 * 5)
    y = librosa.util.normalize(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                         power=1.0, fmin=F_MIN, fmax=F_MAX)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)          # dB : 0 = le plus fort, -80 = le plus faible
    norm = np.clip((mel_db + 80) / 80, 0, 1)                  # entre 0 et 1
    rgb = matplotlib.colormaps["magma"](norm[::-1])[:, :, :3]  # couleurs de specshow, graves en bas
    return (rgb * 255).astype(np.uint8)


def list_files():
    """Renvoie (chemin, label) avec label 0 = normal, 1 = pathologique."""
    # Cas 1 : dataset déjà trié par classe (ourdataset/train/normal et abnormal, créé au TP1)
    if (RAW / "normal").is_dir() and (RAW / "abnormal").is_dir():
        return [(wav, label) for label, cls in ((0, "normal"), (1, "abnormal"))
                for wav in sorted((RAW / cls).glob("*.wav"))]

    # Cas 2 : datasets d'origine PASCAL + PhysioNet
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
