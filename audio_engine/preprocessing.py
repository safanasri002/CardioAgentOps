import numpy as np
import librosa
from pathlib import Path

TARGET_SR = 2000
TARGET_DURATION = 5.0  # secondes

def load_and_preprocess(file_path: str | Path, target_sr: int = TARGET_SR, target_duration: float = TARGET_DURATION):
    """
    Charge un fichier audio, le rééchantillonne à 2000 Hz
    et normalise sa durée (padding ou truncation).
    """
    y, sr = librosa.load(file_path, sr=None)

    # Rééchantillonnage
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Normalisation de durée
    target_length = int(target_sr * target_duration)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)))

    # Normalisation amplitude
    y = librosa.util.normalize(y)

    return y, target_sr