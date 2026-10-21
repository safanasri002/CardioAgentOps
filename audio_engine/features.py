import numpy as np
import librosa

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40
MAX_FEATURES = 1200  # pour rester compatible avec tes features actuelles

def extract_mel_spectrogram(y: np.ndarray, sr: int = 2000) -> np.ndarray:
    """Retourne le Mel Spectrogramme en dB."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def extract_mfcc(y: np.ndarray, sr: int = 2000, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Retourne un vecteur MFCC (moyenne + std)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

def flatten_mel(mel: np.ndarray, max_len: int = MAX_FEATURES) -> np.ndarray:
    """Aplatit le Mel Spectrogramme et le tronque/pad."""
    features = mel.flatten()[:max_len]
    if len(features) < max_len:
        features = np.pad(features, (0, max_len - len(features)))
    return features