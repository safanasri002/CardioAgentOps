from pathlib import Path
import numpy as np
import librosa
import onnxruntime as ort

TARGET_SR = 2000
TARGET_DURATION = 5.0
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128

MODEL_PATH = Path("audio_engine/models/cardio_model.onnx")
_session = None


def _session_onnx():
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def _mel(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    n = int(TARGET_SR * TARGET_DURATION)
    y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel[None, None, :, :]  # (1, 1, 64, T)


def predict(audio_path: str) -> dict:
    session = _session_onnx()
    x = _mel(audio_path)
    logits = session.run(None, {"mel": x})[0][0]
    exp = np.exp(logits - logits.max())
    proba = exp / exp.sum()
    pred = int(np.argmax(proba))
    return {
        "prediction": "abnormal" if pred == 1 else "normal",
        "confidence": float(proba[pred]),
        "probabilities": {"normal": float(proba[0]), "abnormal": float(proba[1])},
    }