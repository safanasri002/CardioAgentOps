"""Tests de l'audio-engine : modèle ML (ONNX) + serveur FastAPI.

    pytest tests/test_audio.py -v
"""
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio-engine"
os.chdir(AUDIO_DIR)                 # le serveur charge "data/cardio_model.onnx" en chemin relatif
sys.path.insert(0, str(AUDIO_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from prepare_data import preprocess  # noqa: E402
from server import app  # noqa: E402

client = TestClient(app)


# ---------- Données / modèle ----------

@pytest.fixture(scope="module")
def test_split():
    """Même découpage que train_svm.py -> on évalue sur des sons jamais vus à l'entraînement."""
    data = np.load("data/features.npz")
    _, X_test, _, y_test = train_test_split(data["X"], data["y"], test_size=0.2,
                                            stratify=data["y"], random_state=42)
    return X_test.astype(np.float32), y_test


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession("data/cardio_model.onnx")


def make_wav(seconds=5.0, sr=4000):
    """Faux battement de cœur : deux bips basse fréquence par seconde (S1/S2)."""
    t = np.arange(int(seconds * sr)) / sr
    signal = np.sin(2 * np.pi * 50 * t) * ((t % 1.0) < 0.1) + 0.5 * np.sin(2 * np.pi * 70 * t) * (((t % 1.0) > 0.4) & ((t % 1.0) < 0.5))
    pcm = (signal / np.abs(signal).max() * 32767 * 0.8).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_onnx_input_shape(session):
    assert session.get_inputs()[0].shape[1] == 40


def test_recall_abnormal(session, test_split):
    """Critère principal : détecter au moins 85 % des sons pathologiques."""
    X, y = test_split
    labels, _ = session.run(None, {session.get_inputs()[0].name: X})
    assert recall_score(y, labels) >= 0.85


def test_probas_valides(session, test_split):
    X, _ = test_split
    _, probas = session.run(None, {session.get_inputs()[0].name: X})
    assert probas.shape == (len(X), 2)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(probas.sum(axis=1), 1, atol=1e-4)


def test_onnx_identique_sklearn(session, test_split):
    joblib = pytest.importorskip("joblib")
    X, _ = test_split
    model = joblib.load("data/svm.joblib")
    labels, _ = session.run(None, {session.get_inputs()[0].name: X})
    assert (labels == model.predict(X)).mean() >= 0.99


def test_preprocess_donne_40_features():
    features = preprocess(io.BytesIO(make_wav()))
    assert features.shape == (40,)
    assert np.all(np.isfinite(features))


def test_preprocess_son_court():
    """Un son de 1 s est complété à 5 s -> toujours 40 features."""
    assert preprocess(io.BytesIO(make_wav(seconds=1))).shape == (40,)


# ---------- Serveur ----------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_wav():
    r = client.post("/predict", files={"file": ("coeur.wav", make_wav(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("normal", "abnormal")
    assert 0 <= body["prob_abnormal"] <= 1


def test_predict_fichier_invalide():
    r = client.post("/predict", files={"file": ("pas_un_son.wav", b"ceci n'est pas un wav", "audio/wav")})
    assert r.status_code == 400
