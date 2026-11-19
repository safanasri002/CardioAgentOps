"""Tests de l'api-gateway : guardrails + routes FastAPI.

Les autres services (audio-engine, agents) sont simulés : pas besoin de les lancer.

    cd api-gateway && pytest tests -v
"""
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

API_DIR = Path(__file__).resolve().parent.parent          # api-gateway/
sys.path.insert(0, str(API_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from guardrails import DISCLAIMER, check_input, check_output  # noqa: E402

client = TestClient(main.app)

WAV = b"RIFF" + b"\x00" * 100          # en-tête minimal d'un .wav
BON_RAPPORT = {"label": "abnormal", "confidence": 0.8, "text": "Souffle possible, à confirmer par un cardiologue."}


# ---------- Guardrail d'entrée ----------

def test_input_wav_valide():
    assert check_input("coeur.wav", WAV) is None


def test_input_mauvaise_extension():
    assert check_input("coeur.mp3", WAV) is not None


def test_input_faux_wav():
    assert check_input("coeur.wav", b"ceci n'est pas un wav") is not None


def test_input_trop_gros():
    assert check_input("coeur.wav", b"RIFF" + b"\x00" * (10 * 1024 * 1024)) is not None


# ---------- Guardrail de sortie ----------

def test_output_valide_avec_disclaimer():
    r = check_output(BON_RAPPORT)
    assert r["label"] == "abnormal"
    assert r["text"] == BON_RAPPORT["text"]
    assert r["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize("mot", ["certain", "100%", "aucun risque", "prescris", "posologie"])
def test_output_mot_interdit_bloque(mot):
    r = check_output({**BON_RAPPORT, "text": f"Le patient est {mot.upper()} malade."})
    assert mot not in r["text"].lower()
    assert "bloquée" in r["text"]
    assert r["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize("champ, valeur", [
    ("label", "malade"),            # seulement normal / abnormal
    ("confidence", 1.5),            # doit être entre 0 et 1
    ("confidence", -0.1),
    ("text", "a" * 2001),           # 2000 caractères max
])
def test_output_invalide(champ, valeur):
    with pytest.raises(ValidationError):
        check_output({**BON_RAPPORT, champ: valeur})


def test_output_champ_manquant():
    with pytest.raises(ValidationError):
        check_output({"label": "normal", "confidence": 0.5})


# ---------- Routes ----------

class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def fake_services(prediction, report):
    """Remplace httpx.post : renvoie la prédiction (audio-engine) puis le rapport (agents)."""
    def fake_post(url, **kwargs):
        return FakeResponse(prediction if url.endswith("/predict") else report)
    return fake_post


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_home(monkeypatch):
    monkeypatch.chdir(API_DIR)          # main.py lit "static/index.html" en chemin relatif
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_analyze_ok(monkeypatch):
    prediction = {"label": "abnormal", "prob_abnormal": 0.8}
    monkeypatch.setattr(main.httpx, "post", fake_services(prediction, BON_RAPPORT))
    r = client.post("/analyze_pcg", files={"file": ("coeur.wav", WAV, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == prediction
    assert body["report"]["label"] == "abnormal"
    assert body["report"]["disclaimer"] == DISCLAIMER


def test_analyze_mauvais_fichier():
    r = client.post("/analyze_pcg", files={"file": ("coeur.txt", b"bonjour", "text/plain")})
    assert r.status_code == 400


def test_analyze_service_indisponible(monkeypatch):
    def down(url, **kwargs):
        raise httpx.ConnectError("service arrêté")
    monkeypatch.setattr(main.httpx, "post", down)
    r = client.post("/analyze_pcg", files={"file": ("coeur.wav", WAV, "audio/wav")})
    assert r.status_code == 503


def test_analyze_rapport_llm_invalide(monkeypatch):
    mauvais = {"label": "peut-être", "confidence": 2, "text": "?"}
    monkeypatch.setattr(main.httpx, "post", fake_services({"label": "normal"}, mauvais))
    r = client.post("/analyze_pcg", files={"file": ("coeur.wav", WAV, "audio/wav")})
    assert r.status_code == 502


def test_analyze_mot_interdit_filtre(monkeypatch):
    dangereux = {**BON_RAPPORT, "text": "Je prescris de l'aspirine, posologie 500 mg."}
    monkeypatch.setattr(main.httpx, "post", fake_services({"label": "abnormal"}, dangereux))
    r = client.post("/analyze_pcg", files={"file": ("coeur.wav", WAV, "audio/wav")})
    assert r.status_code == 200
    assert "prescris" not in r.json()["report"]["text"]
