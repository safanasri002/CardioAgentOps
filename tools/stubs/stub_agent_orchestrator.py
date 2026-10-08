"""Stub minimal du Service 4 (agent-orchestrator), pour tester l'api-gateway
avant que le vrai réseau d'agents LangGraph ne soit développé.

Reçoit le .wav + le prompt depuis la gateway, puis simule l'AudioAgent en
appelant lui-même le Service 3 (audio-engine), comme dans l'architecture cible.

- POST /analyze         -> rapport JSON complet
- POST /analyze/stream  -> flux SSE (étapes des agents, tokens, rapport final)

Lancer :
    uvicorn tools.stubs.stub_agent_orchestrator:app --port 8004
(Variable AUDIO_ENGINE_URL, par défaut http://127.0.0.1:8003)
"""

import asyncio
import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://127.0.0.1:8003")
TOKEN_DELAY_S = float(os.getenv("STUB_TOKEN_DELAY_S", "0.06"))
SOURCES = ["ESC Guidelines 2021 (stub, non réel)"]

app = FastAPI(title="Stub agent-orchestrator")


async def audio_agent(file: UploadFile, request_id: str | None) -> dict[str, Any]:
    """Simule l'AudioAgent : appel du Service 3."""
    wav = await file.read()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                AUDIO_ENGINE_URL.rstrip("/") + "/predict",
                files={"file": (file.filename, wav, "audio/wav")},
                headers={"X-Request-ID": request_id or ""},
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"AudioAgent : audio-engine indisponible ({exc.__class__.__name__})")


def build_report(audio: dict[str, Any], filename: str | None, prompt: str | None) -> dict[str, Any]:
    predicted = audio.get("predicted_class", "inconnu")
    return {
        "patient_file": filename,
        "question_recue": prompt,
        "predicted_class": predicted,
        "confidence": audio.get("probabilities", {}).get(predicted, 0.5),
        "probabilities": audio.get("probabilities"),
        "diagnosis": (
            "Suspicion de souffle cardiaque (réponse simulée)"
            if predicted == "abnormal"
            else "Signal jugé normal (réponse simulée)"
        ),
        "recommendation": (
            "Ceci est une réponse simulée du Service 4 (agent-orchestrator), "
            "à remplacer une fois le graphe LangGraph (Analyste / Cardiologue RAG / Auditeur) implémenté."
        ),
        "sources": SOURCES,
    }


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    x_request_id: str | None = Header(default=None),
):
    audio = await audio_agent(file, x_request_id)
    return build_report(audio, filename or file.filename, prompt)


@app.post("/analyze/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    x_request_id: str | None = Header(default=None),
):
    # L'AudioAgent est appelé avant d'ouvrir le flux : si S3 est en panne, on renvoie
    # un vrai 502 que la gateway transformera en erreur JSON.
    audio = await audio_agent(file, x_request_id)
    report = build_report(audio, filename or file.filename, prompt)

    async def events():
        yield sse("step", {"agent": "AudioAgent", "status": "done",
                           "predicted_class": audio.get("predicted_class"),
                           "probabilities": audio.get("probabilities")})
        yield sse("step", {"agent": "KnowledgeRAGAgent", "status": "running"})
        await asyncio.sleep(TOKEN_DELAY_S * 8)
        yield sse("step", {"agent": "KnowledgeRAGAgent", "status": "done", "sources": SOURCES})
        yield sse("step", {"agent": "ReasoningLLMAgent", "status": "running"})
        text = f"{report['diagnosis']}. {report['recommendation']}"
        for word in text.split(" "):
            await asyncio.sleep(TOKEN_DELAY_S)
            yield sse("token", {"text": word + " "})
        yield sse("step", {"agent": "ReasoningLLMAgent", "status": "done"})
        yield sse("report", report)
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "stub-agent-orchestrator"}
