"""Stub minimal du Service 4 (agent-orchestrator), pour tester l'api-gateway
avant que le vrai réseau d'agents LangGraph ne soit développé.

Lancer :
    uvicorn tools.stubs.stub_agent_orchestrator:app --port 8004
"""

from typing import Any

from fastapi import FastAPI

app = FastAPI(title="Stub agent-orchestrator")


@app.post("/analyze")
async def analyze(payload: dict[str, Any]):
    predicted = payload.get("predicted_class", "inconnu")
    confidence = payload.get("probabilities", {}).get(predicted, 0.5)
    return {
        "patient_file": payload.get("filename"),
        "predicted_class": predicted,
        "confidence": confidence,
        "diagnosis": (
            "Suspicion de souffle cardiaque (réponse simulée)"
            if predicted == "abnormal"
            else "Signal jugé normal (réponse simulée)"
        ),
        "recommendation": (
            "Ceci est une réponse simulée du Service 4 (agent-orchestrator), "
            "à remplacer une fois le graphe LangGraph (Analyste / Cardiologue RAG / Auditeur) implémenté."
        ),
        "sources": ["ESC Guidelines 2021 (stub, non réel)"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "stub-agent-orchestrator"}
