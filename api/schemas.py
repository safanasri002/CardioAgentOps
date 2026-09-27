from typing import Any

from pydantic import BaseModel, ConfigDict


class AudioEngineResponse(BaseModel):
    """Réponse du Service 3 (audio-engine)."""

    model_config = ConfigDict(extra="allow")

    probabilities: dict[str, float]
    predicted_class: str
    features: dict[str, Any] = {}


class OrchestratorRequest(AudioEngineResponse):
    """Payload envoyé au Service 4 (agent-orchestrator)."""

    filename: str


class AnalysisReport(BaseModel):
    """Rapport final du Service 4, relayé tel quel (schéma souple)."""

    model_config = ConfigDict(extra="allow")
