from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from api.config import Settings
from api.errors import UpstreamError
from api.schemas import AnalysisReport, AudioEngineResponse, OrchestratorRequest

T = TypeVar("T", bound=BaseModel)


async def _call(
    client: httpx.AsyncClient,
    service: str,
    url: str,
    model: type[T],
    request_id: str,
    **kwargs,
) -> T:
    try:
        resp = await client.post(url, headers={"X-Request-ID": request_id}, **kwargs)
    except httpx.TimeoutException:
        raise UpstreamError(service, "Timeout du service", 504)
    except httpx.HTTPError as exc:
        raise UpstreamError(service, f"Service injoignable : {exc.__class__.__name__}")

    if resp.status_code >= 400:
        raise UpstreamError(service, f"Le service a répondu {resp.status_code}: {resp.text[:200]}")
    try:
        return model.model_validate(resp.json())
    except (ValueError, ValidationError):
        raise UpstreamError(service, "Réponse invalide du service")


async def call_audio_engine(
    client: httpx.AsyncClient, settings: Settings, wav: bytes, filename: str, request_id: str
) -> AudioEngineResponse:
    return await _call(
        client,
        "audio-engine",
        settings.audio_engine_url.rstrip("/") + settings.audio_engine_path,
        AudioEngineResponse,
        request_id,
        files={"file": (filename, wav, "audio/wav")},
    )


async def call_orchestrator(
    client: httpx.AsyncClient, settings: Settings, payload: OrchestratorRequest, request_id: str
) -> AnalysisReport:
    return await _call(
        client,
        "agent-orchestrator",
        settings.agent_orchestrator_url.rstrip("/") + settings.orchestrator_path,
        AnalysisReport,
        request_id,
        json=payload.model_dump(),
    )
