import json
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError

from api.config import Settings
from api.errors import UpstreamError
from api.schemas import AnalysisReport

SERVICE = "agent-orchestrator"


def _build_request(
    client: httpx.AsyncClient,
    url: str,
    wav: bytes,
    filename: str,
    prompt: str | None,
    request_id: str,
) -> httpx.Request:
    data = {"filename": filename}
    if prompt is not None:
        data["prompt"] = prompt
    return client.build_request(
        "POST",
        url,
        headers={"X-Request-ID": request_id},
        files={"file": (filename, wav, "audio/wav")},
        data=data,
    )


async def _send(client: httpx.AsyncClient, request: httpx.Request, stream: bool) -> httpx.Response:
    try:
        resp = await client.send(request, stream=stream)
    except httpx.TimeoutException:
        raise UpstreamError(SERVICE, "Timeout du service", 504)
    except httpx.HTTPError as exc:
        raise UpstreamError(SERVICE, f"Service injoignable : {exc.__class__.__name__}")

    if resp.status_code >= 400:
        text = (await resp.aread()).decode(errors="replace")
        await resp.aclose()
        raise UpstreamError(SERVICE, f"Le service a répondu {resp.status_code}: {text[:200]}")
    return resp


async def call_orchestrator(
    client: httpx.AsyncClient,
    settings: Settings,
    wav: bytes,
    filename: str,
    prompt: str | None,
    request_id: str,
) -> AnalysisReport:
    """Envoie le .wav et le prompt au Service 4 et attend le rapport complet (JSON).

    C'est l'AudioAgent du Service 4 qui appelle ensuite le Service 3 (audio-engine).
    """
    url = settings.agent_orchestrator_url.rstrip("/") + settings.orchestrator_path
    resp = await _send(client, _build_request(client, url, wav, filename, prompt, request_id), stream=False)
    try:
        return AnalysisReport.model_validate(resp.json())
    except (ValueError, ValidationError):
        raise UpstreamError(SERVICE, "Réponse invalide du service")


async def open_orchestrator_stream(
    client: httpx.AsyncClient,
    settings: Settings,
    wav: bytes,
    filename: str,
    prompt: str | None,
    request_id: str,
) -> httpx.Response:
    """Ouvre le flux SSE du Service 4.

    Les erreurs de connexion et les statuts HTTP d'erreur sont levés ici, avant que la
    réponse au client ne commence : la gateway peut encore renvoyer un 502/504 JSON classique.
    """
    url = settings.agent_orchestrator_url.rstrip("/") + settings.orchestrator_stream_path
    resp = await _send(client, _build_request(client, url, wav, filename, prompt, request_id), stream=True)
    if not resp.headers.get("content-type", "").startswith("text/event-stream"):
        await resp.aclose()
        raise UpstreamError(SERVICE, "Réponse invalide du service : flux text/event-stream attendu")
    return resp


def sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def relay_sse(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Relaie tel quel le flux SSE du Service 4.

    Si le flux est coupé en cours de route, le statut HTTP 200 est déjà parti : l'erreur
    est donc signalée au client par un dernier événement `error`.
    """
    last = b"\n\n"
    try:
        async for chunk in resp.aiter_raw():
            last = chunk
            yield chunk
    except httpx.HTTPError as exc:
        timeout = isinstance(exc, httpx.TimeoutException)
        if not last.endswith(b"\n\n"):
            yield b"\n\n"  # termine proprement l'événement coupé avant d'envoyer l'erreur
        yield sse_event(
            "error",
            {
                "error": "gateway_timeout" if timeout else "bad_gateway",
                "service": SERVICE,
                "detail": "Timeout du service" if timeout else f"Flux interrompu : {exc.__class__.__name__}",
            },
        )
    finally:
        await resp.aclose()
