import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.clients import call_orchestrator, open_orchestrator_stream, relay_sse
from api.config import Settings, get_settings
from api.errors import (
    GuardrailViolation,
    InvalidUpload,
    UpstreamError,
    guardrail_violation_handler,
    invalid_upload_handler,
    upstream_error_handler,
)
from api.guardrails import apply_guardrails
from api.schemas import AnalysisReport
from api.validation import read_and_validate_wav

logger = logging.getLogger("api-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.client = httpx.AsyncClient(timeout=settings.request_timeout_s)
    yield
    await app.state.client.aclose()


app = FastAPI(title="CardioAgent-Ops API Gateway", version="1.0.0", lifespan=lifespan)
app.add_exception_handler(UpstreamError, upstream_error_handler)
app.add_exception_handler(InvalidUpload, invalid_upload_handler)
app.add_exception_handler(GuardrailViolation, guardrail_violation_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _prepare(
    file: UploadFile, prompt: str | None, x_request_id: str | None, settings: Settings
) -> tuple[str, str | None, bytes]:
    """Étapes communes aux deux routes : request-id, guardrails du prompt, validation du .wav."""
    request_id = x_request_id or uuid.uuid4().hex
    clean_prompt, pii_masked = apply_guardrails(prompt, settings.max_prompt_chars)
    if pii_masked:
        logger.info("request_id=%s guardrails: PII masquées %s", request_id, pii_masked)
    wav = await read_and_validate_wav(file, settings.max_upload_mb)
    return request_id, clean_prompt, wav


@app.post("/analyze_pcg", response_model=AnalysisReport)
async def analyze_pcg(
    request: Request,
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    x_request_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
):
    request_id, clean_prompt, wav = await _prepare(file, prompt, x_request_id, settings)

    t0 = time.perf_counter()
    report = await call_orchestrator(
        request.app.state.client, settings, wav, file.filename, clean_prompt, request_id
    )
    logger.info(
        "request_id=%s file=%s orchestrator=%.3fs", request_id, file.filename, time.perf_counter() - t0
    )
    return report


@app.post("/analyze_pcg/stream", response_class=StreamingResponse)
async def analyze_pcg_stream(
    request: Request,
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    x_request_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
):
    """Version streaming : relaie au client les événements SSE émis par le Service 4.

    Les erreurs de validation (422/413) et de connexion au Service 4 (502/504) sont renvoyées
    en JSON comme pour /analyze_pcg ; une fois le flux ouvert, une coupure est signalée par
    un événement SSE `error`.
    """
    request_id, clean_prompt, wav = await _prepare(file, prompt, x_request_id, settings)

    upstream = await open_orchestrator_stream(
        request.app.state.client, settings, wav, file.filename, clean_prompt, request_id
    )
    logger.info("request_id=%s file=%s orchestrator stream ouvert", request_id, file.filename)
    return StreamingResponse(
        relay_sse(upstream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id},
    )
