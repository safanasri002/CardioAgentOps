import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Header, Request, UploadFile

from api.clients import call_audio_engine, call_orchestrator
from api.config import Settings, get_settings
from api.errors import InvalidUpload, UpstreamError, invalid_upload_handler, upstream_error_handler
from api.schemas import AnalysisReport, OrchestratorRequest
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze_pcg", response_model=AnalysisReport)
async def analyze_pcg(
    request: Request,
    file: UploadFile = File(...),
    x_request_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
):
    request_id = x_request_id or uuid.uuid4().hex
    client: httpx.AsyncClient = request.app.state.client

    wav = await read_and_validate_wav(file, settings.max_upload_mb)

    t0 = time.perf_counter()
    audio = await call_audio_engine(client, settings, wav, file.filename, request_id)
    t1 = time.perf_counter()
    report = await call_orchestrator(
        client,
        settings,
        OrchestratorRequest(**audio.model_dump(), filename=file.filename),
        request_id,
    )
    t2 = time.perf_counter()

    logger.info(
        "request_id=%s file=%s audio_engine=%.3fs orchestrator=%.3fs",
        request_id, file.filename, t1 - t0, t2 - t1,
    )
    return report
