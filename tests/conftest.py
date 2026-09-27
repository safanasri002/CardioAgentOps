import httpx
import pytest
import respx

from api.config import Settings, get_settings
from api.main import app

AUDIO_URL = "http://audio-engine.test/predict"
ORCH_URL = "http://orchestrator.test/analyze"

AUDIO_OK = {
    "probabilities": {"normal": 0.12, "abnormal": 0.88},
    "predicted_class": "abnormal",
    "features": {"mfcc": [1.0, 2.0], "zcr": 0.05},
}
REPORT_OK = {"diagnosis": "Suspicion de souffle systolique", "confidence": 0.88, "sources": ["ESC-2021"]}


def make_wav(n_bytes: int = 64) -> bytes:
    return b"RIFF" + (n_bytes).to_bytes(4, "little") + b"WAVE" + b"\x00" * n_bytes


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav()


@pytest.fixture(autouse=True)
def settings_override():
    app.dependency_overrides[get_settings] = lambda: Settings(
        audio_engine_url="http://audio-engine.test",
        agent_orchestrator_url="http://orchestrator.test",
        max_upload_mb=1,
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_upstreams():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client():
    app.state.client = httpx.AsyncClient(timeout=5)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
        yield c
    await app.state.client.aclose()
