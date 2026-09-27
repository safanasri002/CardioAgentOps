import httpx

from tests.conftest import AUDIO_OK, AUDIO_URL, ORCH_URL, REPORT_OK, make_wav


def wav_files(data: bytes, name: str = "heart.wav", ctype: str = "audio/wav"):
    return {"file": (name, data, ctype)}


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_analyze_nominal(client, mock_upstreams, wav_bytes):
    audio = mock_upstreams.post(AUDIO_URL).mock(return_value=httpx.Response(200, json=AUDIO_OK))
    orch = mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, json=REPORT_OK))

    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes), headers={"X-Request-ID": "abc"})

    assert resp.status_code == 200
    assert resp.json() == REPORT_OK
    assert audio.calls.last.request.headers["X-Request-ID"] == "abc"
    sent = orch.calls.last.request.read()
    assert b"heart.wav" in sent and b"abnormal" in sent


async def test_rejects_non_wav_extension(client, wav_bytes):
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes, "heart.mp3"))
    assert resp.status_code == 422


async def test_rejects_bad_content_type(client, wav_bytes):
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes, ctype="text/plain"))
    assert resp.status_code == 422


async def test_rejects_fake_wav(client):
    resp = await client.post("/analyze_pcg", files=wav_files(b"not a wav at all"))
    assert resp.status_code == 422


async def test_rejects_empty_file(client):
    resp = await client.post("/analyze_pcg", files=wav_files(b""))
    assert resp.status_code == 422


async def test_rejects_too_large(client):
    resp = await client.post("/analyze_pcg", files=wav_files(make_wav(1024 * 1024 + 10)))
    assert resp.status_code == 413


async def test_missing_file_field(client):
    resp = await client.post("/analyze_pcg")
    assert resp.status_code == 422


async def test_audio_engine_unreachable(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(side_effect=httpx.ConnectError("down"))
    orch = mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, json=REPORT_OK))

    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))

    assert resp.status_code == 502
    assert resp.json()["service"] == "audio-engine"
    assert not orch.called


async def test_audio_engine_500(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(return_value=httpx.Response(500, text="boom"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 502
    assert resp.json()["service"] == "audio-engine"


async def test_audio_engine_invalid_json(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(return_value=httpx.Response(200, text="not json"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 502


async def test_audio_engine_schema_mismatch(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(return_value=httpx.Response(200, json={"foo": "bar"}))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 502


async def test_orchestrator_unreachable(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(return_value=httpx.Response(200, json=AUDIO_OK))
    mock_upstreams.post(ORCH_URL).mock(side_effect=httpx.ConnectError("down"))

    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))

    assert resp.status_code == 502
    assert resp.json()["service"] == "agent-orchestrator"


async def test_timeout_returns_504(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(AUDIO_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 504
    assert resp.json()["error"] == "gateway_timeout"
