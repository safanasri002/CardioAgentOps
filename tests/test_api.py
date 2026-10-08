import httpx

from tests.conftest import ORCH_URL, REPORT_OK, make_wav


def wav_files(data: bytes, name: str = "heart.wav", ctype: str = "audio/wav"):
    return {"file": (name, data, ctype)}


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_analyze_nominal(client, mock_upstreams, wav_bytes):
    orch = mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, json=REPORT_OK))

    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes), headers={"X-Request-ID": "abc"})

    assert resp.status_code == 200
    assert resp.json() == REPORT_OK
    sent = orch.calls.last.request
    assert sent.headers["X-Request-ID"] == "abc"
    assert sent.headers["content-type"].startswith("multipart/form-data")
    body = sent.read()
    assert b'name="file"; filename="heart.wav"' in body
    assert wav_bytes in body
    assert b'name="prompt"' not in body


async def test_analyze_forwards_prompt(client, mock_upstreams, wav_bytes):
    orch = mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, json=REPORT_OK))

    resp = await client.post(
        "/analyze_pcg",
        files=wav_files(wav_bytes),
        data={"prompt": "Y a-t-il un souffle systolique ?"},
    )

    assert resp.status_code == 200
    body = orch.calls.last.request.read().decode(errors="replace")
    assert 'name="prompt"' in body and "souffle systolique" in body


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


async def test_invalid_file_does_not_call_orchestrator(client, mock_upstreams):
    orch = mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, json=REPORT_OK))
    await client.post("/analyze_pcg", files=wav_files(b"not a wav at all"))
    assert not orch.called


async def test_orchestrator_unreachable(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(ORCH_URL).mock(side_effect=httpx.ConnectError("down"))

    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))

    assert resp.status_code == 502
    assert resp.json()["service"] == "agent-orchestrator"


async def test_orchestrator_500(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(500, text="boom"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 502
    assert "boom" in resp.json()["detail"]


async def test_orchestrator_invalid_json(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(ORCH_URL).mock(return_value=httpx.Response(200, text="not json"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Réponse invalide du service"


async def test_timeout_returns_504(client, mock_upstreams, wav_bytes):
    mock_upstreams.post(ORCH_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    resp = await client.post("/analyze_pcg", files=wav_files(wav_bytes))
    assert resp.status_code == 504
    assert resp.json()["error"] == "gateway_timeout"
