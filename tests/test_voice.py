"""Unit tests for md2short.voice (synthesize, synthesize_to_file).

Uses httpx.MockTransport exclusively -- no real network calls. The module
under test creates its own `httpx.Client()` internally, so tests monkeypatch
`md2short.voice.httpx.Client` to a factory that returns a client wired to a
`MockTransport` handler.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from md2short import voice as voice_mod

VOICE_ID = "v1jVu1Ky28piIPEJqRrm"
API_KEY = "fake-api-key"
FAKE_MP3_BYTES = b"FAKE_MP3_BYTES"


_RealClient = httpx.Client


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, handler):
    def _factory(*args, **kwargs):
        return _RealClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(voice_mod.httpx, "Client", _factory)


def test_synthesize_sets_an_explicit_generous_client_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    """httpx defaults to a 5s timeout, which is far too short to synthesize a
    full narration: every real run failed with "The read operation timed out"
    after burning all 4 retry attempts (each still billable server-side).
    """
    captured_kwargs = {}

    def _factory(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _RealClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=FAKE_MP3_BYTES)
            )
        )

    monkeypatch.setattr(voice_mod.httpx, "Client", _factory)

    voice_mod.synthesize("안녕하세요", VOICE_ID, API_KEY)

    assert captured_kwargs["timeout"] == voice_mod.REQUEST_TIMEOUT_SECONDS
    assert voice_mod.REQUEST_TIMEOUT_SECONDS >= 120.0


def test_synthesize_calls_correct_url_and_header(monkeypatch: pytest.MonkeyPatch):
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)

    result = voice_mod.synthesize("안녕하세요", VOICE_ID, API_KEY)

    assert result == FAKE_MP3_BYTES
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url.path == f"/v1/text-to-speech/{VOICE_ID}"
    assert request.headers["xi-api-key"] == API_KEY
    assert request.url.params["output_format"] == "mp3_44100_128"


def test_synthesize_returns_mocked_mp3_bytes_on_success(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)

    result = voice_mod.synthesize("텍스트", VOICE_ID, API_KEY)
    assert result == FAKE_MP3_BYTES


def test_synthesize_retries_through_post_with_retry_on_transient_500(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)
    monkeypatch.setattr("md2short._retry.time.sleep", lambda *_: None)

    result = voice_mod.synthesize("텍스트", VOICE_ID, API_KEY)

    assert result == FAKE_MP3_BYTES
    assert calls["n"] == 2


def test_synthesize_to_file_writes_bytes_to_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)

    output_path = tmp_path / "audio" / "narration.mp3"
    result = voice_mod.synthesize_to_file("텍스트", VOICE_ID, API_KEY, output_path)

    assert result == output_path
    assert output_path.read_bytes() == FAKE_MP3_BYTES


def test_synthesize_to_file_writes_tts_input_log_when_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)

    output_path = tmp_path / "audio" / "narration.mp3"
    log_path = tmp_path / "logs" / "tts_input.txt"
    text = "이것은 TTS 입력 텍스트입니다"

    voice_mod.synthesize_to_file(
        text, VOICE_ID, API_KEY, output_path, tts_input_log_path=log_path
    )

    assert log_path.read_text(encoding="utf-8") == text


def test_synthesize_to_file_skips_log_when_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FAKE_MP3_BYTES)

    _install_mock_client(monkeypatch, handler)

    output_path = tmp_path / "audio" / "narration.mp3"
    voice_mod.synthesize_to_file("텍스트", VOICE_ID, API_KEY, output_path)

    assert not (tmp_path / "logs").exists()
