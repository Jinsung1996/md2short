"""Unit tests for md2short.transcript (US-006, F-04).

Uses httpx.MockTransport exclusively -- no real network calls. We monkeypatch
`md2short.transcript.httpx.Client` so `transcribe()` routes its request
through a mocked transport while still exercising the real
`_retry.post_with_retry` call path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx

from md2short.transcript import (
    normalize_scribe_response,
    transcribe,
    transcribe_and_save,
)

#: Mirrors the real ElevenLabs Scribe response shape as verified against the
#: live API in US-016: `audio_duration_secs` (not `audio_duration`) carries the
#: duration, and `words` interleaves `type: "spacing"` whitespace entries
#: between the `type: "word"` entries.
MOCK_RAW_RESPONSE = {
    "language_code": "KOR",
    "language_probability": 1.0,
    "text": "금일 미국",
    "words": [
        {"text": "금일", "start": 0.0, "end": 0.32, "type": "word", "logprob": -0.0001},
        {"text": " ", "start": 0.32, "end": 0.33, "type": "spacing", "logprob": -0.0001},
        {"text": "미국", "start": 0.33, "end": 0.59, "type": "word", "logprob": -0.0001},
    ],
    "transcription_id": "4ekPa2QARTr0hgVwIxde",
    "audio_duration_secs": 0.82,
}


_RealClient = httpx.Client


def _mock_client_factory(handler):
    def _factory(*args, **kwargs):
        return _RealClient(transport=httpx.MockTransport(handler))

    return _factory


def test_transcribe_sets_an_explicit_generous_client_timeout():
    """See the matching note in test_voice.py: httpx's 5s default timeout is far
    too short to transcribe a full narration mp3, and a timed-out attempt is
    still billable.
    """
    import md2short.transcript as transcript_mod

    captured_kwargs = {}

    def _factory(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _RealClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=MOCK_RAW_RESPONSE)
            )
        )

    with patch("md2short.transcript.httpx.Client", _factory):
        transcribe(b"fake-audio-bytes", api_key="test-key", language_code="kor")

    assert captured_kwargs["timeout"] == transcript_mod.REQUEST_TIMEOUT_SECONDS
    assert transcript_mod.REQUEST_TIMEOUT_SECONDS >= 120.0


def test_transcribe_sends_correct_request_and_returns_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json=MOCK_RAW_RESPONSE)

    with patch("md2short.transcript.httpx.Client", _mock_client_factory(handler)):
        result = transcribe(b"fake-audio-bytes", api_key="test-key", language_code="kor")

    assert captured["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert captured["headers"]["xi-api-key"] == "test-key"
    assert "multipart/form-data" in captured["content_type"]
    body = captured["body"]
    assert b'name="model_id"' in body
    assert b"scribe_v1" in body
    assert b'name="language_code"' in body
    assert b"kor" in body
    assert b"fake-audio-bytes" in body
    assert result == MOCK_RAW_RESPONSE


def test_normalize_scribe_response_converts_seconds_to_ms():
    normalized = normalize_scribe_response(MOCK_RAW_RESPONSE, language_code="kor")

    assert normalized["language_code"] == "kor"
    assert normalized["words"] == [
        {"text": "금일", "start": 0, "end": 320},
        {"text": "미국", "start": 330, "end": 590},
    ]


def test_normalize_scribe_response_drops_spacing_entries():
    """`type: "spacing"` entries are whitespace, not narration tokens."""
    normalized = normalize_scribe_response(MOCK_RAW_RESPONSE, language_code="kor")

    assert [word["text"] for word in normalized["words"]] == ["금일", "미국"]


def test_normalize_scribe_response_drops_audio_event_entries():
    raw = {
        "words": [
            {"text": "금일", "start": 0.0, "end": 0.32, "type": "word"},
            {"text": "(laughter)", "start": 0.4, "end": 0.9, "type": "audio_event"},
        ]
    }

    normalized = normalize_scribe_response(raw, language_code="kor")

    assert [word["text"] for word in normalized["words"]] == ["금일"]


def test_normalize_scribe_response_duration_from_audio_duration_secs():
    """The live API returns the duration under `audio_duration_secs`."""
    normalized = normalize_scribe_response(MOCK_RAW_RESPONSE, language_code="kor")

    assert normalized["audio_duration_ms"] == 820


def test_normalize_scribe_response_duration_from_max_word_end():
    """With no duration key at all, fall back to the last word's end."""
    raw = {key: value for key, value in MOCK_RAW_RESPONSE.items() if key != "audio_duration_secs"}

    normalized = normalize_scribe_response(raw, language_code="kor")

    assert normalized["audio_duration_ms"] == 590


def test_normalize_scribe_response_duration_from_legacy_field():
    raw = {key: value for key, value in MOCK_RAW_RESPONSE.items() if key != "audio_duration_secs"}
    raw["audio_duration"] = 1.234

    normalized = normalize_scribe_response(raw, language_code="kor")

    assert normalized["audio_duration_ms"] == 1234


def test_normalize_scribe_response_text_fallback_join():
    raw = {
        "words": [
            {"text": "금일", "start": 0.0, "end": 0.32},
            {"text": "미국", "start": 0.33, "end": 0.59},
        ]
    }

    normalized = normalize_scribe_response(raw, language_code="kor")

    assert normalized["text"] == "금일 미국"


def test_transcribe_and_save_writes_raw_and_normalized(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=MOCK_RAW_RESPONSE)

    audio_path = tmp_path / "audio" / "narration.mp3"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"fake-audio-bytes")

    words_json_path = tmp_path / "transcripts" / "words.json"
    raw_log_path = tmp_path / "logs" / "scribe_raw.json"

    with patch("md2short.transcript.httpx.Client", _mock_client_factory(handler)):
        result = transcribe_and_save(
            audio_path=audio_path,
            api_key="test-key",
            words_json_path=words_json_path,
            raw_log_path=raw_log_path,
            language_code="kor",
        )

    assert raw_log_path.exists()
    assert json.loads(raw_log_path.read_text()) == MOCK_RAW_RESPONSE

    assert words_json_path.exists()
    saved_normalized = json.loads(words_json_path.read_text())
    assert saved_normalized == result
    assert saved_normalized["language_code"] == "kor"
    assert saved_normalized["words"] == [
        {"text": "금일", "start": 0, "end": 320},
        {"text": "미국", "start": 330, "end": 590},
    ]
    assert saved_normalized["audio_duration_ms"] == 820
