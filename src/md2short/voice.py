"""ElevenLabs TTS synthesis (`synthesize`, `synthesize_to_file`).

docs/architecture.md section 5.2 [F-03]:
- ElevenLabs endpoint: `/v1/text-to-speech/{voice_id}`.
- default voice_id: `v1jVu1Ky28piIPEJqRrm` (see `schema.DEFAULT_VOICES["ko"]`).
- model: `eleven_multilingual_v2`, output_format: `mp3_44100_128`.
- 429/5xx/timeout/transport error retried via `_retry.post_with_retry()`.
- 출력: `audio/narration.mp3`, `logs/tts_input.txt`.

Request shape implemented here (documented explicitly because this was
written without live access to current ElevenLabs API docs -- flagged as a
risk to double-check during US-016's real end-to-end test):
- `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
- header: `xi-api-key: <api_key>` (ElevenLabs' standard auth header name).
- query param: `output_format` (ElevenLabs' documented convention is that
  `output_format` is a query-string parameter, not a JSON body field).
- JSON body: `{"text": <text>, "model_id": <model>, "voice_settings":
  {"speed": <speed>}}`. `model_id` is the field name ElevenLabs uses for
  the model identifier. `voice_settings.speed` is included on the
  assumption that the current API accepts a `speed` key inside
  `voice_settings` (recent ElevenLabs versions added speed control there);
  if the live API rejects or silently ignores it, that is a known
  limitation to re-verify in US-016, not something this module can confirm
  without network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from md2short._retry import post_with_retry

__all__ = ["synthesize", "synthesize_to_file"]

_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

#: Synthesizing a full narration (~1K Korean characters) takes well over
#: httpx's 5s default timeout, which made every real run fail with "The read
#: operation timed out" after burning all 4 retry attempts -- and each timed-out
#: attempt still costs TTS credits server-side. Must stay generous.
REQUEST_TIMEOUT_SECONDS = 180.0


def synthesize(
    text: str,
    voice_id: str,
    api_key: str,
    speed: float = 1.0,
    model: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
) -> bytes:
    """Call the ElevenLabs TTS endpoint and return the raw mp3 bytes.

    Uses `_retry.post_with_retry` (via a plain `httpx.Client`) so 429/5xx/
    timeout/transport errors are retried per architecture.md's retry policy;
    other 4xx responses raise immediately.
    """
    url = f"{_BASE_URL}/{voice_id}"
    headers = {"xi-api-key": api_key}
    json_body = {
        "text": text,
        "model_id": model,
        "voice_settings": {"speed": speed},
    }
    params = {"output_format": output_format}

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = post_with_retry(
            client,
            url,
            headers=headers,
            json=json_body,
            params=params,
        )
    return response.content


def synthesize_to_file(
    text: str,
    voice_id: str,
    api_key: str,
    output_path: Path,
    speed: float = 1.0,
    model: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    tts_input_log_path: Optional[Path] = None,
) -> Path:
    """Synthesize `text` and write the resulting mp3 bytes to `output_path`.

    Corresponds to the `audio/narration.mp3` output in docs/architecture.md
    section 5.2. `output_path`'s parent directory is created if missing.

    If `tts_input_log_path` is given, the exact `text` string sent to TTS is
    also written there as plain UTF-8 text (corresponds to the
    `logs/tts_input.txt` output); its parent directory is created too. When
    `tts_input_log_path` is omitted (the default), nothing extra is written.
    """
    audio_bytes = synthesize(
        text,
        voice_id,
        api_key,
        speed=speed,
        model=model,
        output_format=output_format,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)

    if tts_input_log_path is not None:
        tts_input_log_path.parent.mkdir(parents=True, exist_ok=True)
        tts_input_log_path.write_text(text, encoding="utf-8")

    return output_path
