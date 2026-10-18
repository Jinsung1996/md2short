"""ElevenLabs Scribe word-timestamp transcription (F-04).

See docs/architecture.md section 5.3 ("[F-04] 단어 단위 타임스탬프 추출") and
section 4.2 (`words.json` schema). This module calls ElevenLabs Speech-to-Text
(`scribe_v1`), normalizes the raw response into the `words.json` shape, and
writes both the raw and normalized artifacts to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from md2short._retry import post_with_retry

__all__ = ["transcribe", "normalize_scribe_response", "transcribe_and_save"]

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL_ID = "scribe_v1"

#: Transcribing a full narration mp3 takes far longer than httpx's 5s default
#: timeout (see the same note in voice.py), and a timed-out attempt still costs
#: STT credits. Higher than the TTS timeout because Scribe uploads the audio
#: and transcribes it in one request.
REQUEST_TIMEOUT_SECONDS = 300.0


def transcribe(audio_bytes: bytes, api_key: str, language_code: str = "kor") -> dict:
    """Call ElevenLabs Speech-to-Text (`scribe_v1`) and return the raw JSON response.

    Sends a multipart/form-data POST to `SCRIBE_URL` with the `xi-api-key`
    header, the audio bytes as the `file` field, and `model_id`/`language_code`
    form fields. Uses `_retry.post_with_retry` for 429/5xx/timeout/transport
    retry behavior. Does not transform the response -- callers pass it to
    `normalize_scribe_response` separately.
    """
    headers = {"xi-api-key": api_key}
    files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
    data = {"model_id": SCRIBE_MODEL_ID, "language_code": language_code}

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = post_with_retry(
            client,
            SCRIBE_URL,
            headers=headers,
            files=files,
            data=data,
        )
    return response.json()


#: Scribe tags every entry in `words` with a `type`. Only `"word"` entries are
#: real words: `"spacing"` entries are the whitespace *between* words (their
#: `text` is literally `" "`) and `"audio_event"` entries are non-speech sounds
#: (laughter, applause). Both must be dropped -- they are not narration tokens,
#: and letting them through would corrupt every downstream word-index
#: calculation (caption chunking, opening duration, emoticon cue placement).
#: Entries with no `type` at all are kept, so a hand-written or older-shaped
#: response still normalizes.
_NON_WORD_ENTRY_TYPES = frozenset({"spacing", "audio_event"})

#: Overall audio length in seconds. This is the field name the live Scribe API
#: actually returns (verified against the real API in US-016); `audio_duration`
#: is accepted as a secondary spelling only.
_DURATION_KEYS = ("audio_duration_secs", "audio_duration")


def normalize_scribe_response(raw_response: dict, language_code: str = "kor") -> dict:
    """Normalize a raw Scribe response into the `words.json` shape.

    The raw ElevenLabs Scribe response shape, as verified against the live API
    during US-016:
      - `raw_response["words"]` is a list of dicts, each with the token text
        under `"text"`, timestamps under `"start"`/`"end"` in **seconds**
        (float) per architecture.md §5.3, a `"type"` discriminator
        (`"word"` / `"spacing"` / `"audio_event"`) and a `"logprob"`.
        Non-`"word"` entries are filtered out here (see
        `_NON_WORD_ENTRY_TYPES`); `"word"` is also read from the legacy
        `"word"` key when `"text"` is absent.
      - the overall duration in seconds is returned under
        `"audio_duration_secs"` (see `_DURATION_KEYS`). When no duration key is
        present, duration falls back to the max word `end` (in ms), which
        *undercounts* by however much trailing silence the mp3 has.
      - `raw_response["text"]` holds the full transcript. If absent, it is
        reconstructed by joining word texts with a single space.

    Converts each word's start/end from seconds to ms via `round(seconds *
    1000)` and sets `language_code` to the normalized value passed in
    (`language_code` param), not whatever raw casing/value Scribe returns.
    """
    raw_words = raw_response.get("words", [])

    words: list[dict[str, Any]] = []
    for raw_word in raw_words:
        if raw_word.get("type") in _NON_WORD_ENTRY_TYPES:
            continue
        text = raw_word.get("text", raw_word.get("word", ""))
        start_ms = round(raw_word["start"] * 1000)
        end_ms = round(raw_word["end"] * 1000)
        words.append({"text": text, "start": start_ms, "end": end_ms})

    duration_key = next((key for key in _DURATION_KEYS if key in raw_response), None)
    if duration_key is not None:
        audio_duration_ms = round(raw_response[duration_key] * 1000)
    elif words:
        audio_duration_ms = max(word["end"] for word in words)
    else:
        audio_duration_ms = 0

    text = raw_response.get("text")
    if text is None:
        text = " ".join(word["text"] for word in words)

    return {
        "language_code": language_code,
        "audio_duration_ms": audio_duration_ms,
        "text": text,
        "words": words,
    }


def transcribe_and_save(
    audio_path: Path,
    api_key: str,
    words_json_path: Path,
    raw_log_path: Path,
    language_code: str = "kor",
) -> dict:
    """Transcribe `audio_path` and persist both raw and normalized outputs.

    Reads audio bytes from `audio_path`, calls `transcribe`, writes the raw
    Scribe response to `raw_log_path` (`logs/scribe_raw.json`), normalizes it
    via `normalize_scribe_response`, writes the normalized result to
    `words_json_path` (`transcripts/words.json`), and returns the normalized
    dict.
    """
    audio_bytes = audio_path.read_bytes()
    raw_response = transcribe(audio_bytes, api_key, language_code)

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.write_text(json.dumps(raw_response, ensure_ascii=False, indent=2))

    normalized = normalize_scribe_response(raw_response, language_code)

    words_json_path.parent.mkdir(parents=True, exist_ok=True)
    words_json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2))

    return normalized
