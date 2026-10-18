"""Shared environment/config loading helpers.

Loads secrets from the project `.env` file via python-dotenv. See
docs/AGENTS.md section 4.3: secrets live only in `.env`, never committed,
and a missing required key must fail immediately with a clear error.

The current `.env` uses the legacy, typo'd key name `ELEVNLABS_API` for the
ElevenLabs API key. `get_elevenlabs_api_key()` prefers the correctly spelled
`ELEVENLABS_API_KEY` and falls back to the legacy name when the former is
absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_loaded = False


def load_env() -> None:
    """Load the project `.env` file into `os.environ` (idempotent)."""
    global _loaded
    if _loaded:
        return
    load_dotenv(dotenv_path=_ENV_PATH)
    _loaded = True


def get_env_var(name: str, *, fallback_names: tuple[str, ...] = ()) -> str:
    """Return the value of `name` from the environment, or a fallback name.

    Raises `RuntimeError` with a readable message if neither `name` nor any
    of `fallback_names` is set to a non-empty value.
    """
    load_env()
    for candidate in (name, *fallback_names):
        value = os.environ.get(candidate)
        if value:
            return value
    names = ", ".join((name, *fallback_names))
    raise RuntimeError(
        f"Missing required environment variable (looked for: {names}). "
        f"Set it in {_ENV_PATH}."
    )


def get_elevenlabs_api_key() -> str:
    """Return the ElevenLabs API key, honoring the legacy `ELEVNLABS_API` name."""
    return get_env_var("ELEVENLABS_API_KEY", fallback_names=("ELEVNLABS_API",))


def get_openai_api_key() -> str:
    """Return the OpenAI API key."""
    return get_env_var("OPENAI_API_KEY")


def get_elevenlabs_voice_id() -> str | None:
    """Return the `ELEVENLABS_VOICE_ID` override, or None when unset.

    Optional, unlike the API keys. `schema.DEFAULT_VOICES["ko"]` (documented in
    docs/voice.md) is a *voice library* voice, and ElevenLabs rejects library
    voices with HTTP 402 `paid_plan_required` on free accounts:

        "Free users cannot use library voices via the API."

    This override lets such an account point the default at a voice it can
    actually synthesize with (e.g. a premade voice) without editing the
    documented project default, which stays correct for paid accounts. The
    `--voice` CLI flag still wins over this value.
    """
    load_env()
    return os.environ.get("ELEVENLABS_VOICE_ID") or None
