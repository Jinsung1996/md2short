"""Unit tests for the env/config loader (docs/AGENTS.md section 4.3)."""

import pytest

from md2short import config


def test_get_env_var_missing_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_REQUIRED_KEY", raising=False)
    config._loaded = True  # skip dotenv reload; rely on monkeypatched environ
    with pytest.raises(RuntimeError, match="SOME_REQUIRED_KEY"):
        config.get_env_var("SOME_REQUIRED_KEY")


def test_get_env_var_present_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_REQUIRED_KEY", "value-123")
    config._loaded = True
    assert config.get_env_var("SOME_REQUIRED_KEY") == "value-123"


def test_get_elevenlabs_api_key_falls_back_to_legacy_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("ELEVNLABS_API", "legacy-value")
    config._loaded = True
    assert config.get_elevenlabs_api_key() == "legacy-value"


def test_get_elevenlabs_api_key_prefers_correct_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "correct-value")
    monkeypatch.setenv("ELEVNLABS_API", "legacy-value")
    config._loaded = True
    assert config.get_elevenlabs_api_key() == "correct-value"


def test_get_elevenlabs_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVNLABS_API", raising=False)
    config._loaded = True
    with pytest.raises(RuntimeError):
        config.get_elevenlabs_api_key()


def test_get_openai_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config._loaded = True
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.get_openai_api_key()


def test_get_openai_api_key_present_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-value")
    config._loaded = True
    assert config.get_openai_api_key() == "openai-value"
