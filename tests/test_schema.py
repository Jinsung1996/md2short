"""Unit tests for md2short.schema (docs/architecture.md section 4.1)."""

import pytest
from pydantic import ValidationError

from md2short.schema import DEFAULT_VOICES, JobConfig


def _base_kwargs() -> dict:
    return {"voice_id": DEFAULT_VOICES["ko"]}


def test_default_voice_id_matches_docs_voice_md() -> None:
    assert DEFAULT_VOICES["ko"] == "v1jVu1Ky28piIPEJqRrm"


@pytest.mark.parametrize("voice_speed", [0.7, 1.0, 1.2])
def test_voice_speed_in_range_accepted(voice_speed: float) -> None:
    config = JobConfig(**_base_kwargs(), voice_speed=voice_speed)
    assert config.voice_speed == voice_speed


@pytest.mark.parametrize("voice_speed", [0.69, 1.21, 0.0, 2.0])
def test_voice_speed_out_of_range_rejected(voice_speed: float) -> None:
    with pytest.raises(ValidationError):
        JobConfig(**_base_kwargs(), voice_speed=voice_speed)


@pytest.mark.parametrize("chunk_size", [1, 14, 24])
def test_chunk_size_in_range_accepted(chunk_size: int) -> None:
    config = JobConfig(**_base_kwargs(), chunk_size=chunk_size)
    assert config.chunk_size == chunk_size


@pytest.mark.parametrize("chunk_size", [0, 25, -1])
def test_chunk_size_out_of_range_rejected(chunk_size: int) -> None:
    with pytest.raises(ValidationError):
        JobConfig(**_base_kwargs(), chunk_size=chunk_size)


def test_language_rejects_non_ko_literal() -> None:
    with pytest.raises(ValidationError):
        JobConfig(**_base_kwargs(), language="en")


def test_engine_rejects_non_remotion_literal() -> None:
    with pytest.raises(ValidationError):
        JobConfig(**_base_kwargs(), engine="ffmpeg")


def test_visual_mode_rejects_invalid_literal() -> None:
    with pytest.raises(ValidationError):
        JobConfig(**_base_kwargs(), visual_mode="video")


def test_visual_mode_accepts_valid_literals() -> None:
    assert JobConfig(**_base_kwargs(), visual_mode="emoticon").visual_mode == "emoticon"
    assert JobConfig(**_base_kwargs(), visual_mode="photo").visual_mode == "photo"


def test_defaults_match_architecture_doc() -> None:
    config = JobConfig(**_base_kwargs())
    assert config.platform == "youtube_shorts"
    assert config.language == "ko"
    assert config.template_id == "tech_explainer"
    assert config.bg_color == "0xFFFFFF"
    assert config.chunk_size == 14
    assert config.engine == "remotion"
    assert config.visual_mode == "emoticon"
    assert config.show_opening is True
    assert config.show_outro is False
