"""Pydantic data models (JobConfig, CaptionWord, VisualMode, ...).

See docs/architecture.md section 4.1 for the canonical JobConfig definition
and section 5.3/5.10/1.5 for the ko-only lookup tables.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

# Convenience alias mirroring JobConfig.visual_mode's Literal, for reuse by
# other modules (e.g. emoticon_map.py, photo_engine.py) without re-declaring it.
VisualMode = Literal["emoticon", "photo"]


class JobConfig(BaseModel):
    """Per-job configuration. Mirrors docs/architecture.md section 4.1 exactly."""

    platform: Literal["youtube_shorts", "instagram_reels", "tiktok"] = "youtube_shorts"
    language: Literal["ko"] = "ko"  # 한국어 락 (v1.1+). en/ja는 v2.
    voice_id: str
    voice_speed: float = 1.0  # valid range: 0.7~1.2
    template_id: str = "tech_explainer"
    bg_color: str = "0xFFFFFF"
    chunk_size: int = 14  # valid range: 1~24
    engine: Literal["remotion"] = "remotion"
    visual_mode: VisualMode = "emoticon"
    show_opening: bool = True
    show_outro: bool = False

    @field_validator("voice_speed")
    @classmethod
    def _validate_voice_speed(cls, value: float) -> float:
        if not (0.7 <= value <= 1.2):
            raise ValueError(f"voice_speed must be between 0.7 and 1.2, got {value}")
        return value

    @field_validator("chunk_size")
    @classmethod
    def _validate_chunk_size(cls, value: int) -> int:
        if not (1 <= value <= 24):
            raise ValueError(f"chunk_size must be between 1 and 24, got {value}")
        return value


# docs/architecture.md section 1.5: 한국어 전용 락 — 아래 사전들은 모두 "ko" 단일 키만 노출.
DEFAULT_VOICES: dict[str, str] = {
    "ko": "v1jVu1Ky28piIPEJqRrm",
}

CTA_TEXT: dict[str, str] = {
    "ko": "구독과 알람 설정, 큰 힘이 됩니다!",
}

# docs/architecture.md section 5.3: ElevenLabs Scribe language_code 매핑 (ko -> kor).
SCRIBE_LANG: dict[str, str] = {
    "ko": "kor",
}

# docs/architecture.md section 5.10 / 2.1: "X월 X일 데이브리핑입니다" opening visual template.
OPENING_TEMPLATES: dict[str, str] = {
    "ko": "{year}년 {month}월 {day}일\n데이브리핑",
}


class CaptionWord(BaseModel):
    """A single word-level timestamp entry, mirroring words.json (architecture.md section 4.2)."""

    text: str
    start: int  # ms
    end: int  # ms
