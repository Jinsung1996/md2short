"""Automated QC checks and cost estimation (F-07).

See docs/architecture.md section 5.6 ("[F-07] 자동 QC") and PRD.md sections 7
(비기능 요구사항) and 8 (성공 지표/KPI) for the exact tolerance and cost
numbers this module encodes:
  - `sync_last_word_vs_audio`: audio duration vs. last word end (trailing
    silence), tolerance +/-1000ms.
  - `final_mp4_vs_audio`: final MP4 duration vs. audio duration, tolerance
    +/-100ms.
  - cost estimate: ElevenLabs TTS $0.30/1K chars, ElevenLabs STT $0.40/hour,
    plus a fixed-rate OpenAI text-model approximation (see
    `OPENAI_GPT55_INPUT_RATE`/`OPENAI_GPT55_OUTPUT_RATE` below) using chars/4
    as a token approximation. gpt-image-2 image-generation cost is
    deliberately excluded from `total_usd` (architecture.md section 5.6/5.7:
    "이미지 생성 비용은 로컬 추정에 합산하지 않고 실제 OpenAI 청구서를 기준으로
    별도 확인") -- check the real OpenAI invoice for that line item instead.

Division of labour: this module only measures and reports. It writes
`qc_report.json` and returns the same dict, but it never calls
`state.validate_transition()` -- that state-transition decision belongs to
`pipeline.py` (US-013), which inspects `overall_status` after calling
`run_qc()`.
"""

from __future__ import annotations

import json
from pathlib import Path

from md2short import render

__all__ = [
    "OPENAI_GPT55_INPUT_RATE",
    "OPENAI_GPT55_OUTPUT_RATE",
    "check_sync_last_word_vs_audio",
    "check_final_mp4_vs_audio",
    "estimate_cost",
    "run_qc",
]

# ElevenLabs pricing (architecture.md section 5.6).
_ELEVENLABS_TTS_RATE_PER_1K_CHARS = 0.30
_ELEVENLABS_STT_RATE_PER_HOUR = 0.40

# Fixed per-1K-token approximate rates for the OpenAI text model used by
# script_engine/photo_engine (architecture.md section 5.6 names these
# `qc.py`'s "고정 근사 단가(OPENAI_GPT55_*)"; the PRD does not pin exact
# figures, so these are deliberately chosen, documented placeholders modeled
# on small-model ("mini"-tier) OpenAI text pricing circa this project's
# writing: $0.25 / 1M input tokens, $2.00 / 1M output tokens -- i.e.
# $0.00025/1K input tokens and $0.002/1K output tokens. Tokens are
# approximated as chars/4 (architecture.md: "char/4 token 근사").
OPENAI_GPT55_INPUT_RATE = 0.00025  # USD per 1K input tokens
OPENAI_GPT55_OUTPUT_RATE = 0.002  # USD per 1K output tokens

_CHARS_PER_TOKEN_APPROX = 4


def check_sync_last_word_vs_audio(
    audio_duration_ms: int, last_word_end_ms: int, tolerance_ms: int = 1000
) -> dict:
    """Compare audio duration against the last spoken word's end time.

    Passes when the trailing-silence gap `|audio_duration_ms -
    last_word_end_ms|` is within `tolerance_ms` (default +/-1000ms per
    architecture.md section 5.6 / PRD.md section 7).
    """
    diff_ms = abs(audio_duration_ms - last_word_end_ms)
    return {
        "check": "sync_last_word_vs_audio",
        "passed": diff_ms <= tolerance_ms,
        "audio_duration_ms": audio_duration_ms,
        "last_word_end_ms": last_word_end_ms,
        "diff_ms": diff_ms,
        "tolerance_ms": tolerance_ms,
    }


def check_final_mp4_vs_audio(
    video_duration_ms: int, audio_duration_ms: int, tolerance_ms: int = 100
) -> dict:
    """Compare the final MP4 duration against the audio duration.

    Passes when `|video_duration_ms - audio_duration_ms|` is within
    `tolerance_ms` (default +/-100ms per architecture.md section 5.6 /
    PRD.md section 7).
    """
    diff_ms = abs(video_duration_ms - audio_duration_ms)
    return {
        "check": "final_mp4_vs_audio",
        "passed": diff_ms <= tolerance_ms,
        "video_duration_ms": video_duration_ms,
        "audio_duration_ms": audio_duration_ms,
        "diff_ms": diff_ms,
        "tolerance_ms": tolerance_ms,
    }


def estimate_cost(
    script_char_count: int,
    tts_char_count: int,
    audio_duration_seconds: float,
    openai_input_chars: int = 0,
    openai_output_chars: int = 0,
) -> dict:
    """Local cost-estimate breakdown (architecture.md section 5.6).

    `script_char_count` is carried through for transparency/debugging only
    (it is not itself priced -- `tts_char_count` is what ElevenLabs bills
    for). gpt-image-2 image-generation cost is intentionally NOT included in
    `total_usd`; see `photo_mode_note`.
    """
    tts_usd = tts_char_count * _ELEVENLABS_TTS_RATE_PER_1K_CHARS / 1000
    stt_usd = audio_duration_seconds / 3600 * _ELEVENLABS_STT_RATE_PER_HOUR

    openai_input_tokens = openai_input_chars / _CHARS_PER_TOKEN_APPROX
    openai_output_tokens = openai_output_chars / _CHARS_PER_TOKEN_APPROX
    openai_input_usd = openai_input_tokens / 1000 * OPENAI_GPT55_INPUT_RATE
    openai_output_usd = openai_output_tokens / 1000 * OPENAI_GPT55_OUTPUT_RATE
    openai_text_usd = openai_input_usd + openai_output_usd

    total_usd = tts_usd + stt_usd + openai_text_usd

    return {
        "script_char_count": script_char_count,
        "tts_char_count": tts_char_count,
        "audio_duration_seconds": audio_duration_seconds,
        "elevenlabs_tts_usd": tts_usd,
        "elevenlabs_stt_usd": stt_usd,
        "openai_input_chars": openai_input_chars,
        "openai_output_chars": openai_output_chars,
        "openai_text_usd": openai_text_usd,
        "total_usd": total_usd,
        "rates": {
            "elevenlabs_tts_per_1k_chars": _ELEVENLABS_TTS_RATE_PER_1K_CHARS,
            "elevenlabs_stt_per_hour": _ELEVENLABS_STT_RATE_PER_HOUR,
            "openai_gpt55_input_per_1k_tokens": OPENAI_GPT55_INPUT_RATE,
            "openai_gpt55_output_per_1k_tokens": OPENAI_GPT55_OUTPUT_RATE,
            "chars_per_token_approx": _CHARS_PER_TOKEN_APPROX,
        },
        "photo_mode_note": (
            "gpt-image-2 image-generation cost is excluded from total_usd -- "
            "confirm it against the actual OpenAI invoice (architecture.md "
            "section 5.6/5.7)."
        ),
    }


def run_qc(
    job_dir: Path,
    audio_path: Path,
    final_mp4_path: Path,
    words_doc: dict,
    script_char_count: int,
    tts_char_count: int,
    openai_input_chars: int = 0,
    openai_output_chars: int = 0,
) -> dict:
    """Top-level QC orchestration: measure, check, price, and persist.

    Writes the assembled report to `job_dir / "qc_report.json"` and returns
    the same dict. Does not call `state.validate_transition()` -- the caller
    (pipeline.py, US-013) decides the state transition from
    `overall_status`.
    """
    audio_duration_ms = render.get_audio_duration_ms(audio_path)
    video_duration_ms = render.get_video_duration_ms(final_mp4_path)

    words = words_doc.get("words") or []
    if words:
        last_word_end_ms = int(words[-1]["end"])
        sync_check = check_sync_last_word_vs_audio(audio_duration_ms, last_word_end_ms)
    else:
        sync_check = {
            "check": "sync_last_word_vs_audio",
            "passed": False,
            "audio_duration_ms": audio_duration_ms,
            "last_word_end_ms": None,
            "diff_ms": None,
            "tolerance_ms": 1000,
            "reason": "words_doc contains no words -- cannot determine last word end time",
        }

    mp4_check = check_final_mp4_vs_audio(video_duration_ms, audio_duration_ms)

    audio_duration_seconds = audio_duration_ms / 1000
    cost_estimate = estimate_cost(
        script_char_count=script_char_count,
        tts_char_count=tts_char_count,
        audio_duration_seconds=audio_duration_seconds,
        openai_input_chars=openai_input_chars,
        openai_output_chars=openai_output_chars,
    )

    checks = [sync_check, mp4_check]
    overall_status = "qc_passed" if all(check["passed"] for check in checks) else "qc_failed"

    report = {
        "checks": checks,
        "cost_estimate": cost_estimate,
        "duration_policy": {"source": "tts_audio_duration"},
        "overall_status": overall_status,
    }

    report_path = job_dir / "qc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
