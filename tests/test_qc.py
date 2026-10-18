"""Tests for md2short.qc (US-012, F-07).

`run_qc` tests monkeypatch `md2short.render.get_audio_duration_ms` /
`get_video_duration_ms` so no real ffprobe process is spawned over a fake
media file -- only the duration measurement is a seam here (render.py/US-011
owns the real ffprobe call; qc.py just consumes it).
"""

from __future__ import annotations

import json

from md2short import qc, render


# --- check_sync_last_word_vs_audio -----------------------------------------


def test_sync_last_word_within_tolerance():
    result = qc.check_sync_last_word_vs_audio(
        audio_duration_ms=30000, last_word_end_ms=29500
    )
    assert result["check"] == "sync_last_word_vs_audio"
    assert result["passed"] is True
    assert result["diff_ms"] == 500
    assert result["tolerance_ms"] == 1000
    assert result["audio_duration_ms"] == 30000
    assert result["last_word_end_ms"] == 29500


def test_sync_last_word_outside_tolerance():
    result = qc.check_sync_last_word_vs_audio(
        audio_duration_ms=30000, last_word_end_ms=28000
    )
    assert result["passed"] is False
    assert result["diff_ms"] == 2000


def test_sync_last_word_boundary_exactly_at_tolerance_passes():
    result = qc.check_sync_last_word_vs_audio(
        audio_duration_ms=30000, last_word_end_ms=29000, tolerance_ms=1000
    )
    assert result["diff_ms"] == 1000
    assert result["passed"] is True


def test_sync_last_word_boundary_one_ms_over_fails():
    result = qc.check_sync_last_word_vs_audio(
        audio_duration_ms=30000, last_word_end_ms=28999, tolerance_ms=1000
    )
    assert result["diff_ms"] == 1001
    assert result["passed"] is False


# --- check_final_mp4_vs_audio ------------------------------------------------


def test_final_mp4_within_tolerance():
    result = qc.check_final_mp4_vs_audio(
        video_duration_ms=30050, audio_duration_ms=30000
    )
    assert result["check"] == "final_mp4_vs_audio"
    assert result["passed"] is True
    assert result["diff_ms"] == 50
    assert result["tolerance_ms"] == 100


def test_final_mp4_outside_tolerance():
    result = qc.check_final_mp4_vs_audio(
        video_duration_ms=30200, audio_duration_ms=30000
    )
    assert result["passed"] is False
    assert result["diff_ms"] == 200


def test_final_mp4_boundary_exactly_at_tolerance_passes():
    result = qc.check_final_mp4_vs_audio(
        video_duration_ms=30100, audio_duration_ms=30000, tolerance_ms=100
    )
    assert result["diff_ms"] == 100
    assert result["passed"] is True


def test_final_mp4_boundary_one_ms_over_fails():
    result = qc.check_final_mp4_vs_audio(
        video_duration_ms=30101, audio_duration_ms=30000, tolerance_ms=100
    )
    assert result["diff_ms"] == 101
    assert result["passed"] is False


# --- estimate_cost ------------------------------------------------------------


def test_estimate_cost_arithmetic_matches_hand_computed_values():
    result = qc.estimate_cost(
        script_char_count=1000,
        tts_char_count=2000,
        audio_duration_seconds=36.0,
        openai_input_chars=4000,
        openai_output_chars=800,
    )

    expected_tts_usd = 2000 * 0.30 / 1000  # 0.6
    expected_stt_usd = 36.0 / 3600 * 0.40  # 0.004
    expected_input_tokens = 4000 / 4  # 1000
    expected_output_tokens = 800 / 4  # 200
    expected_openai_usd = (
        expected_input_tokens / 1000 * qc.OPENAI_GPT55_INPUT_RATE
        + expected_output_tokens / 1000 * qc.OPENAI_GPT55_OUTPUT_RATE
    )
    expected_total = expected_tts_usd + expected_stt_usd + expected_openai_usd

    assert result["elevenlabs_tts_usd"] == expected_tts_usd
    assert result["elevenlabs_stt_usd"] == expected_stt_usd
    assert result["openai_text_usd"] == expected_openai_usd
    assert result["total_usd"] == expected_total
    assert result["script_char_count"] == 1000
    assert result["tts_char_count"] == 2000


def test_estimate_cost_image_cost_not_included_in_total():
    result = qc.estimate_cost(
        script_char_count=500,
        tts_char_count=500,
        audio_duration_seconds=10.0,
    )
    # No image-related cost key feeds total_usd; the note documents the
    # exclusion but carries no dollar amount of its own.
    assert "gpt_image_usd" not in result
    assert "image_usd" not in result
    assert "photo_mode_note" in result

    expected_total = result["elevenlabs_tts_usd"] + result["elevenlabs_stt_usd"] + result["openai_text_usd"]
    assert result["total_usd"] == expected_total


def test_estimate_cost_zero_openai_chars_yields_zero_openai_cost():
    result = qc.estimate_cost(
        script_char_count=100,
        tts_char_count=100,
        audio_duration_seconds=5.0,
    )
    assert result["openai_text_usd"] == 0.0


# --- run_qc -------------------------------------------------------------------


def _words_doc(last_end_ms: int) -> dict:
    return {
        "language_code": "kor",
        "audio_duration_ms": 30000,
        "text": "hello world",
        "words": [
            {"text": "hello", "start": 0, "end": 500},
            {"text": "world", "start": 600, "end": last_end_ms},
        ],
    }


def test_run_qc_all_pass_writes_report_and_passes(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)
    audio_path = tmp_path / "narration.mp3"
    audio_path.write_bytes(b"fake-mp3")
    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(render, "get_audio_duration_ms", lambda path: 30000)
    monkeypatch.setattr(render, "get_video_duration_ms", lambda path: 30050)

    words_doc = _words_doc(last_end_ms=29600)

    report = qc.run_qc(
        job_dir=job_dir,
        audio_path=audio_path,
        final_mp4_path=video_path,
        words_doc=words_doc,
        script_char_count=1000,
        tts_char_count=1000,
    )

    assert report["overall_status"] == "qc_passed"
    assert len(report["checks"]) == 2
    assert all(check["passed"] for check in report["checks"])
    assert report["duration_policy"] == {"source": "tts_audio_duration"}

    report_path = job_dir / "qc_report.json"
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk == report


def test_run_qc_duration_mismatch_fails(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "job2"
    job_dir.mkdir(parents=True)
    audio_path = tmp_path / "narration.mp3"
    audio_path.write_bytes(b"fake-mp3")
    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(render, "get_audio_duration_ms", lambda path: 30000)
    # 500ms off -- outside the +/-100ms final_mp4_vs_audio tolerance.
    monkeypatch.setattr(render, "get_video_duration_ms", lambda path: 30500)

    words_doc = _words_doc(last_end_ms=29600)

    report = qc.run_qc(
        job_dir=job_dir,
        audio_path=audio_path,
        final_mp4_path=video_path,
        words_doc=words_doc,
        script_char_count=1000,
        tts_char_count=1000,
    )

    assert report["overall_status"] == "qc_failed"
    mp4_check = next(c for c in report["checks"] if c["check"] == "final_mp4_vs_audio")
    assert mp4_check["passed"] is False
    sync_check = next(c for c in report["checks"] if c["check"] == "sync_last_word_vs_audio")
    assert sync_check["passed"] is True


def test_run_qc_empty_words_list_handled_without_crashing(tmp_path, monkeypatch):
    job_dir = tmp_path / "jobs" / "job3"
    job_dir.mkdir(parents=True)
    audio_path = tmp_path / "narration.mp3"
    audio_path.write_bytes(b"fake-mp3")
    video_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(render, "get_audio_duration_ms", lambda path: 30000)
    monkeypatch.setattr(render, "get_video_duration_ms", lambda path: 30050)

    words_doc = {
        "language_code": "kor",
        "audio_duration_ms": 30000,
        "text": "",
        "words": [],
    }

    report = qc.run_qc(
        job_dir=job_dir,
        audio_path=audio_path,
        final_mp4_path=video_path,
        words_doc=words_doc,
        script_char_count=0,
        tts_char_count=0,
    )

    assert report["overall_status"] == "qc_failed"
    sync_check = next(c for c in report["checks"] if c["check"] == "sync_last_word_vs_audio")
    assert sync_check["passed"] is False
    assert "reason" in sync_check
    assert (job_dir / "qc_report.json").exists()
