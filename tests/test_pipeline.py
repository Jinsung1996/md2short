"""Tests for md2short.pipeline (US-013, architecture.md section 2.1).

`run_job` is exercised as a fully mocked integration test: every external
boundary (ffmpeg/npx pre-flight, ElevenLabs TTS, ElevenLabs Scribe, the
Remotion render subprocess, ffprobe-backed QC) is replaced by a fake that
writes the same job artifacts the real module would. The script stage runs for
real with no OpenAI key, so it naturally takes the `fallback_extract` path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from md2short import (
    emoticon_map,
    photo_engine,
    pipeline,
    qc,
    render,
    script_engine,
    transcript,
    voice,
)
from md2short.photo_engine import PhotoEngineError
from md2short.schema import JobConfig

JOB_ID_RE = re.compile(r"^sf_\d{8}_[0-9a-f]{6}$")

SAMPLE_MARKDOWN = """# 반도체 시장 브리핑

- 삼성전자 2분기 매출은 74조 원으로 집계됐습니다.
- 코스피 지수는 3.5% 올랐습니다.
- 메모리 수요는 꾸준히 증가하고 있습니다.
"""

# A plausible transcripts/words.json word list (architecture.md section 4.2).
FAKE_WORDS = [
    {"text": "26년", "start": 0, "end": 320},
    {"text": "10월", "start": 330, "end": 620},
    {"text": "11일", "start": 630, "end": 980},
    {"text": "데이브리핑", "start": 990, "end": 1600},
    {"text": "삼성전자", "start": 1700, "end": 2200},
    {"text": "2분기", "start": 2210, "end": 2600},
    {"text": "매출은", "start": 2610, "end": 2980},
    {"text": "74조", "start": 2990, "end": 3400},
    {"text": "원으로", "start": 3410, "end": 3800},
    {"text": "집계됐습니다.", "start": 3810, "end": 4400},
    {"text": "코스피", "start": 4500, "end": 4900},
    {"text": "지수는", "start": 4910, "end": 5300},
    {"text": "3.5%", "start": 5310, "end": 5800},
    {"text": "올랐습니다.", "start": 5810, "end": 6400},
    {"text": "메모리", "start": 6500, "end": 6900},
    {"text": "수요는", "start": 6910, "end": 7300},
    {"text": "증가하고", "start": 7310, "end": 7800},
    {"text": "있습니다.", "start": 7810, "end": 8400},
]

FAKE_AUDIO_DURATION_MS = 8800


def _job_config(**overrides) -> JobConfig:
    base = {"voice_id": "v1jVu1Ky28piIPEJqRrm"}
    base.update(overrides)
    return JobConfig(**base)


def _install_fakes(monkeypatch, *, qc_status: str = "qc_passed") -> dict:
    """Replace every external boundary of `run_job` with an artifact-writing fake."""
    calls: dict[str, int] = {"voice": 0, "transcript": 0, "render": 0, "qc": 0}

    monkeypatch.setattr(render, "ensure_ffmpeg_available", lambda: None)
    monkeypatch.setattr(render, "ensure_remotion_available", lambda remotion_dir: None)

    def fake_synthesize_to_file(
        text, voice_id, api_key, output_path, speed=1.0, tts_input_log_path=None, **kwargs
    ):
        calls["voice"] += 1
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3-fake-mp3-bytes")
        if tts_input_log_path is not None:
            tts_input_log_path = Path(tts_input_log_path)
            tts_input_log_path.parent.mkdir(parents=True, exist_ok=True)
            tts_input_log_path.write_text(text, encoding="utf-8")
        return output_path

    def fake_transcribe_and_save(
        audio_path, api_key, words_json_path, raw_log_path, language_code="kor"
    ):
        calls["transcript"] += 1
        doc = {
            "language_code": language_code,
            "audio_duration_ms": FAKE_AUDIO_DURATION_MS,
            "text": " ".join(word["text"] for word in FAKE_WORDS),
            "words": [dict(word) for word in FAKE_WORDS],
        }
        for path, payload in ((raw_log_path, {"words": FAKE_WORDS}), (words_json_path, doc)):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc

    def fake_render_remotion(job_dir, job_id, job_config, remotion_dir, props, log_path):
        calls["render"] += 1
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake remotion render\n", encoding="utf-8")
        final_mp4 = Path(job_dir) / "edit" / "final.mp4"
        final_mp4.parent.mkdir(parents=True, exist_ok=True)
        final_mp4.write_bytes(b"fake-mp4-bytes")
        calls["props"] = props
        return final_mp4

    def fake_run_qc(job_dir, audio_path, final_mp4_path, words_doc, **kwargs):
        calls["qc"] += 1
        calls["qc_kwargs"] = kwargs
        report = {"checks": [], "cost_estimate": {}, "overall_status": qc_status}
        report_path = Path(job_dir) / "qc_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    monkeypatch.setattr(voice, "synthesize_to_file", fake_synthesize_to_file)
    monkeypatch.setattr(transcript, "transcribe_and_save", fake_transcribe_and_save)
    monkeypatch.setattr(render, "render_remotion", fake_render_remotion)
    monkeypatch.setattr(qc, "run_qc", fake_run_qc)
    return calls


def _run(tmp_path, monkeypatch, *, config=None, **overrides) -> tuple[dict, Path]:
    markdown_path = tmp_path / "sample.md"
    markdown_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    kwargs = {
        "jobs_root": tmp_path / "jobs",
        "remotion_dir": tmp_path / "remotion",
        "elevenlabs_api_key": "el-test",
        "openai_api_key": None,
        "output_path": tmp_path / "output" / "ret.mp4",
    }
    kwargs.update(overrides)
    result = pipeline.run_job(markdown_path, config or _job_config(), **kwargs)
    return result, markdown_path


# --- generate_job_id / create_job_directories / update_status --------------


def test_generate_job_id_format():
    job_id = pipeline.generate_job_id()
    assert JOB_ID_RE.match(job_id), job_id
    assert pipeline.generate_job_id() != job_id


def test_create_job_directories_creates_all_subdirs(tmp_path):
    paths = pipeline.create_job_directories(tmp_path / "jobs", "sf_20260502_abc123")

    for name in ("inputs", "audio", "transcripts", "edit", "logs", "thumbs"):
        assert paths[name].is_dir()
        assert paths[name] == paths["root"] / name

    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert status["status"] == "created"
    assert status["job_id"] == "sf_20260502_abc123"
    assert status["started_at"]


def test_update_status_valid_sequence(tmp_path):
    paths = pipeline.create_job_directories(tmp_path / "jobs", "sf_20260502_abc123")
    job_dir = paths["root"]

    updated = pipeline.update_status(job_dir, "script_generated", script_chars=42)
    assert updated["status"] == "script_generated"
    assert updated["script_chars"] == 42
    assert updated["job_id"] == "sf_20260502_abc123"
    assert json.loads(paths["status"].read_text(encoding="utf-8"))["status"] == "script_generated"

    for next_status in ("voice_generated", "transcribed", "subtitled", "final_rendered"):
        assert pipeline.update_status(job_dir, next_status)["status"] == next_status

    final = pipeline.update_status(job_dir, "qc_passed", qc_passed=True)
    assert final["status"] == "qc_passed"
    assert final["qc_passed"] is True


def test_update_status_rejects_invalid_transition(tmp_path):
    paths = pipeline.create_job_directories(tmp_path / "jobs", "sf_20260502_abc123")

    with pytest.raises(ValueError, match="Invalid state transition"):
        pipeline.update_status(paths["root"], "final_rendered")

    # the rejected write left status.json untouched
    assert json.loads(paths["status"].read_text(encoding="utf-8"))["status"] == "created"


def test_on_stage_without_callback_is_noop():
    assert pipeline.on_stage(None, "photo_generating", job_id="x") is None


def test_on_stage_forwards_event_and_data():
    events: list[tuple[str, dict]] = []
    pipeline.on_stage(lambda event, **data: events.append((event, data)), "photo_fallback", reason="boom")
    assert events == [("photo_fallback", {"reason": "boom"})]


# --- run_job happy path ----------------------------------------------------


def test_run_job_emoticon_happy_path(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch)
    result, markdown_path = _run(tmp_path, monkeypatch)

    assert JOB_ID_RE.match(result["job_id"])
    assert result["status"] == "qc_passed"
    assert result["qc_report"]["overall_status"] == "qc_passed"

    job_dir = result["job_dir"]
    assert job_dir == tmp_path / "jobs" / result["job_id"]

    for relative in (
        "status.json",
        "inputs/source.md",
        "inputs/script.md",
        "inputs/script.json",
        "audio/narration.mp3",
        "transcripts/words.json",
        "edit/emoticon_cues.json",
        "edit/caption_segments.json",
        "edit/number_cards.json",
        "edit/final.mp4",
        "logs/tts_input.txt",
        "logs/scribe_raw.json",
        "logs/remotion.log",
        "qc_report.json",
    ):
        assert (job_dir / relative).is_file(), relative

    # photo-mode-only artifacts must not appear
    assert not (job_dir / "edit" / "photo_cues.json").exists()
    assert not (job_dir / "logs" / "photo_fallback.json").exists()
    # no OpenAI key was supplied, so the fallback was expected, not logged
    assert not (job_dir / "logs" / "fallback.json").exists()

    assert (job_dir / "inputs" / "source.md").read_text(encoding="utf-8") == SAMPLE_MARKDOWN

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "qc_passed"
    assert status["qc_passed"] is True
    assert status["qc_report"] == str(job_dir / "qc_report.json")
    assert status["config"]["visual_mode"] == "emoticon"
    assert status["markdown_input"] == str(markdown_path.resolve())
    assert status["visual_mode"] == "emoticon"
    assert status["script_chars"] > 0

    # output copy (step [8])
    output_copy = tmp_path / "output" / "ret.mp4"
    assert output_copy.is_file()
    assert output_copy.read_bytes() == (job_dir / "edit" / "final.mp4").read_bytes()
    assert result["output_path"] == str(output_copy)

    # every external boundary was hit exactly once
    assert calls["voice"] == 1
    assert calls["transcript"] == 1
    assert calls["render"] == 1
    assert calls["qc"] == 1

    # QC received the measured character counts
    tts_input = (job_dir / "logs" / "tts_input.txt").read_text(encoding="utf-8")
    script_json = json.loads((job_dir / "inputs" / "script.json").read_text(encoding="utf-8"))
    assert calls["qc_kwargs"]["tts_char_count"] == len(tts_input)
    assert calls["qc_kwargs"]["script_char_count"] == len(script_json["full_text"])
    assert script_json["plan"]["engine"] == "fallback_extract"
    # ...and 0 OpenAI usage, because this run took the fallback path (US-012).
    assert calls["qc_kwargs"]["openai_input_chars"] == 0
    assert calls["qc_kwargs"]["openai_output_chars"] == 0

    # props wiring (step [7])
    props = calls["props"]
    assert props["chunkSize"] == 14
    assert props["emoticonCues"] == json.loads(
        (job_dir / "edit" / "emoticon_cues.json").read_text(encoding="utf-8")
    )
    assert props["captionSegments"] == json.loads(
        (job_dir / "edit" / "caption_segments.json").read_text(encoding="utf-8")
    )
    assert props["numberCards"] == json.loads(
        (job_dir / "edit" / "number_cards.json").read_text(encoding="utf-8")
    )
    assert props["showOpening"] is True
    assert props["openingText"] == script_json["opening"]["visual_text"]
    assert props["openingDurationMs"] > 0
    assert props["ctaText"] == "구독과 알람 설정, 큰 힘이 됩니다!"
    assert props["durationInFrames"] == round(FAKE_AUDIO_DURATION_MS / 1000 * 30)


def test_run_job_status_transition_sequence(tmp_path, monkeypatch):
    """Every status write goes through validate_transition in section 3 order."""
    _install_fakes(monkeypatch)
    seen: list[str] = []
    real_update_status = pipeline.update_status

    def recording_update_status(job_dir, new_status, **extra):
        seen.append(new_status)
        return real_update_status(job_dir, new_status, **extra)

    monkeypatch.setattr(pipeline, "update_status", recording_update_status)
    _run(tmp_path, monkeypatch)

    assert seen == [
        "script_generated",
        "voice_generated",
        "transcribed",
        "subtitled",
        "final_rendered",
        "qc_passed",
    ]


def test_run_job_qc_failed_is_terminal_status(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, qc_status="qc_failed")
    result, _ = _run(tmp_path, monkeypatch)

    assert result["status"] == "qc_failed"
    status = json.loads((result["job_dir"] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "qc_failed"
    assert status["qc_passed"] is False


def test_run_job_output_copy_disabled_by_empty_path(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    result, _ = _run(tmp_path, monkeypatch, output_path="")

    assert result["output_path"] is None
    assert not (tmp_path / "output").exists()
    assert (result["job_dir"] / "edit" / "final.mp4").is_file()


# --- OpenAI usage reaches qc.run_qc (US-012) -------------------------------


def test_run_job_forwards_openai_usage_from_script_plan_to_qc(tmp_path, monkeypatch):
    """A real `openai` script call must bill a non-zero openai_text_usd line.

    The whole point of US-012's reopened criterion: pipeline reads the usage
    script_engine recorded on `plan` and hands it to `qc.run_qc()`.
    """
    calls = _install_fakes(monkeypatch)
    real_generate_or_extract = script_engine.generate_or_extract

    def fake_generate_or_extract(markdown_text, job_config, storyboard_cuts=None, **kwargs):
        # api_key=None keeps the script stage offline; the `plan` below stands
        # in for what the real `openai` path would have recorded.
        kwargs["api_key"] = None
        result = real_generate_or_extract(markdown_text, job_config, storyboard_cuts, **kwargs)
        result["plan"] = {
            "engine": "openai",
            "fallback_reason": None,
            "model": "gpt-5-mini",
            "openai_input_chars": 7308,
            "openai_output_chars": 2436,
        }
        return result

    monkeypatch.setattr(script_engine, "generate_or_extract", fake_generate_or_extract)

    result, _ = _run(tmp_path, monkeypatch, openai_api_key="sk-test")

    assert result["status"] == "qc_passed"
    assert calls["qc_kwargs"]["openai_input_chars"] == 7308
    assert calls["qc_kwargs"]["openai_output_chars"] == 2436
    # those counts really do price above zero in the estimator
    estimate = qc.estimate_cost(
        script_char_count=100,
        tts_char_count=100,
        audio_duration_seconds=10.0,
        openai_input_chars=calls["qc_kwargs"]["openai_input_chars"],
        openai_output_chars=calls["qc_kwargs"]["openai_output_chars"],
    )
    assert estimate["openai_text_usd"] > 0


def test_run_job_forwards_zero_openai_usage_for_non_openai_engine(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch)
    real_generate_or_extract = script_engine.generate_or_extract

    def fake_generate_or_extract(markdown_text, job_config, storyboard_cuts=None, **kwargs):
        kwargs["api_key"] = None  # keep the script stage offline
        result = real_generate_or_extract(markdown_text, job_config, storyboard_cuts, **kwargs)
        result["plan"] = {
            "engine": "storyboard",
            "fallback_reason": None,
            "model": None,
            "openai_input_chars": 0,
            "openai_output_chars": 0,
        }
        return result

    monkeypatch.setattr(script_engine, "generate_or_extract", fake_generate_or_extract)

    _run(tmp_path, monkeypatch, openai_api_key="sk-test")

    assert calls["qc_kwargs"]["openai_input_chars"] == 0
    assert calls["qc_kwargs"]["openai_output_chars"] == 0


# --- emoticon mode never touches photo_engine ------------------------------


def test_run_job_emoticon_mode_never_calls_photo_engine(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    plan_photo_cues = MagicMock(name="plan_photo_cues")
    generate_image_prompts = MagicMock(name="generate_image_prompts")
    generate_image_cached = MagicMock(name="generate_image_cached")
    monkeypatch.setattr(photo_engine, "plan_photo_cues", plan_photo_cues)
    monkeypatch.setattr(photo_engine, "generate_image_prompts", generate_image_prompts)
    monkeypatch.setattr(photo_engine, "generate_image_cached", generate_image_cached)

    events: list[str] = []
    result, _ = _run(
        tmp_path,
        monkeypatch,
        config=_job_config(visual_mode="emoticon"),
        openai_api_key=None,
        on_stage_callback=lambda event, **data: events.append(event),
    )

    assert plan_photo_cues.call_count == 0
    assert generate_image_prompts.call_count == 0
    assert generate_image_cached.call_count == 0
    assert events == []
    assert result["status"] == "qc_passed"
    assert (result["job_dir"] / "edit" / "emoticon_cues.json").is_file()


# --- photo mode success + PhotoEngineError fallback ------------------------


def test_run_job_photo_mode_success(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch)
    photo_cues = [
        {
            "emoticonFileName": "jobs/x/photos/seg_00_abc.png",
            "startMs": 0,
            "durationMs": 2000,
            "assetKind": "photo",
            "visualMode": "photo",
            "keyword": None,
        }
    ]
    captured: dict = {}

    def fake_plan_photo_cues(words, api_key, output_dir, chunk_size=14, **kwargs):
        captured["output_dir"] = Path(output_dir)
        captured["words"] = words
        return photo_cues

    monkeypatch.setattr(photo_engine, "plan_photo_cues", fake_plan_photo_cues)

    events: list[tuple[str, dict]] = []
    result, _ = _run(
        tmp_path,
        monkeypatch,
        config=_job_config(visual_mode="photo"),
        openai_api_key="sk-test",
        on_stage_callback=lambda event, **data: events.append((event, data)),
    )

    job_dir = result["job_dir"]
    assert [event for event, _ in events] == ["photo_generating", "photos_generated"]
    assert json.loads((job_dir / "edit" / "photo_cues.json").read_text(encoding="utf-8")) == photo_cues
    assert not (job_dir / "edit" / "emoticon_cues.json").exists()
    assert calls["props"]["emoticonCues"] == photo_cues
    # PNGs are staged straight into remotion/public (architecture.md 2.3/4.4)
    assert captured["output_dir"] == tmp_path / "remotion" / "public" / "jobs" / result["job_id"] / "photos"
    assert result["status"] == "qc_passed"
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["visual_mode"] == "photo"


def test_run_job_photo_engine_error_falls_back_to_emoticon(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch)

    def failing_plan_photo_cues(*args, **kwargs):
        raise PhotoEngineError("all retries exhausted for seg_03")

    monkeypatch.setattr(photo_engine, "plan_photo_cues", failing_plan_photo_cues)

    events: list[tuple[str, dict]] = []
    result, _ = _run(
        tmp_path,
        monkeypatch,
        config=_job_config(visual_mode="photo"),
        openai_api_key="sk-test",
        on_stage_callback=lambda event, **data: events.append((event, data)),
    )

    job_dir = result["job_dir"]

    assert [event for event, _ in events] == ["photo_generating", "photo_fallback"]
    assert "all retries exhausted for seg_03" in events[1][1]["reason"]

    fallback_log = json.loads(
        (job_dir / "logs" / "photo_fallback.json").read_text(encoding="utf-8")
    )
    assert "all retries exhausted for seg_03" in fallback_log["reason"]
    assert fallback_log["requested_visual_mode"] == "photo"
    assert fallback_log["effective_visual_mode"] == "emoticon"

    # the job completed on emoticon cues, it was not marked failed
    assert result["status"] == "qc_passed"
    emoticon_cues = json.loads(
        (job_dir / "edit" / "emoticon_cues.json").read_text(encoding="utf-8")
    )
    assert emoticon_cues == emoticon_map.plan_cues(FAKE_WORDS, chunk_size=14)
    assert not (job_dir / "edit" / "photo_cues.json").exists()
    assert calls["props"]["emoticonCues"] == emoticon_cues

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "qc_passed"
    assert status["visual_mode"] == "emoticon"
    # the progress events never reached status.json (section 3.1)
    assert "photo_generating" not in status.values()
    assert "photo_fallback" not in status.values()


def test_run_job_photo_mode_without_openai_key_uses_emoticon(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    plan_photo_cues = MagicMock(name="plan_photo_cues")
    monkeypatch.setattr(photo_engine, "plan_photo_cues", plan_photo_cues)

    result, _ = _run(
        tmp_path,
        monkeypatch,
        config=_job_config(visual_mode="photo"),
        openai_api_key=None,
    )

    assert plan_photo_cues.call_count == 0
    assert (result["job_dir"] / "edit" / "emoticon_cues.json").is_file()
    assert result["status"] == "qc_passed"


# --- logs/fallback.json ----------------------------------------------------


def test_run_job_logs_fallback_when_api_key_present_but_script_downgraded(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)

    def fake_generate_script(markdown_text, **kwargs):
        return {
            "hook": "",
            "body": "삼성전자 매출은 74조 원입니다.",
            "cta": "",
            "full_text": "삼성전자 매출은 74조 원입니다.",
            "plan": {
                "engine": "fallback_extract",
                "fallback_reason": "api_error",
                "model": "gpt-5-mini",
            },
        }

    monkeypatch.setattr(script_engine, "generate_script", fake_generate_script)

    result, _ = _run(tmp_path, monkeypatch, openai_api_key="sk-test")

    fallback_log = json.loads(
        (result["job_dir"] / "logs" / "fallback.json").read_text(encoding="utf-8")
    )
    assert fallback_log["reason"] == "api_error"
    assert fallback_log["engine"] == "fallback_extract"
    assert fallback_log["model"] == "gpt-5-mini"


# --- failure handling ------------------------------------------------------


def test_run_job_render_failure_marks_job_failed_and_reraises(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)

    def failing_render(*args, **kwargs):
        raise RuntimeError("remotion exited with code 1")

    monkeypatch.setattr(render, "render_remotion", failing_render)

    with pytest.raises(RuntimeError, match="remotion exited with code 1"):
        _run(tmp_path, monkeypatch)

    job_dirs = sorted((tmp_path / "jobs").iterdir())
    assert len(job_dirs) == 1
    status = json.loads((job_dirs[0] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"] == "remotion exited with code 1"
    # the job failed from "subtitled", so earlier artifacts survive for debugging
    assert (job_dirs[0] / "edit" / "emoticon_cues.json").is_file()
    assert not (job_dirs[0] / "qc_report.json").exists()


def test_run_job_tts_failure_marks_job_failed(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)

    def failing_synthesize(*args, **kwargs):
        raise RuntimeError("elevenlabs 401")

    monkeypatch.setattr(voice, "synthesize_to_file", failing_synthesize)

    with pytest.raises(RuntimeError, match="elevenlabs 401"):
        _run(tmp_path, monkeypatch)

    job_dir = next(iter((tmp_path / "jobs").iterdir()))
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"] == "elevenlabs 401"
