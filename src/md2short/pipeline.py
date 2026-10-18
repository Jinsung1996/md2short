"""Single-job orchestration (docs/architecture.md section 2.1, US-013).

`run_job()` wires the already-built stage modules together in the order of the
architecture.md section 2.1 flow diagram:

    [1] script -> [2] voice -> [3] transcript -> [4] numeric_display
    -> [5] visual cue planning -> [6] subtitled -> [7] render
    -> [8] output copy -> [9] qc

Every canonical status write goes through `update_status()`, which gates the
transition with `state.validate_transition()` (section 3). The photo-mode
progress events (`photo_generating`, `photos_generated`, `photo_fallback`) are
*not* state-machine transitions (section 3.1): they are surfaced through the
`on_stage` callback only, plus `logs/photo_fallback.json` on fallback.

This module owns job directory layout (section 4.4), the `output/ret.mp4` copy
(section 5.11) and the job-wide photo->emoticon fallback (section 1.7). It
makes no external API call itself; each stage module owns its own HTTP work.
"""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from md2short import (
    caption_plan,
    data_extract,
    emoticon_map,
    numeric_display,
    photo_engine,
    qc,
    render,
    script,
    script_engine,
    state,
    storyboard,
    transcript,
    tts_text,
    voice,
)
from md2short.photo_engine import PhotoEngineError
from md2short.schema import CTA_TEXT, SCRIBE_LANG, JobConfig

#: architecture.md section 5.11: `pipeline.DEFAULT_OUTPUT_COPY = output/ret.mp4`.
DEFAULT_OUTPUT_COPY = Path("output/ret.mp4")

#: architecture.md section 4.4 job directory layout.
JOB_SUBDIRS: tuple[str, ...] = (
    "inputs",
    "audio",
    "transcripts",
    "edit",
    "logs",
    "thumbs",
)

STATUS_FILE_NAME = "status.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def generate_job_id() -> str:
    """Return a fresh job id in the `sf_<YYYYMMDD>_<6 hex>` format (section 3.2)."""
    return f"sf_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{secrets.token_hex(3)}"


def create_job_directories(jobs_root: Path, job_id: str) -> dict[str, Path]:
    """Create `jobs_root/<job_id>/` with the section 4.4 subdirectories.

    Also writes the initial `status.json` (`status="created"`). Returns a dict
    of named paths: `"root"`, one entry per subdirectory in `JOB_SUBDIRS`, and
    `"status"` for the `status.json` file itself.
    """
    root = Path(jobs_root) / job_id
    paths: dict[str, Path] = {"root": root}

    for name in JOB_SUBDIRS:
        sub = root / name
        sub.mkdir(parents=True, exist_ok=True)
        paths[name] = sub

    paths["status"] = root / STATUS_FILE_NAME
    _write_json(
        paths["status"],
        {
            "status": "created",
            "job_id": job_id,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    return paths


def update_status(job_dir: Path, new_status: str, **extra_fields: Any) -> dict:
    """Validate and persist a status transition for `job_dir`.

    Reads the current `status.json`, calls `state.validate_transition()` (whose
    `ValueError` propagates unchanged), merges `extra_fields` (e.g. `qc_passed`,
    `qc_report`, `error`), stamps `updated_at`, writes the file back and returns
    the updated dict.
    """
    status_path = Path(job_dir) / STATUS_FILE_NAME
    current: dict[str, Any] = json.loads(status_path.read_text(encoding="utf-8"))

    state.validate_transition(str(current.get("status", "")), new_status)

    current.update(extra_fields)
    current["status"] = new_status
    current["updated_at"] = _now_iso()
    _write_json(status_path, current)
    return current


def on_stage(callback: Callable | None, event: str, **data: Any) -> None:
    """Surface a progress event (section 3.1) without touching the state machine.

    No-op when `callback` is None. `photo_generating`/`photos_generated`/
    `photo_fallback` reach the caller only through here — never through
    `status.json` or `state.validate_transition()`.
    """
    if callback is not None:
        callback(event, **data)


def _opening_duration_ms(spoken_text: str | None, words: list[dict[str, Any]]) -> int:
    """Measure the spoken opening against `words.json` (section 5.10).

    The spoken opening is prefixed to the narration, so its tokens are the
    first transcript words; the opening visual holds until the last of them
    ends. Returns 0 when there is no opening or no transcript.
    """
    if not spoken_text or not words:
        return 0
    token_count = min(len(spoken_text.split()), len(words))
    if token_count <= 0:
        return 0
    return int(words[token_count - 1]["end"])


def _highlight_keywords(cues: list[dict[str, Any]] | None) -> list[str]:
    """Collect the caption highlight keywords from emoticon cues (section 2.1).

    Emoticon cues already carry the matched Korean narration keyword, so the
    highlight list is a dedup of those in cue order. Photo cues have no Korean
    keyword (their `primaryVisualSubject` is the English image subject), so
    photo mode yields an empty list.
    """
    keywords: list[str] = []
    for cue in cues or []:
        keyword = cue.get("keyword")
        if keyword and keyword not in keywords:
            keywords.append(str(keyword))
    return keywords


def run_job(
    markdown_path: Path,
    job_config: JobConfig,
    *,
    jobs_root: Path = Path("jobs"),
    remotion_dir: Path = Path("remotion"),
    elevenlabs_api_key: str,
    openai_api_key: str | None,
    output_path: Path | None = DEFAULT_OUTPUT_COPY,
    on_stage_callback: Callable | None = None,
    model: str = "gpt-5-mini",
) -> dict:
    """Run the full section 2.1 pipeline for one markdown input.

    Returns `{"job_id", "job_dir", "status", "qc_report", "output_path"}`.
    Any stage exception marks the job `failed` (with an `error` field) and is
    re-raised so the CLI (US-014) can map it to exit code 2.
    """
    markdown_path = Path(markdown_path)
    remotion_dir = Path(remotion_dir)

    # pre-flight (section 2.1): fail before spending API credit on a box that
    # cannot measure durations or run the renderer.
    render.ensure_ffmpeg_available()
    render.ensure_remotion_available(remotion_dir)

    job_id = generate_job_id()
    job_dirs = create_job_directories(jobs_root, job_id)
    job_dir = job_dirs["root"]
    logs_dir = job_dirs["logs"]
    edit_dir = job_dirs["edit"]

    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
        script.write_source_md(job_dir, markdown_text)

        # --- [1] script -------------------------------------------------
        storyboard_cuts: list[dict[str, Any]] | None = None
        if script.detect_input_kind(markdown_text) == "storyboard":
            storyboard_cuts = storyboard.parse_storyboard(markdown_text)
            _write_json(job_dirs["inputs"] / "storyboard.json", storyboard_cuts)

        script_result = script_engine.generate_or_extract(
            markdown_text,
            job_config,
            storyboard_cuts,
            model=model,
            api_key=openai_api_key,
        )
        script_engine.write_script_outputs(job_dir, script_result)

        plan = script_result.get("plan") or {}
        fallback_reason = plan.get("fallback_reason")
        if openai_api_key and plan.get("engine") == "fallback_extract" and fallback_reason:
            # section 4.4: logs/fallback.json only when a key was available and
            # script_engine still downgraded (a real API/validation failure).
            _write_json(
                logs_dir / "fallback.json",
                {
                    "reason": fallback_reason,
                    "engine": plan.get("engine"),
                    "model": plan.get("model"),
                    "recorded_at": _now_iso(),
                },
            )

        full_text = script_result.get("full_text") or ""
        update_status(
            job_dir,
            "script_generated",
            config=job_config.model_dump(),
            markdown_input=str(markdown_path.resolve()),
            script_chars=len(full_text),
            engine=job_config.engine,
        )

        # --- [2] voice --------------------------------------------------
        tts_input = tts_text.prepare_tts_text(script_result.get("narration") or "")
        audio_path = voice.synthesize_to_file(
            tts_input,
            job_config.voice_id,
            elevenlabs_api_key,
            job_dirs["audio"] / "narration.mp3",
            speed=job_config.voice_speed,
            tts_input_log_path=logs_dir / "tts_input.txt",
        )
        update_status(job_dir, "voice_generated", tts_chars=len(tts_input))

        # --- [3] transcript ---------------------------------------------
        words_json_path = job_dirs["transcripts"] / "words.json"
        words_doc = transcript.transcribe_and_save(
            audio_path,
            elevenlabs_api_key,
            words_json_path,
            logs_dir / "scribe_raw.json",
            language_code=SCRIBE_LANG.get(job_config.language, "kor"),
        )
        update_status(job_dir, "transcribed")

        # --- [4] numeric display ----------------------------------------
        # Rewrites word *text* only (timings untouched) and re-persists
        # words.json, because render.stage_job_assets copies that same file
        # into remotion/public — Remotion captions must show display forms.
        words_doc["words"] = numeric_display.apply_numeric_display_overrides(
            words_doc.get("words") or []
        )
        # Then restore digits Scribe mis-heard ("7,230" -> "7,030"), using the
        # narration actually sent to TTS as the reference. Without this the
        # caption and the NumberCard sticker for the same figure can disagree,
        # because the sticker reads the script text and the caption reads ASR.
        words_doc["words"] = numeric_display.apply_script_number_corrections(
            words_doc["words"], script_result.get("narration") or ""
        )
        _write_json(words_json_path, words_doc)
        words = words_doc["words"]

        # --- [5] visual cue planning ------------------------------------
        effective_visual_mode = job_config.visual_mode
        photo_cues: list[dict[str, Any]] | None = None
        emoticon_cues: list[dict[str, Any]] | None = None

        if effective_visual_mode == "photo" and openai_api_key:
            on_stage(on_stage_callback, "photo_generating", job_id=job_id)
            try:
                photo_cues = photo_engine.plan_photo_cues(
                    words,
                    openai_api_key,
                    # section 2.3/4.4: photo_engine stages PNGs directly under
                    # remotion/public so cue paths are public-relative.
                    remotion_dir / "public" / "jobs" / job_id / "photos",
                    chunk_size=job_config.chunk_size,
                )
            except PhotoEngineError as exc:
                # section 1.7: the whole job falls back to emoticon mode.
                photo_cues = None
                effective_visual_mode = "emoticon"
                on_stage(on_stage_callback, "photo_fallback", job_id=job_id, reason=str(exc))
                _write_json(
                    logs_dir / "photo_fallback.json",
                    {
                        "reason": str(exc),
                        "requested_visual_mode": "photo",
                        "effective_visual_mode": "emoticon",
                        "recorded_at": _now_iso(),
                    },
                )
            else:
                on_stage(
                    on_stage_callback,
                    "photos_generated",
                    job_id=job_id,
                    cue_count=len(photo_cues),
                )
                _write_json(edit_dir / "photo_cues.json", photo_cues)

        if photo_cues is None:
            effective_visual_mode = "emoticon"
            emoticon_cues = emoticon_map.plan_cues(
                words,
                fill_gaps=True,
                max_cues=24,
                chunk_size=job_config.chunk_size,
            )
            _write_json(edit_dir / "emoticon_cues.json", emoticon_cues)

        caption_segments = caption_plan.build_caption_line_segments(
            words, job_config.chunk_size
        )
        if caption_segments:
            _write_json(edit_dir / "caption_segments.json", caption_segments)

        audio_duration_ms = int(words_doc.get("audio_duration_ms") or 0)
        if audio_duration_ms <= 0:
            audio_duration_ms = render.get_audio_duration_ms(audio_path)

        number_cards = data_extract.plan_number_cards(full_text, audio_duration_ms)
        if number_cards:
            _write_json(edit_dir / "number_cards.json", number_cards)

        # --- [6] subtitled ----------------------------------------------
        update_status(
            job_dir,
            "subtitled",
            visual_mode=effective_visual_mode,
            caption_segment_count=len(caption_segments),
        )

        # --- [7] render -------------------------------------------------
        opening = script_result.get("opening") or {}
        props = render.build_props_json(
            job_config,
            job_id,
            words_doc,
            emoticon_cues,
            photo_cues,
            caption_segments,
            number_cards,
            # No module plans company logo cues yet (section 4.4: the file is
            # written only when such cues exist), so none are passed.
            [],
            opening.get("visual_text"),
            _opening_duration_ms(opening.get("spoken_text"), words)
            if opening.get("show")
            else 0,
            CTA_TEXT.get(job_config.language, ""),
            _highlight_keywords(emoticon_cues),
            audio_duration_ms,
        )
        final_mp4_path = render.render_remotion(
            job_dir,
            job_id,
            job_config,
            remotion_dir,
            props,
            logs_dir / "remotion.log",
        )
        update_status(job_dir, "final_rendered")

        # --- [8] output copy --------------------------------------------
        copied_output: str | None = None
        if output_path is not None and str(output_path) != "":
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_mp4_path, destination)
            copied_output = str(destination)

        # --- [9] qc ------------------------------------------------------
        # script_engine records the real OpenAI usage on `plan` (0 on the
        # storyboard/fallback paths, which bill nothing), so the qc_report's
        # openai_text_usd line reflects the call that actually happened.
        qc_report = qc.run_qc(
            job_dir,
            audio_path,
            final_mp4_path,
            words_doc,
            script_char_count=len(full_text),
            tts_char_count=len(tts_input),
            openai_input_chars=int(plan.get("openai_input_chars") or 0),
            openai_output_chars=int(plan.get("openai_output_chars") or 0),
        )
        terminal_status = qc_report.get("overall_status", "qc_failed")
        update_status(
            job_dir,
            terminal_status,
            qc_report=str(job_dir / "qc_report.json"),
            qc_passed=terminal_status == "qc_passed",
            output_copy=copied_output,
        )

        return {
            "job_id": job_id,
            "job_dir": job_dir,
            "status": terminal_status,
            "qc_report": qc_report,
            "output_path": copied_output,
        }

    except Exception as exc:
        # Any non-terminal state may transition to "failed" (section 3).
        try:
            update_status(job_dir, "failed", error=str(exc))
        except (ValueError, OSError, json.JSONDecodeError):
            pass  # never mask the original failure with a bookkeeping error
        raise
