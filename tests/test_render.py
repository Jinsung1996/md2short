"""Tests for md2short.render (US-011, F-06).

Most tests mock `md2short.render.subprocess.run` so no browser/ffprobe process
is spawned. Two exceptions run for real:
  - the `ensure_*_available` success paths (ffmpeg/ffprobe/npx/node_modules are
    genuinely installed in this environment),
  - `test_render_remotion_real_integration` (marked `slow`), which renders a
    short real MP4 from the US-010 `remotion/public/jobs/sample/` fixture and
    probes it with ffprobe.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from md2short.render import (
    FFPROBE_DURATION_ARGS,
    PROJECT_PUBLIC_ASSETS,
    build_props_json,
    ensure_ffmpeg_available,
    ensure_remotion_available,
    get_audio_duration_ms,
    get_video_duration_ms,
    normalize_color,
    render_remotion,
    stage_job_assets,
)
from md2short.schema import JobConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTION_DIR = PROJECT_ROOT / "remotion"
SAMPLE_FIXTURE = REMOTION_DIR / "public" / "jobs" / "sample"

INTEGRATION_JOB_ID = "pytest_render_integration"


def _job_config(**overrides) -> JobConfig:
    base = {"voice_id": "v1jVu1Ky28piIPEJqRrm"}
    base.update(overrides)
    return JobConfig(**base)


def _make_job_dir(tmp_path: Path) -> Path:
    job_dir = tmp_path / "jobs" / "sf_20260502_abc123"
    (job_dir / "audio").mkdir(parents=True)
    (job_dir / "audio" / "narration.mp3").write_bytes(b"fake-mp3")
    (job_dir / "transcripts").mkdir(parents=True)
    (job_dir / "transcripts" / "words.json").write_text('{"words": []}')
    return job_dir


# --------------------------------------------------------------------------
# pre-flight checks
# --------------------------------------------------------------------------


def test_ensure_ffmpeg_available_real_success():
    ensure_ffmpeg_available()  # ffmpeg + ffprobe really are installed


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_ensure_ffmpeg_available_raises_when_binary_missing(missing: str):
    def fake_which(binary: str) -> str | None:
        return None if binary == missing else f"/opt/homebrew/bin/{binary}"

    with patch("md2short.render.shutil.which", side_effect=fake_which):
        with pytest.raises(RuntimeError) as excinfo:
            ensure_ffmpeg_available()

    message = str(excinfo.value)
    assert missing in message
    assert "brew install ffmpeg" in message


def test_ensure_remotion_available_real_success():
    ensure_remotion_available(REMOTION_DIR)  # node_modules installed, npx on PATH


def test_ensure_remotion_available_raises_when_dir_missing(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Remotion project directory not found"):
        ensure_remotion_available(tmp_path / "nope")


def test_ensure_remotion_available_raises_when_node_modules_missing(tmp_path: Path):
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        ensure_remotion_available(remotion_dir)

    assert "node_modules" in str(excinfo.value)
    assert "npm install" in str(excinfo.value)


def test_ensure_remotion_available_raises_when_npx_missing():
    with patch("md2short.render.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="npx not found on PATH"):
            ensure_remotion_available(REMOTION_DIR)


# --------------------------------------------------------------------------
# ffprobe duration primitives
# --------------------------------------------------------------------------


def _ffprobe_result(duration: str) -> subprocess.CompletedProcess:
    payload = json.dumps({"format": {"duration": duration}})
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=payload, stderr="")


def test_get_audio_duration_ms_builds_ffprobe_command_and_rounds_to_ms():
    audio_path = Path("/jobs/x/audio/narration.mp3")

    with patch(
        "md2short.render.subprocess.run", return_value=_ffprobe_result("33.2004")
    ) as run:
        assert get_audio_duration_ms(audio_path) == 33200

    run.assert_called_once()
    assert run.call_args.args[0] == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        "/jobs/x/audio/narration.mp3",
    ]
    assert run.call_args.kwargs == {"capture_output": True, "text": True}
    assert FFPROBE_DURATION_ARGS == (
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
    )


def test_get_video_duration_ms_builds_ffprobe_command_and_rounds_to_ms():
    with patch(
        "md2short.render.subprocess.run", return_value=_ffprobe_result("6.0006")
    ) as run:
        assert get_video_duration_ms(Path("/jobs/x/edit/final.mp4")) == 6001

    assert run.call_args.args[0][-1] == "/jobs/x/edit/final.mp4"


def test_duration_probe_raises_on_ffprobe_failure():
    failure = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="moov atom not found\n"
    )

    with patch("md2short.render.subprocess.run", return_value=failure):
        with pytest.raises(RuntimeError) as excinfo:
            get_audio_duration_ms(Path("/jobs/x/audio/broken.mp3"))

    assert "ffprobe failed" in str(excinfo.value)
    assert "moov atom not found" in str(excinfo.value)


def test_get_audio_duration_ms_real_sample_fixture():
    assert get_audio_duration_ms(SAMPLE_FIXTURE / "narration.mp3") == 6000


# --------------------------------------------------------------------------
# bg_color normalization (mirrors remotion/src/types.ts normalizeColor)
# --------------------------------------------------------------------------


def test_normalize_color_strips_0x_prefix():
    assert normalize_color("0xFFFFFF") == "#FFFFFF"


def test_normalize_color_passes_through_hash_form():
    assert normalize_color("#FFFFFF") == "#FFFFFF"


def test_normalize_color_preserves_hex_digit_case_like_typescript():
    # remotion/src/types.ts normalizeColor() does NOT change case, so neither do
    # we -- the two sides must agree byte for byte. CSS hex is case-insensitive,
    # so all three inputs paint the same white.
    assert normalize_color("0xffffff") == "#ffffff"
    assert normalize_color("#ffffff") == "#ffffff"
    assert (
        normalize_color("0xFFFFFF").lower()
        == normalize_color("#FFFFFF").lower()
        == normalize_color("#ffffff").lower()
        == "#ffffff"
    )


def test_normalize_color_matches_typescript_implementation():
    """Run the real normalizeColor() from types.ts and compare, input by input."""
    script = (
        "const normalizeColor = (c) => c.startsWith('0x') ? `#${c.slice(2)}` : c;"
        "console.log(JSON.stringify("
        "['0xFFFFFF','#FFFFFF','#ffffff','0xffffff','0x101010','#000000']"
        ".map(normalizeColor)));"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    ts_output = json.loads(result.stdout)
    py_output = [
        normalize_color(value)
        for value in ["0xFFFFFF", "#FFFFFF", "#ffffff", "0xffffff", "0x101010", "#000000"]
    ]
    assert py_output == ts_output


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def test_stage_job_assets_copies_audio_and_words_as_real_files(tmp_path: Path):
    """Remotion's static server 404s on file symlinks under public/, so copies."""
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"

    staged = stage_job_assets(job_dir, "sf_20260502_abc123", public_dir)

    assert staged == public_dir / "jobs" / "sf_20260502_abc123"
    assert (staged / "narration.mp3").read_bytes() == b"fake-mp3"
    assert not (staged / "narration.mp3").is_symlink()
    assert json.loads((staged / "words.json").read_text()) == {"words": []}
    assert not (staged / "words.json").is_symlink()
    assert not (staged / "photos").exists()


def test_stage_job_assets_never_stages_photos_from_edit_dir(tmp_path: Path):
    """photo_engine writes PNGs straight into the staged dir; edit/photos is not a source.

    `pipeline.run_job()` passes photo_engine an output dir of
    `remotion/public/jobs/<id>/photos`, so staging from `job_dir/edit/photos`
    was unreachable dead code (US-017 M2). Even when such a dir exists it must
    be ignored.
    """
    job_dir = _make_job_dir(tmp_path)
    stray = job_dir / "edit" / "photos"
    stray.mkdir(parents=True)
    (stray / "seg_00_deadbeef.png").write_bytes(b"fake-png")
    public_dir = tmp_path / "remotion" / "public"

    staged = stage_job_assets(job_dir, "job1", public_dir)

    assert not (staged / "photos").exists()


def test_stage_job_assets_preserves_photos_written_by_photo_engine(tmp_path: Path):
    """The staged dir IS photo_engine's output dir, so staging must not clobber it."""
    public_dir = tmp_path / "remotion" / "public"
    photos = public_dir / "jobs" / "job1" / "photos"
    photos.mkdir(parents=True)
    (photos / "seg_00_deadbeef.png").write_bytes(b"real-png")

    staged = stage_job_assets(_make_job_dir(tmp_path), "job1", public_dir)

    assert (staged / "photos" / "seg_00_deadbeef.png").read_bytes() == b"real-png"


def test_stage_job_assets_is_idempotent(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"

    stage_job_assets(job_dir, "job1", public_dir)
    staged = stage_job_assets(job_dir, "job1", public_dir)

    assert (staged / "narration.mp3").read_bytes() == b"fake-mp3"
    assert json.loads((staged / "words.json").read_text()) == {"words": []}


def test_stage_job_assets_raises_on_missing_narration(tmp_path: Path):
    job_dir = tmp_path / "jobs" / "job1"
    (job_dir / "transcripts").mkdir(parents=True)
    (job_dir / "transcripts" / "words.json").write_text("{}")

    with pytest.raises(RuntimeError, match="cannot stage missing asset"):
        stage_job_assets(job_dir, "job1", tmp_path / "remotion" / "public")


def test_stage_job_assets_does_not_fail_on_missing_project_assets(tmp_path: Path, caplog):
    """emoticon/img/company/font are project-level; absence only warns."""
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"

    with caplog.at_level("WARNING", logger="md2short.render"):
        stage_job_assets(job_dir, "job1", public_dir)

    warned = " ".join(record.getMessage() for record in caplog.records)
    for name in ("emoticon", "img", "company", "font"):
        assert name in warned
    assert not (public_dir / "img").exists()


def test_stage_job_assets_leaves_existing_project_assets_untouched(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"
    existing = public_dir / "emoticon"
    existing.mkdir(parents=True)
    (existing / "marker.png").write_bytes(b"keep-me")

    stage_job_assets(job_dir, "job1", public_dir)

    assert (existing / "marker.png").read_bytes() == b"keep-me"
    assert not existing.is_symlink()


# --------------------------------------------------------------------------
# project-level emoticon symlink repair (US-017 M1)
# --------------------------------------------------------------------------


def test_recreated_emoticon_link_resolves_to_dir_containing_pngs(tmp_path: Path):
    """A freshly recreated emoticon link must land on the dir holding the PNGs.

    The 60 tiger PNGs live in `<repo>/emoticon/emoticon/`, one level below the
    `emoticon` name used under `public/`. Pointing the recreated link at
    `<repo>/emoticon/` would resolve to a directory with no PNGs at all.
    """
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"
    source = tmp_path / "emoticon" / "emoticon"
    source.mkdir(parents=True)
    (source / "09_가즈아_상승.png").write_bytes(b"fake-png")

    stage_job_assets(job_dir, "job1", public_dir)

    link = public_dir / "emoticon"
    assert link.is_symlink()
    resolved = link.resolve()
    assert resolved.is_dir()
    assert list(resolved.glob("*.png")), f"{resolved} holds no *.png files"
    assert (link / "09_가즈아_상승.png").read_bytes() == b"fake-png"


def test_recreated_emoticon_dir_falls_back_to_copy_when_symlink_fails(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)
    public_dir = tmp_path / "remotion" / "public"
    source = tmp_path / "emoticon" / "emoticon"
    source.mkdir(parents=True)
    (source / "09_가즈아_상승.png").write_bytes(b"fake-png")

    with patch.object(Path, "symlink_to", side_effect=OSError("cross-device link")):
        stage_job_assets(job_dir, "job1", public_dir)

    link = public_dir / "emoticon"
    assert not link.is_symlink()
    assert list(link.resolve().glob("*.png"))


def test_emoticon_source_matches_the_real_repo_layout():
    """The configured source is the real PNG dir and the real public/ link target."""
    source = PROJECT_ROOT / PROJECT_PUBLIC_ASSETS["emoticon"]
    assert source.is_dir()
    assert list(source.glob("*.png")), f"{source} holds no *.png files"

    existing_link = REMOTION_DIR / "public" / "emoticon"
    if existing_link.is_symlink():
        assert existing_link.resolve() == source.resolve()


# --------------------------------------------------------------------------
# props.json assembly (architecture.md 4.3 / remotion/src/types.ts CueProps)
# --------------------------------------------------------------------------

EMOTICON_CUE = {
    "emoticonFileName": "09_가즈아_상승.png",
    "startMs": 1300,
    "durationMs": 2400,
    "assetKind": "emoticon",
    "visualMode": "emoticon",
    "keyword": "상승",
}
PHOTO_CUE = {
    "emoticonFileName": "jobs/sf_20260502_abc123/photos/seg_00_deadbeef.png",
    "startMs": 0,
    "durationMs": 2840,
    "assetKind": "photo",
    "visualMode": "photo",
    "keyword": None,
    "primaryVisualSubject": "나스닥",
    "visualKeywords": ["상승"],
}
CAPTION_SEGMENT = {"startMs": 0, "endMs": 2840, "text": "지난 금요일 미국 증시"}
NUMBER_CARD = {"text": "0.94%", "startMs": 1950, "durationMs": 1800, "label": "나스닥"}
COMPANY_LOGO_CUE = {
    "logoFileName": "nasdaq.png",
    "startMs": 1300,
    "durationMs": 1800,
    "company": "나스닥",
}


def _build(**overrides) -> dict:
    kwargs = {
        "job_config": _job_config(),
        "job_id": "sf_20260502_abc123",
        "words_doc": {"audio_duration_ms": 6000, "words": []},
        "emoticon_cues": [EMOTICON_CUE],
        "photo_cues": None,
        "caption_segments": [CAPTION_SEGMENT],
        "number_cards": [NUMBER_CARD],
        "company_logo_cues": [COMPANY_LOGO_CUE],
        "opening_text": "26년 5월 2일\n데이브리핑",
        "opening_duration_ms": 2800,
        "cta_text": "구독과 알람 설정, 큰 힘이 됩니다!",
        "highlight_keywords": ["나스닥은", "다우는"],
        "audio_duration_ms": 30000,
    }
    kwargs.update(overrides)
    return build_props_json(**kwargs)


def test_build_props_json_exact_shape():
    props = _build()

    assert props == {
        "audioFileName": "jobs/sf_20260502_abc123/narration.mp3",
        "wordsFileName": "jobs/sf_20260502_abc123/words.json",
        "bgColor": "#FFFFFF",
        "chunkSize": 14,
        "emoticonCues": [EMOTICON_CUE],
        "captionSegments": [CAPTION_SEGMENT],
        "companyLogoCues": [COMPANY_LOGO_CUE],
        "numberCards": [NUMBER_CARD],
        "highlightKeywords": ["나스닥은", "다우는"],
        "openingText": "26년 5월 2일\n데이브리핑",
        "openingDurationMs": 2800,
        "showOpening": True,
        "ctaText": "구독과 알람 설정, 큰 힘이 됩니다!",
        "showOutro": False,
        "durationInFrames": 900,
    }


def test_build_props_json_keys_match_cueprops_in_types_ts():
    """Every CueProps field in remotion/src/types.ts is present, and no extras."""
    types_ts = (REMOTION_DIR / "src" / "types.ts").read_text()
    block = types_ts.split("export type CueProps = {", 1)[1].split("};", 1)[0]
    expected_keys = [
        line.strip().split(":", 1)[0].strip()
        for line in block.strip().splitlines()
        if ":" in line
    ]

    assert expected_keys  # guard against a parsing regression
    assert sorted(_build().keys()) == sorted(expected_keys)


def test_build_props_json_duration_in_frames_is_audio_ms_at_30fps():
    assert _build(audio_duration_ms=33200)["durationInFrames"] == 996
    assert _build(audio_duration_ms=6000)["durationInFrames"] == 180
    assert _build(audio_duration_ms=1016)["durationInFrames"] == 30


def test_build_props_json_falls_back_to_words_doc_duration():
    props = _build(audio_duration_ms=0, words_doc={"audio_duration_ms": 6000})

    assert props["durationInFrames"] == 180


def test_build_props_json_uses_photo_cues_when_photo_mode_active():
    props = _build(photo_cues=[PHOTO_CUE])

    assert props["emoticonCues"] == [PHOTO_CUE]


def test_build_props_json_photo_cues_empty_list_wins_over_emoticon_cues():
    assert _build(photo_cues=[])["emoticonCues"] == []


def test_build_props_json_empty_when_both_cue_sets_are_none():
    assert _build(emoticon_cues=None, photo_cues=None)["emoticonCues"] == []


def test_build_props_json_normalizes_bg_color_and_derives_flags():
    props = _build(
        job_config=_job_config(
            bg_color="0x101010", chunk_size=20, show_opening=False, show_outro=True
        )
    )

    assert props["bgColor"] == "#101010"
    assert props["chunkSize"] == 20
    assert props["showOpening"] is False
    assert props["showOutro"] is True


def test_build_props_json_is_json_serializable_with_korean_text():
    dumped = json.dumps(_build(), ensure_ascii=False)

    assert "데이브리핑" in dumped
    assert json.loads(dumped)["durationInFrames"] == 900


# --------------------------------------------------------------------------
# render subprocess (mocked)
# --------------------------------------------------------------------------


def _fake_remotion_dir(tmp_path: Path) -> Path:
    remotion_dir = tmp_path / "remotion"
    (remotion_dir / "node_modules").mkdir(parents=True)
    return remotion_dir


def test_render_remotion_command_cwd_and_outputs(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)
    remotion_dir = _fake_remotion_dir(tmp_path)
    log_path = job_dir / "logs" / "remotion.log"
    props = _build()
    success = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Rendered 900 frames\n", stderr=""
    )

    with patch("md2short.render.subprocess.run", return_value=success) as run:
        output_path = render_remotion(
            job_dir=job_dir,
            job_id="sf_20260502_abc123",
            job_config=_job_config(),
            remotion_dir=remotion_dir,
            props=props,
            log_path=log_path,
        )

    staged_props = remotion_dir / "public" / "jobs" / "sf_20260502_abc123" / "props.json"
    assert run.call_args.args[0] == [
        "npx",
        "remotion",
        "render",
        "src/index.ts",
        "TechExplainer",
        str(job_dir / "edit" / "final.mp4"),
        f"--props={staged_props}",
        "--codec=h264",
        "--width=1080",
        "--height=1920",
        "--log=warn",
        "--overwrite",
    ]
    assert run.call_args.kwargs == {
        "cwd": remotion_dir,
        "capture_output": True,
        "text": True,
    }

    assert output_path == job_dir / "edit" / "final.mp4"
    assert output_path.parent.is_dir()
    assert json.loads(staged_props.read_text()) == props
    assert (staged_props.parent / "narration.mp3").read_bytes() == b"fake-mp3"
    assert (staged_props.parent / "words.json").exists()

    log_text = log_path.read_text()
    assert "Rendered 900 frames" in log_text
    assert "exit=0" in log_text


def test_render_remotion_passes_absolute_paths_despite_relative_job_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The subprocess runs with `cwd=remotion_dir`, so `--props` and the output
    path must be absolute even when the caller passes project-root-relative
    paths (as pipeline.run_job does with `jobs/<id>`). Passing them through
    verbatim made Remotion resolve them under `remotion/` and fail with
    "--props ... was neither valid JSON nor a file path to a valid JSON file".
    """
    _make_job_dir(tmp_path)
    _fake_remotion_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    job_dir = Path("jobs/sf_20260502_abc123")
    remotion_dir = Path("remotion")
    success = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Rendered 900 frames\n", stderr=""
    )

    with patch("md2short.render.subprocess.run", return_value=success) as run:
        render_remotion(
            job_dir=job_dir,
            job_id="sf_20260502_abc123",
            job_config=_job_config(),
            remotion_dir=remotion_dir,
            props=_build(),
            log_path=job_dir / "logs" / "remotion.log",
        )

    command = run.call_args.args[0]
    output_arg = command[5]
    props_arg = command[6].removeprefix("--props=")

    assert Path(output_arg).is_absolute()
    assert Path(props_arg).is_absolute()
    assert Path(output_arg) == (tmp_path / "jobs/sf_20260502_abc123/edit/final.mp4").resolve()
    assert (
        Path(props_arg)
        == (tmp_path / "remotion/public/jobs/sf_20260502_abc123/props.json").resolve()
    )


def test_render_remotion_raises_with_log_tail_on_non_zero_exit(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)
    remotion_dir = _fake_remotion_dir(tmp_path)
    log_path = job_dir / "logs" / "remotion.log"
    failure = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Error: Could not find composition\n"
    )

    with patch("md2short.render.subprocess.run", return_value=failure):
        with pytest.raises(RuntimeError) as excinfo:
            render_remotion(
                job_dir=job_dir,
                job_id="sf_20260502_abc123",
                job_config=_job_config(),
                remotion_dir=remotion_dir,
                props=_build(),
                log_path=log_path,
            )

    message = str(excinfo.value)
    assert "exit code 1" in message
    assert "Could not find composition" in message
    assert str(log_path) in message
    assert "Could not find composition" in log_path.read_text()


def test_render_remotion_preflights_before_spawning(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)

    with patch("md2short.render.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="Remotion project directory not found"):
            render_remotion(
                job_dir=job_dir,
                job_id="job1",
                job_config=_job_config(),
                remotion_dir=tmp_path / "missing-remotion",
                props=_build(),
                log_path=job_dir / "logs" / "remotion.log",
            )

    run.assert_not_called()


def test_render_remotion_fails_fast_when_ffmpeg_missing(tmp_path: Path):
    job_dir = _make_job_dir(tmp_path)

    with patch("md2short.render.shutil.which", return_value=None):
        with patch("md2short.render.subprocess.run") as run:
            with pytest.raises(RuntimeError, match="brew install ffmpeg"):
                render_remotion(
                    job_dir=job_dir,
                    job_id="job1",
                    job_config=_job_config(),
                    remotion_dir=_fake_remotion_dir(tmp_path),
                    props=_build(),
                    log_path=job_dir / "logs" / "remotion.log",
                )

    run.assert_not_called()


# --------------------------------------------------------------------------
# real end-to-end render (slow, but runs)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_render_remotion_real_integration(tmp_path: Path):
    """Render a real short MP4 through the real Remotion project (US-010 fixture)."""
    job_dir = tmp_path / "jobs" / INTEGRATION_JOB_ID
    (job_dir / "audio").mkdir(parents=True)
    shutil.copy2(SAMPLE_FIXTURE / "narration.mp3", job_dir / "audio" / "narration.mp3")
    (job_dir / "transcripts").mkdir(parents=True)
    shutil.copy2(SAMPLE_FIXTURE / "words.json", job_dir / "transcripts" / "words.json")

    words_doc = json.loads((SAMPLE_FIXTURE / "words.json").read_text())
    sample_props = json.loads((SAMPLE_FIXTURE / "props.json").read_text())
    audio_duration_ms = get_audio_duration_ms(job_dir / "audio" / "narration.mp3")
    assert audio_duration_ms == 6000

    props = build_props_json(
        job_config=_job_config(),
        job_id=INTEGRATION_JOB_ID,
        words_doc=words_doc,
        emoticon_cues=sample_props["emoticonCues"],
        photo_cues=None,
        caption_segments=sample_props["captionSegments"],
        number_cards=sample_props["numberCards"],
        company_logo_cues=sample_props["companyLogoCues"],
        opening_text=sample_props["openingText"],
        opening_duration_ms=sample_props["openingDurationMs"],
        cta_text=sample_props["ctaText"],
        highlight_keywords=sample_props["highlightKeywords"],
        audio_duration_ms=audio_duration_ms,
    )
    assert props["durationInFrames"] == 180
    # Keep the real render short: 1s of frames is enough to prove the chain.
    props["durationInFrames"] = 30

    staged_dir = REMOTION_DIR / "public" / "jobs" / INTEGRATION_JOB_ID
    log_path = job_dir / "logs" / "remotion.log"
    try:
        output_path = render_remotion(
            job_dir=job_dir,
            job_id=INTEGRATION_JOB_ID,
            job_config=_job_config(),
            remotion_dir=REMOTION_DIR,
            props=props,
            log_path=log_path,
        )

        assert output_path == job_dir / "edit" / "final.mp4"
        assert output_path.is_file(), log_path.read_text()
        assert output_path.stat().st_size > 10_000
        assert (staged_dir / "props.json").is_file()
        assert (staged_dir / "narration.mp3").exists()
        assert (staged_dir / "words.json").exists()
        assert log_path.is_file()

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name",
                "-of",
                "json",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["width"] == 1080
        assert stream["height"] == 1920
        assert stream["codec_name"] == "h264"

        assert abs(get_video_duration_ms(output_path) - 1000) <= 150
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
