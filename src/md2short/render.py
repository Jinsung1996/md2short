"""Remotion staging, props.json assembly, and the render subprocess (F-06).

See docs/architecture.md section 5.5 ("[F-06] Remotion 렌더링 + ffmpeg/ffprobe
보조"), section 2.3 (the "Remotion assets는 `render_remotion()`이 다음을 보장함"
contract) and section 4.3 (the `props.json` shape).

Division of labour mirrored from architecture.md section 1.1/1.3:
  - this module *stages* assets and *writes* props.json, then shells out to
    `npx remotion render`. It never composites captions/overlays with ffmpeg.
  - ffmpeg/ffprobe stay in the auxiliary role: duration measurement only
    (`get_audio_duration_ms`, `get_video_duration_ms`; the latter is the
    primitive QC in US-012 consumes).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "ensure_ffmpeg_available",
    "ensure_remotion_available",
    "get_audio_duration_ms",
    "get_video_duration_ms",
    "normalize_color",
    "stage_job_assets",
    "build_props_json",
    "render_remotion",
]

logger = logging.getLogger(__name__)

# Fixed 9:16 output contract -- mirrors remotion/src/layout.ts.
FPS = 30
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

# remotion/src/index.ts registers Root.tsx, whose single composition id is this.
REMOTION_ENTRY = "src/index.ts"
COMPOSITION_ID = "TechExplainer"

# `ffprobe -v error -show_entries format=duration -of json <path>`
FFPROBE_DURATION_ARGS = ("-v", "error", "-show_entries", "format=duration", "-of", "json")

# Project-level (NOT per-job) public asset dirs from architecture.md section 2.3,
# mapped to their repo-root-relative source. They are set up once; a missing one
# degrades the render (text wordmark instead of the logo, etc.) but must never
# fail it.
#
# `emoticon` is nested one level deeper than its public/ name: the 60 PNGs live
# in `<repo>/emoticon/emoticon/`, not `<repo>/emoticon/`. The real link US-010
# created by hand is `remotion/public/emoticon -> ../../emoticon/emoticon`, and a
# recreated link must resolve to that same directory -- pointing it at the parent
# would serve `emoticon/emoticon/<file>.png` paths that the cue planner never
# emits, so every emoticon would 404.
PROJECT_PUBLIC_ASSETS: dict[str, str] = {
    "emoticon": "emoticon/emoticon",
    "img": "img",
    "company": "company",
    "font": "font",
}

# How much of the render log is inlined into the raised RuntimeError.
_LOG_TAIL_CHARS = 2000


def ensure_ffmpeg_available() -> None:
    """Pre-flight: both `ffmpeg` and `ffprobe` must be on PATH.

    architecture.md section 5.5: "공통 pre-flight: ffmpeg, ffprobe 필수."
    """
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not found on PATH -- "
            "install via `brew install ffmpeg` (macOS) or your distro's ffmpeg package"
        )


def ensure_remotion_available(remotion_dir: Path) -> None:
    """Pre-flight: the Remotion project, its node_modules, and `npx` must exist.

    architecture.md section 5.5: "Remotion pre-flight: remotion/,
    remotion/node_modules, npx 확인."
    """
    if not remotion_dir.is_dir():
        raise RuntimeError(f"Remotion project directory not found: {remotion_dir}")

    node_modules = remotion_dir / "node_modules"
    if not node_modules.is_dir():
        raise RuntimeError(
            f"Remotion dependencies not installed: {node_modules} is missing -- "
            f"run `npm install` in {remotion_dir}"
        )

    if shutil.which("npx") is None:
        raise RuntimeError(
            "npx not found on PATH -- install Node.js (`brew install node`) so that "
            "`npx remotion render` can run"
        )


def _probe_duration_ms(media_path: Path) -> int:
    """Measure a media file's duration in ms via ffprobe's `format=duration`."""
    command = ["ffprobe", *FFPROBE_DURATION_ARGS, str(media_path)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {media_path} (exit {result.returncode}): "
            f"{(result.stderr or '').strip()}"
        )

    payload = json.loads(result.stdout)
    seconds = float(payload["format"]["duration"])
    return round(seconds * 1000)


def get_audio_duration_ms(audio_path: Path) -> int:
    """Return `audio_path`'s duration in milliseconds (ffprobe).

    This is the single source of the timeline length (architecture.md section
    1.4: "음성 길이 기준 타임라인").
    """
    return _probe_duration_ms(audio_path)


def get_video_duration_ms(video_path: Path) -> int:
    """Return `video_path`'s duration in milliseconds (ffprobe).

    Used by QC's `final_mp4_vs_audio` check (architecture.md section 5.6).
    """
    return _probe_duration_ms(video_path)


def normalize_color(color: str) -> str:
    """Mirror `normalizeColor()` in remotion/src/types.ts exactly.

    JobConfig stores `0xFFFFFF`, props.json/CSS wants `#FFFFFF`; anything that
    already starts with `#` is passed through untouched. Hex digit case is
    preserved on both sides (CSS hex colors are case-insensitive, so `#ffffff`
    and `#FFFFFF` paint identically).
    """
    return f"#{color[2:]}" if color.startswith("0x") else color


def _clear_dest(source: Path, dest: Path) -> None:
    """Validate `source` and remove a pre-existing `dest` (dangling links too)."""
    if not source.exists():
        raise RuntimeError(f"cannot stage missing asset: {source}")

    if dest.is_symlink() or dest.exists():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()


def _stage_file(source: Path, dest: Path) -> None:
    """Copy a single staged file into `public/`.

    Measured behavior of Remotion 4.0.534's render-time static server: a
    *file* symlink under `public/` is not served (`404`, which aborts the
    render), while a *directory* symlink is. So the two small per-job files
    (narration.mp3, words.json) are real copies -- that is the "copy fallback"
    half of architecture.md section 2.3, taken unconditionally because the
    symlink half provably does not work for files.
    """
    _clear_dest(source, dest)
    shutil.copy2(source, dest)


def _link_or_copy_dir(source: Path, dest: Path) -> None:
    """Symlink a directory into `public/`, falling back to a copy on OSError.

    Directory symlinks are served correctly, so large payloads (photos/,
    emoticon/, img/, company/) are linked rather than duplicated. The copy
    fallback covers cross-filesystem and no-symlink-permission cases
    (architecture.md section 2.3: "symlink 또는 copy fallback").
    """
    _clear_dest(source, dest)
    try:
        dest.symlink_to(source.resolve())
    except OSError:
        shutil.copytree(source, dest)


def _ensure_project_public_assets(remotion_public_dir: Path) -> None:
    """Best-effort check of the project-level `public/` asset dirs.

    `emoticon`/`img`/`company`/`font` are staged once for the whole project
    (architecture.md section 2.3), not per job, so an existing entry is left
    exactly as it is. A missing entry is linked from its `PROJECT_PUBLIC_ASSETS`
    source under the repo root when that dir exists, and otherwise only logged
    -- a missing logo or company directory degrades the video, it must not abort
    the render.
    """
    project_root = remotion_public_dir.parent.parent

    for name, relative_source in PROJECT_PUBLIC_ASSETS.items():
        dest = remotion_public_dir / name
        if dest.exists():
            continue

        source = project_root / relative_source
        if not source.is_dir():
            logger.warning(
                "remotion public asset %r is missing (no %s either) -- "
                "render continues with degraded visuals",
                str(dest),
                source,
            )
            continue

        try:
            _link_or_copy_dir(source, dest)
            logger.info("staged project-level remotion asset %s -> %s", dest, source)
        except (OSError, RuntimeError) as exc:  # never fail the render over this
            logger.warning("could not stage remotion public asset %s: %s", dest, exc)


def stage_job_assets(job_dir: Path, job_id: str, remotion_public_dir: Path) -> Path:
    """Stage one job's render inputs under `remotion_public_dir/jobs/<job_id>/`.

    Copies `narration.mp3` and `words.json` (see `_stage_file` for why files are
    not symlinked). props.json is written separately by `render_remotion`.
    Returns the staged job dir.

    Photos are deliberately NOT staged here. `pipeline.run_job()` hands
    `photo_engine.plan_photo_cues()` an output dir of
    `remotion/public/jobs/<job_id>/photos` (architecture.md section 2.3/4.4), so
    in photo mode the PNGs are written straight into this same staged dir before
    the render and the cue paths are already public-relative. Nothing ever
    populates `job_dir/edit/photos` -- `edit/` only receives the cue JSON -- so
    the branch that used to copy from there was unreachable. The
    `mkdir(parents=True, exist_ok=True)` below preserves a photos/ dir
    photo_engine already wrote into `staged_dir`.
    """
    staged_dir = remotion_public_dir / "jobs" / job_id
    staged_dir.mkdir(parents=True, exist_ok=True)

    _stage_file(job_dir / "audio" / "narration.mp3", staged_dir / "narration.mp3")
    _stage_file(job_dir / "transcripts" / "words.json", staged_dir / "words.json")

    _ensure_project_public_assets(remotion_public_dir)

    return staged_dir


def build_props_json(
    job_config: Any,
    job_id: str,
    words_doc: dict,
    emoticon_cues: list[dict] | None,
    photo_cues: list[dict] | None,
    caption_segments: list[dict],
    number_cards: list[dict],
    company_logo_cues: list[dict],
    opening_text: str | None,
    opening_duration_ms: int,
    cta_text: str,
    highlight_keywords: list[str],
    audio_duration_ms: int,
) -> dict:
    """Assemble the `props.json` dict of architecture.md section 4.3.

    `job_id` builds the two `public/`-relative asset paths. `emoticonCues`
    carries whichever visual mode is active: `photo_cues` when it is not None
    (visual_mode=photo), else `emoticon_cues` -- the field name is shared by
    both kinds on purpose (architecture.md section 1.7). `words_doc` is only a
    fallback source for the timeline length when `audio_duration_ms` is not a
    positive measurement.
    """
    if audio_duration_ms <= 0:
        audio_duration_ms = int(words_doc.get("audio_duration_ms") or 0)

    visual_cues = photo_cues if photo_cues is not None else (emoticon_cues or [])

    return {
        "audioFileName": f"jobs/{job_id}/narration.mp3",
        "wordsFileName": f"jobs/{job_id}/words.json",
        "bgColor": normalize_color(job_config.bg_color),
        "chunkSize": job_config.chunk_size,
        "emoticonCues": list(visual_cues),
        "captionSegments": list(caption_segments or []),
        "companyLogoCues": list(company_logo_cues or []),
        "numberCards": list(number_cards or []),
        "highlightKeywords": list(highlight_keywords or []),
        "openingText": opening_text or "",
        "openingDurationMs": int(opening_duration_ms),
        "showOpening": bool(job_config.show_opening),
        "ctaText": cta_text,
        "showOutro": bool(job_config.show_outro),
        "durationInFrames": round(audio_duration_ms / 1000 * FPS),
    }


def render_remotion(
    job_dir: Path,
    job_id: str,
    job_config: Any,
    remotion_dir: Path,
    props: dict,
    log_path: Path,
) -> Path:
    """Stage assets, write props.json, and run `npx remotion render`.

    Returns `job_dir/edit/final.mp4` (architecture.md section 5.5 출력). The
    full subprocess output is written to `log_path` (`logs/remotion.log`) and a
    non-zero exit raises a RuntimeError carrying the log tail.

    `job_config` is accepted for pipeline symmetry (US-013); every
    config-derived value already lives in `props` via `build_props_json`.
    """
    ensure_ffmpeg_available()
    ensure_remotion_available(remotion_dir)

    staged_dir = stage_job_assets(job_dir, job_id, remotion_dir / "public")

    staged_props_path = staged_dir / "props.json"
    staged_props_path.write_text(
        json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_path = job_dir / "edit" / "final.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The subprocess runs with `cwd=remotion_dir`, so every path handed to it
    # must be absolute: `staged_props_path`/`output_path` are relative to the
    # *project root*, and passing them through verbatim made Remotion resolve
    # them under `remotion/` -- it could not find props.json there, reported
    # "--props ... was neither valid JSON nor a file path to a valid JSON file",
    # and would have written final.mp4 to `remotion/jobs/...` as well.
    command = [
        "npx",
        "remotion",
        "render",
        REMOTION_ENTRY,
        COMPOSITION_ID,
        str(output_path.resolve()),
        f"--props={staged_props_path.resolve()}",
        "--codec=h264",
        f"--width={VIDEO_WIDTH}",
        f"--height={VIDEO_HEIGHT}",
        "--log=warn",
        "--overwrite",
    ]

    result = subprocess.run(command, cwd=remotion_dir, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n(cwd={remotion_dir})\nexit={result.returncode}\n"
        f"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    if result.returncode != 0:
        tail = (stderr or stdout).strip()[-_LOG_TAIL_CHARS:]
        raise RuntimeError(
            f"`npx remotion render` failed with exit code {result.returncode} "
            f"(full log: {log_path})\n{tail}"
        )

    return output_path
