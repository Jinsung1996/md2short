"""md2short Typer CLI entry point.

Implements `run`/`status`/`clean` per docs/architecture.md section 6 (US-014).
All CLI-level input validation (lang/chunk-size/engine/visual-mode/missing
env vars/missing markdown file) exits with code 1 before `pipeline.run_job()`
is ever called; any exception raised *by* `run_job()` is caught here and
mapped to exit code 2 (section 6.4).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import typer
from pydantic import ValidationError

from md2short import config, pipeline, schema

app = typer.Typer(name="md2short", help="Markdown -> Korean 9:16 shortform video CLI.")

_DURATION_RE = re.compile(r"^(\d+)([smhd])?$")


def _parse_duration_seconds(value: str) -> int:
    """Parse a `--older-than` value (e.g. `24h`, `7d`, `30m`, `3600s`, `3600`) into seconds.

    Supports suffixes `s`/`m`/`h`/`d`; no suffix means seconds. Raises
    `ValueError` with a human-readable message on malformed input.
    """
    match = _DURATION_RE.match(value.strip())
    if not match:
        raise ValueError(
            f"invalid duration {value!r} (expected e.g. '24h', '7d', '30m', '3600s', "
            "or a bare integer number of seconds)"
        )
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * multiplier


def _job_age_seconds(job_dir: Path, now: float) -> float:
    """Return how many seconds old `job_dir` is, for `--older-than` comparison.

    Uses `status.json`'s own filesystem mtime when present (it is rewritten
    on every `pipeline.update_status()` call, so its mtime tracks the job's
    last activity). Falls back to the job directory's own mtime when
    `status.json` is missing or unreadable -- a reasonable approximation of
    job age when the status file never got written.
    """
    status_path = job_dir / pipeline.STATUS_FILE_NAME
    try:
        mtime = status_path.stat().st_mtime
    except OSError:
        mtime = job_dir.stat().st_mtime
    return now - mtime


@app.command()
def run(
    markdown_file: str = typer.Argument(..., help="Path to the input markdown file."),
    lang: str = typer.Option("ko", "--lang", "-l", help="Language lock. Only 'ko' is supported."),
    voice: str | None = typer.Option(None, "--voice", "-v", help="ElevenLabs voice id."),
    speed: float = typer.Option(1.0, "--speed", help="TTS speed, 0.7-1.2."),
    template: str = typer.Option(
        "tech_explainer", "--template", "-t", help="Template id (passthrough; single composition)."
    ),
    chunk_size: int = typer.Option(14, "--chunk-size", help="Caption chunk size safety cap, 1-24."),
    bg: str = typer.Option("0xFFFFFF", "--bg", help="Remotion main-section background color."),
    jobs_root: str = typer.Option("jobs", "--jobs-root", help="Job output root directory."),
    engine: str = typer.Option("remotion", "--engine", help="Render engine. Only 'remotion' is supported."),
    visual_mode: str = typer.Option(
        "emoticon", "--visual-mode", help="Visual mode: 'emoticon' or 'photo' (photo requires OPENAI_API_KEY)."
    ),
    model: str = typer.Option("gpt-5-mini", "--model", help="OpenAI model for script generation."),
    output: str = typer.Option(
        "output/ret.mp4", "--output", "-o", help="Final MP4 copy path. Empty string disables the copy."
    ),
) -> None:
    """Generate a shortform video from a markdown file."""
    markdown_path = Path(markdown_file)
    if not markdown_path.exists():
        typer.echo(f"Error: markdown file not found: {markdown_file}", err=True)
        raise typer.Exit(code=1)

    if lang != "ko":
        typer.echo(f"Error: --lang must be 'ko' (got {lang!r}); only Korean is supported.", err=True)
        raise typer.Exit(code=1)

    if engine != "remotion":
        typer.echo(f"Error: --engine must be 'remotion' (got {engine!r}).", err=True)
        raise typer.Exit(code=1)

    if visual_mode not in ("emoticon", "photo"):
        typer.echo(f"Error: --visual-mode must be 'emoticon' or 'photo' (got {visual_mode!r}).", err=True)
        raise typer.Exit(code=1)

    # script_engine can still use OpenAI for script generation in emoticon
    # mode if the key happens to be present, so only treat it as missing
    # when we actually fail to load it -- never force it to None here.
    try:
        openai_api_key: str | None = config.get_openai_api_key()
    except RuntimeError:
        openai_api_key = None

    if visual_mode == "photo" and openai_api_key is None:
        typer.echo("Error: photo 모드는 OPENAI_API_KEY가 필요합니다.", err=True)
        raise typer.Exit(code=1)

    try:
        elevenlabs_api_key = config.get_elevenlabs_api_key()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # --voice wins, then the optional ELEVENLABS_VOICE_ID env override (for
    # accounts whose plan cannot use the documented library voice), then the
    # documented project default.
    voice_id = voice or config.get_elevenlabs_voice_id() or schema.DEFAULT_VOICES["ko"]

    try:
        job_config = schema.JobConfig(
            language=lang,
            voice_id=voice_id,
            voice_speed=speed,
            template_id=template,
            bg_color=bg,
            chunk_size=chunk_size,
            engine=engine,
            visual_mode=visual_mode,
        )
    except ValidationError as exc:
        typer.echo(f"Error: invalid job configuration: {exc}", err=True)
        raise typer.Exit(code=1)

    output_path = None if output == "" else Path(output)

    try:
        result = pipeline.run_job(
            markdown_path,
            job_config,
            jobs_root=Path(jobs_root),
            remotion_dir=Path("remotion"),
            elevenlabs_api_key=elevenlabs_api_key,
            openai_api_key=openai_api_key,
            output_path=output_path,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - any run_job() failure maps to exit code 2
        typer.echo(f"Error: run_job failed: {exc}", err=True)
        raise typer.Exit(code=2)

    qc_report = result.get("qc_report") or {}
    typer.echo(f"job_id: {result['job_id']}")
    typer.echo(f"status: {result['status']}")
    typer.echo(f"qc: {qc_report.get('overall_status', 'unknown')}")
    if result.get("output_path"):
        typer.echo(f"output: {result['output_path']}")
    raise typer.Exit(code=0)


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job id to inspect."),
    jobs_root: str = typer.Option("jobs", "--jobs-root", help="Job output root directory."),
) -> None:
    """Show job status (status.json fields, plus qc_report.json if present)."""
    job_dir = Path(jobs_root) / job_id
    status_path = job_dir / pipeline.STATUS_FILE_NAME
    if not job_dir.exists() or not status_path.exists():
        typer.echo(f"Error: job not found: {job_id} (looked in {job_dir})", err=True)
        raise typer.Exit(code=1)

    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    typer.echo(f"job_id: {status_data.get('job_id', job_id)}")
    for key, value in status_data.items():
        if key == "job_id":
            continue
        typer.echo(f"{key}: {value}")

    qc_report_path = job_dir / "qc_report.json"
    if qc_report_path.exists():
        qc_report = json.loads(qc_report_path.read_text(encoding="utf-8"))
        typer.echo("")
        typer.echo("qc checks:")
        for check in qc_report.get("checks", []):
            name = check.get("check", "unknown")
            verdict = "PASS" if check.get("passed") else "FAIL"
            typer.echo(f"  - {name}: {verdict}")
        total_usd = (qc_report.get("cost_estimate") or {}).get("total_usd")
        if total_usd is not None:
            typer.echo(f"cost estimate (usd): {total_usd:.4f}")
        typer.echo(f"overall_status: {qc_report.get('overall_status', 'unknown')}")

    raise typer.Exit(code=0)


@app.command()
def clean(
    older_than: str = typer.Option(
        ..., "--older-than", help="Age threshold, e.g. '24h', '7d', '30m', '3600s', or a bare integer."
    ),
    jobs_root: str = typer.Option("jobs", "--jobs-root", help="Job output root directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List candidates without deleting them."),
) -> None:
    """Delete job directories older than `--older-than` (or just list them with `--dry-run`)."""
    try:
        cutoff_seconds = _parse_duration_seconds(older_than)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    jobs_root_path = Path(jobs_root)
    if not jobs_root_path.exists():
        typer.echo(f"No jobs_root directory found at {jobs_root_path}; nothing to clean.")
        raise typer.Exit(code=0)

    now = time.time()
    candidates = [
        job_dir
        for job_dir in sorted(jobs_root_path.iterdir())
        if job_dir.is_dir() and _job_age_seconds(job_dir, now) >= cutoff_seconds
    ]

    if dry_run:
        if not candidates:
            typer.echo("No job directories match --older-than; nothing would be deleted.")
        else:
            typer.echo("Would delete:")
            for job_dir in candidates:
                typer.echo(f"  {job_dir}")
        raise typer.Exit(code=0)

    if not candidates:
        typer.echo("No job directories match --older-than; nothing deleted.")
        raise typer.Exit(code=0)

    for job_dir in candidates:
        shutil.rmtree(job_dir)
        typer.echo(f"Deleted: {job_dir}")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
