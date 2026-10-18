"""Tests for `md2short clean` and its duration parser (US-014, architecture.md section 6.3)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from md2short.cli import _parse_duration_seconds, app

runner = CliRunner()


@pytest.mark.parametrize(
    ("value", "expected_seconds"),
    [
        ("24h", 24 * 3600),
        ("7d", 7 * 86400),
        ("30m", 30 * 60),
        ("3600s", 3600),
        ("3600", 3600),
    ],
)
def test_parse_duration_seconds_valid(value: str, expected_seconds: int) -> None:
    assert _parse_duration_seconds(value) == expected_seconds


@pytest.mark.parametrize("value", ["", "abc", "24x", "-5h", "5 h", "h5"])
def test_parse_duration_seconds_malformed_raises(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_duration_seconds(value)


def test_clean_malformed_older_than_exits_1(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()

    result = runner.invoke(app, ["clean", "--older-than", "garbage", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 1


def _make_job_dir(jobs_root: Path, job_id: str, age_seconds: float) -> Path:
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    status_path = job_dir / "status.json"
    status_path.write_text("{}", encoding="utf-8")
    old_time = time.time() - age_seconds
    os.utime(status_path, (old_time, old_time))
    return job_dir


def test_clean_dry_run_does_not_delete(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    old_job = _make_job_dir(jobs_root, "sf_old", age_seconds=48 * 3600)

    result = runner.invoke(
        app, ["clean", "--older-than", "24h", "--jobs-root", str(jobs_root), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "sf_old" in result.output
    assert old_job.exists()


def test_clean_deletes_old_job_but_keeps_fresh_job(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    old_job = _make_job_dir(jobs_root, "sf_old", age_seconds=48 * 3600)
    fresh_job = _make_job_dir(jobs_root, "sf_fresh", age_seconds=10)

    result = runner.invoke(app, ["clean", "--older-than", "24h", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 0
    assert not old_job.exists()
    assert fresh_job.exists()


def test_clean_falls_back_to_directory_mtime_when_status_json_missing(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_dir = jobs_root / "sf_no_status"
    job_dir.mkdir()
    old_time = time.time() - 48 * 3600
    os.utime(job_dir, (old_time, old_time))

    result = runner.invoke(app, ["clean", "--older-than", "24h", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 0
    assert not job_dir.exists()
