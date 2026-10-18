"""Tests for `md2short status` (US-014, architecture.md section 6.2)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from md2short.cli import app

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_prints_fields_and_qc_report(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "sf_20261011_abc123"
    _write_json(
        job_dir / "status.json",
        {
            "job_id": "sf_20261011_abc123",
            "status": "qc_passed",
            "started_at": "2026-10-11T00:00:00+00:00",
            "updated_at": "2026-10-11T00:05:00+00:00",
        },
    )
    _write_json(
        job_dir / "qc_report.json",
        {
            "checks": [
                {"check": "sync_last_word_vs_audio", "passed": True},
                {"check": "final_mp4_vs_audio", "passed": False},
            ],
            "cost_estimate": {"total_usd": 0.1234},
            "overall_status": "qc_failed",
        },
    )

    result = runner.invoke(app, ["status", "sf_20261011_abc123", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 0
    assert "sf_20261011_abc123" in result.output
    assert "qc_passed" in result.output
    assert "sync_last_word_vs_audio" in result.output
    assert "PASS" in result.output
    assert "FAIL" in result.output
    assert "0.1234" in result.output
    assert "qc_failed" in result.output


def test_status_without_qc_report_still_prints_status_fields(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "sf_20261011_noqc"
    _write_json(
        job_dir / "status.json",
        {"job_id": "sf_20261011_noqc", "status": "transcribed"},
    )

    result = runner.invoke(app, ["status", "sf_20261011_noqc", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 0
    assert "transcribed" in result.output


def test_status_nonexistent_job_id_exits_1(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()

    result = runner.invoke(app, ["status", "sf_does_not_exist", "--jobs-root", str(jobs_root)])

    assert result.exit_code == 1
