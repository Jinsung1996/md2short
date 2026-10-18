"""Tests for `md2short run` (US-014, architecture.md section 6.1/6.4).

`pipeline.run_job` is mocked throughout -- these tests exercise only CLI-level
argument validation and exit-code mapping, never real API calls or renders.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from md2short import cli, config
from md2short.cli import app

runner = CliRunner()


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text("# 제목\n\n본문입니다.\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _default_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default, both API keys resolve fine so happy-path tests don't fail."""
    monkeypatch.setattr(config, "get_elevenlabs_api_key", lambda: "fake-elevenlabs-key")
    monkeypatch.setattr(config, "get_openai_api_key", lambda: "fake-openai-key")


def test_lang_other_than_ko_rejected(sample_markdown: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown), "--lang", "en"])

    assert result.exit_code == 1
    assert "lang" in result.output.lower() or "ko" in result.output.lower()
    mock_run_job.assert_not_called()


@pytest.mark.parametrize("chunk_size", ["0", "25"])
def test_chunk_size_out_of_range_rejected(
    sample_markdown: Path, monkeypatch: pytest.MonkeyPatch, chunk_size: str
) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown), "--chunk-size", chunk_size])

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" not in result.output
    mock_run_job.assert_not_called()


def test_engine_other_than_remotion_rejected(sample_markdown: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown), "--engine", "ffmpeg"])

    assert result.exit_code == 1
    mock_run_job.assert_not_called()


def test_visual_mode_photo_without_openai_key_rejected(
    sample_markdown: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)
    monkeypatch.setattr(config, "get_openai_api_key", MagicMock(side_effect=RuntimeError("missing key")))

    result = runner.invoke(app, ["run", str(sample_markdown), "--visual-mode", "photo"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    mock_run_job.assert_not_called()


def test_missing_markdown_file_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    missing_path = tmp_path / "does_not_exist.md"
    result = runner.invoke(app, ["run", str(missing_path)])

    assert result.exit_code == 1
    mock_run_job.assert_not_called()


def test_missing_elevenlabs_key_rejected(sample_markdown: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run_job = MagicMock()
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)
    monkeypatch.setattr(config, "get_elevenlabs_api_key", MagicMock(side_effect=RuntimeError("missing key")))

    result = runner.invoke(app, ["run", str(sample_markdown)])

    assert result.exit_code == 1
    mock_run_job.assert_not_called()


def test_happy_path_exits_zero_and_prints_job_id(
    sample_markdown: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_run_job = MagicMock(
        return_value={
            "job_id": "sf_20261011_abcdef",
            "job_dir": Path("jobs/sf_20261011_abcdef"),
            "status": "qc_passed",
            "qc_report": {"overall_status": "qc_passed"},
            "output_path": "output/ret.mp4",
        }
    )
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown)])

    assert result.exit_code == 0
    assert "sf_20261011_abcdef" in result.output
    mock_run_job.assert_called_once()


def test_run_job_exception_maps_to_exit_code_2(
    sample_markdown: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_run_job = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown)])

    assert result.exit_code == 2
    assert "boom" in result.output


def test_empty_output_passes_none_to_run_job(
    sample_markdown: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_run_job = MagicMock(
        return_value={
            "job_id": "sf_20261011_abcdef",
            "job_dir": Path("jobs/sf_20261011_abcdef"),
            "status": "qc_passed",
            "qc_report": {"overall_status": "qc_passed"},
            "output_path": None,
        }
    )
    monkeypatch.setattr(cli.pipeline, "run_job", mock_run_job)

    result = runner.invoke(app, ["run", str(sample_markdown), "--output", ""])

    assert result.exit_code == 0
    _, kwargs = mock_run_job.call_args
    assert kwargs["output_path"] is None
