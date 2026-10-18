"""Smoke tests for the md2short CLI entry point."""

import subprocess

from typer.testing import CliRunner

from md2short.cli import app

runner = CliRunner()


def test_help_exits_zero_via_typer_runner() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "md2short" in result.output.lower() or "Usage" in result.output


def test_help_exits_zero_via_uv_run_subprocess() -> None:
    result = subprocess.run(
        ["uv", "run", "md2short", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
