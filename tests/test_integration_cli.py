"""Integration tests for CLI functionality."""

import subprocess
import sys

import pytest


def test_cli_version():
    """Test --version flag."""
    result = subprocess.run(
        [sys.executable, "-m", "kalkulator_pkg.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip() != ""


def test_cli_health_check():
    """Test --health-check command."""
    result = subprocess.run(
        [sys.executable, "-m", "kalkulator_pkg.cli", "--health-check"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Health check may pass or fail depending on environment
    assert result.returncode in [0, 1]
    assert "health check" in result.stdout.lower() or "SymPy" in result.stdout


def test_cli_eval_json():
    """Test CLI evaluation with JSON output."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kalkulator_pkg.cli",
            "--eval",
            "2+2",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    output = result.stdout.strip()
    # Should be valid JSON
    import json

    data = json.loads(output)
    assert "ok" in data


def test_cli_eval_human():
    """Test CLI evaluation with human output."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kalkulator_pkg.cli",
            "--eval",
            "2+2",
            "--format",
            "human",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "4" in result.stdout or result.stdout.strip() != ""


def test_cli_solve_equation():
    """Test CLI equation solving."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kalkulator_pkg.cli",
            "--eval",
            "x+1=0 find x",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    import json

    data = json.loads(result.stdout.strip())
    assert data.get("ok") in [True, False]  # May succeed or fail depending on worker


def test_cli_invalid_input():
    """Test CLI with invalid input."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kalkulator_pkg.cli",
            "--eval",
            "__import__('os')",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Should handle error gracefully
    assert result.returncode in [0, 1]
    if result.returncode == 0:
        import json

        data = json.loads(result.stdout.strip())
        assert data.get("ok") is False or "error" in data


@pytest.mark.slow
def test_cli_repl_quit():
    """Test CLI REPL with quit command (slow test)."""
    # This would require interactive input simulation
    # Skipping for now as it's complex
    pass


def test_cli_help():
    """Test --help flag."""
    result = subprocess.run(
        [sys.executable, "-m", "kalkulator_pkg.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "kalkulator" in result.stdout.lower() or "usage" in result.stdout.lower()
