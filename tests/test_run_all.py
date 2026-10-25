"""Tests for run_all.py (command-line behaviour only; the steps have their own tests)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "run_all.py"), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_missing_config_fails_cleanly():
    result = run("--config", "does_not_exist.yml")
    assert result.returncode != 0
    assert "Config file not found" in result.stderr


def test_skipping_every_step_runs_nothing():
    result = run("--config", "config.yml", "--skip", "main", "sensitivity", "cells")
    assert result.returncode == 0
    assert result.stdout.count("skipped") == 3
