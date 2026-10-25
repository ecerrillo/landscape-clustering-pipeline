#!/usr/bin/env python3
"""
Run the whole analysis with one command.

Steps, in order (each later step reads the outputs of `main.py`):

1. ``main``         -> main.py: DEM derivatives, terrain classes, HDBSCAN
                       clustering, enriched points and run report.
2. ``sensitivity``  -> sensitivity_analysis.py: robustness of the clustering
                       to positional error, cost-model settings and HDBSCAN
                       parameters.
3. ``cells``        -> cell_density_model.py: cell-scale negative binomial
                       model of prehistoric vs historic finds.

Each step runs as a separate process with the same Python interpreter and
config file, so its behaviour is exactly that of running the script on its
own. The run stops at the first failing step.

Usage
-----
    python run_all.py --config config.yml
    python run_all.py --config config.yml --quick              # reduced sensitivity sweep
    python run_all.py --config config.yml --skip sensitivity   # skip one or more steps
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STEPS = {
    "main": "main.py",
    "sensitivity": "sensitivity_analysis.py",
    "cells": "cell_density_model.py",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config.yml"), help="Path to config.yml")
    ap.add_argument("--quick", action="store_true",
                    help="Pass --quick to sensitivity_analysis.py (fast smoke test, not for final results)")
    ap.add_argument("--skip", nargs="+", choices=list(STEPS), default=[], help="Steps to skip")
    return ap.parse_args()


def run_step(name: str, script: str, config: Path, quick: bool) -> float:
    """Run one step as a subprocess; raise SystemExit if it fails."""
    cmd = [sys.executable, str(ROOT / script), "--config", str(config)]
    if quick and name == "sensitivity":
        cmd.append("--quick")
    print(f"\n=== [{name}] {' '.join(cmd[1:])}", flush=True)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise SystemExit(f"Step '{name}' failed (exit code {result.returncode}); later steps were not run.")
    print(f"=== [{name}] done in {elapsed:.0f} s", flush=True)
    return elapsed


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    started = time.time()
    for name, script in STEPS.items():
        if name in args.skip:
            print(f"\n=== [{name}] skipped", flush=True)
            continue
        run_step(name, script, args.config, args.quick)

    # List the reports written during this run
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reports = Path(config.get("paths", {}).get("output_dir", "outputs")) / "reports"
    new = sorted(p for p in reports.glob("*.md") if p.stat().st_mtime >= started) if reports.exists() else []
    print(f"\nAll steps finished in {time.time() - started:.0f} s.")
    if new:
        print("Reports:")
        for p in new:
            print(f"  {p}")


if __name__ == "__main__":
    main()
