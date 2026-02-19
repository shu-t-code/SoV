"""Headless CLI workflows that avoid importing GUI dependencies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .services import MonteCarloSettings, StepSelection, apply_steps, load_project, run_monte_carlo


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_headless_smoke(
    csv_path: str | Path,
    n_trials: int = 100,
    seed: int | None = 42,
    out_dir: str | Path | None = None,
    no_open3d: bool = False,
) -> int:
    path = Path(csv_path).expanduser()
    output = Path(out_dir if out_dir is not None else "out_headless").expanduser()
    output.mkdir(parents=True, exist_ok=True)

    started_at = _iso_now()
    started_perf = perf_counter()
    rc = 1
    notes: str | None = None
    output_files: list[str] = ["summary.json"]

    summary: dict[str, Any] = {
        "status": "error",
        "csv_path": str(path),
        "mc_n": int(n_trials),
        "seed": seed,
        "no_open3d": bool(no_open3d),
        "out_dir": str(output),
        "started_at": started_at,
        "finished_at": started_at,
        "elapsed_seconds": 0.0,
        "output_files": output_files,
        "notes": None,
    }

    try:
        if not path.exists():
            rc = 2
            notes = f"CSV file not found: {path}"
            return rc

        app_state = load_project(path)
        steps_mask = [False] * len(app_state.flow.steps)
        if steps_mask:
            steps_mask[0] = True

        apply_steps(app_state, StepSelection(steps_mask=steps_mask, seed=seed if seed is not None else 42))
        results = run_monte_carlo(
            app_state,
            MonteCarloSettings(n_trials=n_trials, steps_mask=steps_mask, seed=seed if seed is not None else 42),
        )

        csv_name = "mc_results.csv"
        results.to_csv(output / csv_name, index=False)
        output_files.append(csv_name)

        rc = 0 if len(results) == n_trials else 1
        if rc != 0:
            notes = f"Expected {n_trials} MC rows but got {len(results)}"
        return rc
    finally:
        finished_at = _iso_now()
        elapsed_seconds = perf_counter() - started_perf
        summary["status"] = "ok" if rc == 0 else "error"
        summary["finished_at"] = finished_at
        summary["elapsed_seconds"] = elapsed_seconds
        summary["output_files"] = sorted(set(output_files))
        summary["notes"] = notes
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = ["run_headless_smoke"]
