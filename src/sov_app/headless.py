"""Headless CLI workflows that avoid importing GUI dependencies."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .services import MonteCarloSettings, StepSelection, apply_steps, load_project, run_monte_carlo


def run_headless_smoke(
    csv_path: str | Path,
    n_trials: int = 100,
    seed: int = 42,
    out_dir: str | Path | None = None,
    no_open3d: bool = False,
) -> int:
    started_ts = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output = Path(out_dir).expanduser() if out_dir is not None else None
    output_files: list[str] = []
    csv_out_path: str | None = None

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    def _write_summary(status: str, *, error: str | None = None, tb: str | None = None) -> None:
        if output is None:
            return
        summary_path = output / "summary.json"
        finished_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "status": status,
            "csv_path": str(csv_out_path or ""),
            "mc_n": int(n_trials),
            "seed": int(seed) if seed is not None else None,
            "no_open3d": bool(no_open3d),
            "out_dir": str(output),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": float(time.perf_counter() - started_ts),
            "output_files": sorted({*output_files, "summary.json"}),
        }
        if error is not None:
            payload["error"] = error
        if tb is not None:
            payload["traceback"] = tb

        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    path = Path(csv_path).expanduser()
    if not path.exists():
        _write_summary(status="error", error=f"CSV file not found: {path}")
        return 2

    try:
        app_state = load_project(path)
        steps_mask = [False] * len(app_state.flow.steps)
        if steps_mask:
            steps_mask[0] = True

        apply_steps(app_state, StepSelection(steps_mask=steps_mask, seed=seed))
        results = run_monte_carlo(
            app_state,
            MonteCarloSettings(n_trials=n_trials, steps_mask=steps_mask, seed=seed),
        )

        rc = 0 if len(results) == n_trials else 1
        if output is not None:
            csv_file = output / "mc_results.csv"
            results.to_csv(csv_file, index=False)
            csv_out_path = str(csv_file)
            output_files.append(csv_file.name)
        _write_summary(status="ok" if rc == 0 else "error")
        return rc
    except Exception as exc:
        _write_summary(
            status="error",
            error=str(exc),
            tb="".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)),
        )
        return 1


__all__ = ["run_headless_smoke"]
