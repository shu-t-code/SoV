"""Dedicated headless runner with standard outputs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import HeadlessConfig
from .services import MonteCarloSettings, StepSelection, apply_steps, load_project, run_monte_carlo

logger = logging.getLogger("sov_app")


def _make_steps_mask(n_steps: int) -> list[bool]:
    """Select a deterministic minimal-safe set of steps (first step only)."""
    mask = [False] * n_steps
    if mask:
        mask[0] = True
    return mask


def _build_summary(config: HeadlessConfig, timings: dict[str, float], mc_results: pd.DataFrame, steps_mask: list[bool]) -> dict[str, Any]:
    metric_summary: dict[str, Any] = {}
    for col in mc_results.columns:
        if col == "trial":
            continue
        series = mc_results[col]
        metric_summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return {
        "status": "ok",
        "config": {
            **asdict(config),
            "csv_path": str(config.csv_path),
            "out_dir": str(config.out_dir),
            "mc_config": {
                "n": config.mc_config.n,
                "seed": config.mc_config.seed,
                "effective_seed": config.mc_config.effective_seed,
            },
        },
        "counts": {
            "steps_total": len(steps_mask),
            "steps_selected": int(sum(steps_mask)),
            "trials_requested": config.mc_config.n,
            "trials_completed": int(len(mc_results)),
        },
        "timings_sec": timings,
        "metrics": metric_summary,
    }


def run_headless(config: HeadlessConfig) -> int:
    """Run headless pipeline and persist standardized output artifacts."""
    csv_path = config.csv_path.expanduser()
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return 2

    out_dir = config.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    logger.info("Starting headless run for %s", csv_path)
    logger.debug("Headless configuration: %s", config)

    t0 = time.perf_counter()
    app_state = load_project(csv_path)
    t_load = time.perf_counter() - t0

    steps_mask = _make_steps_mask(len(app_state.flow.steps))
    seed = config.mc_config.effective_seed

    t1 = time.perf_counter()
    apply_steps(app_state, StepSelection(steps_mask=steps_mask, seed=seed))
    t_apply = time.perf_counter() - t1

    t2 = time.perf_counter()
    mc_results = run_monte_carlo(
        app_state,
        MonteCarloSettings(n_trials=config.mc_config.n, steps_mask=steps_mask, seed=seed),
    )
    t_mc = time.perf_counter() - t2

    result_path = out_dir / "mc_results.csv"
    mc_results.to_csv(result_path, index=False)

    timings = {
        "load_project": round(t_load, 6),
        "apply_steps": round(t_apply, 6),
        "run_monte_carlo": round(t_mc, 6),
        "total": round(time.perf_counter() - start, 6),
    }
    summary = _build_summary(config, timings, mc_results, steps_mask)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Headless run complete: %s", out_dir)
    logger.info("Wrote artifacts: %s, %s", summary_path.name, result_path.name)

    if len(mc_results) != config.mc_config.n:
        logger.error("Monte Carlo produced %s rows (expected %s)", len(mc_results), config.mc_config.n)
        return 1
    return 0


__all__ = ["run_headless"]
