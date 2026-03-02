"""Headless smoke flow built on top of the services facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .engine.monte_carlo import MonteCarloSimulator
from .services import MonteCarloSettings, StepSelection, apply_steps, build_trial_state, load_project, run_monte_carlo

DIM_COLUMNS = ("L_ab", "L_dc", "H_ad", "H_bc", "L", "H")


def _pick_default_dims_instance(app_state: Any) -> str | None:
    """Pick the first geometry instance as default for headless dims output."""
    inst_ids = app_state.geom.get_instance_ids()
    return inst_ids[0] if inst_ids else None


def _resolve_dims_instance(app_state: Any, requested_inst_id: str | None) -> str | None:
    if requested_inst_id:
        if requested_inst_id in app_state.geom.instances:
            return requested_inst_id
        fallback = _pick_default_dims_instance(app_state)
        print(
            f"[headless] --dims-inst '{requested_inst_id}' was not found; "
            f"falling back to '{fallback}'."
        )
        return fallback
    return _pick_default_dims_instance(app_state)


def _inject_realized_dims_columns(
    app_state: Any,
    results: pd.DataFrame,
    steps_mask: list[bool],
    n_trials: int,
    seed: int,
    dims_inst: str | None,
) -> pd.DataFrame:
    out = results.copy()
    if out.empty:
        for col in DIM_COLUMNS:
            out[col] = float("nan")
        return out

    dims_rows: list[dict[str, float]] = []
    for trial in range(n_trials):
        state = build_trial_state(app_state, steps_mask, trial, seed)
        dims = state.get_realized_dims(dims_inst) if dims_inst else {}
        dims_rows.append({col: float(dims.get(col, float("nan"))) for col in DIM_COLUMNS})

    dims_df = pd.DataFrame(dims_rows)
    for col in DIM_COLUMNS:
        out[col] = dims_df[col].values if col in dims_df.columns else float("nan")
    return out


def run_headless_smoke(csv_path: str | Path, n_trials: int = 100, seed: int = 42) -> int:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return 2

    app_state = load_project(path)
    steps_mask = [False] * len(app_state.flow.steps)
    if steps_mask:
        steps_mask[0] = True

    apply_steps(app_state, StepSelection(steps_mask=steps_mask, seed=seed))
    results = run_monte_carlo(
        app_state,
        MonteCarloSettings(n_trials=n_trials, steps_mask=steps_mask, seed=seed),
    )
    return 0 if len(results) == n_trials else 1


def run_headless_smoke_results(
    csv_path: str | Path,
    n_trials: int = 100,
    seed: int = 42,
    dims_inst: str | None = None,
    out_dir: Path | None = None,
    trace: bool = False,
) -> tuple[int, pd.DataFrame | None]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return 2, None

    app_state = load_project(path)
    steps_mask = [False] * len(app_state.flow.steps)
    if steps_mask:
        steps_mask[0] = True

    sim = MonteCarloSimulator(app_state.geom, app_state.flow)
    results = sim.run(n_trials=n_trials, steps_mask=steps_mask, seed=seed, out_dir=out_dir, trace=trace)
    resolved_dims_inst = _resolve_dims_instance(app_state, dims_inst)
    results = _inject_realized_dims_columns(
        app_state,
        results,
        steps_mask=steps_mask,
        n_trials=n_trials,
        seed=seed,
        dims_inst=resolved_dims_inst,
    )
    rc = 0 if len(results) == n_trials else 1
    return rc, results


__all__ = [
    "DIM_COLUMNS",
    "run_headless_smoke",
    "run_headless_smoke_results",
    "_inject_realized_dims_columns",
    "_pick_default_dims_instance",
]
