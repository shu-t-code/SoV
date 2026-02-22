"""Headless smoke flow built on top of the services facade."""

from __future__ import annotations

from pathlib import Path

from ..services import MonteCarloSettings, StepSelection, apply_steps, load_project, run_monte_carlo


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


__all__ = ["run_headless_smoke"]
