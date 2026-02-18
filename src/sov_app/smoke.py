"""Backward-compatible smoke helpers wrapping the headless runner."""

from __future__ import annotations

from pathlib import Path

from .config import HeadlessConfig, MCConfig
from .headless import run_headless


def run_headless_smoke(csv_path: str | Path, n_trials: int = 100, seed: int = 42) -> int:
    config = HeadlessConfig(
        csv_path=Path(csv_path).expanduser(),
        mc_config=MCConfig(n=n_trials, seed=seed),
    )
    return run_headless(config)


__all__ = ["run_headless_smoke"]
