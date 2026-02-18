"""Runtime configuration objects for SoV CLI and headless execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MC_N = 100
DEFAULT_MC_SEED = 42
DEFAULT_HEADLESS_OUT_DIR = Path("out_headless")
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class MCConfig:
    """Monte Carlo execution settings."""

    n: int = DEFAULT_MC_N
    seed: int | None = None

    @property
    def effective_seed(self) -> int:
        return DEFAULT_MC_SEED if self.seed is None else self.seed


@dataclass(frozen=True)
class HeadlessConfig:
    """Top-level config for a headless run."""

    csv_path: Path
    out_dir: Path = field(default_factory=lambda: DEFAULT_HEADLESS_OUT_DIR)
    mc_config: MCConfig = field(default_factory=MCConfig)
    no_open3d: bool = False
    log_level: str = DEFAULT_LOG_LEVEL


__all__ = [
    "DEFAULT_HEADLESS_OUT_DIR",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_MC_N",
    "DEFAULT_MC_SEED",
    "HeadlessConfig",
    "MCConfig",
]
