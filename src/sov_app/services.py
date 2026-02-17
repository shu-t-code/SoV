from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .core_models import AssemblyState, FlowModel, GeometryModel, Validator
from .env import USE_OPEN3D
from .io_csv import load_data_from_csv, nested_dict_to_csv_rows
from .monte_carlo import MonteCarloSimulator, build_state_for_trial as _build_state_for_trial, run_pair_distance_trials
from .process_engine import ProcessEngine
from .visualize import MatplotlibVisualizer, Open3DVisualizer

ProcessEngineFactory = Callable[[GeometryModel, FlowModel, np.random.Generator], ProcessEngine]
DEFAULT_PROCESS_ENGINE_FACTORY: ProcessEngineFactory = ProcessEngine


@dataclass
class AppState:
    geom: GeometryModel
    flow: FlowModel


@dataclass
class MonteCarloSettings:
    n_trials: int
    steps_mask: List[bool]
    seed: int = 42
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY


@dataclass
class VisualizerConfig:
    use_open3d: bool = USE_OPEN3D


def load_csv(path: str | Path) -> AppState:
    csv_path = Path(path).expanduser()
    geom_data, flow_data = load_data_from_csv(csv_path)
    return AppState(geom=GeometryModel(geom_data), flow=FlowModel(flow_data))


def from_dicts(geom_data: dict, flow_data: dict) -> AppState:
    return AppState(geom=GeometryModel(geom_data), flow=FlowModel(flow_data))


def save_csv(state: AppState, path: str | Path) -> None:
    rows = nested_dict_to_csv_rows(state.geom.raw, "geometry") + nested_dict_to_csv_rows(state.flow.raw, "flow")
    pd.DataFrame(rows).to_csv(Path(path).expanduser(), index=False)


def apply_steps(
    state: AppState,
    selected_steps: List[bool],
    seed: int = 42,
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY,
) -> AssemblyState:
    assembly_state = AssemblyState(state.geom)
    rng = np.random.default_rng(seed)
    engine = process_engine_factory(state.geom, state.flow, rng)
    engine.apply_steps(assembly_state, selected_steps)
    return assembly_state


def run_monte_carlo(state: AppState, settings: MonteCarloSettings) -> pd.DataFrame:
    sim = MonteCarloSimulator(state.geom, state.flow, settings.process_engine_factory)
    return sim.run(settings.n_trials, settings.steps_mask, settings.seed)


def build_state_for_trial(
    state: AppState,
    selected_steps: List[bool],
    trial: int,
    seed_base: int,
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY,
) -> AssemblyState:
    return _build_state_for_trial(
        state.geom,
        state.flow,
        selected_steps,
        trial,
        seed_base,
        process_engine_factory,
    )


def run_pair_distance(
    state: AppState,
    selected_steps: List[bool],
    p1_instance: str,
    p1_ref: str,
    p2_instance: str,
    p2_ref: str,
    n_trials: int,
    seed: int,
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY,
) -> np.ndarray:
    return run_pair_distance_trials(
        state.geom,
        state.flow,
        selected_steps,
        p1_instance,
        p1_ref,
        p2_instance,
        p2_ref,
        n_trials,
        seed,
        process_engine_factory,
    )


def build_visualizer(config: VisualizerConfig):
    if config.use_open3d:
        return Open3DVisualizer()
    return MatplotlibVisualizer()


def validate_models(state: AppState) -> List[Dict[str, str]]:
    return Validator.validate(state.geom, state.flow)


__all__ = [
    "AppState",
    "DEFAULT_PROCESS_ENGINE_FACTORY",
    "MonteCarloSettings",
    "ProcessEngineFactory",
    "VisualizerConfig",
    "apply_steps",
    "from_dicts",
    "build_state_for_trial",
    "build_visualizer",
    "load_csv",
    "run_monte_carlo",
    "run_pair_distance",
    "save_csv",
    "validate_models",
]
