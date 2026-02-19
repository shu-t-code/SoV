from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .core_models import AssemblyState, FlowModel, GeometryModel, Validator
from .io_csv import load_data_from_csv, nested_dict_to_csv_rows
from .monte_carlo import MonteCarloSimulator, build_state_for_trial as _build_state_for_trial, run_pair_distance_trials
from .process_engine import ProcessEngine

ProcessEngineFactory = Callable[[GeometryModel, FlowModel, np.random.Generator], ProcessEngine]
DEFAULT_PROCESS_ENGINE_FACTORY: ProcessEngineFactory = ProcessEngine
MCResults = pd.DataFrame


@dataclass
class AppState:
    geom: GeometryModel
    flow: FlowModel


@dataclass
class StepSelection:
    steps_mask: List[bool]
    seed: int = 42
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY


@dataclass
class MonteCarloSettings:
    n_trials: int
    steps_mask: List[bool]
    seed: int = 42
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY


def load_project(csv_path: str | Path) -> AppState:
    path = Path(csv_path).expanduser()
    geom_data, flow_data = load_data_from_csv(path)
    return AppState(geom=GeometryModel(geom_data), flow=FlowModel(flow_data))


def save_project(state: AppState, csv_path: str | Path) -> None:
    rows = nested_dict_to_csv_rows(state.geom.raw, "geometry") + nested_dict_to_csv_rows(state.flow.raw, "flow")
    pd.DataFrame(rows).to_csv(Path(csv_path).expanduser(), index=False)


def create_project(geom_data: dict[str, Any], flow_data: dict[str, Any]) -> AppState:
    return AppState(geom=GeometryModel(geom_data), flow=FlowModel(flow_data))


def apply_steps(state: AppState, selection: StepSelection | List[bool], seed: int = 42, process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY) -> AssemblyState:
    if isinstance(selection, StepSelection):
        steps_mask = selection.steps_mask
        seed_value = selection.seed
        engine_factory = selection.process_engine_factory
    else:
        steps_mask = selection
        seed_value = seed
        engine_factory = process_engine_factory

    assembly_state = AssemblyState(state.geom)
    rng = np.random.default_rng(seed_value)
    engine = engine_factory(state.geom, state.flow, rng)
    engine.apply_steps(assembly_state, steps_mask)
    return assembly_state


def run_monte_carlo(state: AppState, mc_config: MonteCarloSettings) -> MCResults:
    sim = MonteCarloSimulator(state.geom, state.flow, mc_config.process_engine_factory)
    return sim.run(mc_config.n_trials, mc_config.steps_mask, mc_config.seed)


def build_trial_state(
    state: AppState,
    steps_mask: List[bool],
    trial: int,
    seed_base: int,
    process_engine_factory: ProcessEngineFactory = DEFAULT_PROCESS_ENGINE_FACTORY,
) -> AssemblyState:
    return _build_state_for_trial(
        state.geom,
        state.flow,
        steps_mask,
        trial,
        seed_base,
        process_engine_factory,
    )


def run_pair_distance(
    state: AppState,
    steps_mask: List[bool],
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
        steps_mask,
        p1_instance,
        p1_ref,
        p2_instance,
        p2_ref,
        n_trials,
        seed,
        process_engine_factory,
    )


def validate_models(state: AppState) -> List[Dict[str, str]]:
    return Validator.validate(state.geom, state.flow)


# Backward compatible aliases.
load_csv = load_project
save_csv = save_project
from_dicts = create_project
build_state_for_trial = build_trial_state


__all__ = [
    "AppState",
    "DEFAULT_PROCESS_ENGINE_FACTORY",
    "MCResults",
    "MonteCarloSettings",
    "ProcessEngineFactory",
    "StepSelection",
    "apply_steps",
    "build_state_for_trial",
    "build_trial_state",
    "create_project",
    "from_dicts",
    "load_csv",
    "load_project",
    "run_monte_carlo",
    "run_pair_distance",
    "save_csv",
    "save_project",
    "validate_models",
]
