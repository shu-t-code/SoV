"""Legacy compatibility facade.

This module intentionally re-exports symbols from the refactored modules so
older imports keep working while the package uses the src-layout modules.
"""

from __future__ import annotations

import warnings

warnings.warn("Deprecated compatibility module; import from sov_app.services or dedicated modules.", DeprecationWarning, stacklevel=2)

from .core_models import (
    AssemblyState,
    DistributionSampler,
    FlowModel,
    GeometryModel,
    Validator,
    get_world_point,
    rpy_to_rotation_matrix,
)
from .env import USE_OPEN3D
from .io_csv import csv_to_nested_dict, load_data_from_csv, nested_dict_to_csv_rows
from .monte_carlo import (
    MonteCarloSimulator,
    build_state_for_trial,
    print_all_edge_stds_after_cutting,
    run_pair_distance_trials,
)
from .process_engine import ProcessEngine
from .util_logging import FileChangeHandler, HAS_WATCHDOG, Observer, log_env, setup_font
from .visualize import DistanceHistogramWidget, InteractivePointSelector, MatplotlibVisualizer, Open3DVisualizer

__all__ = [
    "AssemblyState",
    "DistanceHistogramWidget",
    "DistributionSampler",
    "FileChangeHandler",
    "FlowModel",
    "GeometryModel",
    "HAS_WATCHDOG",
    "InteractivePointSelector",
    "MatplotlibVisualizer",
    "MonteCarloSimulator",
    "build_state_for_trial",
    "Observer",
    "Open3DVisualizer",
    "ProcessEngine",
    "USE_OPEN3D",
    "Validator",
    "csv_to_nested_dict",
    "get_world_point",
    "load_data_from_csv",
    "log_env",
    "nested_dict_to_csv_rows",
    "print_all_edge_stds_after_cutting",
    "run_pair_distance_trials",
    "rpy_to_rotation_matrix",
    "setup_font",
]
