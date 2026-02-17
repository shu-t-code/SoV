# Refactor notes: shrinking `legacy_impl.py`

## Current compatibility surface in `legacy_impl.py`

`legacy_impl.py` should stay a thin facade that only re-exports symbols from dedicated modules.

## Move map (source of truth)

- **`core_models.py`**
  - `AssemblyState`
  - `DistributionSampler`
  - `FlowModel`
  - `GeometryModel`
  - `Validator`
  - `get_world_point`
  - `rpy_to_rotation_matrix`

- **`process_engine.py`**
  - `ProcessEngine`
  - Step-application behavior (`apply_step` and per-op handlers)
  - Multi-step apply helper (`apply_steps`)

- **`monte_carlo.py`**
  - `MonteCarloSimulator`
  - Trial state reconstruction helper (`build_state_for_trial`)
  - Pair-distance trial helper (`run_pair_distance_trials`)
  - CLI/analysis utility (`print_all_edge_stds_after_cutting`)

- **`visualize.py`**
  - `DistanceHistogramWidget`
  - `InteractivePointSelector`
  - `MatplotlibVisualizer`
  - `Open3DVisualizer`

- **`util_logging.py`**
  - `FileChangeHandler`
  - `HAS_WATCHDOG`
  - `Observer`
  - `log_env`
  - `setup_font`

- **`io_csv.py`**
  - `csv_to_nested_dict`
  - `load_data_from_csv`
  - `nested_dict_to_csv_rows`

- **`env.py`**
  - `USE_OPEN3D`

## MainWindow dependency direction

- `main_window.py` imports directly from dedicated modules (`core_models`, `process_engine`, `monte_carlo`, `visualize`, `io_csv`, `util_logging`).
- UI code should not rely on `legacy_impl.py` for runtime behavior.
- `legacy_impl.py` remains for backward-compatible imports only.
