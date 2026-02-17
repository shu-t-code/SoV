# Refactor dependency map

## 1) `legacy_impl.py` public symbols (compatibility exports)

- `AssemblyState`
- `DistanceHistogramWidget`
- `DistributionSampler`
- `FileChangeHandler`
- `FlowModel`
- `GeometryModel`
- `HAS_WATCHDOG`
- `InteractivePointSelector`
- `MatplotlibVisualizer`
- `MonteCarloSimulator`
- `Observer`
- `Open3DVisualizer`
- `ProcessEngine`
- `USE_OPEN3D`
- `Validator`
- `csv_to_nested_dict`
- `get_world_point`
- `load_data_from_csv`
- `log_env`
- `nested_dict_to_csv_rows`
- `print_all_edge_stds_after_cutting`
- `rpy_to_rotation_matrix`
- `setup_font`

## 2) Symbols imported from `legacy_impl.py` by target files

- `__main__.py`: *(no direct import from `legacy_impl.py`)*
- `main_window.py`: *(no direct import from `legacy_impl.py`)*
- `widgets.py`: *(no direct import from `legacy_impl.py`)*

## 3) `main_window.py` runtime symbols and current sources

- `core_models`: `AssemblyState`, `FlowModel`, `GeometryModel`, `Validator`, `get_world_point`.
- `io_csv`: `load_data_from_csv`, `nested_dict_to_csv_rows`.
- `monte_carlo`: `MonteCarloSimulator`.
- `process_engine`: `ProcessEngine`.
- `util_logging`: `FileChangeHandler`, `HAS_WATCHDOG`, `Observer`.
- `visualize`: `DistanceHistogramWidget`, `InteractivePointSelector`, `MatplotlibVisualizer`, `Open3DVisualizer`.
