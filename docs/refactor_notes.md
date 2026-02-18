# Refactor notes: services facade only

## Completed architecture shift

`src/sov_app/services.py` is now the single application business-logic facade consumed by UI/headless paths.

- `legacy_impl.py` has been removed.
- `main_window.py` now calls the services facade for loading, saving, step application, Monte Carlo, pair-distance trials, and render orchestration.
- Headless execution uses `smoke.run_headless_smoke`, which itself only uses `services`.

## Stable facade surface

- `load_project(csv_path) -> AppState`
- `save_project(state, csv_path) -> None`
- `create_project(geom_data, flow_data) -> AppState`
- `apply_steps(state, selection) -> AssemblyState`
- `run_monte_carlo(state, mc_config) -> pandas.DataFrame`
- `build_trial_state(state, steps_mask, trial, seed_base) -> AssemblyState`
- `run_pair_distance(state, steps_mask, p1_instance, p1_ref, p2_instance, p2_ref, n_trials, seed) -> numpy.ndarray`
- `build_visualizer(config)`
- `render(visualizer, state, assembly_state, render_config) -> None`
- `show_rendered_scene(visualizer, title, width, height) -> bool`
- `validate_models(state) -> list[dict[str, str]]`

Backward-compatible aliases remain in `services.py` (`load_csv`, `save_csv`, `from_dicts`, `build_state_for_trial`) to reduce downstream breakage while keeping runtime entrypoints on the facade.

## Optional dependencies

Open3D remains optional.

- If Open3D is available and enabled, `build_visualizer` returns the Open3D visualizer and `show_rendered_scene` opens an Open3D window.
- Otherwise, the Matplotlib/Qt visualizer path is used.
