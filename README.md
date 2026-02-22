# SoV (PySide6 GUI)

## Supported commands

```bash
python -m sov_app "path/to/model_onefile.csv"
python -m sov_app --headless "path/to/model_onefile.csv"
```

## Notes

- The GUI command launches the main application window and loads the CSV model.
- The `--headless` command runs a smoke flow (load + one safe step + Monte Carlo with `N=100`) and exits with code `0` when successful.
- Open3D is optional. When installed/enabled, 3D rendering opens in an Open3D window; otherwise the app uses the Matplotlib/Qt fallback visualizer.

## Engine purity check

```bash
python tools/check_engine_purity.py
pytest -q tests/test_engine_purity.py tests/test_engine_imports.py
```

- `tools/check_engine_purity.py` performs an AST-based scan of `src/sov_app/engine/` and fails if GUI/visualization imports are detected.
- The pytest checks enforce the same rule in CI and confirm engine modules import cleanly when GUI dependencies are unavailable.
