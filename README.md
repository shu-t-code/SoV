# SoV (PySide6 GUI)

## Supported commands

```bash
python -m sov_app "path/to/model_onefile.csv"
python -m sov_app --headless "path/to/model_onefile.csv"
python -m sov_app --headless "path/to/model_onefile.csv" --mc-n 20 --out out_headless_ci --no-open3d
```

## Notes

- The GUI command launches the main application window and loads the CSV model.
- The `--headless` command runs a smoke flow (load + one safe step + Monte Carlo), writes `mc_results.csv` and `summary.json` in the output directory, and exits with code `0` when successful.
- Headless options: `--mc-n` (trial count), `--out` (output directory for `mc_results.csv` + `summary.json`), and `--no-open3d` (forces Open3D off for CI).
- Open3D is optional. When installed/enabled, 3D rendering opens in an Open3D window; otherwise the app uses the Matplotlib/Qt fallback visualizer.
