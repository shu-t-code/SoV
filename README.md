# SoV (PySide6 GUI)

## Supported commands

```bash
# GUI launch
python -m sov_app "path/to/model_onefile.csv"

# Headless launch with defaults (N=100, out=./out_headless)
python -m sov_app --headless "path/to/model_onefile.csv"

# Headless with explicit options
python -m sov_app --headless "path/to/model_onefile.csv" \
  --mc-n 250 \
  --seed 123 \
  --out out_headless \
  --no-open3d \
  --log-level DEBUG
```

## Headless outputs

Each headless run writes standard artifacts to `--out`:

- `summary.json`: run status, config, trial counts, timings, and aggregate metrics.
- `mc_results.csv`: per-trial Monte Carlo results.
- `run.log`: detailed log output for the run.

## Notes

- The GUI command launches the main application window and loads the CSV model.
- The `--headless` command runs a deterministic smoke flow (load + minimal step selection + Monte Carlo).
- Open3D is optional. Use `--no-open3d` (or env `SOV_USE_OPEN3D=0`) to force-disable it.
- `python -m sov_app --help` shows full CLI documentation.
