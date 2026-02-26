# Engine structure integrity check (SoV)

## Conclusion
- Current state: **Almost YES** (`ほぼYES`)
- `src/sov_app/engine` itself is clean from GUI/UI dependencies.
- However, the headless entrypoint is not yet aligned with the target contract (`--out` + `mc_results.csv`), and runtime checks are currently blocked unless core numerical deps are installed.

## PASS/FAIL matrix

1. `engine` does not import `PySide6/open3d/matplotlib`: **PASS**
2. `engine` does not import `sov_app.services` / `sov_app.ui`: **PASS**
3. `sov_app/__init__.py` is lightweight and does not force GUI deps: **PASS**
4. Headless smoke contract (`--headless <csv> --out <dir>` and `mc_results.csv`) is satisfied: **FAIL**
5. Engine API direct use without `services` is clearly available/intended: **PASS (partial constraints)**

## Evidence summary

- `engine` package has only internal imports + numpy/pandas, no UI modules.
- `sov_app/__init__.py` is docstring-only.
- `__main__.py` imports PySide6 lazily inside GUI-only code path, but parser only supports `--headless <csv>` and has no `--out`.
- `smoke.py` runs through `services` and returns status code only; there is no `mc_results.csv` export in that path.
- CSV export for Monte Carlo exists in GUI main window code, not in headless flow.

## Minimal fixes to reach full YES

1. Add `--out` option in `src/sov_app/__main__.py` and pass it to headless runner.
2. In `src/sov_app/smoke.py`, persist Monte Carlo DataFrame to `<out>/mc_results.csv`.
3. (Optional) Add a tiny regression test that runs a minimal synthetic CSV through headless mode and asserts `<out>/mc_results.csv` exists.

These changes can stay small and do not require large refactoring.
