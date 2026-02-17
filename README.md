# SoV (PySide6 GUI)

## Run

```bash
pip install -e .
python -m sov_app "path/to/model_onefile.csv"
```

CSV パスを省略した場合はファイル選択ダイアログが開きます。

```bash
python -m sov_app
```

## Headless smoke path

```bash
python -m sov_app --headless "path/to/model_onefile.csv"
```

`--headless` は CSV をロードし、最小ステップ適用 + 小規模 Monte Carlo（3試行）を実行して終了します。

## Minimal smoke checks

```bash
python -m compileall src/sov_app
PYTHONPATH=src python -m sov_app --headless model_onefile.csv
PYTHONPATH=src python -c "from sov_app.services import load_csv, apply_steps; print('services import ok')"
```

## README smoke test commands

```bash
python -m sov_app
python -m sov_app "<csv_path>"
python -m sov_app --headless "<csv_path>"
# Windows
py -m sov_app "<csv_path>"
```
