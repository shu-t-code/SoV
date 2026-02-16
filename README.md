# app_onefile.py

## Run (Python 3)

```bash
python3 app_onefile.py
```

## CSV I/O の簡易動作確認

`model_onefile.csv` を使って、CSV（カンマ/タブ自動判定込み）から geometry/flow を読めるか確認できます。

```bash
PYTHONPATH=src python3 -c "from pathlib import Path; from sov_app.io_csv import load_data_from_csv; g, f = load_data_from_csv(Path('model_onefile.csv')); print(type(g), type(f))"
```
