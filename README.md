# SoV (PySide6 GUI)

## Run

```bash
pip install -e .
python -m sov_app path/to/model_onefile.csv
```

CSV パスを省略した場合はファイル選択ダイアログが開きます。

```bash
python -m sov_app
```

## Minimal smoke checks

```bash
python -m compileall src/sov_app
PYTHONPATH=src python -m sov_app --help
```

> `--help` は専用オプションではなく、存在しないパスとして扱われます。エラーハンドリング確認用の簡易チェックです。

## CSV I/O の簡易動作確認

```bash
PYTHONPATH=src python -c "from pathlib import Path; from sov_app.io_csv import load_data_from_csv; g, f = load_data_from_csv(Path('model_onefile.csv')); print(type(g), type(f))"
```
