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
- `fitup_pair_chain` の入力は `/steps/N/base` と `/steps/N/guest` のみをサポートします（`chain` は廃止）。1 step = 1ペア（base/guest）で記述してください。

## Engine purity check

```bash
python tools/check_engine_purity.py
pytest -q tests/test_engine_purity.py tests/test_engine_imports.py
```

- `tools/check_engine_purity.py` performs an AST-based scan of `src/sov_app/engine/` and fails if GUI/visualization imports are detected.
- The pytest checks enforce the same rule in CI and confirm engine modules import cleanly when GUI dependencies are unavailable.

## Manual check (Open3D separation)

```bash
python -m sov_app "path/to/model_onefile.csv"
# GUI起動後に「ビュー更新」を押す
# Open3Dを直接確認する場合
python -m sov_app.tools.view_open3d /tmp/sov_scene_example.ply
```

- GUI起動時にOpen3Dウィンドウは自動起動しません。
- Open3D表示は別プロセスで起動するため、GUI操作は継続できます。

## WindowsローカルCSVをGitHubへ載せる手順

`python -m sov_app "C:\\...\\model_onefile_buttpair_single_steps.csv"` のように実行できている場合でも、
Codex実行環境からは通常そのWindowsローカルパスへ直接アクセスできません。
GitHubへ載せるには、CSVをこのリポジトリ配下へ配置してコミットしてください。

```bash
# 例: リポジトリ直下で実行
mkdir -p data
cp "C:/Users/.../model_onefile_buttpair_single_steps.csv" data/
git add data/model_onefile_buttpair_single_steps.csv
git commit -m "Add model_onefile_buttpair_single_steps.csv"
git push
```

その後は相対パスで起動できます。

```bash
python -m sov_app "data/model_onefile_buttpair_single_steps.csv"
```
