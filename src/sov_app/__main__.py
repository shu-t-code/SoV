"""sov_app package entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

USAGE = "Usage: python -m sov_app <path_to_model_onefile.csv>"
logger = logging.getLogger("sov_app")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SoV application launcher")
    parser.add_argument("csv", nargs="?", help="Path to model_onefile.csv")
    parser.add_argument("--headless", dest="headless_csv", metavar="CSV", help="Run a headless smoke flow with the given CSV")
    parser.add_argument("--out", dest="out_dir", metavar="OUT_DIR", help="Output directory for headless results")
    parser.add_argument("--dims-inst", dest="dims_inst", metavar="INST_ID", help="Instance ID used for realized-dims columns in headless output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing headless output files")
    return parser


def _pick_csv_path() -> Path | None:
    from PySide6.QtWidgets import QFileDialog

    selected_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open model_onefile.csv",
        "",
        "CSV Files (*.csv);;All Files (*)",
    )
    if not selected_path:
        return None
    return Path(selected_path).expanduser()


def _resolve_headless_output_dir(csv_path: Path, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    # Default policy: create "sov_headless_out" in current working directory.
    return (Path.cwd() / "sov_headless_out").resolve()


def _pick_output_file(base_file: Path, overwrite: bool) -> Path:
    if overwrite or not base_file.exists():
        return base_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_file.with_name(f"{base_file.stem}_{timestamp}{base_file.suffix}")


def _write_headless_outputs(results: Any, output_dir: Path, csv_path: Path, seed: int, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_results_file = _pick_output_file(output_dir / "mc_results.csv", overwrite)
    results.to_csv(mc_results_file, index=False)

    numeric_cols = results.select_dtypes(include=["number"])
    summary_payload = {
        "runs": int(len(results)),
        "seed": int(seed),
        "input_csv": str(csv_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            col: {
                "mean": float(numeric_cols[col].mean()),
                "std": float(numeric_cols[col].std(ddof=1)) if len(numeric_cols[col]) > 1 else 0.0,
                "min": float(numeric_cols[col].min()),
                "max": float(numeric_cols[col].max()),
            }
            for col in numeric_cols.columns
        },
    }
    summary_file = _pick_output_file(output_dir / "summary.json", overwrite)
    summary_file.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_headless(csv_path: Path, out_dir: str | None, overwrite: bool, dims_inst: str | None = None) -> int:
    from .smoke import run_headless_smoke_results

    seed = 42
    output_dir = _resolve_headless_output_dir(csv_path, out_dir)
    trace_enabled = out_dir is not None
    print(f"Headless output dir: {output_dir}")

    rc, results = run_headless_smoke_results(
        csv_path,
        n_trials=100,
        seed=seed,
        dims_inst=dims_inst,
        out_dir=output_dir,
        trace=trace_enabled,
    )
    if rc == 0 and results is not None:
        _write_headless_outputs(results, output_dir, csv_path, seed, overwrite)
    if rc == 2:
        logger.error("CSV file not found: %s", csv_path)
    elif rc != 0:
        logger.error("Headless smoke failed with code=%s", rc)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    parsed = _build_parser().parse_args(args)

    if parsed.headless_csv:
        return _run_headless(Path(parsed.headless_csv).expanduser(), parsed.out_dir, parsed.overwrite, parsed.dims_inst)

    try:
        from PySide6.QtWidgets import QApplication
        from .ui.main_window import MainWindow
        from .util_logging import log_env, setup_font
    except ModuleNotFoundError as exc:
        logger.error("Required dependency is missing: %s", exc.name)
        return 1

    log_env()

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    if parsed.csv:
        csv_path = Path(parsed.csv).expanduser()
    else:
        picked = _pick_csv_path()
        if picked is None:
            logger.info("No CSV file selected from dialog. Exiting without launching the app.")
            return 0
        csv_path = picked

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        logger.error(USAGE)
        return 2

    app.setFont(setup_font())

    window = MainWindow(csv_path)
    window.show()

    if created:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
