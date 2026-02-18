"""sov_app package entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_HEADLESS_OUT_DIR, DEFAULT_LOG_LEVEL, DEFAULT_MC_N, HeadlessConfig, MCConfig

USAGE = "Usage: python -m sov_app <path_to_model_onefile.csv>"
logger = logging.getLogger("sov_app")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SoV launcher (GUI by default, or headless batch mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", nargs="?", help="Path to model_onefile.csv")
    parser.add_argument("--headless", action="store_true", help="Run the headless pipeline (no GUI)")
    parser.add_argument("--mc-n", type=int, default=DEFAULT_MC_N, help="Monte Carlo trial count for headless mode")
    parser.add_argument("--seed", type=int, help="Random seed for headless mode")
    parser.add_argument("--out", type=Path, default=DEFAULT_HEADLESS_OUT_DIR, help="Output directory for headless artifacts")
    parser.add_argument("--no-open3d", action="store_true", help="Disable Open3D even when installed")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO"], default=DEFAULT_LOG_LEVEL, help="Logging verbosity")
    return parser


def _setup_logging(log_level: str, run_log_path: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    console_exists = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not console_exists:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root.addHandler(console)

    if run_log_path is not None:
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == run_log_path:
                break
        else:
            file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            root.addHandler(file_handler)


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


def _run_headless_mode(args: argparse.Namespace) -> int:
    if not args.csv:
        logger.error("Headless mode requires a CSV path.\n%s", USAGE)
        return 2
    if args.mc_n <= 0:
        logger.error("--mc-n must be > 0 (got %s)", args.mc_n)
        return 2

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_level, out_dir / "run.log")

    config = HeadlessConfig(
        csv_path=Path(args.csv).expanduser(),
        out_dir=out_dir,
        mc_config=MCConfig(n=args.mc_n, seed=args.seed),
        no_open3d=args.no_open3d,
        log_level=args.log_level,
    )
    from .headless import run_headless

    return run_headless(config)


def main(argv: Sequence[str] | None = None) -> int:
    parsed = _build_parser().parse_args(list(argv if argv is not None else sys.argv[1:]))

    if parsed.no_open3d:
        os.environ["SOV_USE_OPEN3D"] = "0"

    _setup_logging(parsed.log_level)

    if parsed.headless:
        try:
            return _run_headless_mode(parsed)
        except Exception:
            logger.exception("Unhandled error during headless execution")
            return 1

    try:
        from PySide6.QtWidgets import QApplication
        from .main_window import MainWindow
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
