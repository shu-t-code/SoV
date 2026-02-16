"""sov_app package entrypoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

USAGE = "Usage: python -m sov_app <path_to_model_onefile.csv>"
logger = logging.getLogger("sov_app")


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


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv)

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

    if len(args) >= 2:
        csv_path = Path(args[1]).expanduser()
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
