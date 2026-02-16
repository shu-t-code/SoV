"""sov_app package entrypoint.

Usage examples:
    python -m sov_app C:\\path\\to\\model_onefile.csv
    python -m sov_app ./model_onefile.csv
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Sequence


USAGE = "Usage: python -m sov_app <path_to_model_onefile.csv>"
logger = logging.getLogger("sov_app")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv)

    from PySide6.QtWidgets import QApplication
    from sov_app.main_window import MainWindow

    import app_onefile

    app_onefile.log_env()

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    if len(args) >= 2:
        csv_path = Path(args[1]).expanduser()
    else:
        from PySide6.QtWidgets import QFileDialog

        selected_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open model_onefile.csv",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not selected_path:
            logger.info("No CSV file selected from dialog. Exiting without launching the app.")
            return 0
        csv_path = Path(selected_path).expanduser()

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        logger.error(USAGE)
        return 2

    app.setFont(app_onefile.setup_font())

    app_onefile.MODEL_ONEFILE_CSV_PATH = csv_path

    app_onefile.print_all_edge_stds_after_cutting(
        app_onefile.MODEL_ONEFILE_CSV_PATH,
        n_trials=5000,
        seed=42,
    )

    window = getattr(app_onefile, "_APP_WINDOW", None)
    if window is not None:
        window.close()
        window.deleteLater()
        app.processEvents()

    window = MainWindow(csv_path)
    app_onefile._APP_WINDOW = window
    window.show()

    if created:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
