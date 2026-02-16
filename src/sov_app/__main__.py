"""sov_app package entrypoint.

Usage examples:
    python -m sov_app C:\\path\\to\\model_onefile.csv
    python -m sov_app ./model_onefile.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


USAGE = "Usage: python -m sov_app <path_to_model_onefile.csv>"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv)

    if len(args) < 2:
        print(f"[ERROR] CSV path is required.\n{USAGE}", file=sys.stderr)
        return 2

    csv_path = Path(args[1]).expanduser()
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        return 2

    from PySide6.QtWidgets import QApplication

    import app_onefile

    app_onefile.log_env()

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
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

    window = app_onefile.MainWindow()
    app_onefile._APP_WINDOW = window
    window.show()

    if created:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
