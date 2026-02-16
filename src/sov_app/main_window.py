"""Main window entrypoint for sov_app GUI.

Step 1 extracts the GUI skeleton so later splits (widgets/viz/engine) can
move away from ``app_onefile.py`` incrementally without behavior changes.
"""

from __future__ import annotations

import app_onefile


class MainWindow(app_onefile.MainWindow):
    """Thin compatibility wrapper around the legacy one-file MainWindow."""

    pass
