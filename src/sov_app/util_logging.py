"""Shared utility helpers for environment logging and Qt defaults."""

from __future__ import annotations

import platform
import sys
import time
from typing import Callable

from PySide6.QtGui import QFont

from .env import USE_WATCHDOG

try:
    import matplotlib
except Exception:  # pragma: no cover - optional dependency
    matplotlib = None


def log_env() -> None:
    o3d_ver = "N/A"
    try:
        import open3d as _o3d

        o3d_ver = getattr(_o3d, "__version__", "unknown")
    except Exception:
        pass

    mpl_ver = getattr(matplotlib, "__version__", "N/A")

    from PySide6 import QtCore

    print(f"[ENV] Python {sys.version.split()[0]} on {platform.system()}")
    print(f"[ENV] PySide6 {QtCore.__version__}, Matplotlib {mpl_ver}, Open3D {o3d_ver}")


def setup_font() -> QFont:
    for font_name in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        font = QFont(font_name, 9)
        if font.exactMatch():
            return font
    return QFont()


if USE_WATCHDOG:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        HAS_WATCHDOG = True

        class FileChangeHandler(FileSystemEventHandler):
            def __init__(self, callback: Callable[[], None]):
                super().__init__()
                self.callback = callback
                self.last: dict[str, float] = {}

            def on_modified(self, event):  # type: ignore[override]
                if event.is_directory:
                    return
                now = time.time()
                if event.src_path in self.last and (now - self.last[event.src_path] < 0.8):
                    return
                self.last[event.src_path] = now
                if event.src_path.endswith((".json", ".csv")):
                    self.callback()

    except Exception:  # pragma: no cover - optional dependency
        HAS_WATCHDOG = False
        Observer = None
        FileChangeHandler = None
else:
    HAS_WATCHDOG = False
    Observer = None
    FileChangeHandler = None


__all__ = ["FileChangeHandler", "HAS_WATCHDOG", "Observer", "log_env", "setup_font"]
