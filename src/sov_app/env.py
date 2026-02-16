"""Environment and optional dependency detection helpers for SoV."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("sov_app")

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _parse_bool_env(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    logger.warning("Invalid boolean override for %s: %r", name, value)
    return None


def detect_open3d() -> bool:
    """Return True when open3d is importable."""
    try:
        importlib.import_module("open3d")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("open3d detection failed: %s", exc)
        return False


def detect_watchdog() -> bool:
    """Return True when watchdog is importable."""
    try:
        importlib.import_module("watchdog")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("watchdog detection failed: %s", exc)
        return False


def _resolve_bool_override(name: str, detector: Callable[[], bool]) -> bool:
    override = _parse_bool_env(name)
    if override is not None:
        return override
    return detector()


def set_mpl_backend(prefer_qt: bool = True) -> str:
    """Set a safe matplotlib backend and return the resulting backend name."""
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover - matplotlib unavailable
        logger.debug("matplotlib import failed: %s", exc)
        return os.getenv("SOV_MPL_BACKEND", "Agg")

    override_backend = os.getenv("SOV_MPL_BACKEND")
    candidates: list[str] = []

    if override_backend:
        candidates.append(override_backend)
    elif prefer_qt:
        candidates.extend(["QtAgg", "Qt5Agg", "Agg"])
    else:
        candidates.append("Agg")

    if "Agg" not in candidates:
        candidates.append("Agg")

    current_backend = ""
    try:
        current_backend = str(matplotlib.get_backend())
    except Exception as exc:
        logger.debug("Unable to query matplotlib backend before set: %s", exc)

    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            return str(matplotlib.get_backend())
        except Exception as exc:
            logger.debug("Failed to set matplotlib backend %s: %s", backend, exc)

    try:
        return str(matplotlib.get_backend())
    except Exception as exc:
        logger.debug("Unable to query matplotlib backend after attempts: %s", exc)
        return current_backend


USE_OPEN3D: bool = _resolve_bool_override("SOV_USE_OPEN3D", detect_open3d)
USE_WATCHDOG: bool = _resolve_bool_override("SOV_USE_WATCHDOG", detect_watchdog)
MPL_BACKEND: str = set_mpl_backend(prefer_qt=True)


__all__ = [
    "MPL_BACKEND",
    "USE_OPEN3D",
    "USE_WATCHDOG",
    "detect_open3d",
    "detect_watchdog",
    "set_mpl_backend",
]
