"""Backward-compatible import shim for headless smoke flow."""

from __future__ import annotations

from .headless import run_headless_smoke

__all__ = ["run_headless_smoke"]
