import pytest
import importlib
import importlib.abc
import sys
from contextlib import contextmanager

BLOCKED = ("PySide6", "PyQt5", "PyQt6", "qtpy", "open3d", "matplotlib", "pyqtgraph")


class BlockedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(BLOCKED):
            raise ModuleNotFoundError(fullname)
        return None


@contextmanager
def blocked_gui_imports():
    finder = BlockedImportFinder()
    old_modules = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith(BLOCKED)}
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(old_modules)


def test_engine_modules_import_without_gui_deps() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    with blocked_gui_imports():
        importlib.import_module("sov_app.engine")
        importlib.import_module("sov_app.engine.core_models")
        importlib.import_module("sov_app.engine.io_csv")
        importlib.import_module("sov_app.engine.process_engine")
        importlib.import_module("sov_app.engine.monte_carlo")
