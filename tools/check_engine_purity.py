#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED_IMPORT_PREFIXES = (
    "PySide6",
    "PyQt",
    "qtpy",
    "open3d",
    "matplotlib",
    "pyqtgraph",
)


def _is_banned(module_name: str) -> bool:
    return module_name.startswith(BANNED_IMPORT_PREFIXES)


def check_engine_purity(engine_dir: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for py_file in sorted(engine_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_banned(alias.name):
                        violations.append((py_file, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _is_banned(node.module):
                    violations.append((py_file, node.lineno, node.module))
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    engine_dir = repo_root / "src" / "sov_app" / "engine"
    violations = check_engine_purity(engine_dir)
    if violations:
        print("Engine purity check failed:")
        for path, lineno, module in violations:
            rel = path.relative_to(repo_root)
            print(f"  - {rel}:{lineno} imports forbidden module '{module}'")
        return 1

    print("Engine purity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
