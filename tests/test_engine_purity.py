from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_checker_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "tools" / "check_engine_purity.py"
    spec = spec_from_file_location("check_engine_purity", mod_path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_engine_has_no_gui_imports() -> None:
    checker = _load_checker_module()
    repo_root = Path(__file__).resolve().parents[1]
    engine_dir = repo_root / "src" / "sov_app" / "engine"
    violations = checker.check_engine_purity(engine_dir)
    assert violations == []
