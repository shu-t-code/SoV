from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from sov_app import __main__
from sov_app.smoke import run_headless_smoke_results


def test_run_headless_enables_trace_only_with_explicit_out_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = Path("data/model_onefile_buttpair_single_steps.csv")
    captured: dict[str, object] = {}

    def fake_run_headless_smoke_results(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 0, None

    monkeypatch.setattr("sov_app.smoke.run_headless_smoke_results", fake_run_headless_smoke_results)

    rc = __main__._run_headless(csv_path, str(tmp_path), overwrite=True)

    assert rc == 0
    assert captured["kwargs"]["trace"] is True
    assert captured["kwargs"]["out_dir"] == tmp_path.resolve()


def test_run_headless_disables_trace_without_explicit_out_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = Path("data/model_onefile_buttpair_single_steps.csv")
    captured: dict[str, object] = {}

    def fake_run_headless_smoke_results(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 0, None

    monkeypatch.setattr("sov_app.smoke.run_headless_smoke_results", fake_run_headless_smoke_results)
    monkeypatch.chdir(tmp_path)

    rc = __main__._run_headless(csv_path, None, overwrite=True)

    assert rc == 0
    assert captured["kwargs"]["trace"] is False
    assert captured["kwargs"]["out_dir"] == (tmp_path / "sov_headless_out").resolve()


def test_headless_smoke_results_writes_trace_files_when_enabled(tmp_path: Path) -> None:
    csv_path = Path("data/model_onefile_buttpair_with_fillet_attach.csv")

    rc, results = run_headless_smoke_results(csv_path, n_trials=2, seed=42, out_dir=tmp_path, trace=True)

    assert rc == 0
    assert results is not None
    expected = [
        "00_10_cutting",
        "01_20_fitup_C1_on_A1_center",
        "02_21_fitup_C2_on_A2_center",
        "03_30_mid_fitup_butt_A1_A2",
        "04_40_mid_welding",
        "05_50_inspection",
    ]
    for step in expected:
        assert (tmp_path / f"mc_trace_{step}__vertices.csv").exists()
        assert (tmp_path / f"mc_trace_worst_{step}__vertices.csv").exists()

    trace_files = [p for p in tmp_path.glob("mc_trace_*__vertices.csv") if not p.name.startswith("mc_trace_worst_")]
    assert len(trace_files) == len(expected)
    assert len(list(tmp_path.glob("mc_trace_worst_*__vertices.csv"))) == len(expected)
    assert (tmp_path / "mc_worstcase_summary.csv").exists()
