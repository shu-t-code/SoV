from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from sov_app import smoke


class _DummyState:
    def __init__(self, dims_by_inst: dict[str, dict[str, float]]):
        self._dims = dims_by_inst

    def get_realized_dims(self, inst_id: str):
        return self._dims.get(inst_id, {})


def test_inject_realized_dims_columns_uses_selected_instance(monkeypatch) -> None:
    app_state = SimpleNamespace(
        geom=SimpleNamespace(
            instances={"p1": {}, "p2": {}},
            get_instance_ids=lambda: ["p1", "p2"],
        )
    )
    base = pd.DataFrame({"trial": [0, 1], "edge_gap_x": [1.0, 2.0]})

    def _fake_build_trial_state(_app_state, _steps_mask, trial, _seed):
        if trial == 0:
            return _DummyState({"p2": {"L": 101.0, "H": 52.0, "L_ab": 100.0}})
        return _DummyState({"p2": {"L": 102.0, "H": 53.0, "H_bc": 51.0}})

    monkeypatch.setattr(smoke, "build_trial_state", _fake_build_trial_state)

    out = smoke._inject_realized_dims_columns(
        app_state,
        base,
        steps_mask=[True],
        n_trials=2,
        seed=42,
        dims_inst="p2",
    )

    assert list(out["L"]) == [101.0, 102.0]
    assert list(out["H"]) == [52.0, 53.0]
    assert out.loc[0, "L_ab"] == 100.0
    assert pd.isna(out.loc[0, "H_ad"])
    assert out.loc[1, "H_bc"] == 51.0
