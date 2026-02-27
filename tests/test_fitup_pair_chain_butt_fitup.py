import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel
from sov_app.engine.process_engine import ProcessEngine


def _build_geom_two_pairs() -> GeometryModel:
    return GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate",
                    "dims": {"L": 200.0, "H": 100.0, "t": 10.0},
                    "features": {
                        "points": {
                            "A": [0.0, 0.0, 0.0],
                            "B": [200.0, 0.0, 0.0],
                            "C": [200.0, 100.0, 0.0],
                            "D": [0.0, 100.0, 0.0],
                        }
                    },
                }
            ],
            "instances": [
                {"id": "A1", "prototype": "plate", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "A2", "prototype": "plate", "frame": {"parent": "world", "origin": [300.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "G1", "prototype": "plate", "frame": {"parent": "world", "origin": [30.0, 10.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "G2", "prototype": "plate", "frame": {"parent": "world", "origin": [340.0, -20.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
            ],
        }
    )


def test_butt_fitup_new_mode_records_metrics_and_shares_delta_y() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "fitup_step",
                    "op": "fitup_pair_chain",
                    "chain": [
                        {
                            "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                            "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
                        },
                        {
                            "base": {"instance": "A2", "p0": "points.B", "p1": "points.C"},
                            "guest": {"instance": "G2", "q0": "points.A", "q1": "points.D"},
                        },
                    ],
                    "model": {
                        "butt_fitup": {
                            "d_nom": 73.0,
                            "g0": 4.0,
                            "w0": 150.0,
                            "L_dist": {"type": "LogNormalLinear", "mean": 4.0, "std": 1.5},
                            "eps_mA": {"type": "NormalLinear", "mean": 0.0, "std": 0.5},
                            "eps_mB": {"type": "NormalLinear", "mean": 0.0, "std": 0.5},
                            "eps_cA": {"type": "NormalLinear", "mean": 0.05, "std": 0.1},
                            "eps_cB": {"type": "NormalLinear", "mean": 0.05, "std": 0.1},
                            "delta_y": {"type": "NormalLinear", "mean": 0.0, "std": 1.5},
                        }
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(123))

    engine.apply_steps(state)

    assert state.point_offsets["G1"] == {}
    assert state.point_offsets["G2"] == {}

    metrics = state.butt_fitup_metrics["fitup_step"]
    assert len(metrics) == 2
    assert metrics[0]["delta_y"] == metrics[1]["delta_y"]
    assert not np.isclose(metrics[0]["w"], metrics[1]["w"])
    assert metrics[0]["interferes"] != metrics[1]["interferes"] or not np.isclose(metrics[0]["g_real"], metrics[1]["g_real"])


def test_butt_fitup_enforce_nonnegative_gap_keeps_interference_flag() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "fitup_step",
                    "op": "fitup_pair_chain",
                    "chain": [
                        {
                            "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                            "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
                        }
                    ],
                    "model": {
                        "butt_fitup": {
                            "d_nom": 73.0,
                            "g0": 4.0,
                            "w0": 10.0,
                            "L_dist": {"type": "Fixed", "value": 0.0},
                            "eps_mA": {"type": "Fixed", "value": 0.0},
                            "eps_mB": {"type": "Fixed", "value": 0.0},
                            "eps_cA": {"type": "Fixed", "value": 0.0},
                            "eps_cB": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                            "enforce_nonnegative_gap": True,
                        }
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(state)

    metric = state.butt_fitup_metrics["fitup_step"][0]
    assert metric["g_real"] == 0.0
    assert metric["interferes"] is True


def test_legacy_mode_without_butt_fitup_is_still_used() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "legacy_fitup",
                    "op": "fitup_pair_chain",
                    "chain": [
                        {
                            "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                            "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
                        }
                    ],
                    "model": {
                        "dx0_logn": {"type": "Fixed", "value": 4.0},
                        "dx1_logn": {"type": "Fixed", "value": 2.0},
                        "dy_norm": {"type": "Fixed", "value": 1.0},
                    },
                    "constraints": {"inplane_y": {"direction": "joint_perp"}},
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(state)

    assert np.linalg.norm(state.get_point_offset("G1", "D")) > 0.0
    assert not hasattr(state, "butt_fitup_metrics")
