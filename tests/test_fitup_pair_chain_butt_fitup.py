import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.process_engine import ProcessEngine
from sov_app.services import summarize_butt_fitup_metrics


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
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
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
    assert len(metrics) == 1
    assert "w" in metrics[0]
    assert "delta_y" in metrics[0]




def test_butt_fitup_with_q1_uses_marking_line_basis_for_two_point_alignment() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "fitup_step",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
                    "model": {
                        "butt_fitup": {
                            "d_nom": 10.0,
                            "g0": 0.0,
                            "w0": 0.0,
                            "L_dist": {"type": "Fixed", "value": 0.0},
                            "eps_mA": {"type": "Fixed", "value": 0.0},
                            "eps_mB": {"type": "Fixed", "value": 0.0},
                            "eps_cA": {"type": "Fixed", "value": 0.0},
                            "eps_cB": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                        }
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    # Samples per pair: [L, eps_mA, eps_mB, eps_cA, eps_cB]
    # delta_y is Fixed(0), so _sample is not called for delta_y.
    # pair0 => w=0, dA=dB=0 => edge shift 0
    # pair1 => w=10, dA=dB=0 => edge shift -10 * v
    sample_seq = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    engine._sample = lambda spec: sample_seq.pop(0)

    engine.apply_steps(state)

    p0 = get_world_point(geom, state, "A1", "points.B")
    p1 = get_world_point(geom, state, "A1", "points.C")
    v = np.array([-1.0, 0.0, 0.0], dtype=float)

    metric = state.butt_fitup_metrics["fitup_step"][0]
    q0_target = p0 + (metric["dA_0"] + metric["dB_0"] - metric["w_0"]) * v
    q1_target = p1 + (metric["dA_1"] + metric["dB_1"] - metric["w_1"]) * v

    q0 = get_world_point(geom, state, "G1", "points.A")
    q1 = get_world_point(geom, state, "G1", "points.D")

    assert np.allclose(q0, q0_target, atol=1e-6)
    assert np.allclose(q1, q1_target, atol=1e-6)
    assert state.point_offsets["G1"] == {}

def test_butt_fitup_enforce_nonnegative_gap_keeps_interference_flag() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "fitup_step",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
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
    assert metric["clipped_0"] is True
    assert metric["clipped_1"] is True


def test_summarize_butt_fitup_metrics_counts_clipped_and_interferes() -> None:
    states = [
        {
            "butt_fitup_metrics": {
                "fitup_step": [
                    {"interferes_0": True, "clipped_0": True, "interferes_1": True, "clipped_1": False}
                ]
            }
        },
        {
            "butt_fitup_metrics": {
                "fitup_step": [
                    {"interferes_0": False, "clipped_0": False, "interferes_1": None, "clipped_1": None}
                ]
            }
        },
    ]

    summary = summarize_butt_fitup_metrics(states)

    assert summary["fitup_step"]["n"] == 2
    assert summary["fitup_step"]["pair0"] == {"n": 2, "interferes": 1, "clipped": 1}
    assert summary["fitup_step"]["pair1"] == {"n": 1, "interferes": 1, "clipped": 0}


def test_legacy_mode_without_butt_fitup_is_still_used() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "legacy_fitup",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
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


def test_butt_fitup_missing_required_keys_raises_value_error() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "fitup_step",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "G1", "q0": "points.A", "q1": "points.D"},
                    "model": {
                        "butt_fitup": {
                            "d_nom": 73.0,
                            "g0": 4.0,
                            "eps_mA": {"type": "Fixed", "value": 0.0},
                            "eps_mB": {"type": "Fixed", "value": 0.0},
                            "eps_cA": {"type": "Fixed", "value": 0.0},
                            "eps_cB": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                        }
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    with pytest.raises(ValueError, match="butt_fitup missing required keys"):
        engine.apply_steps(state)


def test_fitup_pair_chain_rejects_legacy_chain_input() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "legacy_chain_step",
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
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    with pytest.raises(ValueError, match="chain は廃止。base/guest に移行して下さい"):
        engine.apply_steps(state)


def test_welding_distortion_applies_pairwise_butt_transverse_shrinkage_from_fitup_metrics() -> None:
    geom = GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate_with_marks",
                    "dims": {"L": 200.0, "H": 100.0, "t": 10.0},
                    "features": {
                        "points": {
                            "A": [0.0, 0.0, 0.0],
                            "B": [200.0, 0.0, 0.0],
                            "C": [200.0, 100.0, 0.0],
                            "D": [0.0, 100.0, 0.0],
                            "MK_AB": [120.0, 0.0, 0.0],
                            "MK_CD": [40.0, 100.0, 0.0],
                            "MID": [100.0, 50.0, 0.0],
                        }
                    },
                }
            ],
            "instances": [
                {"id": "G1", "prototype": "plate_with_marks", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
            ],
        }
    )
    flow = FlowModel(
        {
            "selectors": {"GUEST": {"ids": ["G1"]}},
            "steps": [
                {
                    "id": "weld_step",
                    "op": "welding_distortion",
                    "target": {"selector": "GUEST"},
                    "model": {
                        "outplane_dz": {"type": "Fixed", "value": 0.0},
                        "weak_bending_about_x": {"type": "Fixed", "value": 0.0},
                    },
                }
            ],
        }
    )

    state = AssemblyState(geom)
    state.butt_fitup_metrics = {
        "fitup_step": [
            {
                "guest_instance": "G1",
                "transverse_dir_world": [1.0, 0.0, 0.0],
                "g_real_0": 10.0,
                "g_real_1": 5.0,
            }
        ]
    }

    engine = ProcessEngine(geom, flow, np.random.default_rng(0))
    engine.apply_steps(state)

    expected_lower = np.array([-1.8, 0.0, 0.0], dtype=float)
    expected_upper = np.array([-0.9, 0.0, 0.0], dtype=float)

    assert np.allclose(state.get_point_offset("G1", "A"), expected_lower, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "B"), expected_lower, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "MK_AB"), expected_lower, atol=1e-8)

    assert np.allclose(state.get_point_offset("G1", "C"), expected_upper, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "D"), expected_upper, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "MK_CD"), expected_upper, atol=1e-8)

    assert np.allclose(state.get_point_offset("G1", "MID"), np.zeros(3, dtype=float), atol=1e-8)


def test_welding_distortion_butt_transverse_shrinkage_clamps_negative_gap_to_zero() -> None:
    geom = _build_geom_two_pairs()
    flow = FlowModel(
        {
            "selectors": {"GUEST": {"ids": ["G1"]}},
            "steps": [
                {
                    "id": "weld_step",
                    "op": "welding_distortion",
                    "target": {"selector": "GUEST"},
                    "model": {
                        "outplane_dz": {"type": "Fixed", "value": 0.0},
                        "weak_bending_about_x": {"type": "Fixed", "value": 0.0},
                    },
                }
            ],
        }
    )

    state = AssemblyState(geom)
    state.butt_fitup_metrics = {
        "fitup_step": [
            {
                "guest_instance": "G1",
                "transverse_dir_world": [1.0, 0.0, 0.0],
                "g_real_0": -2.0,
                "g_real_1": -3.0,
            }
        ]
    }

    engine = ProcessEngine(geom, flow, np.random.default_rng(0))
    engine.apply_steps(state)

    assert state.point_offsets["G1"] == {}
