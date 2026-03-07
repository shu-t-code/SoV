from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine


MIN_CSV = Path("data/model_min_marking_attach.csv")


def _load_models() -> tuple[GeometryModel, FlowModel]:
    geom_dict, flow_dict = load_data_from_csv(MIN_CSV)
    return GeometryModel(geom_dict), FlowModel(flow_dict)


def test_min_csv_geometry_integrity() -> None:
    geom, _ = _load_models()

    assert set(geom.instances.keys()) == {"A1", "A2", "C1", "C2"}

    proto_a = geom.get_prototype("plate_A_proto")
    proto_c = geom.get_prototype("longi_C_proto")

    assert "MK1_P0" in proto_a["features"]["points"]
    assert "MK1_P1" in proto_a["features"]["points"]
    assert proto_a["features"]["lines"]["MK1"] == {"p0": "points.MK1_P0", "p1": "points.MK1_P1"}

    assert "c1" in proto_c["features"]["points"]
    assert "c4" in proto_c["features"]["points"]


def test_marking_line_world_point_follows_a2_transform() -> None:
    geom, _ = _load_models()
    state = AssemblyState(geom)

    before_p0 = get_world_point(geom, state, "A2", "points.MK1_P0")
    before_p1 = get_world_point(geom, state, "A2", "points.MK1_P1")

    tr = state.get_transform("A2")
    state.set_transform("A2", tr["origin"] + np.array([100.0, 0.0, 0.0], dtype=float), tr["rpy_deg"])

    after_p0 = get_world_point(geom, state, "A2", "points.MK1_P0")
    after_p1 = get_world_point(geom, state, "A2", "points.MK1_P1")

    assert np.allclose(after_p0 - before_p0, np.array([100.0, 0.0, 0.0], dtype=float), atol=1e-8)
    assert np.allclose(after_p1 - before_p1, np.array([100.0, 0.0, 0.0], dtype=float), atol=1e-8)


def test_marking_line_attach_min_flow_aligns_c2_ref_line_to_a2_marking_line() -> None:
    geom, flow = _load_models()
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(state)

    p0 = get_world_point(geom, state, "A2", "points.MK1_P0")
    p1 = get_world_point(geom, state, "A2", "points.MK1_P1")
    q0 = get_world_point(geom, state, "C2", "points.c1")
    q1 = get_world_point(geom, state, "C2", "points.c4")

    assert np.linalg.norm(p0 - q0) < 1e-6
    assert np.linalg.norm(p1 - q1) < 1e-6


def test_existing_fitup_pair_chain_still_operates() -> None:
    geom = GeometryModel(
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
                {"id": "G1", "prototype": "plate", "frame": {"parent": "world", "origin": [40.0, 20.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
            ],
        }
    )
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

    q0 = get_world_point(geom, state, "G1", "points.A")
    q1 = get_world_point(geom, state, "G1", "points.D")
    assert np.isfinite(q0).all()
    assert np.isfinite(q1).all()


def test_marking_line_butt_shrinkage_is_applied_via_welding_distortion_flow_step() -> None:
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
                {"id": "BASE", "prototype": "plate_with_marks", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "G1", "prototype": "plate_with_marks", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
            ],
        }
    )
    flow = FlowModel(
        {
            "selectors": {"GUEST": {"ids": ["G1"]}},
            "steps": [
                {
                    "id": "fitup_pair0",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "BASE", "p0": "points.A", "p1": "points.B"},
                    "guest": {"instance": "G1", "q0": "points.A"},
                    "model": {
                        "butt_fitup": {
                            "d_nom": 10.0,
                            "g0": 4.0,
                            "w0": 20.0,
                            "L_dist": {"type": "Fixed", "value": 4.0},
                            "eps_mA": {"type": "Fixed", "value": 0.0},
                            "eps_mB": {"type": "Fixed", "value": 0.0},
                            "eps_cA": {"type": "Fixed", "value": 0.0},
                            "eps_cB": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                        }
                    },
                },
                {
                    "id": "fitup_pair1",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "BASE", "p0": "points.C", "p1": "points.D"},
                    "guest": {"instance": "G1", "q0": "points.C"},
                    "model": {
                        "butt_fitup": {
                            "d_nom": 10.0,
                            "g0": 1.0,
                            "w0": 20.0,
                            "L_dist": {"type": "Fixed", "value": 1.0},
                            "eps_mA": {"type": "Fixed", "value": 0.0},
                            "eps_mB": {"type": "Fixed", "value": 0.0},
                            "eps_cA": {"type": "Fixed", "value": 0.0},
                            "eps_cB": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                        }
                    },
                },
                {
                    "id": "weld_step",
                    "op": "welding_distortion",
                    "target": {"selector": "GUEST"},
                    "model": {
                        "outplane_dz": {"type": "Fixed", "value": 0.0},
                        "weak_bending_about_x": {"type": "Fixed", "value": 0.0},
                    },
                },
            ],
        }
    )

    fitup_only = AssemblyState(geom)
    fitup_with_weld = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(fitup_only, steps_mask=[True, True, False])
    engine.apply_steps(fitup_with_weld)

    pair0_metric = fitup_with_weld.butt_fitup_metrics["fitup_pair0"][0]
    pair1_metric = fitup_with_weld.butt_fitup_metrics["fitup_pair1"][0]
    assert pair0_metric["pair_index"] == 0
    assert pair1_metric["pair_index"] == 1
    assert pair0_metric["g_real_0"] == pytest.approx(4.0)
    assert pair1_metric["g_real_0"] == pytest.approx(1.0)
    assert pair1_metric["g_real_1"] is None
    assert pair0_metric["weld_x_local_0"] == pytest.approx(0.0)
    assert pair1_metric["weld_x_local_0"] == pytest.approx(200.0)

    delta_a = fitup_with_weld.get_point_offset("G1", "A") - fitup_only.get_point_offset("G1", "A")
    delta_b = fitup_with_weld.get_point_offset("G1", "B") - fitup_only.get_point_offset("G1", "B")
    delta_mk_ab = fitup_with_weld.get_point_offset("G1", "MK_AB") - fitup_only.get_point_offset("G1", "MK_AB")
    delta_c = fitup_with_weld.get_point_offset("G1", "C") - fitup_only.get_point_offset("G1", "C")
    delta_d = fitup_with_weld.get_point_offset("G1", "D") - fitup_only.get_point_offset("G1", "D")
    delta_mk_cd = fitup_with_weld.get_point_offset("G1", "MK_CD") - fitup_only.get_point_offset("G1", "MK_CD")
    delta_mid = fitup_with_weld.get_point_offset("G1", "MID") - fitup_only.get_point_offset("G1", "MID")

    expected_ab = np.array([-0.72, 0.0, 0.0], dtype=float)
    expected_cd = np.array([-0.18, 0.0, 0.0], dtype=float)
    expected_none = np.zeros(3, dtype=float)

    assert np.allclose(delta_a, expected_ab, atol=1e-8)
    assert np.allclose(delta_b, expected_ab, atol=1e-8)
    assert np.allclose(delta_mk_ab, expected_ab, atol=1e-8)
    assert np.allclose(delta_c, expected_cd, atol=1e-8)
    assert np.allclose(delta_d, expected_cd, atol=1e-8)
    assert np.allclose(delta_mk_cd, expected_cd, atol=1e-8)
    assert np.allclose(delta_mid, expected_none, atol=1e-8)


def test_marking_line_butt_shrinkage_direction_moves_points_toward_weld_line() -> None:
    geom = GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate_with_marks",
                    "dims": {"L": 200.0, "H": 100.0, "t": 10.0},
                    "features": {
                        "points": {
                            "A": [40.0, 0.0, 0.0],
                            "B": [160.0, 0.0, 0.0],
                            "C": [160.0, 100.0, 0.0],
                            "D": [40.0, 100.0, 0.0],
                            "MK_AB": [20.0, 0.0, 0.0],
                            "MK_CD": [180.0, 100.0, 0.0],
                            "ON_AB": [100.0, 0.0, 0.0],
                            "ON_CD": [100.0, 100.0, 0.0],
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
        "fitup_pair0": [
            {
                "guest_instance": "G1",
                "pair_index": 0,
                "g_real_0": 4.0,
                "weld_x_local_0": 100.0,
            }
        ],
        "fitup_pair1": [
            {
                "guest_instance": "G1",
                "pair_index": 1,
                "g_real_0": 2.0,
                "weld_x_local_0": 100.0,
            }
        ],
    }

    engine = ProcessEngine(geom, flow, np.random.default_rng(0))
    engine.apply_steps(state)

    s0 = 0.18 * 4.0
    s1 = 0.18 * 2.0
    expected_neg_side_ab = np.array([+s0, 0.0, 0.0], dtype=float)
    expected_pos_side_ab = np.array([-s0, 0.0, 0.0], dtype=float)
    expected_neg_side_cd = np.array([+s1, 0.0, 0.0], dtype=float)
    expected_pos_side_cd = np.array([-s1, 0.0, 0.0], dtype=float)
    expected_none = np.zeros(3, dtype=float)

    assert np.allclose(state.get_point_offset("G1", "A"), expected_neg_side_ab, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "B"), expected_pos_side_ab, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "MK_AB"), expected_neg_side_ab, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "ON_AB"), expected_none, atol=1e-8)

    assert np.allclose(state.get_point_offset("G1", "D"), expected_neg_side_cd, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "C"), expected_pos_side_cd, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "MK_CD"), expected_pos_side_cd, atol=1e-8)
    assert np.allclose(state.get_point_offset("G1", "ON_CD"), expected_none, atol=1e-8)

    assert np.allclose(state.get_point_offset("G1", "MID"), expected_none, atol=1e-8)
