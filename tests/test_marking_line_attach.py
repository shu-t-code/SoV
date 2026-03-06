import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point, rpy_to_rotation_matrix
from sov_app.engine.process_engine import ProcessEngine


def _build_geom_with_mark_line() -> GeometryModel:
    return GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate_A_proto",
                    "dims": {"L": 1998.0, "H": 1000.0, "t": 10.0},
                    "features": {
                        "points": {
                            "A": [0.0, 0.0, 0.0],
                            "B": [1998.0, 0.0, 0.0],
                            "C": [1998.0, 1000.0, 0.0],
                            "D": [0.0, 1000.0, 0.0],
                            "MK1_P0": [1000.0, 0.0, 0.0],
                            "MK1_P1": [1000.0, 1000.0, 0.0],
                        },
                        "lines": {
                            "MK1": {"p0": "points.MK1_P0", "p1": "points.MK1_P1"},
                        },
                    },
                },
                {
                    "id": "longi_C_proto",
                    "dims": {"L": 500.0, "H": 1000.0, "t": 10.0},
                    "features": {
                        "points": {
                            "c1": [0.0, 0.0, 0.0],
                            "c2": [500.0, 0.0, 0.0],
                            "c3": [500.0, 1000.0, 0.0],
                            "c4": [0.0, 1000.0, 0.0],
                        }
                    },
                },
            ],
            "instances": [
                {"id": "A1", "prototype": "plate_A_proto", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "A2", "prototype": "plate_A_proto", "frame": {"parent": "world", "origin": [2100.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "C2", "prototype": "longi_C_proto", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 30.0]}},
            ],
        }
    )


def test_marking_line_world_follows_transform_and_cutting_noise_is_not_written_to_marking_points() -> None:
    geom = _build_geom_with_mark_line()
    flow = FlowModel(
        {
            "selectors": {"A2_only": {"ids": ["A2"]}},
            "steps": [
                {
                    "id": "10_cutting",
                    "op": "apply_variation",
                    "target": {"union": ["A2_only"]},
                    "model": {
                        "per_point_xy_noise": True,
                        "point_dx": {"type": "Fixed", "value": 1.0},
                        "point_dy": {"type": "Fixed", "value": -2.0},
                        "point_dz": {"type": "Fixed", "value": 0.0},
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    state.set_transform("A2", np.array([2500.0, 200.0, 0.0], dtype=float), [0.0, 0.0, 90.0])
    engine.apply_steps(state)

    tr = state.get_transform("A2")
    r = rpy_to_rotation_matrix(*tr["rpy_deg"])
    expected_p0 = r @ np.array([1000.0, 0.0, 0.0], dtype=float) + tr["origin"]
    expected_p1 = r @ np.array([1000.0, 1000.0, 0.0], dtype=float) + tr["origin"]

    p0 = get_world_point(geom, state, "A2", "lines.MK1.p0")
    p1 = get_world_point(geom, state, "A2", "lines.MK1.p1")
    assert np.allclose(p0, expected_p0, atol=1e-6)
    assert np.allclose(p1, expected_p1, atol=1e-6)

    assert np.allclose(state.get_point_offset("A2", "MK1_P0"), np.zeros(3), atol=1e-12)
    assert np.allclose(state.get_point_offset("A2", "MK1_P1"), np.zeros(3), atol=1e-12)
    assert np.allclose(state.get_point_offset("A2", "A"), np.array([1.0, -2.0, 0.0]), atol=1e-12)


def test_fitup_attach_to_marking_line_aligns_guest_ref_line_to_current_base_mark_line() -> None:
    geom = _build_geom_with_mark_line()
    flow = FlowModel(
        {
            "selectors": {},
            "steps": [
                {
                    "id": "butt_move_A2",
                    "op": "fitup_pair_chain",
                    "base": {"instance": "A1", "p0": "points.B", "p1": "points.C"},
                    "guest": {"instance": "A2", "q0": "points.A", "q1": "points.D"},
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
                },
                {
                    "id": "attach_c2_to_a2_marking",
                    "op": "fitup_attach_to_marking_line",
                    "base": {
                        "instance": "A2",
                        "mark_line": {"p0": "points.MK1_P0", "p1": "points.MK1_P1"},
                    },
                    "guest": {
                        "instance": "C2",
                        "ref_line": {"p0": "points.c1", "p1": "points.c4"},
                    },
                },
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(state)

    p0 = get_world_point(geom, state, "A2", "points.MK1_P0")
    p1 = get_world_point(geom, state, "A2", "points.MK1_P1")
    q0 = get_world_point(geom, state, "C2", "points.c1")
    q1 = get_world_point(geom, state, "C2", "points.c4")

    assert np.allclose(q0, p0, atol=1e-6)
    assert np.allclose(q1, p1, atol=1e-6)


def test_fitup_array_attach_still_works() -> None:
    geom = _build_geom_with_mark_line()
    flow = FlowModel(
        {
            "selectors": {"C2_only": {"ids": ["C2"]}},
            "steps": [
                {
                    "id": "legacy_attach",
                    "op": "fitup_array_attach",
                    "base": {"instance": "A2"},
                    "guests": {
                        "selector": "C2_only",
                        "pattern": {"start_offset_mm": 998.0, "pitch_on_base_mm": 0.0},
                    },
                    "constraints": {"set_rpy_deg": [0.0, -90.0, 0.0]},
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    engine.apply_steps(state)

    a2_origin = state.get_transform("A2")["origin"]
    c2 = state.get_transform("C2")
    assert np.allclose(c2["origin"], a2_origin + np.array([998.0, 0.0, 0.0]), atol=1e-6)
    assert c2["rpy_deg"] == [0.0, -90.0, 0.0]
