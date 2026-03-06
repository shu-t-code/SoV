from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine


MIN_CSV = Path("data/model_min_marking_attach.csv")
BUTT_MIN_CSV = Path("data/model_min_marking_attach_butt_shrink.csv")


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


def test_min_csv_butt_transverse_shrinkage_moves_ab_cd_and_marking_points_independently() -> None:
    g0 = 4.0
    g1 = 1.0
    s0 = 0.18 * max(g0, 0.0)
    s1 = 0.18 * max(g1, 0.0)

    geom_dict, flow_dict = load_data_from_csv(BUTT_MIN_CSV)
    geom = GeometryModel(geom_dict)
    flow = FlowModel(flow_dict)
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    ab_before = get_world_point(geom, state, "AB", "points.A")
    mk_ab_before = get_world_point(geom, state, "AB", "points.MK_AB")
    cd_before = get_world_point(geom, state, "CD", "points.A")
    mk_cd_before = get_world_point(geom, state, "CD", "points.MK_CD")
    mid_before = get_world_point(geom, state, "MID", "points.A")

    engine.apply_steps(state)

    ab_after = get_world_point(geom, state, "AB", "points.A")
    mk_ab_after = get_world_point(geom, state, "AB", "points.MK_AB")
    cd_after = get_world_point(geom, state, "CD", "points.A")
    mk_cd_after = get_world_point(geom, state, "CD", "points.MK_CD")
    mid_after = get_world_point(geom, state, "MID", "points.A")

    ab_disp = ab_after - ab_before
    mk_ab_disp = mk_ab_after - mk_ab_before
    cd_disp = cd_after - cd_before
    mk_cd_disp = mk_cd_after - mk_cd_before

    assert np.allclose(ab_disp, np.array([-s0, 0.0, 0.0], dtype=float), atol=1e-8)
    assert np.allclose(mk_ab_disp, np.array([-s0, 0.0, 0.0], dtype=float), atol=1e-8)

    assert np.allclose(cd_disp, np.array([-s1, 0.0, 0.0], dtype=float), atol=1e-8)
    assert np.allclose(mk_cd_disp, np.array([-s1, 0.0, 0.0], dtype=float), atol=1e-8)

    assert not np.isclose(abs(ab_disp[0]), abs(cd_disp[0]))
    assert np.allclose(mid_after - mid_before, np.zeros(3, dtype=float), atol=1e-8)
