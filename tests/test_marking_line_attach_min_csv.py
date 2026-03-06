from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine


MIN_CSV = Path("data/model_min_marking_attach.csv")
MIN_BUTT_SHRINK_CSV = Path("data/model_min_marking_attach_butt_shrink.csv")


def _load_models() -> tuple[GeometryModel, FlowModel]:
    geom_dict, flow_dict = load_data_from_csv(MIN_CSV)
    return GeometryModel(geom_dict), FlowModel(flow_dict)


def _load_butt_shrink_models() -> tuple[GeometryModel, FlowModel]:
    geom_dict, flow_dict = load_data_from_csv(MIN_BUTT_SHRINK_CSV)
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
    geom, flow = _load_butt_shrink_models()
    assert [step["op"] for step in flow.steps] == ["fitup_pair_chain", "fitup_pair_chain", "welding_distortion"]

    state_fitup = AssemblyState(geom)
    engine_fitup = ProcessEngine(geom, flow, np.random.default_rng(0))
    engine_fitup.apply_steps(state_fitup, steps_mask=[True, True, False])

    fitup_metrics = state_fitup.butt_fitup_metrics
    pair0_metric = fitup_metrics["fitup_pair0"][0]
    pair1_metric = fitup_metrics["fitup_pair1"][0]

    s0 = 0.18 * max(float(pair0_metric["g_real_0"]), 0.0)
    s1 = 0.18 * max(float(pair1_metric["g_real_0"]), 0.0)
    v = np.array(pair0_metric["transverse_dir_world"], dtype=float)
    v /= np.linalg.norm(v)
    expected_ab = -s0 * v
    expected_cd = -s1 * v

    l_dists = [float(step["model"]["butt_fitup"]["L_dist"]) for step in flow.steps[:2]]
    assert l_dists == [4.0, 1.0]
    assert l_dists[0] != pytest.approx(float(np.linalg.norm(expected_ab)))
    assert l_dists[1] != pytest.approx(float(np.linalg.norm(expected_cd)))

    state_full = AssemblyState(geom)
    engine_full = ProcessEngine(geom, flow, np.random.default_rng(0))
    engine_full.apply_steps(state_full)

    def _delta(instance: str, ref: str) -> np.ndarray:
        return get_world_point(geom, state_full, instance, ref) - get_world_point(geom, state_fitup, instance, ref)

    ab_a = _delta("G1", "points.A")
    ab_b = _delta("G1", "points.B")
    ab_mk = _delta("G1", "points.MK_AB")
    cd_c = _delta("G1", "points.C")
    cd_d = _delta("G1", "points.D")
    cd_mk = _delta("G1", "points.MK_CD")
    mid = _delta("G1", "points.MID")

    assert np.allclose(ab_a, expected_ab, atol=1e-8)
    assert np.allclose(ab_b, expected_ab, atol=1e-8)
    assert np.allclose(ab_mk, expected_ab, atol=1e-8)

    assert np.allclose(cd_c, expected_cd, atol=1e-8)
    assert np.allclose(cd_d, expected_cd, atol=1e-8)
    assert np.allclose(cd_mk, expected_cd, atol=1e-8)

    assert np.allclose(mid, np.zeros(3, dtype=float), atol=1e-8)
