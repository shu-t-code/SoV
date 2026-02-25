import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel
from sov_app.engine.process_engine import ProcessEngine


def _build_plate_geom(rpy_deg: list[float]) -> GeometryModel:
    return GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate",
                    "dims": {"L": 100.0, "H": 50.0, "t": 1.0},
                    "features": {
                        "points": {
                            "A": [0.0, 0.0, 0.0],
                            "B": [100.0, 0.0, 0.0],
                            "C": [100.0, 50.0, 0.0],
                            "D": [0.0, 50.0, 0.0],
                        }
                    },
                }
            ],
            "instances": [
                {
                    "id": "p1",
                    "prototype": "plate",
                    "frame": {"parent": "world", "origin": [10.0, -20.0, 5.0], "rpy_deg": rpy_deg},
                }
            ],
        }
    )


def test_recompute_realized_dims_rpy_zero_world_equals_local() -> None:
    geom = _build_plate_geom([0.0, 0.0, 0.0])
    flow = FlowModel({"selectors": {}, "steps": []})
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    state.set_point_offset("p1", "A", np.array([0.0, 0.0, 0.0]))
    state.set_point_offset("p1", "B", np.array([2.0, 0.0, 0.0]))
    state.set_point_offset("p1", "C", np.array([3.0, 1.0, 0.0]))
    state.set_point_offset("p1", "D", np.array([0.0, 4.0, 0.0]))

    engine._recompute_realized_dims_from_points("p1", state)

    dims = state.get_realized_dims("p1")
    assert dims["L_ab"] == 102.0
    assert dims["L_dc"] == 103.0
    assert dims["H_ad"] == 54.0
    assert dims["H_bc"] == 51.0
    assert dims["L"] == 102.5
    assert dims["H"] == 52.5


def test_recompute_realized_dims_uses_local_components_with_rotation() -> None:
    geom = _build_plate_geom([0.0, 0.0, 33.0])
    flow = FlowModel({"selectors": {}, "steps": []})
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    state.set_point_offset("p1", "A", np.array([0.0, 0.0, 0.0]))
    state.set_point_offset("p1", "B", np.array([1.0, 2.0, 0.0]))
    state.set_point_offset("p1", "C", np.array([3.0, 4.0, 0.0]))
    state.set_point_offset("p1", "D", np.array([2.0, 5.0, 0.0]))

    engine._recompute_realized_dims_from_points("p1", state)

    dims = state.get_realized_dims("p1")
    assert dims["L_ab"] == 101.0
    assert dims["L_dc"] == 101.0
    assert dims["H_ad"] == 55.0
    assert dims["H_bc"] == 52.0
    assert dims["L"] == 101.0
    assert dims["H"] == 53.5


def test_cutting_ignores_dim_variations_and_recomputes_from_points() -> None:
    geom = _build_plate_geom([0.0, 0.0, 0.0])
    flow = FlowModel(
        {
            "selectors": {"plate_sel": {"ids": ["p1"]}},
            "steps": [
                {
                    "id": "10_cutting",
                    "op": "apply_variation",
                    "target": {"union": ["plate_sel"]},
                    "model": {
                        "inplane_dx": {"type": "Fixed", "value": 10.0},
                        "inplane_dy": {"type": "Fixed", "value": -5.0},
                        "outplane_dz": {"type": "Fixed", "value": 3.0},
                        "per_point_xy_noise": True,
                        "point_dx": {"type": "Fixed", "value": 1.0},
                        "point_dy": {"type": "Fixed", "value": 2.0},
                        "point_dz": {"type": "Fixed", "value": 0.0},
                        "dims_variation": {
                            "L": {"type": "Fixed", "value": 999.0},
                            "H": {"type": "Fixed", "value": 888.0},
                        },
                    },
                }
            ],
        }
    )
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    original_origin = state.get_transform("p1")["origin"].copy()
    engine.apply_steps(state)

    dims = state.get_realized_dims("p1")
    # All points receive same xy noise, so edge lengths remain nominal.
    assert dims["L_ab"] == 100.0
    assert dims["L_dc"] == 100.0
    assert dims["H_ad"] == 50.0
    assert dims["H_bc"] == 50.0
    assert dims["L"] == 100.0
    assert dims["H"] == 50.0
    assert np.allclose(state.get_transform("p1")["origin"], original_origin)
