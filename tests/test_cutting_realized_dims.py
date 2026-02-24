import numpy as np

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel
from sov_app.engine.process_engine import ProcessEngine


def test_recompute_realized_dims_uses_local_components() -> None:
    geom = GeometryModel(
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
                    "frame": {"parent": "world", "origin": [10.0, -20.0, 5.0], "rpy_deg": [0.0, 0.0, 33.0]},
                }
            ],
        }
    )
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
