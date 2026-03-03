import pytest

np = pytest.importorskip("numpy")

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel
from sov_app.engine.process_engine import ProcessEngine


def test_fillet_fitup_array_attach_uses_independent_x_distributions_for_lower_and_upper() -> None:
    geom = GeometryModel(
        {
            "prototypes": [
                {
                    "id": "plate",
                    "dims": {"L": 100.0, "H": 50.0, "t": 10.0},
                    "features": {
                        "points": {
                            "A": [0.0, 0.0, 0.0],
                            "B": [100.0, 10.0, 0.0],
                            "C": [100.0, 40.0, 10.0],
                            "D": [0.0, 50.0, 10.0],
                        }
                    },
                }
            ],
            "instances": [
                {"id": "BASE", "prototype": "plate", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
                {"id": "G1", "prototype": "plate", "frame": {"parent": "world", "origin": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}},
            ],
        }
    )
    flow = FlowModel(
        {
            "selectors": {"guest_sel": {"ids": ["G1"]}},
            "steps": [
                {
                    "id": "20_fitup_C1_on_A1_center",
                    "op": "fitup_array_attach",
                    "base": {"instance": "BASE"},
                    "guests": {"selector": "guest_sel", "pattern": {"pitch_on_base_mm": 0.0, "start_offset_mm": 0.0}},
                    "constraints": {"set_rpy_deg": [0.0, 0.0, 0.0]},
                    "model": {
                        "fillet_fitup": {
                            "delta_mA": {"type": "Fixed", "value": 0.0},
                            "delta_mB": {"type": "Fixed", "value": 0.0},
                            "x_lower": {"type": "Fixed", "value": 0.0},
                            "x_upper": {"type": "Fixed", "value": 0.0},
                            "delta_y": {"type": "Fixed", "value": 0.0},
                            "z_lower": {"type": "Fixed", "value": 0.0},
                        }
                    },
                }
            ],
        }
    )

    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(0))

    # dm_a, dm_b, then (x_lower, x_upper, z_lower) for each of the 2 lower/upper pairs.
    samples = [1.0, 2.0, 10.0, 30.0, 0.0, 11.0, 31.0, 0.0]
    engine._sample = lambda spec: samples.pop(0)

    engine.apply_steps(state)

    offsets = state.point_offsets["G1"]
    lower_dx = sorted(float(offsets[name][0]) for name in ("A", "B"))
    upper_dx = sorted(float(offsets[name][0]) for name in ("C", "D"))

    assert lower_dx == [11.0, 12.0]
    assert upper_dx == [31.0, 32.0]
