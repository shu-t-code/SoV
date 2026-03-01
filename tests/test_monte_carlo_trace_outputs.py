from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from sov_app.engine.core_models import FlowModel, GeometryModel
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.monte_carlo import MonteCarloSimulator


def test_monte_carlo_trace_outputs_are_generated(tmp_path: Path) -> None:
    geom_dict, flow_dict = load_data_from_csv(Path("data/model_onefile_buttpair_single_steps.csv"))
    geom = GeometryModel(geom_dict)
    flow = FlowModel(flow_dict)
    sim = MonteCarloSimulator(geom, flow)

    steps_mask = [True for _ in flow.steps]
    results = sim.run(n_trials=2, steps_mask=steps_mask, seed=42, out_dir=tmp_path, trace=True)

    mc_results = tmp_path / "mc_results.csv"
    results.to_csv(mc_results, index=False)

    assert mc_results.exists()
    assert "trial" in results.columns

    trace_files = sorted(tmp_path.glob("mc_trace_*__vertices.csv"))
    assert len(trace_files) == len(flow.steps)

    required_cols = {
        "trial",
        "seed_used",
        "step_idx",
        "step_id",
        "op",
        "instance_id",
        "vertex",
        "x_before",
        "y_before",
        "z_before",
        "x_after",
        "y_after",
        "z_after",
        "model_spec_json",
        "model_dists_json",
    }
    trace_df = pd.read_csv(trace_files[0])
    assert required_cols.issubset(set(trace_df.columns))

    summary_file = tmp_path / "mc_worstcase_summary.csv"
    assert summary_file.exists()
    summary_df = pd.read_csv(summary_file)
    assert len(summary_df) == len(flow.steps)

    worst_files = sorted(tmp_path.glob("mc_trace_worst_*__vertices.csv"))
    assert len(worst_files) == len(flow.steps)
