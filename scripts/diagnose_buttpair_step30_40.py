from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine

CSV_PATH = Path("data/model_onefile_buttpair_with_fillet_attach.csv")
REPORT_PATH = Path("reports/diagnosis_buttpair_step30_40.md")


def _state_after(last_step_index: int, seed: int = 123) -> tuple[GeometryModel, FlowModel, AssemblyState]:
    geom_dict, flow_dict = load_data_from_csv(CSV_PATH)
    geom = GeometryModel(geom_dict)
    flow = FlowModel(flow_dict)
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(seed))
    mask = [i <= last_step_index for i in range(len(flow.steps))]
    engine.apply_steps(state, steps_mask=mask)
    return geom, flow, state


def _world_points(geom: GeometryModel, state: AssemblyState) -> dict[str, np.ndarray]:
    refs = [
        ("C1", "points.c1"),
        ("C1", "points.c2"),
        ("C1", "points.c3"),
        ("C1", "points.c4"),
        ("C2", "points.c1"),
        ("C2", "points.c2"),
        ("C2", "points.c3"),
        ("C2", "points.c4"),
        ("A1", "points.B"),
        ("A1", "points.C"),
        ("A2", "points.A"),
        ("A2", "points.B"),
        ("A2", "points.C"),
        ("A2", "points.D"),
    ]
    out: dict[str, np.ndarray] = {}
    for inst, ref in refs:
        out[f"{inst}:{ref}"] = np.asarray(get_world_point(geom, state, inst, ref), dtype=float)
    return out


def _distance_stats(last_step_index: int, n: int = 200) -> tuple[float, float, float, float]:
    dists = []
    for seed in range(n):
        geom, _, state = _state_after(last_step_index=last_step_index, seed=seed)
        p1 = np.asarray(get_world_point(geom, state, "C1", "points.c2"), dtype=float)
        p2 = np.asarray(get_world_point(geom, state, "C2", "points.c2"), dtype=float)
        dists.append(float(np.linalg.norm(p2 - p1)))
    arr = np.asarray(dists, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def main() -> None:
    geom_dict, flow_dict = load_data_from_csv(CSV_PATH)
    flow = FlowModel(flow_dict)

    step30 = flow.steps[3]
    step40 = flow.steps[4]

    geom21, _, state21 = _state_after(2, seed=123)
    geom30, _, state30 = _state_after(3, seed=123)
    geom40, _, state40 = _state_after(4, seed=123)

    points21 = _world_points(geom21, state21)
    points30 = _world_points(geom30, state30)
    points40 = _world_points(geom40, state40)

    transforms = {
        "step21": {k: state21.get_transform(k) for k in ("A1", "A2", "C1", "C2")},
        "step30": {k: state30.get_transform(k) for k in ("A1", "A2", "C1", "C2")},
        "step40": {k: state40.get_transform(k) for k in ("A1", "A2", "C1", "C2")},
    }

    deltas_21_30 = {k: (points30[k] - points21[k]) for k in points21}
    deltas_30_40 = {k: (points40[k] - points30[k]) for k in points30}

    stats21 = _distance_stats(last_step_index=2, n=200)
    stats30 = _distance_stats(last_step_index=3, n=200)
    stats40 = _distance_stats(last_step_index=4, n=200)

    lines: list[str] = []
    lines.append("# Diagnosis: step30/40 effect on C1:points.c2 - C2:points.c2")
    lines.append("")
    lines.append("## 1) Loaded dict (flow.steps[3], flow.steps[4])")
    lines.append("### step[3] 30_mid_fitup_butt_A1_A2")
    lines.append("```json")
    lines.append(json.dumps(step30, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("### step[4] 40_butt_welding_A1_A2")
    lines.append("```json")
    lines.append(json.dumps(step40, ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append("## 2) Transforms (seed=123)")
    lines.append("```json")
    lines.append(json.dumps(transforms, ensure_ascii=False, indent=2, default=lambda x: np.asarray(x).tolist()))
    lines.append("```")

    lines.append("## 3) World point deltas (seed=123)")
    lines.append("### step21 -> step30")
    lines.append("```json")
    lines.append(json.dumps({k: v.tolist() for k, v in deltas_21_30.items()}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("### step30 -> step40")
    lines.append("```json")
    lines.append(json.dumps({k: v.tolist() for k, v in deltas_30_40.items()}, ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append("## 4) Distance stats for C1:points.c2 - C2:points.c2 (N=200)")
    lines.append("- step21まで: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(*stats21))
    lines.append("- step30まで: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(*stats30))
    lines.append("- step40まで: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(*stats40))

    lines.append("")
    lines.append("## 5) butt_fitup_metrics / point_offsets (seed=123)")
    lines.append("### step30 metrics")
    lines.append("```json")
    lines.append(json.dumps(getattr(state30, "butt_fitup_metrics", {}), ensure_ascii=False, indent=2, default=lambda x: np.asarray(x).tolist()))
    lines.append("```")
    lines.append("### step40 metrics")
    lines.append("```json")
    lines.append(json.dumps(getattr(state40, "butt_fitup_metrics", {}), ensure_ascii=False, indent=2, default=lambda x: np.asarray(x).tolist()))
    lines.append("```")
    lines.append("### step40 point offsets (A2)")
    lines.append("```json")
    lines.append(
        json.dumps(
            {k: v.tolist() for k, v in state40.point_offsets.get("A2", {}).items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
