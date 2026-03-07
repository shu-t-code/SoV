#!/usr/bin/env python3
"""Debug fillet_fitup effective point classification for marking-line steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine

CSV_PATH = Path("data/model_onefile_buttpair_with_fillet_attach.csv")
TARGET_STEP_IDS = {
    "10_cutting",
    "20_fitup_C1_on_A1_marking_line",
    "21_fitup_C2_on_A2_marking_line",
}
TARGET_FITUP_IDS = [
    "20_fitup_C1_on_A1_marking_line",
    "21_fitup_C2_on_A2_marking_line",
]


def _build_models() -> tuple[GeometryModel, FlowModel]:
    geom_dict, flow_dict = load_data_from_csv(CSV_PATH)
    steps = [s for s in flow_dict.get("steps", []) if str(s.get("id", "")) in TARGET_STEP_IDS]
    flow_dict = dict(flow_dict)
    flow_dict["steps"] = steps
    return GeometryModel(geom_dict), FlowModel(flow_dict)


def _run_once(seed: int) -> dict[str, dict[str, object]]:
    geom, flow = _build_models()
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, np.random.default_rng(seed))
    engine.apply_steps(state)

    steps_by_id = {str(step.get("id", "")): step for step in flow.steps}
    result: dict[str, dict[str, object]] = {}
    for sid in TARGET_FITUP_IDS:
        step = steps_by_id.get(sid, {})
        fillet = (step.get("model") or {}).get("fillet_fitup") or {}
        lower = list(fillet.get("lower_points_effective", []))
        upper = list(fillet.get("upper_points_effective", []))
        pairs = list(fillet.get("pair_effective", []))
        c2_class = "upper" if "points.c2" in upper else "lower" if "points.c2" in lower else "none"
        result[sid] = {
            "lower_points_effective": lower,
            "upper_points_effective": upper,
            "pair_effective": pairs,
            "points.c2_class": c2_class,
        }
    return result


def main() -> None:
    print(f"csv: {CSV_PATH}")
    print("target_steps:", sorted(TARGET_STEP_IDS))

    once = _run_once(seed=0)
    print("\n=== single run (seed=0) ===")
    for sid in TARGET_FITUP_IDS:
        item = once.get(sid, {})
        print(f"step_id: {sid}")
        print("lower_points_effective:", item.get("lower_points_effective"))
        print("upper_points_effective:", item.get("upper_points_effective"))
        print("pair_effective:", item.get("pair_effective"))
        print("points.c2 class:", item.get("points.c2_class"))
        print()

    tally = {sid: {"upper": 0, "lower": 0, "none": 0} for sid in TARGET_FITUP_IDS}
    n = 200
    for seed in range(n):
        per_run = _run_once(seed=seed)
        for sid in TARGET_FITUP_IDS:
            klass = str(per_run.get(sid, {}).get("points.c2_class", "none"))
            if klass not in ("upper", "lower", "none"):
                klass = "none"
            tally[sid][klass] += 1

    print("=== tally over seeds 0..199 ===")
    for sid in TARGET_FITUP_IDS:
        counts = tally[sid]
        print(f"{sid}: points.c2 upper={counts['upper']} lower={counts['lower']} none={counts['none']} / {n}")


if __name__ == "__main__":
    main()
