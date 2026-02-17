"""Monte Carlo simulation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from .io_csv import load_data_from_csv
from .process_engine import ProcessEngine


class MonteCarloSimulator:
    def __init__(self, geom: GeometryModel, flow: FlowModel):
        self.geom = geom
        self.flow = flow

    def run(self, n_trials: int, steps_mask: List[bool], seed: int = 42) -> pd.DataFrame:
        results: List[Dict[str, float]] = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            state = AssemblyState(self.geom)
            engine = ProcessEngine(self.geom, self.flow, rng)
            for i, step in enumerate(self.flow.steps):
                if i < len(steps_mask) and steps_mask[i]:
                    engine.apply_step(step, state)
            metrics = self._compute_metrics(state)
            metrics["trial"] = trial
            results.append(metrics)
        return pd.DataFrame(results)

    def _compute_metrics(self, state: AssemblyState) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if "A1" in state.transforms and "A2" in state.transforms:
            a1 = state.get_transform("A1")
            a2 = state.get_transform("A2")
            a1_dims = state.get_realized_dims("A1")
            proto = self.geom.get_prototype(self.geom.get_instance("A1")["prototype"])
            l_size = a1_dims.get("L", float(proto.get("dims", {}).get("L", 2000.0)))
            metrics["edge_gap_x"] = float(a2["origin"][0] - (a1["origin"][0] + l_size))
            metrics["flush_z"] = float(abs(a2["origin"][2] - a1["origin"][2]))

        for m in self.flow.measurements:
            name = m.get("metric_name", m.get("id", "dist"))
            p1, p2 = m["p1"], m["p2"]
            v1 = get_world_point(self.geom, state, p1["instance"], p1["ref"])
            v2 = get_world_point(self.geom, state, p2["instance"], p2["ref"])
            metrics[name] = float(np.linalg.norm(v2 - v1))
        return metrics


def print_all_edge_stds_after_cutting(csv_path: Path, n_trials: int = 5000, seed: int = 42) -> None:
    geom_dict, flow_dict = load_data_from_csv(csv_path)
    geom = GeometryModel(geom_dict)
    flow = FlowModel(flow_dict)

    cutting_step = next((s for s in flow.steps if s.get("id") == "10_cutting"), None)
    if cutting_step is None:
        raise RuntimeError('Step id="10_cutting" not found in flow.steps')

    edge_defs = []
    for inst_id, inst in geom.instances.items():
        proto = geom.get_prototype(inst.get("prototype", ""))
        for edge_name, e in proto.get("features", {}).get("edges", {}).items():
            endpoints = e.get("endpoints", [])
            if len(endpoints) >= 2:
                edge_defs.append((inst_id, edge_name, endpoints[0], endpoints[1]))

    if not edge_defs:
        print("[EDGE STD] No edges found in prototypes.features.edges")
        return

    values = {(inst_id, edge_name): [] for (inst_id, edge_name, _, _) in edge_defs}
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        state = AssemblyState(geom)
        ProcessEngine(geom, flow, rng).apply_step(cutting_step, state)
        for inst_id, edge_name, ep0, ep1 in edge_defs:
            p0 = get_world_point(geom, state, inst_id, f"points.{ep0}")
            p1 = get_world_point(geom, state, inst_id, f"points.{ep1}")
            values[(inst_id, edge_name)].append(float(np.linalg.norm(p1 - p0)))

    print(f"[EDGE STD] After 10_cutting  (n_trials={n_trials}, seed={seed})")
    for inst_id, edge_name in sorted(values.keys()):
        arr = np.asarray(values[(inst_id, edge_name)], dtype=float)
        print(f"  {inst_id}:{edge_name:<6}  mean={float(arr.mean()):10.6f} mm   std={float(arr.std(ddof=1)):10.6f} mm")


__all__ = ["MonteCarloSimulator", "print_all_edge_stds_after_cutting"]
