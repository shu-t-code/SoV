"""Monte Carlo simulation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from .io_csv import load_data_from_csv
from .process_engine import ProcessEngine

ProcessEngineFactory = Callable[[GeometryModel, FlowModel, np.random.Generator], ProcessEngine]


def build_state_for_trial(
    geom: GeometryModel,
    flow: FlowModel,
    steps_mask: List[bool],
    trial: int,
    seed_base: int,
    process_engine_factory: ProcessEngineFactory = ProcessEngine,
) -> AssemblyState:
    rng = np.random.default_rng(seed_base + trial)
    state = AssemblyState(geom)
    engine = process_engine_factory(geom, flow, rng)
    engine.apply_steps(state, steps_mask)
    return state


def run_pair_distance_trials(
    geom: GeometryModel,
    flow: FlowModel,
    steps_mask: List[bool],
    p1_instance: str,
    p1_ref: str,
    p2_instance: str,
    p2_ref: str,
    n_trials: int,
    seed: int,
    process_engine_factory: ProcessEngineFactory = ProcessEngine,
) -> np.ndarray:
    vals: List[float] = []
    for trial in range(n_trials):
        state = build_state_for_trial(geom, flow, steps_mask, trial, seed, process_engine_factory)
        p1 = get_world_point(geom, state, p1_instance, p1_ref)
        p2 = get_world_point(geom, state, p2_instance, p2_ref)
        vals.append(float(np.linalg.norm(p2 - p1)))
    return np.asarray(vals, dtype=float)


class MonteCarloSimulator:
    def __init__(
        self,
        geom: GeometryModel,
        flow: FlowModel,
        process_engine_factory: ProcessEngineFactory = ProcessEngine,
    ):
        self.geom = geom
        self.flow = flow
        self.process_engine_factory = process_engine_factory

    def run(
        self,
        n_trials: int,
        steps_mask: List[bool],
        seed: int = 42,
        out_dir: Path | None = None,
        trace: bool = False,
    ) -> pd.DataFrame:
        results: List[Dict[str, float]] = []
        trace_enabled = bool(trace and out_dir is not None)
        trace_context: Dict[str, Any] | None = None
        if trace_enabled:
            trace_context = self._init_trace_context(Path(out_dir), steps_mask)

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            state = AssemblyState(self.geom)
            engine = self.process_engine_factory(self.geom, self.flow, rng)
            if trace_context is None:
                engine.apply_steps(state, steps_mask)
            else:
                trace_context["trial"] = trial
                trace_context["seed_used"] = seed + trial
                trace_context["before_snapshots"].clear()

                def on_step_before(step_idx: int, step: Dict[str, Any], current_state: AssemblyState) -> None:
                    trace_context["before_snapshots"][step_idx] = self._capture_step_vertices(step_idx, current_state)

                def on_step_after(step_idx: int, step: Dict[str, Any], current_state: AssemblyState) -> None:
                    self._record_step_trace(trace_context, step_idx, step, current_state)

                engine.apply_steps(state, steps_mask, on_step_before=on_step_before, on_step_after=on_step_after)

            metrics = self._compute_metrics(state)
            metrics["trial"] = trial
            results.append(metrics)

        if trace_context is not None:
            self._finalize_trace(trace_context)
        return pd.DataFrame(results)

    def _init_trace_context(self, out_dir: Path, steps_mask: List[bool]) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        enabled_step_indices = [
            idx for idx, _ in enumerate(self.flow.steps) if idx < len(steps_mask) and steps_mask[idx]
        ]

        context: Dict[str, Any] = {
            "out_dir": out_dir,
            "trace_files": {},
            "trace_writers": {},
            "trace_handles": {},
            "before_snapshots": {},
            "trial": -1,
            "seed_used": -1,
            "worst": {},
            "steps_mask": steps_mask,
            "step_meta": {},
            "nominal_after": self._compute_nominal_after(steps_mask),
        }

        for step_idx in enabled_step_indices:
            step = self.flow.steps[step_idx]
            safe_step_id = self._sanitize_for_filename(step.get("id", "noid"))
            trace_name = f"mc_trace_{step_idx:02d}_{safe_step_id}__vertices.csv"
            trace_path = out_dir / trace_name
            handle = trace_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(
                [
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
                    "x_after_nominal",
                    "y_after_nominal",
                    "z_after_nominal",
                    "dx",
                    "dy",
                    "dz",
                    "model_spec_json",
                    "model_dists_json",
                ]
            )
            context["trace_files"][step_idx] = trace_path
            context["trace_writers"][step_idx] = writer
            context["trace_handles"][step_idx] = handle
            context["step_meta"][step_idx] = {
                "step_id": str(step.get("id", "noid") or "noid"),
                "op": str(step.get("op", "")),
            }
        return context

    def _compute_nominal_after(self, steps_mask: List[bool]) -> Dict[tuple[int, str, str], np.ndarray]:
        nominal_after: Dict[tuple[int, str, str], np.ndarray] = {}
        nominal_state = AssemblyState(self.geom)
        nominal_engine = self.process_engine_factory(self.geom, self.flow, np.random.default_rng(0))
        original_sample = nominal_engine._sample

        def _sample_nominal(spec: Any) -> float:
            return self._sample_nominal_value(spec)

        nominal_engine._sample = _sample_nominal
        try:
            for step_idx, step in enumerate(self.flow.steps):
                if step_idx >= len(steps_mask) or not steps_mask[step_idx]:
                    continue
                nominal_engine.apply_step(step, nominal_state)
                nominal_vertices = self._capture_step_vertices(step_idx, nominal_state)
                for key, pos in nominal_vertices.items():
                    nominal_after[(step_idx, key[0], key[1])] = pos
        finally:
            nominal_engine._sample = original_sample
        return nominal_after

    def _sample_nominal_value(self, spec: Any) -> float:
        if isinstance(spec, str):
            return self._sample_nominal_value(self.flow.dists.get(spec, 0.0))
        if isinstance(spec, (int, float)):
            return float(spec)
        if not isinstance(spec, dict):
            return 0.0

        dtype = str(spec.get("type", "Fixed"))
        if dtype == "Fixed":
            return float(spec.get("value", 0.0))
        if dtype in {"NormalLinear", "LogNormalLinear"}:
            return float(spec.get("mean", 0.0))
        if "mean" in spec:
            return float(spec.get("mean", 0.0))
        if "value" in spec:
            return float(spec.get("value", 0.0))
        return 0.0

    def _capture_step_vertices(self, step_idx: int, state: AssemblyState) -> Dict[tuple[str, str], np.ndarray]:
        captures: Dict[tuple[str, str], np.ndarray] = {}
        for inst_id in self.geom.get_instance_ids():
            proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
            for vertex in proto.get("features", {}).get("points", {}).keys():
                captures[(inst_id, str(vertex))] = get_world_point(self.geom, state, inst_id, f"points.{vertex}")
        return captures

    def _record_step_trace(
        self,
        trace_context: Dict[str, Any],
        step_idx: int,
        step: Dict[str, Any],
        state: AssemblyState,
    ) -> None:
        before = trace_context["before_snapshots"].get(step_idx, {})
        after = self._capture_step_vertices(step_idx, state)
        writer = trace_context["trace_writers"].get(step_idx)
        if writer is None:
            return

        model_spec_json = json.dumps(step.get("model", {}), ensure_ascii=False, sort_keys=True)
        model_dists_json = json.dumps(self._resolve_model_dists(step.get("model", {})), ensure_ascii=False, sort_keys=True)
        rows: List[List[Any]] = []
        max_vertex_delta = -1.0
        for key, pos_after in after.items():
            inst_id, vertex = key
            pos_before = before.get(key, pos_after)
            nominal_after = trace_context["nominal_after"].get((step_idx, inst_id, vertex), pos_after)
            delta = float(np.linalg.norm(pos_after - pos_before))
            max_vertex_delta = max(max_vertex_delta, delta)
            row = [
                trace_context["trial"],
                trace_context["seed_used"],
                step_idx,
                str(step.get("id", "noid") or "noid"),
                str(step.get("op", "")),
                inst_id,
                vertex,
                float(pos_before[0]),
                float(pos_before[1]),
                float(pos_before[2]),
                float(pos_after[0]),
                float(pos_after[1]),
                float(pos_after[2]),
                float(nominal_after[0]),
                float(nominal_after[1]),
                float(nominal_after[2]),
                float(pos_after[0] - nominal_after[0]),
                float(pos_after[1] - nominal_after[1]),
                float(pos_after[2] - nominal_after[2]),
                model_spec_json,
                model_dists_json,
            ]
            writer.writerow(row)
            rows.append(row)

        current_worst = trace_context["worst"].get(step_idx)
        if current_worst is None or max_vertex_delta > current_worst["value_worst"]:
            trace_context["worst"][step_idx] = {
                "criterion": "max_vertex_delta",
                "trial_worst": trace_context["trial"],
                "value_worst": max_vertex_delta,
                "rows": rows,
            }

    def _finalize_trace(self, trace_context: Dict[str, Any]) -> None:
        for handle in trace_context["trace_handles"].values():
            handle.close()

        summary_path = trace_context["out_dir"] / "mc_worstcase_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["step_idx", "step_id", "op", "criterion", "trial_worst", "value_worst"])
            for step_idx in sorted(trace_context["worst"].keys()):
                meta = trace_context["step_meta"][step_idx]
                worst = trace_context["worst"][step_idx]
                writer.writerow(
                    [
                        step_idx,
                        meta["step_id"],
                        meta["op"],
                        worst["criterion"],
                        worst["trial_worst"],
                        worst["value_worst"],
                    ]
                )

                safe_step_id = self._sanitize_for_filename(meta["step_id"])
                worst_trace_path = trace_context["out_dir"] / f"mc_trace_worst_{step_idx:02d}_{safe_step_id}__vertices.csv"
                with worst_trace_path.open("w", newline="", encoding="utf-8") as worst_fp:
                    worst_writer = csv.writer(worst_fp)
                    worst_writer.writerow(
                        [
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
                            "x_after_nominal",
                            "y_after_nominal",
                            "z_after_nominal",
                            "dx",
                            "dy",
                            "dz",
                            "model_spec_json",
                            "model_dists_json",
                        ]
                    )
                    worst_writer.writerows(worst["rows"])

    def _resolve_model_dists(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.flow.dists.get(value, value)
        if isinstance(value, list):
            return [self._resolve_model_dists(item) for item in value]
        if isinstance(value, dict):
            return {k: self._resolve_model_dists(v) for k, v in value.items()}
        return value

    def _sanitize_for_filename(self, value: Any) -> str:
        text = str(value or "noid")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
        safe = safe.strip("_")
        return safe or "noid"

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
        process_engine = ProcessEngine(geom, flow, rng)
        process_engine.apply_step(cutting_step, state)
        for inst_id, edge_name, ep0, ep1 in edge_defs:
            p0 = get_world_point(geom, state, inst_id, f"points.{ep0}")
            p1 = get_world_point(geom, state, inst_id, f"points.{ep1}")
            values[(inst_id, edge_name)].append(float(np.linalg.norm(p1 - p0)))

    print(f"[EDGE STD] After 10_cutting  (n_trials={n_trials}, seed={seed})")
    for inst_id, edge_name in sorted(values.keys()):
        arr = np.asarray(values[(inst_id, edge_name)], dtype=float)
        print(f"  {inst_id}:{edge_name:<6}  mean={float(arr.mean()):10.6f} mm   std={float(arr.std(ddof=1)):10.6f} mm")


__all__ = [
    "MonteCarloSimulator",
    "build_state_for_trial",
    "print_all_edge_stds_after_cutting",
    "run_pair_distance_trials",
]
