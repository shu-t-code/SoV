#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sov_app.engine.core_models import AssemblyState, FlowModel, GeometryModel, get_world_point
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine


def _step_desc(step: dict[str, Any], idx: int) -> str:
    return f"idx={idx}, id={step.get('id', '<no-id>')}, op={step.get('op', '<no-op>')}"


def _resolve_step(flow: FlowModel, step_idx: int | None, step_id: str | None) -> tuple[int, dict[str, Any]]:
    if (step_idx is None) == (step_id is None):
        raise ValueError("Specify exactly one of --step-idx or --step-id.")

    if step_idx is not None:
        if step_idx < 0 or step_idx >= len(flow.steps):
            raise ValueError(f"--step-idx out of range: {step_idx} (n_steps={len(flow.steps)})")
        return step_idx, flow.steps[step_idx]

    matches = [(i, s) for i, s in enumerate(flow.steps) if str(s.get("id", "")) == str(step_id)]
    if not matches:
        all_ids = [str(s.get("id", "")) for s in flow.steps]
        raise ValueError(f"--step-id '{step_id}' not found. available_step_ids={all_ids}")
    if len(matches) > 1:
        indices = [i for i, _ in matches]
        raise ValueError(f"--step-id '{step_id}' is not unique. matching_indices={indices}")
    return matches[0]


def _resolve_vertex_names(step: dict[str, Any], geom: GeometryModel, guest_id: str) -> list[str]:
    inst = geom.get_instance(guest_id)
    if not inst:
        raise ValueError(f"Unknown guest instance: {guest_id}")

    proto_id = str(inst.get("prototype", ""))
    proto = geom.get_prototype(proto_id)
    all_point_names = sorted(proto.get("features", {}).get("points", {}).keys())
    if not all_point_names:
        raise ValueError(f"Guest '{guest_id}' prototype '{proto_id}' has no points.")

    ref_points: set[str] = set()
    guest_def = step.get("guest", {})
    ref_line = guest_def.get("ref_line", {}) if isinstance(guest_def, dict) else {}
    if isinstance(ref_line, dict):
        for key in ("p0", "p1"):
            ref = ref_line.get(key)
            if isinstance(ref, str) and ref.startswith("points."):
                ref_points.add(ref.split(".", 1)[1])

    without_ref = [p for p in all_point_names if p not in ref_points]
    if len(without_ref) == 4:
        vertex_names = without_ref
    else:
        vertex_names = all_point_names

    if len(vertex_names) != 4:
        raise ValueError(
            f"Expected exactly 4 physical vertices for guest='{guest_id}'. "
            f"all_points={all_point_names}, ref_points={sorted(ref_points)}, selected={vertex_names}"
        )
    return vertex_names


def _effective_grouping(geom: GeometryModel, state: AssemblyState, guest_id: str, vertex_names: list[str]) -> dict[str, Any]:
    world_points = {v: get_world_point(geom, state, guest_id, f"points.{v}") for v in vertex_names}
    by_z = sorted(vertex_names, key=lambda v: (float(world_points[v][2]), v))
    lower_vertices = by_z[:2]
    upper_vertices = by_z[2:]
    lower_by_y = sorted(lower_vertices, key=lambda v: (float(world_points[v][1]), v))
    upper_by_y = sorted(upper_vertices, key=lambda v: (float(world_points[v][1]), v))

    first_lower = lower_by_y[0]
    first_upper = min(upper_by_y, key=lambda v: (abs(float(world_points[first_lower][1] - world_points[v][1])), v))
    second_lower = lower_by_y[1]
    second_upper = upper_by_y[0] if upper_by_y[0] != first_upper else upper_by_y[1]
    pair_vertices = [(first_lower, first_upper), (second_lower, second_upper)]

    return {
        "lower_vertices": lower_vertices,
        "upper_vertices": upper_vertices,
        "pair_vertices": pair_vertices,
        "world_points": world_points,
    }


def _matrix_df(mat: np.ndarray, vertex_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(mat, index=vertex_names, columns=vertex_names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check empirical x-direction correlation of 4 fillet-fitup vertices")
    parser.add_argument("--csv", required=True, help="Path to onefile CSV")
    parser.add_argument("--step-id", default=None, help="Target step id")
    parser.add_argument("--step-idx", type=int, default=None, help="Target step index (0-based)")
    parser.add_argument("--guest-id", required=True, help="Target guest instance id")
    parser.add_argument("--trials", type=int, default=2000, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--outdir", default="_tmp/check_fillet_x_corr_out/", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.trials < 2:
        raise ValueError("--trials must be >= 2 to compute covariance/correlation.")

    geom_data, flow_data = load_data_from_csv(csv_path)
    geom = GeometryModel(geom_data)
    flow = FlowModel(flow_data)

    step_idx, step = _resolve_step(flow, args.step_idx, args.step_id)
    model = step.get("model", {})
    if not isinstance(model, dict) or not isinstance(model.get("fillet_fitup"), dict):
        raise ValueError(
            "Target step does not contain model.fillet_fitup. "
            f"target=({_step_desc(step, step_idx)})"
        )

    vertex_names = _resolve_vertex_names(step, geom, args.guest_id)
    dx_samples = np.zeros((args.trials, 4), dtype=float)

    effective_first: dict[str, Any] | None = None
    effective_variation_count = 0

    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        state = AssemblyState(geom)
        engine = ProcessEngine(geom, flow, rng)

        for i in range(step_idx):
            engine.apply_step(flow.steps[i], state)

        eff_before = _effective_grouping(geom, state, args.guest_id, vertex_names)
        x_before = np.array([eff_before["world_points"][v][0] for v in vertex_names], dtype=float)

        engine.apply_step(step, state)

        x_after = np.array([get_world_point(geom, state, args.guest_id, f"points.{v}")[0] for v in vertex_names], dtype=float)
        dx_samples[trial, :] = x_after - x_before

        eff_sig = (
            tuple(eff_before["lower_vertices"]),
            tuple(eff_before["upper_vertices"]),
            tuple(tuple(p) for p in eff_before["pair_vertices"]),
        )
        if effective_first is None:
            effective_first = {
                "lower_vertices": list(eff_before["lower_vertices"]),
                "upper_vertices": list(eff_before["upper_vertices"]),
                "pair_vertices": [tuple(p) for p in eff_before["pair_vertices"]],
                "sig": eff_sig,
            }
        elif effective_first["sig"] != eff_sig:
            effective_variation_count += 1

    cov = np.cov(dx_samples, rowvar=False, ddof=1)
    corr = np.corrcoef(dx_samples, rowvar=False)

    row_mean = dx_samples.mean(axis=1, keepdims=True)
    residual = dx_samples - row_mean
    residual_corr = np.corrcoef(residual, rowvar=False)

    common_mode = dx_samples.mean(axis=1)
    common_mode_corr = []
    for i, v in enumerate(vertex_names):
        c = float(np.corrcoef(common_mode, dx_samples[:, i])[0, 1])
        common_mode_corr.append({"vertex": v, "corr_with_common_mode": c})
    common_mode_corr_df = pd.DataFrame(common_mode_corr)

    dx_cols = [f"dx_{v}" for v in vertex_names]
    samples_df = pd.DataFrame(dx_samples, columns=dx_cols)
    samples_df.insert(0, "vertex_4", vertex_names[3])
    samples_df.insert(0, "vertex_3", vertex_names[2])
    samples_df.insert(0, "vertex_2", vertex_names[1])
    samples_df.insert(0, "vertex_1", vertex_names[0])
    samples_df.insert(0, "trial", np.arange(args.trials, dtype=int))

    summary_stats = pd.DataFrame({
        "vertex": vertex_names,
        "mean_dx": dx_samples.mean(axis=0),
        "std_dx": dx_samples.std(axis=0, ddof=1),
    })

    cov_df = _matrix_df(cov, vertex_names)
    corr_df = _matrix_df(corr, vertex_names)
    residual_corr_df = _matrix_df(residual_corr, vertex_names)

    samples_df.to_csv(outdir / "x_dx_samples.csv", index=False)
    cov_df.to_csv(outdir / "x_cov_matrix.csv")
    corr_df.to_csv(outdir / "x_corr_matrix.csv")
    residual_corr_df.to_csv(outdir / "x_residual_corr_matrix.csv")
    common_mode_corr_df.to_csv(outdir / "x_common_mode_corr.csv", index=False)

    lines: list[str] = []
    lines.append("=== fillet x-correlation check ===")
    lines.append(f"csv={csv_path}")
    lines.append(f"step_id={step.get('id', '')}")
    lines.append(f"step_idx={step_idx}")
    lines.append(f"guest_id={args.guest_id}")
    lines.append(f"vertex_order={vertex_names}")
    if effective_first is not None:
        lines.append(f"lower_vertices_effective={effective_first['lower_vertices']}")
        lines.append(f"upper_vertices_effective={effective_first['upper_vertices']}")
        lines.append(f"pair_vertices_effective={effective_first['pair_vertices']}")
        lines.append(f"effective_partition_changed_trials={effective_variation_count}")
    lines.append(f"trials={args.trials}")
    lines.append(f"seed={args.seed}")
    lines.append("")
    lines.append("--- mean/std per vertex ---")
    lines.append(summary_stats.to_string(index=False))
    lines.append("")
    lines.append("--- covariance matrix (x) ---")
    lines.append(cov_df.to_string())
    lines.append("")
    lines.append("--- correlation matrix (x) ---")
    lines.append(corr_df.to_string())
    lines.append("")
    lines.append("--- residual correlation matrix (x, after removing per-trial 4pt mean) ---")
    lines.append(residual_corr_df.to_string())
    lines.append("")
    lines.append("--- correlation with common_mode (= per-trial mean of 4 dx values) ---")
    lines.append(common_mode_corr_df.to_string(index=False))

    summary_text = "\n".join(lines)
    print(summary_text)
    (outdir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
