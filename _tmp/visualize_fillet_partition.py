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

from sov_app.engine.core_models import (
    AssemblyState,
    FlowModel,
    GeometryModel,
    get_world_point,
    rotation_matrix_to_rpy_deg,
    rpy_to_rotation_matrix,
)
from sov_app.engine.io_csv import load_data_from_csv
from sov_app.engine.process_engine import ProcessEngine


def _resolve_step(flow: FlowModel, step_idx: int | None, step_id: str | None) -> tuple[int, dict[str, Any]]:
    if (step_idx is None) == (step_id is None):
        raise ValueError("Specify exactly one of --step-idx or --step-id.")
    if step_idx is not None:
        if step_idx < 0 or step_idx >= len(flow.steps):
            raise ValueError(f"--step-idx out of range: {step_idx} (n_steps={len(flow.steps)})")
        return step_idx, flow.steps[step_idx]

    matches = [(i, s) for i, s in enumerate(flow.steps) if str(s.get("id", "")) == str(step_id)]
    if not matches:
        raise ValueError(f"--step-id '{step_id}' not found")
    if len(matches) > 1:
        raise ValueError(f"--step-id '{step_id}' is not unique: {[i for i, _ in matches]}")
    return matches[0]


def _resolve_physical_4_vertices(step: dict[str, Any], geom: GeometryModel, guest_id: str) -> list[str]:
    inst = geom.get_instance(guest_id)
    if not inst:
        raise ValueError(f"Unknown guest instance: {guest_id}")
    proto = geom.get_prototype(str(inst.get("prototype", "")))
    all_point_names = sorted(proto.get("features", {}).get("points", {}).keys())

    guest = step.get("guest", {})
    ref_line = guest.get("ref_line", {}) if isinstance(guest, dict) else {}
    ref_line_points: list[str] = []
    if isinstance(ref_line, dict):
        for key in ("p0", "p1"):
            ref = ref_line.get(key)
            if isinstance(ref, str) and ref.startswith("points."):
                ref_line_points.append(ref.split(".", 1)[1])
    ref_line_points = sorted(dict.fromkeys(ref_line_points))

    candidate_without_ref = [name for name in all_point_names if name not in ref_line_points]
    candidate_point_names = candidate_without_ref if len(candidate_without_ref) == 4 else all_point_names

    if len(candidate_point_names) != 4:
        raise ValueError(
            "Expected exactly 4 physical vertices. "
            f"guest_id={guest_id}, all_point_names={all_point_names}, "
            f"candidate_point_names={candidate_point_names}, ref_line_points={ref_line_points}"
        )
    return candidate_point_names


def _partition_from_world_points(world_points: dict[str, np.ndarray], vertex_names: list[str]) -> dict[str, Any]:
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
        "by_z": by_z,
        "lower_vertices": lower_vertices,
        "upper_vertices": upper_vertices,
        "lower_by_y": lower_by_y,
        "upper_by_y": upper_by_y,
        "first_lower": first_lower,
        "first_upper": first_upper,
        "second_lower": second_lower,
        "second_upper": second_upper,
        "pair_vertices": pair_vertices,
    }


def _world_points(geom: GeometryModel, state: AssemblyState, guest_id: str, vertex_names: list[str]) -> dict[str, np.ndarray]:
    return {v: get_world_point(geom, state, guest_id, f"points.{v}") for v in vertex_names}


def _simulate_attach_to_marking_until_partition(
    engine: ProcessEngine,
    step: dict[str, Any],
    state: AssemblyState,
    guest_id_target: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    base = step.get("base", {})
    guest = step.get("guest", {})
    if not isinstance(base, dict) or not isinstance(guest, dict):
        raise ValueError("fitup_attach_to_marking_line requires base/guest as dict")

    base_id = str(base.get("instance", ""))
    guest_id = str(guest.get("instance", ""))
    if guest_id != guest_id_target:
        raise ValueError(f"--guest-id '{guest_id_target}' does not match step guest.instance '{guest_id}'")

    mark_line = base.get("mark_line", {})
    ref_line = guest.get("ref_line", {})
    if not isinstance(mark_line, dict) or not isinstance(ref_line, dict):
        raise ValueError("fitup_attach_to_marking_line requires base.mark_line and guest.ref_line")

    base_p0 = str(mark_line.get("p0", ""))
    base_p1 = str(mark_line.get("p1", ""))
    guest_q0 = str(ref_line.get("p0", ""))
    guest_q1 = str(ref_line.get("p1", ""))

    p0 = get_world_point(engine.geom, state, base_id, base_p0)
    p1 = get_world_point(engine.geom, state, base_id, base_p1)
    q0 = get_world_point(engine.geom, state, guest_id, guest_q0)
    q1 = get_world_point(engine.geom, state, guest_id, guest_q1)

    base_dir = engine._unit(p1 - p0)
    guest_dir = engine._unit(q1 - q0)
    if float(np.linalg.norm(base_dir)) < 1e-12 or float(np.linalg.norm(guest_dir)) < 1e-12:
        raise ValueError("fitup_attach_to_marking_line requires non-degenerate lines")

    axis = np.cross(guest_dir, base_dir)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(guest_dir, base_dir), -1.0, 1.0))

    if axis_norm < 1e-12:
        if dot < 0.0:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(guest_dir, fallback))) > 0.99:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            axis = engine._safe_unit_from_cross(guest_dir, fallback, np.array([0.0, 0.0, 1.0], dtype=float))
            theta = float(np.pi)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
            theta = 0.0
    else:
        axis = axis / axis_norm
        theta = float(np.arccos(dot))

    tr = state.get_transform(guest_id)
    origin_old = np.array(tr["origin"], dtype=float)
    r_old = rpy_to_rotation_matrix(*tr["rpy_deg"])
    q0_local = r_old.T @ (q0 - origin_old)

    r_delta = engine._rotation_matrix_from_axis_angle(axis, theta)
    r_align = r_delta @ r_old
    r_new = r_align

    constraints = step.get("constraints", {})
    set_rpy_deg = constraints.get("set_rpy_deg") if isinstance(constraints, dict) else None
    if set_rpy_deg is not None:
        r_hint = rpy_to_rotation_matrix(float(set_rpy_deg[0]), float(set_rpy_deg[1]), float(set_rpy_deg[2]))
        best_phi = 0.0
        best_cost = float("inf")
        for phi_deg in np.linspace(-180.0, 180.0, 721):
            r_spin = engine._rotation_matrix_from_axis_angle(base_dir, np.deg2rad(float(phi_deg)))
            r_candidate = r_spin @ r_align
            cost = float(np.linalg.norm(r_candidate - r_hint, ord="fro"))
            if cost < best_cost:
                best_cost = cost
                best_phi = float(phi_deg)
        r_spin = engine._rotation_matrix_from_axis_angle(base_dir, np.deg2rad(best_phi))
        r_new = r_spin @ r_align

    origin_new = p0 - r_new @ q0_local
    state.set_transform(guest_id, origin_new, rotation_matrix_to_rpy_deg(r_new))

    model = step.get("model", {})
    fillet_fitup = model.get("fillet_fitup") if isinstance(model, dict) else None
    if not isinstance(fillet_fitup, dict):
        raise ValueError("target step has no model.fillet_fitup")

    y_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    dy = engine._sample(fillet_fitup["delta_y"])
    gtr = state.get_transform(guest_id)
    state.set_transform(guest_id, gtr["origin"] + dy * y_dir, gtr["rpy_deg"])

    _ = engine._sample(fillet_fitup["delta_mA"])
    _ = engine._sample(fillet_fitup["delta_mB"])

    vertex_names = _resolve_physical_4_vertices(step, engine.geom, guest_id)
    wp = _world_points(engine.geom, state, guest_id, vertex_names)
    partition = _partition_from_world_points(wp, vertex_names)
    return vertex_names, wp, partition


def _simulate_array_attach_until_partition(
    engine: ProcessEngine,
    step: dict[str, Any],
    state: AssemblyState,
    guest_id_target: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    base_info = step.get("base", {})
    guests_info = step.get("guests", {})
    model = step.get("model", {})
    constraints = step.get("constraints", {})
    fillet_fitup = model.get("fillet_fitup") if isinstance(model, dict) else None
    if not isinstance(fillet_fitup, dict):
        raise ValueError("target step has no model.fillet_fitup")

    base_inst_id = base_info["instance"]
    guest_ids = engine.flow.resolve_selector(guests_info["selector"], engine.geom)
    if guest_id_target not in guest_ids:
        raise ValueError(f"--guest-id '{guest_id_target}' is not selected by this step: {guest_ids}")

    pitch = float(guests_info["pattern"].get("pitch_on_base_mm", 0.0))
    start_offset = float(guests_info["pattern"].get("start_offset_mm", 0.0))
    base_tr = state.get_transform(base_inst_id)

    y_translation_applied_guests: set[str] = set()
    point_offsets_applied_guests: set[str] = set()

    y_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    x_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    z_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    for i, gid in enumerate(guest_ids):
        gap = engine._sample(constraints.get("coincident_1D", {}).get("gap_dist", 0.0)) if "coincident_1D" in constraints else 0.0
        dy_attach = (
            engine._sample(constraints["inplane_y"].get("dist"))
            if "inplane_y" in constraints and "dist" in constraints["inplane_y"]
            else 0.0
        )
        new_origin = base_tr["origin"] + np.array([start_offset + i * pitch + gap, dy_attach, 0.0], dtype=float)
        state.set_transform(gid, new_origin, list(constraints.get("set_rpy_deg", [0.0, 0.0, 0.0])))

        if gid not in y_translation_applied_guests:
            dy = engine._sample(fillet_fitup["delta_y"])
            gtr = state.get_transform(gid)
            state.set_transform(gid, gtr["origin"] + dy * y_dir, gtr["rpy_deg"])
            y_translation_applied_guests.add(gid)

        if gid in point_offsets_applied_guests:
            continue

        dm_a = engine._sample(fillet_fitup["delta_mA"])
        dm_b = engine._sample(fillet_fitup["delta_mB"])

        inst = engine.geom.get_instance(gid)
        proto = engine.geom.get_prototype(inst.get("prototype", ""))
        vertex_names = sorted(proto.get("features", {}).get("points", {}).keys())
        if len(vertex_names) != 4:
            raise ValueError(f"fillet_fitup requires 4 vertices for guest={gid}, n={len(vertex_names)}")

        wp = _world_points(engine.geom, state, gid, vertex_names)
        partition = _partition_from_world_points(wp, vertex_names)

        if gid == guest_id_target:
            return vertex_names, wp, partition

        for lower_v, upper_v in partition["pair_vertices"]:
            x_lower_i = dm_b + engine._sample(fillet_fitup["x_lower"])
            dx_lower_local = engine._world_vec_to_local(gid, state, (x_lower_i - dm_a) * x_dir)
            x_upper_i = dm_b + engine._sample(fillet_fitup["x_upper"])
            dx_upper_local = engine._world_vec_to_local(gid, state, (x_upper_i - dm_a) * x_dir)
            z_i = max(0.0, float(engine._sample(fillet_fitup["z_lower"])))
            dz_local = engine._world_vec_to_local(gid, state, z_i * z_dir)
            state.add_point_offset(gid, lower_v, dx_lower_local)
            state.add_point_offset(gid, lower_v, dz_local)
            state.add_point_offset(gid, upper_v, dx_upper_local)
            state.add_point_offset(gid, upper_v, dz_local)

        point_offsets_applied_guests.add(gid)

    raise RuntimeError("Internal error: target guest not reached")


def _partition_at_decision_timing(
    geom: GeometryModel,
    flow: FlowModel,
    step_idx: int,
    step: dict[str, Any],
    guest_id: str,
    seed: int,
    trial: int,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed + trial)
    state = AssemblyState(geom)
    engine = ProcessEngine(geom, flow, rng)

    for i in range(step_idx):
        engine.apply_step(flow.steps[i], state)

    op = str(step.get("op", ""))
    if op == "fitup_attach_to_marking_line":
        return _simulate_attach_to_marking_until_partition(engine, step, state, guest_id)
    if op == "fitup_array_attach":
        return _simulate_array_attach_until_partition(engine, step, state, guest_id)

    raise ValueError(f"Unsupported target op for this script: {op}")


def _pattern_signature(partition: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    return (
        tuple(partition["lower_vertices"]),
        tuple(partition["upper_vertices"]),
        tuple(tuple(p) for p in partition["pair_vertices"]),
    )


def _write_detail_files(
    outdir: Path,
    detail_trial: int,
    vertex_names: list[str],
    wp: dict[str, np.ndarray],
    partition: dict[str, Any],
) -> None:
    z_rank = {v: i + 1 for i, v in enumerate(partition["by_z"])}
    lower_y_rank = {v: i + 1 for i, v in enumerate(partition["lower_by_y"])}
    upper_y_rank = {v: i + 1 for i, v in enumerate(partition["upper_by_y"])}

    pair_meta: dict[str, tuple[int, str]] = {}
    for i, (lv, uv) in enumerate(partition["pair_vertices"]):
        pair_meta[lv] = (i, "lower")
        pair_meta[uv] = (i, "upper")

    rows = []
    for v in vertex_names:
        group = "lower" if v in partition["lower_vertices"] else "upper"
        y_rank = lower_y_rank[v] if group == "lower" else upper_y_rank[v]
        pair_id, pair_role = pair_meta[v]
        rows.append({
            "vertex": v,
            "world_x": float(wp[v][0]),
            "world_y": float(wp[v][1]),
            "world_z": float(wp[v][2]),
            "z_rank": z_rank[v],
            "group": group,
            "y_rank_within_group": y_rank,
            "pair_id": pair_id,
            "pair_role": pair_role,
        })

    pd.DataFrame(rows).to_csv(outdir / f"trial_{detail_trial}_partition_detail.csv", index=False)

    trace = [
        f"by_z={partition['by_z']}",
        f"lower_vertices={partition['lower_vertices']}",
        f"upper_vertices={partition['upper_vertices']}",
        f"lower_by_y={partition['lower_by_y']}",
        f"upper_by_y={partition['upper_by_y']}",
        f"first_lower={partition['first_lower']}",
        f"first_upper={partition['first_upper']}",
        f"second_lower={partition['second_lower']}",
        f"second_upper={partition['second_upper']}",
        f"pair_vertices={partition['pair_vertices']}",
    ]
    (outdir / f"trial_{detail_trial}_sort_trace.txt").write_text("\n".join(trace) + "\n", encoding="utf-8")


def _save_scatter_plots(
    outdir: Path,
    detail_trial: int,
    step_id: str,
    guest_id: str,
    wp: dict[str, np.ndarray],
    partition: dict[str, Any],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    def draw(ax: Any, x_idx: int, y_idx: int, xlab: str, ylab: str) -> None:
        for v, xyz in wp.items():
            is_lower = v in partition["lower_vertices"]
            color = "tab:blue" if is_lower else "tab:orange"
            ax.scatter(float(xyz[x_idx]), float(xyz[y_idx]), color=color, s=50)

            zr = partition["by_z"].index(v) + 1
            yr = None
            if is_lower:
                yr = partition["lower_by_y"].index(v) + 1
            else:
                yr = partition["upper_by_y"].index(v) + 1
            ax.annotate(f"{v} (z{zr}, y{yr})", (float(xyz[x_idx]), float(xyz[y_idx])), xytext=(5, 5), textcoords="offset points")

        for lv, uv in partition["pair_vertices"]:
            p0, p1 = wp[lv], wp[uv]
            ax.plot([float(p0[x_idx]), float(p1[x_idx])], [float(p0[y_idx]), float(p1[y_idx])], "k--", linewidth=1.0)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    title = f"trial={detail_trial}, step_id={step_id}, guest_id={guest_id}"

    fig, ax = plt.subplots(figsize=(6, 5))
    draw(ax, 1, 2, "world_y", "world_z")
    ax.set_title("YZ partition view\n" + title)
    fig.tight_layout()
    fig.savefig(outdir / f"trial_{detail_trial}_yz.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw(ax, 0, 1, "world_x", "world_y")
    ax.set_title("XY view\n" + title)
    fig.tight_layout()
    fig.savefig(outdir / f"trial_{detail_trial}_xy.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw(ax, 0, 2, "world_x", "world_z")
    ax.set_title("XZ view\n" + title)
    fig.tight_layout()
    fig.savefig(outdir / f"trial_{detail_trial}_xz.png", dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize fillet lower/upper/pair partition timing and patterns")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--step-id", default=None)
    parser.add_argument("--step-idx", type=int, default=None)
    parser.add_argument("--guest-id", required=True)
    parser.add_argument("--trial-detail", type=int, default=0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="_tmp/visualize_fillet_partition_out/")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.trial_detail < 0 or args.trial_detail >= args.trials:
        raise ValueError(f"--trial-detail must be in [0, trials-1], got {args.trial_detail}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geom_data, flow_data = load_data_from_csv(Path(args.csv))
    geom = GeometryModel(geom_data)
    flow = FlowModel(flow_data)
    step_idx, step = _resolve_step(flow, args.step_idx, args.step_id)

    op = str(step.get("op", ""))
    if op not in {"fitup_attach_to_marking_line", "fitup_array_attach"}:
        raise ValueError(f"Unsupported op='{op}'. Use fitup_attach_to_marking_line or fitup_array_attach")
    if not isinstance(step.get("model", {}).get("fillet_fitup"), dict):
        raise ValueError("Target step does not contain model.fillet_fitup")

    trial_records: list[dict[str, Any]] = []
    patterns: dict[tuple[Any, ...], dict[str, Any]] = {}

    detail_payload: tuple[list[str], dict[str, np.ndarray], dict[str, Any]] | None = None

    for t in range(args.trials):
        vertex_names, wp, partition = _partition_at_decision_timing(
            geom=geom,
            flow=flow,
            step_idx=step_idx,
            step=step,
            guest_id=args.guest_id,
            seed=args.seed,
            trial=t,
        )

        sig = _pattern_signature(partition)
        if sig not in patterns:
            patterns[sig] = {
                "lower_vertices": list(partition["lower_vertices"]),
                "upper_vertices": list(partition["upper_vertices"]),
                "pair_vertices": [list(p) for p in partition["pair_vertices"]],
                "count": 0,
                "first_trial": t,
            }
        patterns[sig]["count"] += 1

        row: dict[str, Any] = {
            "trial": t,
            "vertex_order": "|".join(vertex_names),
            "lower_vertices": "|".join(partition["lower_vertices"]),
            "upper_vertices": "|".join(partition["upper_vertices"]),
            "pair_vertices": "|".join([f"{lv}-{uv}" for lv, uv in partition["pair_vertices"]]),
        }
        for v in vertex_names:
            row[f"x_{v}"] = float(wp[v][0])
            row[f"y_{v}"] = float(wp[v][1])
            row[f"z_{v}"] = float(wp[v][2])
        trial_records.append(row)

        if t == args.trial_detail:
            detail_payload = (vertex_names, wp, partition)

    if detail_payload is None:
        raise RuntimeError("Internal error: detail trial payload not found")

    vertex_names_detail, wp_detail, partition_detail = detail_payload
    _write_detail_files(outdir, args.trial_detail, vertex_names_detail, wp_detail, partition_detail)
    _save_scatter_plots(
        outdir=outdir,
        detail_trial=args.trial_detail,
        step_id=str(step.get("id", "")),
        guest_id=args.guest_id,
        wp=wp_detail,
        partition=partition_detail,
        dpi=args.dpi,
    )

    all_trials_df = pd.DataFrame(trial_records)
    all_trials_df.to_csv(outdir / "all_trials_partition_log.csv", index=False)

    pattern_rows = []
    sorted_patterns = sorted(patterns.items(), key=lambda kv: (-kv[1]["count"], kv[1]["first_trial"]))
    for i, (_sig, meta) in enumerate(sorted_patterns):
        pattern_rows.append({
            "pattern_id": i,
            "lower_vertices": "|".join(meta["lower_vertices"]),
            "upper_vertices": "|".join(meta["upper_vertices"]),
            "pair_vertices": "|".join([f"{p[0]}-{p[1]}" for p in meta["pair_vertices"]]),
            "count": int(meta["count"]),
            "first_trial": int(meta["first_trial"]),
        })
    pd.DataFrame(pattern_rows).to_csv(outdir / "partition_patterns.csv", index=False)

    changed_trials = args.trials - sorted_patterns[0][1]["count"]
    physical_vertices = _resolve_physical_4_vertices(step, geom, args.guest_id)

    summary_lines = [
        "=== visualize fillet partition ===",
        f"csv={args.csv}",
        f"step_id={step.get('id', '')}",
        f"step_idx={step_idx}",
        f"guest_id={args.guest_id}",
        f"target_op={op}",
        f"detail_trial={args.trial_detail}",
        f"trials={args.trials}",
        f"seed={args.seed}",
        f"physical_4_vertices={physical_vertices}",
        f"n_unique_partition_patterns={len(sorted_patterns)}",
        f"partition_changed_trials={changed_trials}",
        f"partition_changed={'yes' if changed_trials > 0 else 'no'}",
        "",
        "--- partition patterns ---",
    ]
    for row in pattern_rows:
        summary_lines.append(
            f"pattern_id={row['pattern_id']}, count={row['count']}, first_trial={row['first_trial']}, "
            f"lower={row['lower_vertices']}, upper={row['upper_vertices']}, pairs={row['pair_vertices']}"
        )

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (outdir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
