"""Process step application engine."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

import numpy as np

from .core_models import (
    AssemblyState,
    DistributionSampler,
    FlowModel,
    GeometryModel,
    get_world_point,
    rotation_matrix_to_rpy_deg,
    rpy_to_rotation_matrix,
)


class ProcessEngine:
    def __init__(self, geom: GeometryModel, flow: FlowModel, rng: np.random.Generator):
        self.geom = geom
        self.flow = flow
        self.rng = rng

    def _extract_pair_index(self, value: Any) -> int | None:
        if isinstance(value, (int, np.integer)):
            pair_index = int(value)
            return pair_index if pair_index in (0, 1) else None
        if isinstance(value, str):
            match = re.search(r"pair([01])", value)
            if match:
                return int(match.group(1))
        return None

    def _extract_realized_gap(self, metric: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            if key in metric and metric.get(key) is not None:
                try:
                    return float(metric.get(key))
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _sample(self, spec: Any) -> float:
        return DistributionSampler.sample(spec, self.rng, self.flow.dists)

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.zeros(3, dtype=float)
        return v / n

    def _safe_unit_from_cross(self, a: np.ndarray, b: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        c = np.cross(a, b)
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            return self._unit(fallback)
        return c / n

    def _get_plane_normal_world(self, inst_id: str, state: AssemblyState) -> np.ndarray:
        tr = state.get_transform(inst_id)
        return self._unit(rpy_to_rotation_matrix(*tr["rpy_deg"]) @ np.array([0.0, 0.0, 1.0], dtype=float))

    def _world_vec_to_local(self, inst_id: str, state: AssemblyState, vec_world: np.ndarray) -> np.ndarray:
        tr = state.get_transform(inst_id)
        return rpy_to_rotation_matrix(*tr["rpy_deg"]).T @ np.array(vec_world, dtype=float)

    def _rotation_matrix_from_axis_angle(self, axis: np.ndarray, theta: float) -> np.ndarray:
        axis_unit = self._unit(np.array(axis, dtype=float))
        if float(np.linalg.norm(axis_unit)) < 1e-12 or abs(theta) < 1e-12:
            return np.eye(3, dtype=float)
        ax, ay, az = axis_unit
        skew = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
        outer = np.outer(axis_unit, axis_unit)
        return np.eye(3, dtype=float) * np.cos(theta) + (1.0 - np.cos(theta)) * outer + np.sin(theta) * skew

    def _parse_point_name(self, ref: str) -> str:
        toks = str(ref).split(".")
        if len(toks) >= 2 and toks[0] == "points":
            return toks[1]
        raise ValueError(f"ref must be points.<name>, got: {ref}")

    def _recompute_realized_dims_from_points(self, inst_id: str, state: AssemblyState):
        try:
            a = get_world_point(self.geom, state, inst_id, "points.A")
            b = get_world_point(self.geom, state, inst_id, "points.B")
            c = get_world_point(self.geom, state, inst_id, "points.C")
            d = get_world_point(self.geom, state, inst_id, "points.D")
        except (KeyError, ValueError, TypeError):
            return

        l_ab = float(self._world_vec_to_local(inst_id, state, b - a)[0])
        l_dc = float(self._world_vec_to_local(inst_id, state, c - d)[0])
        h_ad = float(self._world_vec_to_local(inst_id, state, d - a)[1])
        h_bc = float(self._world_vec_to_local(inst_id, state, c - b)[1])
        state.set_realized_dim(inst_id, "L_ab", l_ab)
        state.set_realized_dim(inst_id, "L_dc", l_dc)
        state.set_realized_dim(inst_id, "H_ad", h_ad)
        state.set_realized_dim(inst_id, "H_bc", h_bc)
        state.set_realized_dim(inst_id, "L", 0.5 * (l_ab + l_dc))
        state.set_realized_dim(inst_id, "H", 0.5 * (h_ad + h_bc))

    def apply_step(self, step: Dict[str, Any], state: AssemblyState):
        op = step.get("op", "")
        if op == "apply_variation":
            self._apply_variation(step, state)
        elif op == "fitup_array_attach":
            self._fitup_array_attach(step, state)
        elif op == "welding_distortion":
            self._welding_distortion(step, state)
        elif op == "fitup_pair_chain":
            self._fitup_pair_chain(step, state)
        elif op == "fitup_attach_to_marking_line":
            self._fitup_attach_to_marking_line(step, state)

    def apply_steps(
        self,
        state: AssemblyState,
        steps_mask: List[bool] | None = None,
        on_step_before: Callable[[int, Dict[str, Any], AssemblyState], None] | None = None,
        on_step_after: Callable[[int, Dict[str, Any], AssemblyState], None] | None = None,
    ):
        """Apply enabled flow steps to ``state`` in flow order."""
        for idx, step in enumerate(self.flow.steps):
            if steps_mask is None or (idx < len(steps_mask) and steps_mask[idx]):
                if on_step_before is not None:
                    on_step_before(idx, step, state)
                self.apply_step(step, state)
                if on_step_after is not None:
                    on_step_after(idx, step, state)

    def _apply_variation(self, step: Dict[str, Any], state: AssemblyState):
        target = step.get("target", {})
        model = step.get("model", {})
        step_id = str(step.get("id", ""))

        inst_ids: List[str] = []
        for sel_name in target.get("union", []):
            inst_ids.extend(self.flow.resolve_selector(sel_name, self.geom))
        inst_ids = list(dict.fromkeys(inst_ids))

        is_cutting_step = step_id == "10_cutting"

        for inst_id in inst_ids:
            if bool(model.get("per_point_xy_noise", False)):
                proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
                pts = list(proto.get("features", {}).get("points", {}).keys())
                dist_x = model.get("point_dx", 0.0)
                dist_y = model.get("point_dy", 0.0)
                dist_z = model.get("point_dz", 0.0)
                for pnm in pts:
                    state.set_point_offset(inst_id, pnm, np.array([self._sample(dist_x), self._sample(dist_y), self._sample(dist_z)], dtype=float))

            if is_cutting_step:
                self._recompute_realized_dims_from_points(inst_id, state)

            dim_variations = model.get("dim_variations")
            if dim_variations is None:
                dim_variations = model.get("dims_variation")
            if dim_variations and not is_cutting_step:
                proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
                for dim_name, variation_spec in dim_variations.items():
                    if dim_name in proto.get("dims", {}):
                        nominal = float(proto["dims"][dim_name])
                        state.set_realized_dim(inst_id, dim_name, nominal + self._sample(variation_spec))

    def _fitup_array_attach(self, step: Dict[str, Any], state: AssemblyState):
        base_info = step.get("base", {})
        guests_info = step.get("guests", {})
        model = step.get("model", {})
        constraints = step.get("constraints", {})
        base_inst_id = base_info["instance"]
        guest_ids = self.flow.resolve_selector(guests_info["selector"], self.geom)
        pitch = float(guests_info["pattern"].get("pitch_on_base_mm", 0.0))
        start_offset = float(guests_info["pattern"].get("start_offset_mm", 0.0))
        base_tr = state.get_transform(base_inst_id)

        for i, gid in enumerate(guest_ids):
            gap = self._sample(constraints.get("coincident_1D", {}).get("gap_dist", 0.0)) if "coincident_1D" in constraints else 0.0
            dy = self._sample(constraints["inplane_y"].get("dist")) if "inplane_y" in constraints and "dist" in constraints["inplane_y"] else 0.0
            new_origin = base_tr["origin"] + np.array([start_offset + i * pitch + gap, dy, 0.0], dtype=float)
            state.set_transform(gid, new_origin, list(constraints.get("set_rpy_deg", [0.0, 0.0, 0.0])))

        fillet_fitup = model.get("fillet_fitup")
        if not isinstance(fillet_fitup, dict):
            return

        required_keys = ("delta_mA", "delta_mB", "x_lower", "x_upper", "delta_y", "z_lower")
        missing_keys = [k for k in required_keys if k not in fillet_fitup]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"fillet_fitup missing required keys: {missing}; check CSV /steps/N/model/fillet_fitup/..."
            )

        axis_mode = str(fillet_fitup.get("axis_mode", "world"))
        if axis_mode != "world":
            raise ValueError("fillet_fitup axis_mode currently supports only 'world'.")

        x_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        y_dir = np.array([0.0, 1.0, 0.0], dtype=float)
        z_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        step_id = str(step.get("id", ""))

        y_translation_applied_guests = set()
        point_offsets_applied_guests = set()
        for gid in guest_ids:
            if gid not in y_translation_applied_guests:
                dy = self._sample(fillet_fitup["delta_y"])
                gtr = state.get_transform(gid)
                state.set_transform(gid, gtr["origin"] + dy * y_dir, gtr["rpy_deg"])
                y_translation_applied_guests.add(gid)

            if gid in point_offsets_applied_guests:
                continue

            dm_a = self._sample(fillet_fitup["delta_mA"])
            dm_b = self._sample(fillet_fitup["delta_mB"])

            inst = self.geom.get_instance(gid)
            proto_name = inst.get("prototype", "")
            proto = self.geom.get_prototype(proto_name)
            vertex_names = sorted(proto.get("features", {}).get("points", {}).keys())
            if len(vertex_names) != 4:
                raise ValueError(
                    f"fillet_fitup supports exactly 4 vertices: step_id='{step_id}', guest_id='{gid}', "
                    f"n_vertices={len(vertex_names)}"
                )

            world_points = {vnm: get_world_point(self.geom, state, gid, f"points.{vnm}") for vnm in vertex_names}
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

            fillet_fitup["lower_points_effective"] = [f"points.{v}" for v in lower_vertices]
            fillet_fitup["upper_points_effective"] = [f"points.{v}" for v in upper_vertices]
            fillet_fitup["pair_effective"] = [
                {"lower": f"points.{lv}", "upper": f"points.{uv}"} for lv, uv in pair_vertices
            ]

            for lower_v, upper_v in pair_vertices:
                x_lower_i = dm_b + self._sample(fillet_fitup["x_lower"])
                dx_lower_i = x_lower_i - dm_a
                dx_lower_local = self._world_vec_to_local(gid, state, dx_lower_i * x_dir)

                x_upper_i = dm_b + self._sample(fillet_fitup["x_upper"])
                dx_upper_i = x_upper_i - dm_a
                dx_upper_local = self._world_vec_to_local(gid, state, dx_upper_i * x_dir)

                z_i = max(0.0, float(self._sample(fillet_fitup["z_lower"])))
                dz_local = self._world_vec_to_local(gid, state, z_i * z_dir)

                state.add_point_offset(gid, lower_v, dx_lower_local)
                state.add_point_offset(gid, lower_v, dz_local)
                state.add_point_offset(gid, upper_v, dx_upper_local)
                state.add_point_offset(gid, upper_v, dz_local)

            point_offsets_applied_guests.add(gid)

    def _welding_distortion(self, step: Dict[str, Any], state: AssemblyState):
        model = step.get("model", {})
        inst_ids = self.flow.resolve_selector(step.get("target", {}).get("selector", ""), self.geom)
        for inst_id in inst_ids:
            tr = state.get_transform(inst_id)
            o = tr["origin"].copy()
            rpy = tr["rpy_deg"][:]
            o[2] += self._sample(model.get("outplane_dz", 0.0))
            rpy[0] += self._sample(model.get("weak_bending_about_x", 0.0))
            state.set_transform(inst_id, o, rpy)
            self._apply_butt_transverse_shrinkage(inst_id, state)

    def _apply_butt_transverse_shrinkage(self, inst_id: str, state: AssemblyState):
        metrics_by_step = getattr(state, "butt_fitup_metrics", None)
        if not isinstance(metrics_by_step, dict):
            return

        bound_metrics: List[tuple[Dict[str, Any], int | None]] = []
        unbound_metrics: List[tuple[Dict[str, Any], int | None]] = []
        for step_id, metrics in metrics_by_step.items():
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                pair_index = self._extract_pair_index(metric.get("pair_index"))
                if pair_index is None:
                    pair_index = self._extract_pair_index(step_id)
                if metric.get("guest_instance") == inst_id:
                    bound_metrics.append((metric, pair_index))
                elif metric.get("guest_instance") in (None, ""):
                    unbound_metrics.append((metric, pair_index))

        selected_metrics = bound_metrics
        if not selected_metrics and len(unbound_metrics) == 1:
            selected_metrics = unbound_metrics
        if not selected_metrics:
            return

        proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
        points = proto.get("features", {}).get("points", {})
        if not all(name in points for name in ("A", "B", "C", "D")):
            return

        child_ids = list(getattr(state, "_children", {}).get(inst_id, []))
        point_x_values = [float(xyz[0]) for xyz in points.values() if isinstance(xyz, (list, tuple)) and len(xyz) >= 1]
        min_local_x = min(point_x_values) if point_x_values else 0.0
        max_local_x = max(point_x_values) if point_x_values else 0.0

        y_ab = 0.5 * (float(points["A"][1]) + float(points["B"][1]))
        y_cd = 0.5 * (float(points["C"][1]) + float(points["D"][1]))
        tol = 1e-9

        lower_points = [name for name, xyz in points.items() if abs(float(xyz[1]) - y_ab) <= tol]
        upper_points = [name for name, xyz in points.items() if abs(float(xyz[1]) - y_cd) <= tol]

        def _signed_local_x_shrinkage(point_name: str, shrink: float, weld_x_local: float | None) -> np.ndarray:
            if shrink <= 0.0:
                return np.zeros(3, dtype=float)
            if weld_x_local is None:
                return np.array([-shrink, 0.0, 0.0], dtype=float)
            point_x_local = float(points.get(point_name, [0.0, 0.0, 0.0])[0])
            dx = point_x_local - weld_x_local
            if abs(dx) <= tol:
                return np.zeros(3, dtype=float)
            sign = -1.0 if dx > 0.0 else 1.0
            return np.array([sign * shrink, 0.0, 0.0], dtype=float)

        def _equivalent_rigid_local_x(shrink: float, weld_x_local: float | None) -> float:
            if shrink <= 0.0:
                return 0.0
            if weld_x_local is None:
                return -shrink
            center_x = 0.5 * (min_local_x + max_local_x)
            return -shrink if float(weld_x_local) <= center_x else shrink

        def _propagate_child_rigid_shift(local_dx: float) -> None:
            if abs(local_dx) <= tol or not child_ids:
                return
            parent_tr = state.get_transform(inst_id)
            d_world = rpy_to_rotation_matrix(*parent_tr["rpy_deg"]) @ np.array([local_dx, 0.0, 0.0], dtype=float)
            for child_id in child_ids:
                child_tr = state.get_transform(child_id)
                state.set_transform(child_id, np.array(child_tr["origin"], dtype=float) + d_world, child_tr["rpy_deg"])

        for pair_metric, pair_index in selected_metrics:
            if pair_index == 0:
                g_real = self._extract_realized_gap(pair_metric, "g_real_0", "g_real_1", "g_real")
                shrink = 0.18 * max(g_real, 0.0)
                if shrink <= 0.0:
                    continue
                weld_x_local = pair_metric.get("weld_x_local_0")
                if weld_x_local is not None:
                    weld_x_local = float(weld_x_local)
                for pname in lower_points:
                    d_local = _signed_local_x_shrinkage(pname, shrink, weld_x_local)
                    if np.any(d_local):
                        state.add_point_offset(inst_id, pname, d_local)
                _propagate_child_rigid_shift(_equivalent_rigid_local_x(shrink, weld_x_local))
                continue

            if pair_index == 1:
                g_real = self._extract_realized_gap(pair_metric, "g_real_1", "g_real_0", "g_real")
                shrink = 0.18 * max(g_real, 0.0)
                if shrink <= 0.0:
                    continue
                weld_x_local = pair_metric.get("weld_x_local_1")
                if weld_x_local is None:
                    weld_x_local = pair_metric.get("weld_x_local_0")
                if weld_x_local is not None:
                    weld_x_local = float(weld_x_local)
                for pname in upper_points:
                    d_local = _signed_local_x_shrinkage(pname, shrink, weld_x_local)
                    if np.any(d_local):
                        state.add_point_offset(inst_id, pname, d_local)
                _propagate_child_rigid_shift(_equivalent_rigid_local_x(shrink, weld_x_local))
                continue

            s0 = 0.18 * max(self._extract_realized_gap(pair_metric, "g_real_0", "g_real_1", "g_real"), 0.0)
            s1 = 0.18 * max(self._extract_realized_gap(pair_metric, "g_real_1", "g_real_0", "g_real"), 0.0)

            if s0 > 0.0:
                weld_x_local_0 = pair_metric.get("weld_x_local_0")
                if weld_x_local_0 is not None:
                    weld_x_local_0 = float(weld_x_local_0)
                for pname in lower_points:
                    d0_local = _signed_local_x_shrinkage(pname, s0, weld_x_local_0)
                    if np.any(d0_local):
                        state.add_point_offset(inst_id, pname, d0_local)
            if s1 > 0.0:
                weld_x_local_1 = pair_metric.get("weld_x_local_1")
                if weld_x_local_1 is None:
                    weld_x_local_1 = pair_metric.get("weld_x_local_0")
                if weld_x_local_1 is not None:
                    weld_x_local_1 = float(weld_x_local_1)
                for pname in upper_points:
                    d1_local = _signed_local_x_shrinkage(pname, s1, weld_x_local_1)
                    if np.any(d1_local):
                        state.add_point_offset(inst_id, pname, d1_local)

            rigid_shifts = []
            if s0 > 0.0:
                rigid_shifts.append(_equivalent_rigid_local_x(s0, pair_metric.get("weld_x_local_0")))
            if s1 > 0.0:
                weld_x_local_1 = pair_metric.get("weld_x_local_1")
                if weld_x_local_1 is None:
                    weld_x_local_1 = pair_metric.get("weld_x_local_0")
                rigid_shifts.append(_equivalent_rigid_local_x(s1, weld_x_local_1))
            if rigid_shifts:
                _propagate_child_rigid_shift(float(np.mean(rigid_shifts)))

    def _fitup_attach_to_marking_line(self, step: Dict[str, Any], state: AssemblyState):
        base = step.get("base", {})
        guest = step.get("guest", {})
        if not isinstance(base, dict) or not isinstance(guest, dict):
            raise ValueError("fitup_attach_to_marking_line requires /steps/N/base and /steps/N/guest as dict.")

        base_id = str(base.get("instance", ""))
        guest_id = str(guest.get("instance", ""))
        if not base_id or not guest_id:
            raise ValueError("fitup_attach_to_marking_line requires base.instance and guest.instance")
        if base_id not in self.geom.instances:
            raise ValueError(f"Unknown base instance: {base_id}")
        if guest_id not in self.geom.instances:
            raise ValueError(f"Unknown guest instance: {guest_id}")

        mark_line = base.get("mark_line", {})
        ref_line = guest.get("ref_line", {})
        if not isinstance(mark_line, dict) or not isinstance(ref_line, dict):
            raise ValueError("fitup_attach_to_marking_line requires base.mark_line and guest.ref_line as dict.")

        base_p0 = str(mark_line.get("p0", ""))
        base_p1 = str(mark_line.get("p1", ""))
        guest_q0 = str(ref_line.get("p0", ""))
        guest_q1 = str(ref_line.get("p1", ""))
        if not all((base_p0, base_p1, guest_q0, guest_q1)):
            raise ValueError("fitup_attach_to_marking_line requires p0/p1 for both base.mark_line and guest.ref_line")

        p0 = get_world_point(self.geom, state, base_id, base_p0)
        p1 = get_world_point(self.geom, state, base_id, base_p1)
        q0 = get_world_point(self.geom, state, guest_id, guest_q0)
        q1 = get_world_point(self.geom, state, guest_id, guest_q1)

        base_dir = self._unit(p1 - p0)
        guest_dir = self._unit(q1 - q0)
        if float(np.linalg.norm(base_dir)) < 1e-12 or float(np.linalg.norm(guest_dir)) < 1e-12:
            raise ValueError("fitup_attach_to_marking_line requires non-degenerate mark_line/ref_line")

        axis = np.cross(guest_dir, base_dir)
        axis_norm = float(np.linalg.norm(axis))
        dot = float(np.clip(np.dot(guest_dir, base_dir), -1.0, 1.0))
        if axis_norm < 1e-12:
            if dot < 0.0:
                fallback = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(float(np.dot(guest_dir, fallback))) > 0.99:
                    fallback = np.array([0.0, 1.0, 0.0], dtype=float)
                axis = self._safe_unit_from_cross(guest_dir, fallback, np.array([0.0, 0.0, 1.0], dtype=float))
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
        q_line_local = r_old.T @ (q1 - q0)
        q_line_local_u = self._unit(q_line_local)

        r_delta = self._rotation_matrix_from_axis_angle(axis, theta)
        r_align = r_delta @ r_old
        r_new = r_align

        constraints = step.get("constraints", {})
        set_rpy_deg = constraints.get("set_rpy_deg") if isinstance(constraints, dict) else None
        if set_rpy_deg is not None:
            if not isinstance(set_rpy_deg, (list, tuple)) or len(set_rpy_deg) != 3:
                raise ValueError("fitup_attach_to_marking_line constraints.set_rpy_deg must be [roll, pitch, yaw]")
            r_hint = rpy_to_rotation_matrix(float(set_rpy_deg[0]), float(set_rpy_deg[1]), float(set_rpy_deg[2]))
            best_phi = 0.0
            best_cost = float("inf")
            for phi_deg in np.linspace(-180.0, 180.0, 721):
                r_spin = self._rotation_matrix_from_axis_angle(base_dir, np.deg2rad(float(phi_deg)))
                r_candidate = r_spin @ r_align
                # Minimize geodesic-like matrix distance to legacy orientation hint.
                cost = float(np.linalg.norm(r_candidate - r_hint, ord="fro"))
                if cost < best_cost:
                    best_cost = cost
                    best_phi = float(phi_deg)
            r_spin = self._rotation_matrix_from_axis_angle(base_dir, np.deg2rad(best_phi))
            r_new = r_spin @ r_align

        origin_new = p0 - r_new @ q0_local
        state.set_transform(guest_id, origin_new, rotation_matrix_to_rpy_deg(r_new))

        model = step.get("model", {})
        fillet_fitup = model.get("fillet_fitup") if isinstance(model, dict) else None
        if not isinstance(fillet_fitup, dict):
            return

        required_keys = ("delta_mA", "delta_mB", "x_lower", "x_upper", "delta_y", "z_lower")
        missing_keys = [k for k in required_keys if k not in fillet_fitup]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"fillet_fitup missing required keys: {missing}; check CSV /steps/N/model/fillet_fitup/..."
            )

        axis_mode = str(fillet_fitup.get("axis_mode", "world"))
        if axis_mode != "world":
            raise ValueError("fillet_fitup axis_mode currently supports only 'world'.")

        x_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        y_dir = np.array([0.0, 1.0, 0.0], dtype=float)
        z_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        dy = self._sample(fillet_fitup["delta_y"])
        gtr = state.get_transform(guest_id)
        state.set_transform(guest_id, gtr["origin"] + dy * y_dir, gtr["rpy_deg"])

        dm_a = self._sample(fillet_fitup["delta_mA"])
        dm_b = self._sample(fillet_fitup["delta_mB"])

        inst = self.geom.get_instance(guest_id)
        proto_name = inst.get("prototype", "")
        proto = self.geom.get_prototype(proto_name)
        all_point_names = sorted(proto.get("features", {}).get("points", {}).keys())

        ref_line_points = []
        for ref in (guest_q0, guest_q1):
            if isinstance(ref, str) and ref.startswith("points."):
                ref_line_points.append(ref.split(".", 1)[1])
        ref_line_points = sorted(dict.fromkeys(ref_line_points))

        candidate_without_ref = [name for name in all_point_names if name not in ref_line_points]
        if len(candidate_without_ref) == 4:
            candidate_point_names = candidate_without_ref
            excluded_point_names = [name for name in ref_line_points if name in all_point_names]
        else:
            candidate_point_names = all_point_names
            excluded_point_names = []

        if len(candidate_point_names) != 4:
            step_id = str(step.get("id", ""))
            raise ValueError(
                "fillet_fitup supports exactly 4 physical vertices: "
                f"step_id='{step_id}', guest_id='{guest_id}', "
                f"all_point_names={all_point_names}, "
                f"candidate_point_names={candidate_point_names}, "
                f"excluded_point_names={excluded_point_names}, "
                f"ref_line_point_names={ref_line_points}"
            )

        world_points = {vnm: get_world_point(self.geom, state, guest_id, f"points.{vnm}") for vnm in candidate_point_names}
        by_z = sorted(candidate_point_names, key=lambda v: (float(world_points[v][2]), v))
        lower_vertices = by_z[:2]
        upper_vertices = by_z[2:]
        lower_by_y = sorted(lower_vertices, key=lambda v: (float(world_points[v][1]), v))
        upper_by_y = sorted(upper_vertices, key=lambda v: (float(world_points[v][1]), v))

        first_lower = lower_by_y[0]
        first_upper = min(upper_by_y, key=lambda v: (abs(float(world_points[first_lower][1] - world_points[v][1])), v))
        second_lower = lower_by_y[1]
        second_upper = upper_by_y[0] if upper_by_y[0] != first_upper else upper_by_y[1]
        pair_vertices = [(first_lower, first_upper), (second_lower, second_upper)]

        fillet_fitup["lower_points_effective"] = [f"points.{v}" for v in lower_vertices]
        fillet_fitup["upper_points_effective"] = [f"points.{v}" for v in upper_vertices]
        fillet_fitup["pair_effective"] = [{"lower": f"points.{lv}", "upper": f"points.{uv}"} for lv, uv in pair_vertices]

        for lower_v, upper_v in pair_vertices:
            x_lower_i = dm_b + self._sample(fillet_fitup["x_lower"])
            dx_lower_i = x_lower_i - dm_a
            dx_lower_local = self._world_vec_to_local(guest_id, state, dx_lower_i * x_dir)

            x_upper_i = dm_b + self._sample(fillet_fitup["x_upper"])
            dx_upper_i = x_upper_i - dm_a
            dx_upper_local = self._world_vec_to_local(guest_id, state, dx_upper_i * x_dir)

            z_i = max(0.0, float(self._sample(fillet_fitup["z_lower"])))
            dz_local = self._world_vec_to_local(guest_id, state, z_i * z_dir)

            state.add_point_offset(guest_id, lower_v, dx_lower_local)
            state.add_point_offset(guest_id, lower_v, dz_local)
            state.add_point_offset(guest_id, upper_v, dx_upper_local)
            state.add_point_offset(guest_id, upper_v, dz_local)

    def _fitup_pair_chain(self, step: Dict[str, Any], state: AssemblyState):
        model = step.get("model", {})
        chain = step.get("chain")
        if chain:
            raise ValueError("chain は廃止。base/guest に移行して下さい。1 step = 1ペア（base/guest）です。")
        if not isinstance(step.get("base"), dict) or not isinstance(step.get("guest"), dict):
            raise ValueError(
                "fitup_pair_chain requires /steps/N/base and /steps/N/guest as dict; "
                "1 step = 1 pair (base/guest)."
            )
        chain_like = [{"base": step["base"], "guest": step["guest"]}]
        constraints = step.get("constraints", {})
        butt_fitup = model.get("butt_fitup")
        fillet_fitup = model.get("fillet_fitup")

        if isinstance(butt_fitup, dict):
            required_keys = ("d_nom", "g0", "L_dist", "eps_mA", "eps_mB", "eps_cA", "eps_cB", "delta_y")
            missing_keys = [k for k in required_keys if k not in butt_fitup]
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(
                    f"butt_fitup missing required keys: {missing}; check CSV /steps/3/model/butt_fitup/..."
                )

            d_nom = float(butt_fitup["d_nom"])
            g0 = float(butt_fitup["g0"])
            w0_default = 2.0 * d_nom + g0
            w0_model = float(butt_fitup.get("w0", w0_default))
            enforce_nonnegative_gap = bool(butt_fitup.get("enforce_nonnegative_gap", False))
            delta_y = self._sample(butt_fitup["delta_y"])
            step_id = str(step.get("id", ""))
            pair_index = self._extract_pair_index(step_id)
            metrics_by_step = getattr(state, "butt_fitup_metrics", None)
            if metrics_by_step is None:
                metrics_by_step = {}
                state.butt_fitup_metrics = metrics_by_step
            metrics_for_step = metrics_by_step.setdefault(step_id, [])
            delta_y_applied_guests = set()

            def _extract_local_x_from_point_ref(instance_id: str, point_ref: Any) -> float | None:
                if not isinstance(point_ref, str) or not point_ref.startswith("points."):
                    return None
                point_name = point_ref.split(".", 1)[1]
                proto_id = self.geom.get_instance(instance_id).get("prototype", "")
                proto = self.geom.get_prototype(proto_id)
                point = proto.get("features", {}).get("points", {}).get(point_name)
                if not isinstance(point, (list, tuple)) or len(point) < 1:
                    return None
                return float(point[0])

            def _sample_pair_fitup() -> Dict[str, Any]:
                l_fit = self._sample(butt_fitup["L_dist"])
                em_a = self._sample(butt_fitup["eps_mA"])
                em_b = self._sample(butt_fitup["eps_mB"])
                w = w0_model + l_fit + (em_b - em_a)

                d_a = d_nom + self._sample(butt_fitup["eps_cA"])
                d_b = d_nom + self._sample(butt_fitup["eps_cB"])
                g_real = w - (d_a + d_b)
                interferes = bool(g_real < 0.0)
                clipped = False
                if enforce_nonnegative_gap and g_real < 0.0:
                    w = d_a + d_b
                    g_real = 0.0
                    interferes = True
                    clipped = True
                return {
                    "w": float(w),
                    "L": float(l_fit),
                    "emA": float(em_a),
                    "emB": float(em_b),
                    "dA": float(d_a),
                    "dB": float(d_b),
                    "g_real": float(g_real),
                    "interferes": interferes,
                    "clipped": clipped,
                }

            for pair in chain_like:
                base = pair.get("base", {})
                guest = pair.get("guest", {})
                if not isinstance(base, dict) or not isinstance(guest, dict):
                    continue
                base_id, guest_id = base["instance"], guest["instance"]
                base_p0, base_p1 = base.get("p0", "points.A"), base.get("p1", "points.D")
                guest_q0 = guest.get("q0", "points.B")
                guest_q1 = guest.get("q1")
                weld_x_local_0 = _extract_local_x_from_point_ref(guest_id, guest_q0)
                weld_x_local_1 = _extract_local_x_from_point_ref(guest_id, guest_q1) if guest_q1 else None

                p0 = get_world_point(self.geom, state, base_id, base_p0)
                p1 = get_world_point(self.geom, state, base_id, base_p1)
                # t_dir: marking-line direction, v: transverse (gap) direction.
                t_dir = self._unit(p1 - p0)
                n = self._get_plane_normal_world(base_id, state)
                v = self._safe_unit_from_cross(n, t_dir, np.array([0.0, 1.0, 0.0], dtype=float))

                pair0 = _sample_pair_fitup()
                # q0/q1 must be satisfied by one rigid-body transform, so both targets
                # use the same in-plane offset along v.
                pair1 = dict(pair0) if guest_q1 else None

                if guest_id not in delta_y_applied_guests:
                    gtr = state.get_transform(guest_id)
                    state.set_transform(guest_id, gtr["origin"] + delta_y * t_dir, gtr["rpy_deg"])
                    delta_y_applied_guests.add(guest_id)

                q0 = get_world_point(self.geom, state, guest_id, guest_q0)
                # Butt fitup targets are built in marking-line basis.
                # When explicit marking lines are not provided in step dict, we construct
                # virtual marking lines from edge refs + sampled dA/dB:
                #   M_A = E_A + dA * v
                #   M_B = E_B - dB * v
                # and enforce: M_B_target = M_A - w * v
                # => E_B_target = E_A + (dA + dB - w) * v = E_A - g_real * v
                # This keeps w as marking-line spacing (not edge absolute offset).
                q0_target = p0 + (pair0["dA"] + pair0["dB"] - pair0["w"]) * v
                t = q0_target - q0
                gtr_after_align = state.get_transform(guest_id)
                state.set_transform(guest_id, gtr_after_align["origin"] + t, gtr_after_align["rpy_deg"])

                if guest_q1 and pair1 is not None:
                    q1_target = p1 + (pair1["dA"] + pair1["dB"] - pair1["w"]) * v
                    gtr_aligned = state.get_transform(guest_id)
                    origin_old = np.array(gtr_aligned["origin"], dtype=float)
                    rpy_old = gtr_aligned["rpy_deg"]
                    r_old = rpy_to_rotation_matrix(*rpy_old)
                    q0_local = r_old.T @ (q0_target - origin_old)

                    q1 = get_world_point(self.geom, state, guest_id, guest_q1)
                    a = q1 - q0_target
                    b = q1_target - q0_target
                    a_proj = a - float(np.dot(a, n)) * n
                    b_proj = b - float(np.dot(b, n)) * n
                    a_norm = float(np.linalg.norm(a_proj))
                    b_norm = float(np.linalg.norm(b_proj))
                    if a_norm > 1e-12 and b_norm > 1e-12:
                        theta = float(np.arctan2(np.dot(n, np.cross(a_proj, b_proj)), np.dot(a_proj, b_proj)))
                        r_delta = self._rotation_matrix_from_axis_angle(n, theta)
                        r_new = r_delta @ r_old
                        origin_new = q0_target - r_new @ q0_local
                        rpy_new = rotation_matrix_to_rpy_deg(r_new)
                        state.set_transform(guest_id, origin_new, rpy_new)

                metric_entry: Dict[str, Any] = {
                    "guest_instance": guest_id,
                    "pair_index": pair_index,
                    "transverse_dir_world": v.tolist(),
                    "weld_x_local_0": weld_x_local_0,
                    "weld_x_local_1": weld_x_local_1,
                    "w": pair0["w"],
                    "w0": float(w0_model),
                    "L": pair0["L"],
                    "emA": pair0["emA"],
                    "emB": pair0["emB"],
                    "dA": pair0["dA"],
                    "dB": pair0["dB"],
                    "g_real": pair0["g_real"],
                    "interferes": pair0["interferes"],
                    "delta_y": float(delta_y),
                    "w_0": pair0["w"],
                    "L_0": pair0["L"],
                    "emA_0": pair0["emA"],
                    "emB_0": pair0["emB"],
                    "dA_0": pair0["dA"],
                    "dB_0": pair0["dB"],
                    "g_real_0": pair0["g_real"],
                    "interferes_0": pair0["interferes"],
                    "clipped_0": pair0["clipped"],
                    "w_1": None,
                    "L_1": None,
                    "emA_1": None,
                    "emB_1": None,
                    "dA_1": None,
                    "dB_1": None,
                    "g_real_1": None,
                    "interferes_1": None,
                    "clipped_1": None,
                }
                if pair1 is not None:
                    metric_entry.update(
                        {
                            "w_1": pair1["w"],
                            "L_1": pair1["L"],
                            "emA_1": pair1["emA"],
                            "emB_1": pair1["emB"],
                            "dA_1": pair1["dA"],
                            "dB_1": pair1["dB"],
                            "g_real_1": pair1["g_real"],
                            "interferes_1": pair1["interferes"],
                            "clipped_1": pair1["clipped"],
                        }
                    )
                metrics_for_step.append(metric_entry)
            return

        if isinstance(fillet_fitup, dict):
            required_keys = ("delta_mA", "delta_mB", "x_lower", "x_upper", "delta_y", "z_lower")
            missing_keys = [k for k in required_keys if k not in fillet_fitup]
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(
                    f"fillet_fitup missing required keys: {missing}; check CSV /steps/N/model/fillet_fitup/..."
                )

            axis_mode = str(fillet_fitup.get("axis_mode", "world"))
            if axis_mode != "world":
                raise ValueError("fillet_fitup axis_mode currently supports only 'world'.")
            x_dir = np.array([1.0, 0.0, 0.0], dtype=float)
            y_dir = np.array([0.0, 1.0, 0.0], dtype=float)
            z_dir = np.array([0.0, 0.0, 1.0], dtype=float)

            lower_points = fillet_fitup.get("lower_points", ["points.A", "points.B"])
            upper_points = fillet_fitup.get("upper_points", ["points.D", "points.C"])
            if isinstance(lower_points, str):
                lower_points = [lower_points]
            if isinstance(upper_points, str):
                upper_points = [upper_points]

            y_translation_applied_guests = set()
            point_offsets_applied_guests = set()
            for pair in chain_like:
                base = pair.get("base", {})
                guest = pair.get("guest", {})
                if not isinstance(base, dict) or not isinstance(guest, dict):
                    continue
                guest_id = guest["instance"]

                if guest_id not in y_translation_applied_guests:
                    dy = self._sample(fillet_fitup["delta_y"])
                    gtr = state.get_transform(guest_id)
                    state.set_transform(guest_id, gtr["origin"] + dy * y_dir, gtr["rpy_deg"])
                    y_translation_applied_guests.add(guest_id)

                if guest_id in point_offsets_applied_guests:
                    continue

                dm_a = self._sample(fillet_fitup["delta_mA"])
                dm_b = self._sample(fillet_fitup["delta_mB"])

                for ref in lower_points:
                    x_i = dm_b + self._sample(fillet_fitup["x_lower"])
                    dx_i = x_i - dm_a
                    dx_local = self._world_vec_to_local(guest_id, state, dx_i * x_dir)
                    state.add_point_offset(guest_id, self._parse_point_name(ref), dx_local)

                    z_i = max(0.0, float(self._sample(fillet_fitup["z_lower"])))
                    dz_local = self._world_vec_to_local(guest_id, state, z_i * z_dir)
                    state.add_point_offset(guest_id, self._parse_point_name(ref), dz_local)

                for ref in upper_points:
                    x_i = dm_b + self._sample(fillet_fitup["x_upper"])
                    dx_i = x_i - dm_a
                    dx_local = self._world_vec_to_local(guest_id, state, dx_i * x_dir)
                    state.add_point_offset(guest_id, self._parse_point_name(ref), dx_local)

                point_offsets_applied_guests.add(guest_id)
            return

        has_butt = all(k in model for k in ("dx0_logn", "dx1_logn", "dy_norm"))
        if has_butt:
            for pair in chain_like:
                base = pair.get("base", {})
                guest = pair.get("guest", {})
                if not isinstance(base, dict) or not isinstance(guest, dict):
                    continue
                base_id, guest_id = base["instance"], guest["instance"]
                base_p0, base_p1 = base.get("p0", "points.A"), base.get("p1", "points.D")
                guest_q0, guest_q1 = guest.get("q0", "points.B"), guest.get("q1", "points.C")

                p0 = get_world_point(self.geom, state, base_id, base_p0)
                p1 = get_world_point(self.geom, state, base_id, base_p1)
                q0 = get_world_point(self.geom, state, guest_id, guest_q0)
                u = self._unit(p1 - p0)
                n = self._get_plane_normal_world(base_id, state)
                v = self._safe_unit_from_cross(n, u, np.array([0.0, 1.0, 0.0], dtype=float))

                dx0 = self._sample(model["dx0_logn"])
                dx1 = self._sample(model["dx1_logn"])
                dy = self._sample(model["dy_norm"])
                q0_target = p0 + dx0 * u + dy * v if constraints.get("inplane_y", {}).get("direction") == "joint_perp" else p0 + dx0 * u
                t = q0_target - q0
                gtr = state.get_transform(guest_id)
                state.set_transform(guest_id, gtr["origin"] + t, gtr["rpy_deg"])

                q1_now = get_world_point(self.geom, state, guest_id, guest_q1)
                delta_u = float(np.dot((p1 + dx1 * u) - q1_now, u))
                delta_local = self._world_vec_to_local(guest_id, state, delta_u * u)
                state.add_point_offset(guest_id, self._parse_point_name(guest_q1), delta_local)
            return

        for pair in chain_like:
            base = pair.get("base") if isinstance(pair, dict) else None
            guest = pair.get("guest") if isinstance(pair, dict) else None
            if isinstance(base, dict):
                base_id = base.get("instance")
            elif isinstance(base, list) and base:
                base_id = base[0]
            else:
                base_id = None

            if isinstance(guest, dict):
                guest_id = guest.get("instance")
            elif isinstance(guest, list) and guest:
                guest_id = guest[0]
            else:
                guest_id = None

            if not base_id or not guest_id:
                continue
            base_tr = state.get_transform(base_id)
            base_dims = state.get_realized_dims(base_id)
            proto = self.geom.get_prototype(self.geom.get_instance(base_id)["prototype"])
            l_size = base_dims.get("L", float(proto.get("dims", {}).get("L", 2000.0)))
            gap = self._sample(constraints.get("coincident_1D", {}).get("gap_dist", 0.0)) if "coincident_1D" in constraints else 0.0
            state.set_transform(guest_id, base_tr["origin"] + np.array([l_size + gap, 0.0, 0.0], dtype=float), base_tr["rpy_deg"])
__all__ = ["ProcessEngine"]
