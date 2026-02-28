"""Process step application engine."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .core_models import AssemblyState, DistributionSampler, FlowModel, GeometryModel, get_world_point, rpy_to_rotation_matrix


class ProcessEngine:
    def __init__(self, geom: GeometryModel, flow: FlowModel, rng: np.random.Generator):
        self.geom = geom
        self.flow = flow
        self.rng = rng

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

    def apply_steps(self, state: AssemblyState, steps_mask: List[bool] | None = None):
        """Apply enabled flow steps to ``state`` in flow order."""
        for idx, step in enumerate(self.flow.steps):
            if steps_mask is None or (idx < len(steps_mask) and steps_mask[idx]):
                self.apply_step(step, state)

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

    def _fitup_pair_chain(self, step: Dict[str, Any], state: AssemblyState):
        model = step.get("model", {})
        chain = step.get("chain", [])
        if chain:
            chain_like = chain
        elif isinstance(step.get("base"), dict) and isinstance(step.get("guest"), dict):
            chain_like = [{"base": step["base"], "guest": step["guest"]}]
        else:
            raise ValueError(
                "fitup_pair_chain requires either /steps/N/chain/... entries or both /steps/N/base and /steps/N/guest; "
                "e.g. /steps/3/base/instance and /steps/3/guest/instance."
            )
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
            metrics_by_step = getattr(state, "butt_fitup_metrics", None)
            if metrics_by_step is None:
                metrics_by_step = {}
                state.butt_fitup_metrics = metrics_by_step
            metrics_for_step = metrics_by_step.setdefault(step_id, [])
            delta_y_applied_guests = set()

            def _sample_pair_fitup() -> Dict[str, Any]:
                l_fit = self._sample(butt_fitup["L_dist"])
                em_a = self._sample(butt_fitup["eps_mA"])
                em_b = self._sample(butt_fitup["eps_mB"])
                w = w0_model + l_fit + (em_b - em_a)

                d_a = d_nom + self._sample(butt_fitup["eps_cA"])
                d_b = d_nom + self._sample(butt_fitup["eps_cB"])
                g_real = w - (d_a + d_b)
                interferes = bool(g_real < 0.0)
                if enforce_nonnegative_gap and g_real < 0.0:
                    w = d_a + d_b
                    g_real = 0.0
                    interferes = True
                return {
                    "w": float(w),
                    "L": float(l_fit),
                    "emA": float(em_a),
                    "emB": float(em_b),
                    "dA": float(d_a),
                    "dB": float(d_b),
                    "g_real": float(g_real),
                    "interferes": interferes,
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

                p0 = get_world_point(self.geom, state, base_id, base_p0)
                p1 = get_world_point(self.geom, state, base_id, base_p1)
                u = self._unit(p1 - p0)
                n = self._get_plane_normal_world(base_id, state)
                # u is the welding-line direction and v is the in-plane direction orthogonal to u.
                v = self._safe_unit_from_cross(n, u, np.array([0.0, 1.0, 0.0], dtype=float))

                pair0 = _sample_pair_fitup()
                pair1 = _sample_pair_fitup() if guest_q1 else None

                if guest_id not in delta_y_applied_guests:
                    gtr = state.get_transform(guest_id)
                    state.set_transform(guest_id, gtr["origin"] + delta_y * u, gtr["rpy_deg"])
                    delta_y_applied_guests.add(guest_id)

                q0 = get_world_point(self.geom, state, guest_id, guest_q0)
                # PR1/PR2 convention: w is interpreted as the translation amount along v.
                q0_target = p0 + pair0["w"] * v
                t = q0_target - q0
                gtr_after_align = state.get_transform(guest_id)
                state.set_transform(guest_id, gtr_after_align["origin"] + t, gtr_after_align["rpy_deg"])

                if guest_q1 and pair1 is not None:
                    q1 = get_world_point(self.geom, state, guest_id, guest_q1)
                    q1_target = p1 + pair1["w"] * v
                    delta1_world = q1_target - q1
                    delta1_world = float(np.dot(delta1_world, v)) * v
                    delta1_local = self._world_vec_to_local(guest_id, state, delta1_world)
                    state.add_point_offset(guest_id, self._parse_point_name(guest_q1), delta1_local)

                metric_entry: Dict[str, Any] = {
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
                    "w_1": None,
                    "L_1": None,
                    "emA_1": None,
                    "emB_1": None,
                    "dA_1": None,
                    "dB_1": None,
                    "g_real_1": None,
                    "interferes_1": None,
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
