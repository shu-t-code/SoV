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

    def _apply_variation(self, step: Dict[str, Any], state: AssemblyState):
        target = step.get("target", {})
        model = step.get("model", {})
        step_id = str(step.get("id", ""))

        inst_ids: List[str] = []
        for sel_name in target.get("union", []):
            inst_ids.extend(self.flow.resolve_selector(sel_name, self.geom))
        inst_ids = list(dict.fromkeys(inst_ids))

        no_rigid_origin_on_cutting = bool(model.get("no_rigid_origin_on_cutting", True))
        is_cutting_step = step_id == "10_cutting"

        for inst_id in inst_ids:
            tr = state.get_transform(inst_id)
            o = tr["origin"].copy()
            if is_cutting_step and no_rigid_origin_on_cutting:
                dx = dy = dz = 0.0
            else:
                dx = self._sample(model.get("inplane_dx", 0.0))
                dy = self._sample(model.get("inplane_dy", 0.0))
                dz = self._sample(model.get("outplane_dz", 0.0))
            state.set_transform(inst_id, o + np.array([dx, dy, dz], dtype=float), tr["rpy_deg"])

            if bool(model.get("per_point_xy_noise", False)):
                proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
                pts = list(proto.get("features", {}).get("points", {}).keys())
                dist_x = model.get("point_dx", model.get("inplane_dx", 0.0))
                dist_y = model.get("point_dy", model.get("inplane_dy", 0.0))
                dist_z = model.get("point_dz", 0.0)
                for pnm in pts:
                    state.set_point_offset(inst_id, pnm, np.array([self._sample(dist_x), self._sample(dist_y), self._sample(dist_z)], dtype=float))

            if "dim_variations" in model:
                proto = self.geom.get_prototype(self.geom.get_instance(inst_id).get("prototype", ""))
                for dim_name, variation_spec in model["dim_variations"].items():
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
        constraints = step.get("constraints", {})
        has_butt = all(k in model for k in ("dx0_logn", "dx1_logn", "dy_norm"))
        if has_butt:
            for pair in chain:
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

        for pair in chain:
            base_id = pair["base"][0]
            guest_id = pair["guest"][0]
            base_tr = state.get_transform(base_id)
            base_dims = state.get_realized_dims(base_id)
            proto = self.geom.get_prototype(self.geom.get_instance(base_id)["prototype"])
            l_size = base_dims.get("L", float(proto.get("dims", {}).get("L", 2000.0)))
            gap = self._sample(constraints.get("coincident_1D", {}).get("gap_dist", 0.0)) if "coincident_1D" in constraints else 0.0
            state.set_transform(guest_id, base_tr["origin"] + np.array([l_size + gap, 0.0, 0.0], dtype=float), base_tr["rpy_deg"])


__all__ = ["ProcessEngine"]
