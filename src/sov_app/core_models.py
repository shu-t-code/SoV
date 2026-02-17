"""Core geometry/flow/state models and validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.deg2rad([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


class DistributionSampler:
    @staticmethod
    def sample(dist_def: Any, rng: np.random.Generator, registry: Optional[Dict[str, Any]] = None) -> float:
        if isinstance(dist_def, str):
            if registry and dist_def in registry:
                dist_def = registry[dist_def]
            else:
                raise ValueError(f"Unknown dist name: {dist_def}")
        if isinstance(dist_def, (int, float)):
            return float(dist_def)
        if not isinstance(dist_def, dict):
            return 0.0

        dtype = dist_def.get("type", "Fixed")
        if dtype == "Fixed":
            return float(dist_def.get("value", 0.0))
        if dtype == "NormalLinear":
            mean = float(dist_def.get("mean", 0.0))
            std = float(dist_def.get("std", 1.0))
            return float(rng.normal(mean, std))
        if dtype == "LogNormalLinear":
            mean = float(dist_def.get("mean", 1.0))
            std = float(dist_def.get("std", 0.5))
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            return float(rng.lognormal(mu, sigma))
        return 0.0


class GeometryModel:
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.units = data.get("units", "mm")
        self.datums = data.get("datums", {})
        self.prototypes = {p["id"]: p for p in data.get("prototypes", [])}
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.arrays = data.get("arrays", [])

        for inst in data.get("instances", []):
            self.instances[inst["id"]] = inst
        for arr in self.arrays:
            self._expand_array(arr)

    def _expand_array(self, arr: Dict[str, Any]):
        proto_id = arr["prototype"]
        count = arr["count"]
        pattern = arr["id_pattern"]
        placement = arr["placement"]
        base_origin = np.array(placement["base_origin"], dtype=float)
        delta = placement.get("delta_per_index", {"dx": 0, "dy": 0, "dz": 0, "d_rpy_deg": [0, 0, 0]})
        dx, dy, dz = delta.get("dx", 0.0), delta.get("dy", 0.0), delta.get("dz", 0.0)
        d_rpy = delta.get("d_rpy_deg", [0, 0, 0])

        for i in range(count):
            inst_id = pattern.replace("{index}", str(i + 1))
            origin = base_origin + np.array([dx * i, dy * i, dz * i], dtype=float)
            rpy = [d_rpy[j] * i for j in range(3)]
            tags = arr.get("tags_each", [])
            self.instances[inst_id] = {
                "id": inst_id,
                "prototype": proto_id,
                "frame": {"parent": placement["parent"], "origin": origin.tolist(), "rpy_deg": rpy},
                "tags": tags,
            }

    def get_instance_ids(self) -> List[str]:
        return list(self.instances.keys())

    def get_prototype(self, proto_id: str) -> Dict[str, Any]:
        return self.prototypes.get(proto_id, {})

    def get_instance(self, inst_id: str) -> Dict[str, Any]:
        return self.instances.get(inst_id, {})

    def get_available_refs_for_instance(self, inst_id: str) -> List[str]:
        refs = []
        inst = self.instances.get(inst_id)
        if not inst:
            return refs
        proto = self.get_prototype(inst["prototype"])
        feats = proto.get("features", {})
        refs.extend([f"points.{k}" for k in feats.get("points", {}).keys()])
        refs.extend([f"edges.{k}.mid" for k in feats.get("edges", {}).keys()])
        if "points" in feats and all(k in feats["points"] for k in ["A", "B", "C", "D"]):
            refs.append("face.center")
        return refs


class FlowModel:
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.units = data.get("units", "mm")
        self.selectors = data.get("selectors", {})
        self.dists = data.get("dists", {})
        self.steps = data.get("steps", [])
        self.measurements = data.get("measurements", [])

    def resolve_selector(self, selector_name: str, geom: GeometryModel) -> List[str]:
        sel_def = self.selectors.get(selector_name, {})
        result: List[str] = []
        if "ids" in sel_def:
            result.extend(sel_def["ids"])
        if "tags_any" in sel_def:
            tags = sel_def["tags_any"]
            for inst_id, inst in geom.instances.items():
                if any(t in inst.get("tags", []) for t in tags):
                    result.append(inst_id)
        if "id_glob" in sel_def:
            prefix = sel_def["id_glob"].replace("*", "")
            for inst_id in geom.instances.keys():
                if inst_id.startswith(prefix):
                    result.append(inst_id)
        return list(dict.fromkeys(result))


class AssemblyState:
    def __init__(self, geom: GeometryModel):
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.realized_dims: Dict[str, Dict[str, float]] = {}
        self.point_offsets: Dict[str, Dict[str, np.ndarray]] = {}

        for inst_id, inst in geom.instances.items():
            fr = inst["frame"]
            self.transforms[inst_id] = {"origin": np.array(fr["origin"], dtype=float), "rpy_deg": list(fr["rpy_deg"])}
            proto = geom.get_prototype(inst["prototype"])
            self.realized_dims[inst_id] = dict(proto.get("dims", {}))
            self.point_offsets[inst_id] = {}

    def get_transform(self, inst_id: str) -> Dict[str, Any]:
        return self.transforms.get(inst_id, {"origin": np.zeros(3), "rpy_deg": [0, 0, 0]})

    def set_transform(self, inst_id: str, origin: np.ndarray, rpy_deg: List[float]):
        self.transforms[inst_id] = {"origin": np.array(origin, dtype=float), "rpy_deg": list(rpy_deg)}

    def get_realized_dims(self, inst_id: str) -> Dict[str, float]:
        return self.realized_dims.get(inst_id, {})

    def set_realized_dim(self, inst_id: str, dim_name: str, value: float):
        if inst_id not in self.realized_dims:
            self.realized_dims[inst_id] = {}
        self.realized_dims[inst_id][dim_name] = value

    def set_point_offset(self, inst_id: str, point_name: str, dxyz: np.ndarray):
        if inst_id not in self.point_offsets:
            self.point_offsets[inst_id] = {}
        self.point_offsets[inst_id][point_name] = np.array(dxyz, dtype=float)

    def add_point_offset(self, inst_id: str, point_name: str, dxyz_add: np.ndarray):
        if inst_id not in self.point_offsets:
            self.point_offsets[inst_id] = {}
        cur = self.point_offsets[inst_id].get(point_name, np.zeros(3, dtype=float))
        self.point_offsets[inst_id][point_name] = cur + np.array(dxyz_add, dtype=float)

    def get_point_offset(self, inst_id: str, point_name: str) -> np.ndarray:
        return self.point_offsets.get(inst_id, {}).get(point_name, np.zeros(3, dtype=float))


def _get_local_point_from_ref_with_dims(
    proto: Dict[str, Any],
    ref: str,
    dims: Dict[str, float],
    point_offsets: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    feats = proto.get("features", {})
    dims_nominal = proto.get("dims", {})
    tokens = ref.split(".")
    point_offsets = point_offsets or {}

    l_real = dims.get("L", dims_nominal.get("L", 1000.0))
    h_real = dims.get("H", dims_nominal.get("H", 1000.0))
    t_real = dims.get("t", dims_nominal.get("t", 10.0))

    l_nom = dims_nominal.get("L", 1000.0)
    h_nom = dims_nominal.get("H", 1000.0)
    t_nom = dims_nominal.get("t", 10.0)

    def scale_point(p_nom: np.ndarray) -> np.ndarray:
        return np.array([
            p_nom[0] * (l_real / max(l_nom, 0.001)),
            p_nom[1] * (h_real / max(h_nom, 0.001)),
            p_nom[2] * (t_real / max(t_nom, 0.001)),
        ], dtype=float)

    if tokens[0] == "points":
        pname = tokens[1]
        return scale_point(np.array(feats["points"][pname], dtype=float)) + point_offsets.get(pname, np.zeros(3, dtype=float))
    if tokens[0] == "edges":
        edge_name = tokens[1]
        e = feats["edges"][edge_name]
        p0 = scale_point(np.array(feats["points"][e["endpoints"][0]], dtype=float))
        p1 = scale_point(np.array(feats["points"][e["endpoints"][1]], dtype=float))
        if len(tokens) >= 3 and tokens[2] == "mid":
            t_param = 0.5
        elif len(tokens) >= 3 and tokens[2].startswith("t="):
            t_param = float(tokens[2][2:])
        else:
            t_param = 0.0
        return (1.0 - t_param) * p0 + t_param * p1
    if tokens[0] == "face" and tokens[1] == "center":
        corners = []
        for k in ("A", "B", "C", "D"):
            if k in feats.get("points", {}):
                corners.append(scale_point(np.array(feats["points"][k], dtype=float)) + point_offsets.get(k, np.zeros(3, dtype=float)))
        if not corners:
            raise ValueError("face.center requires corner points A,B,C,D in prototype.features.points")
        return np.mean(corners, axis=0)
    raise ValueError(f"Unsupported ref: {ref}")


def get_world_point(geom: GeometryModel, state: AssemblyState, inst_id: str, ref: str) -> np.ndarray:
    inst = geom.get_instance(inst_id)
    proto = geom.get_prototype(inst.get("prototype", ""))
    dims = state.get_realized_dims(inst_id) or proto.get("dims", {})
    local = _get_local_point_from_ref_with_dims(proto, ref, dims)

    tokens = ref.split(".")
    feats = proto.get("features", {})
    if len(tokens) >= 2 and tokens[0] == "points":
        local = local + state.get_point_offset(inst_id, tokens[1])
    elif len(tokens) >= 2 and tokens[0] == "edges":
        e = feats.get("edges", {}).get(tokens[1], {})
        endpoints = e.get("endpoints", [])
        if len(endpoints) >= 2:
            t_param = 0.5 if len(tokens) >= 3 and tokens[2] == "mid" else float(tokens[2][2:]) if len(tokens) >= 3 and tokens[2].startswith("t=") else 0.0
            local = local + (1.0 - t_param) * state.get_point_offset(inst_id, endpoints[0]) + t_param * state.get_point_offset(inst_id, endpoints[1])
    elif tokens[0] == "face" and len(tokens) >= 2 and tokens[1] == "center":
        corners = [k for k in ("A", "B", "C", "D") if k in feats.get("points", {})]
        if corners:
            local = local + np.mean([state.get_point_offset(inst_id, k) for k in corners], axis=0)

    tr = state.get_transform(inst_id)
    return rpy_to_rotation_matrix(*tr["rpy_deg"]) @ local + tr["origin"]


class Validator:
    @staticmethod
    def validate(geom: GeometryModel, flow: FlowModel) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        for inst_id, inst in geom.instances.items():
            pid = inst.get("prototype", "")
            if pid not in geom.prototypes:
                issues.append({"level": "error", "message": f"Instance '{inst_id}' references unknown prototype '{pid}'"})
        for step in flow.steps:
            if step.get("op") == "fitup_array_attach":
                sel = step.get("guests", {}).get("selector", "")
                if sel and sel not in flow.selectors:
                    issues.append({"level": "warning", "message": f"Step '{step.get('id')}' references unknown selector '{sel}'"})
        return issues


__all__ = [
    "AssemblyState",
    "DistributionSampler",
    "FlowModel",
    "GeometryModel",
    "Validator",
    "get_world_point",
    "rpy_to_rotation_matrix",
]
