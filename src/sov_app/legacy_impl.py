#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QPushButton, QLabel, QCheckBox, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QSpinBox, QFileDialog,
    QLineEdit, QComboBox, QScrollArea, QDialog, QRadioButton, QButtonGroup,
    QSlider, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont

# Matplotlib for embedding in Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ===== Open3D 可視化（フォールバックは Matplotlib）==========================
USE_OPEN3D = True
try:
    import open3d as o3d
    import open3d.visualization.gui as gui
except Exception:
    USE_OPEN3D = False
    print("[WARN] Open3D GUI not available; falling back to Matplotlib 3D")

if not USE_OPEN3D:
    pass  # Matplotlib 3D は後で使用

# ===== ファイル監視（任意）====================================================
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except Exception:
    HAS_WATCHDOG = False
    print("[WARN] watchdog not available; hot-reload disabled")


def log_env():
    import platform
    o3d_ver = "N/A"
    try:
        import open3d as _o3d
        o3d_ver = getattr(_o3d, "__version__", "unknown")
    except Exception:
        pass
    from PySide6 import QtCore
    print(f"[ENV] Python {sys.version.split()[0]} on {platform.system()}")
    print(f"[ENV] PySide6 {QtCore.__version__}, Matplotlib {matplotlib.__version__}, Open3D {o3d_ver}")


# =============================================================================
# CSV I/O ユーティリティ
# =============================================================================
from .io_csv import csv_to_nested_dict, load_data_from_csv, nested_dict_to_csv_rows

# =============================================================================
# フォント / 幾何ユーティリティ
# =============================================================================
def setup_font() -> QFont:
    for font_name in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        f = QFont(font_name, 9)
        if f.exactMatch():
            return f
    return QFont()

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


# =============================================================================
# 色補間ユーティリティ（機能1用）
# =============================================================================
def deviation_color_map(deviation: float, tol_mm: float) -> np.ndarray:
    """
    偏差量に応じて青→赤の色を返す
    deviation: 偏差距離 [mm]
    tol_mm: 許容公差 [mm]
    """
    ratio = min(deviation / max(tol_mm, 0.001), 1.0)
    blue = np.array([0.2, 0.4, 1.0])
    red = np.array([1.0, 0.0, 0.0])
    return (1.0 - ratio) * blue + ratio * red


# =============================================================================
# 分布サンプラー（辞書/名前/数値を許容）
# =============================================================================
class DistributionSampler:
    @staticmethod
    def sample(dist_def: Any, rng: np.random.Generator, registry: Optional[Dict[str, Any]] = None) -> float:
        """
        dist_def: dict | str | number
          - dict: {"type": "NormalLinear", ...}
          - str : "N_cut_xy" -> registry から解決
          - number: そのまま定数扱い
        """
        # 文字列（分布名）はレジストリで解決
        if isinstance(dist_def, str):
            if registry and dist_def in registry:
                dist_def = registry[dist_def]
            else:
                raise ValueError(f"Unknown dist name: {dist_def}")
        # 数値は定数
        if isinstance(dist_def, (int, float)):
            return float(dist_def)

        if not isinstance(dist_def, dict):
            return 0.0

        dtype = dist_def.get("type", "Fixed")
        if dtype == "Fixed":
            return float(dist_def.get("value", 0.0))
        elif dtype == "NormalLinear":
            mean = float(dist_def.get("mean", 0.0))
            std  = float(dist_def.get("std", 1.0))
            return float(rng.normal(mean, std))
        elif dtype == "LogNormalLinear":
            mean = float(dist_def.get("mean", 1.0))
            std  = float(dist_def.get("std",  0.5))
            mu    = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + std**2 / mean**2))
            return float(rng.lognormal(mu, sigma))
        else:
            return 0.0


# =============================================================================
# 幾何モデル
# =============================================================================
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
        delta = placement.get("delta_per_index", {"dx":0,"dy":0,"dz":0,"d_rpy_deg":[0,0,0]})
        dx, dy, dz = delta.get("dx",0.0), delta.get("dy",0.0), delta.get("dz",0.0)
        d_rpy = delta.get("d_rpy_deg", [0,0,0])

        for i in range(count):
            inst_id = pattern.replace("{index}", str(i+1))
            origin = base_origin + np.array([dx*i, dy*i, dz*i], dtype=float)
            rpy = [d_rpy[j]*i for j in range(3)]
            tags = arr.get("tags_each", [])
            self.instances[inst_id] = {
                "id": inst_id,
                "prototype": proto_id,
                "frame": {"parent": placement["parent"], "origin": origin.tolist(), "rpy_deg": rpy},
                "tags": tags
            }

    def get_instance_ids(self) -> List[str]:
        return list(self.instances.keys())

    def get_prototype(self, proto_id: str) -> Dict[str, Any]:
        return self.prototypes.get(proto_id, {})

    def get_instance(self, inst_id: str) -> Dict[str, Any]:
        return self.instances.get(inst_id, {})

    def get_available_refs_for_instance(self, inst_id: str) -> List[str]:
        """
        指定インスタンスの参照可能な点リストを返す
        例: ["points.A", "points.B", "edges.AB.mid", "face.center", ...]
        """
        refs = []
        inst = self.instances.get(inst_id)
        if not inst:
            return refs
        proto = self.get_prototype(inst["prototype"])
        feats = proto.get("features", {})

        # points
        for pt_name in feats.get("points", {}).keys():
            refs.append(f"points.{pt_name}")

        # edges (端点 + 中点)
        for edge_name in feats.get("edges", {}).keys():
            refs.append(f"edges.{edge_name}.mid")

        # face center
        if "points" in feats:
            pts = feats["points"]
            if all(k in pts for k in ["A", "B", "C", "D"]):
                refs.append("face.center")

        return refs


# =============================================================================
# Flow モデル（steps + 任意計測 measurements）
# =============================================================================
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


# =============================================================================
# アセンブリ状態（実現寸法を追加）
# =============================================================================
class AssemblyState:
    def __init__(self, geom: GeometryModel):
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.realized_dims: Dict[str, Dict[str, float]] = {}

        # 点ごとの誤差（ローカル座標系で保持）
        # point_offsets[inst_id][point_name] = np.array([dx, dy, dz])
        self.point_offsets: Dict[str, Dict[str, np.ndarray]] = {}

        for inst_id, inst in geom.instances.items():
            fr = inst["frame"]
            self.transforms[inst_id] = {
                "origin": np.array(fr["origin"], dtype=float),
                "rpy_deg": list(fr["rpy_deg"])
            }

            # 初期化時は公称寸法をコピー
            proto = geom.get_prototype(inst["prototype"])
            self.realized_dims[inst_id] = dict(proto.get("dims", {}))

            # 点誤差テーブルを初期化
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

    # ===== 点ごとの誤差API =====
    def set_point_offset(self, inst_id: str, point_name: str, dxyz: np.ndarray):
        """点オフセットを上書き（注意：既存を消す）"""
        if inst_id not in self.point_offsets:
            self.point_offsets[inst_id] = {}
        self.point_offsets[inst_id][point_name] = np.array(dxyz, dtype=float)

    def add_point_offset(self, inst_id: str, point_name: str, dxyz_add: np.ndarray):
        """点オフセットを加算（切断誤差を保持したまま追加補正できる）"""
        if inst_id not in self.point_offsets:
            self.point_offsets[inst_id] = {}
        cur = self.point_offsets[inst_id].get(point_name, np.zeros(3, dtype=float))
        self.point_offsets[inst_id][point_name] = cur + np.array(dxyz_add, dtype=float)

    def get_point_offset(self, inst_id: str, point_name: str) -> np.ndarray:
        return self.point_offsets.get(inst_id, {}).get(point_name, np.zeros(3, dtype=float))



# =============================================================================
# 点参照 → ワールド座標（実寸法を考慮）【修正版】
# =============================================================================
def _get_local_point_from_ref_with_dims(
    proto: Dict[str, Any],
    ref: str,
    dims: Dict[str, float],
    point_offsets: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    寸法ばらつき +（任意で）頂点ごとのオフセットを考慮した局所座標計算。
    - points.X は point_offsets[X] を加算
    - edges.*.mid は “オフセット済み端点” から中点を計算（←重要）
    - face.center も “オフセット済み四隅” の平均
    """
    feats = proto.get("features", {})
    dims_nominal = proto.get("dims", {})
    tokens = ref.split(".")
    point_offsets = point_offsets or {}

    # 実寸法（無ければ公称）
    L_real = dims.get("L", dims_nominal.get("L", 1000.0))
    H_real = dims.get("H", dims_nominal.get("H", 1000.0))
    t_real = dims.get("t", dims_nominal.get("t", 10.0))

    # 公称寸法（スケーリング用）
    L_nom = dims_nominal.get("L", 1000.0)
    H_nom = dims_nominal.get("H", 1000.0)
    t_nom = dims_nominal.get("t", 10.0)

    def scale_point(p_nom: np.ndarray) -> np.ndarray:
        return np.array([
            p_nom[0] * (L_real / max(L_nom, 0.001)),
            p_nom[1] * (H_real / max(H_nom, 0.001)),
            p_nom[2] * (t_real / max(t_nom, 0.001)),
        ], dtype=float)

    if tokens[0] == "points":
        pname = tokens[1]
        p_nominal = np.array(feats["points"][pname], dtype=float)
        p_real = scale_point(p_nominal)
        off = point_offsets.get(pname, np.zeros(3, dtype=float))
        return p_real + off

    elif tokens[0] == "edges":
        e = feats["edges"][tokens[1]]
        ep0 = e["endpoints"][0]
        ep1 = e["endpoints"][1]

        p0_nominal = np.array(feats["points"][ep0], dtype=float)
        p1_nominal = np.array(feats["points"][ep1], dtype=float)

        p0_real = scale_point(p0_nominal) + point_offsets.get(ep0, np.zeros(3, dtype=float))
        p1_real = scale_point(p1_nominal) + point_offsets.get(ep1, np.zeros(3, dtype=float))

        # パラメータ t の取得
        if len(tokens) >= 3 and tokens[2] == "mid":
            t_param = 0.5
        elif len(tokens) >= 3 and tokens[2].startswith("t="):
            t_param = float(tokens[2][2:])
        else:
            t_param = 0.0

        return (1.0 - t_param) * p0_real + t_param * p1_real

    elif tokens[0] == "face" and tokens[1] == "center":
        pts = feats.get("points", {})
        corners = []
        for k in ("A", "B", "C", "D"):
            if k in pts:
                c_nom = np.array(pts[k], dtype=float)
                c_real = scale_point(c_nom) + point_offsets.get(k, np.zeros(3, dtype=float))
                corners.append(c_real)
        if not corners:
            raise ValueError("face.center requires corner points A,B,C,D in prototype.features.points")
        return np.mean(corners, axis=0)

    else:
        raise ValueError(f"Unsupported ref: {ref}")

def get_world_point(geom: GeometryModel, state: AssemblyState, inst_id: str, ref: str) -> np.ndarray:
    """
    実現された寸法 + 点ごとの誤差（per-point）を考慮してワールド座標を計算
    """
    inst = geom.get_instance(inst_id)
    proto = geom.get_prototype(inst.get("prototype", ""))

    # 実現寸法（なければ公称）
    dims = state.get_realized_dims(inst_id)
    if not dims:
        dims = proto.get("dims", {})

    # 寸法を考慮したローカル点
    local = _get_local_point_from_ref_with_dims(proto, ref, dims)

    tokens = ref.split(".")
    feats = proto.get("features", {})

    # ===== ここから：点ごとの誤差を加える（追加） =====
    if len(tokens) >= 2 and tokens[0] == "points":
        # points.<name>
        pnm = tokens[1]
        local = local + state.get_point_offset(inst_id, pnm)

    elif len(tokens) >= 2 and tokens[0] == "edges":
        # edges.<edge>.mid or edges.<edge>.t=...
        edge_name = tokens[1]
        e = feats.get("edges", {}).get(edge_name, {})
        endpoints = e.get("endpoints", [])
        if len(endpoints) >= 2:
            ep0, ep1 = endpoints[0], endpoints[1]

            # t を決める
            if len(tokens) >= 3 and tokens[2] == "mid":
                t_param = 0.5
            elif len(tokens) >= 3 and tokens[2].startswith("t="):
                t_param = float(tokens[2][2:])
            else:
                t_param = 0.0

            d0 = state.get_point_offset(inst_id, ep0)
            d1 = state.get_point_offset(inst_id, ep1)
            local = local + (1.0 - t_param) * d0 + t_param * d1

    elif tokens[0] == "face" and len(tokens) >= 2 and tokens[1] == "center":
        # face.center を使う場合：四隅の offset を平均して足す（任意だが整合性が良い）
        pts = feats.get("points", {})
        corners = [k for k in ("A", "B", "C", "D") if k in pts]
        if corners:
            d = np.mean([state.get_point_offset(inst_id, k) for k in corners], axis=0)
            local = local + d
    # ===== 追加ここまで =====

    # 変換（回転 + 平行移動）
    tr = state.get_transform(inst_id)
    R = rpy_to_rotation_matrix(*tr["rpy_deg"])
    o = tr["origin"]
    return R @ local + o


# =============================================================================
# 工程エンジン（寸法誤差を追加）
# =============================================================================
class ProcessEngine:
    def __init__(self, geom: GeometryModel, flow: FlowModel, rng: np.random.Generator):
        self.geom = geom
        self.flow = flow
        self.rng = rng

    def _sample(self, spec: Any) -> float:
        return DistributionSampler.sample(spec, self.rng, self.flow.dists)

    # ===== ベクトル/座標補助 =====
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
        """簡易：板ローカル+Zを板面法線とみなす（rpyで回転）"""
        tr = state.get_transform(inst_id)
        R = rpy_to_rotation_matrix(*tr["rpy_deg"])
        return self._unit(R @ np.array([0.0, 0.0, 1.0], dtype=float))

    def _world_vec_to_local(self, inst_id: str, state: AssemblyState, vec_world: np.ndarray) -> np.ndarray:
        """point_offsets はローカル保持なので world補正を local に戻す"""
        tr = state.get_transform(inst_id)
        R = rpy_to_rotation_matrix(*tr["rpy_deg"])
        return (R.T @ np.array(vec_world, dtype=float))

    def _parse_point_name(self, ref: str) -> str:
        """
        ref="points.D" -> "D"
        refがpoints.*でない場合は ValueError
        """
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
            # ★ここで butt joint も従来モードもまとめて処理する
            self._fitup_pair_chain(step, state)
        elif op == "inspection":
            pass

    def _apply_variation(self, step: Dict[str, Any], state: AssemblyState):
        """
        位置誤差（origin）と寸法誤差に加え、
        点ごとの誤差（per-point）をオプションで適用する。
        """
        target = step.get("target", {})
        model  = step.get("model", {})
        step_id = str(step.get("id", ""))

        # 対象インスタンス
        inst_ids: List[str] = []
        for sel_name in target.get("union", []):
            inst_ids.extend(self.flow.resolve_selector(sel_name, self.geom))
        inst_ids = list(dict.fromkeys(inst_ids))

        # cutting工程では origin ノイズを入れない（形状誤差だけ）
        no_rigid_origin_on_cutting = bool(model.get("no_rigid_origin_on_cutting", True))
        is_cutting_step = (step_id == "10_cutting")

        for inst_id in inst_ids:
            tr = state.get_transform(inst_id)
            o  = tr["origin"].copy()

            # (1) 板全体の剛体ズレ（origin）
            if is_cutting_step and no_rigid_origin_on_cutting:
                dx = dy = dz = 0.0
            else:
                dx = self._sample(model.get("inplane_dx", 0.0))
                dy = self._sample(model.get("inplane_dy", 0.0))
                dz = self._sample(model.get("outplane_dz", 0.0))

            o += np.array([dx, dy, dz], dtype=float)
            state.set_transform(inst_id, o, tr["rpy_deg"])

            # (2) 点ごとの誤差（per-point offsets）
            if bool(model.get("per_point_xy_noise", False)):
                inst = self.geom.get_instance(inst_id)
                proto = self.geom.get_prototype(inst.get("prototype", ""))

                pts = list(proto.get("features", {}).get("points", {}).keys())

                dist_x = model.get("point_dx", model.get("inplane_dx", 0.0))
                dist_y = model.get("point_dy", model.get("inplane_dy", 0.0))
                dist_z = model.get("point_dz", 0.0)

                for pnm in pts:
                    ddx = self._sample(dist_x)
                    ddy = self._sample(dist_y)
                    ddz = self._sample(dist_z)
                    state.set_point_offset(inst_id, pnm, np.array([ddx, ddy, ddz], dtype=float))

            # (3) 寸法誤差（realized_dims）
            if "dim_variations" in model:
                inst = self.geom.get_instance(inst_id)
                proto = self.geom.get_prototype(inst.get("prototype", ""))

                for dim_name, variation_spec in model["dim_variations"].items():
                    if dim_name in proto.get("dims", {}):
                        nominal = float(proto["dims"][dim_name])
                        err = self._sample(variation_spec)
                        state.set_realized_dim(inst_id, dim_name, nominal + err)

    def _fitup_array_attach(self, step: Dict[str, Any], state: AssemblyState):
        base_info = step.get("base", {})
        guests_info = step.get("guests", {})
        constraints = step.get("constraints", {})
        base_inst_id = base_info["instance"]

        guest_ids = self.flow.resolve_selector(guests_info["selector"], self.geom)
        pattern = guests_info["pattern"]
        pitch = float(pattern.get("pitch_on_base_mm", 0.0))
        start_offset = float(pattern.get("start_offset_mm", 0.0))

        base_tr = state.get_transform(base_inst_id)
        base_o  = base_tr["origin"]
        set_rpy = constraints.get("set_rpy_deg", None)

        for i, gid in enumerate(guest_ids):
            gap = 0.0
            if "coincident_1D" in constraints:
                gap_def = constraints["coincident_1D"].get("gap_dist", 0.0)
                gap = self._sample(gap_def)
            dy = 0.0
            if "inplane_y" in constraints and "dist" in constraints["inplane_y"]:
                dy = self._sample(constraints["inplane_y"]["dist"])

            nx = start_offset + i * pitch + gap
            new_origin = base_o + np.array([nx, dy, 0.0], dtype=float)

            if set_rpy is not None:
                rpy = list(set_rpy)
            else:
                rpy = [0.0, 0.0, 0.0]

            state.set_transform(gid, new_origin, rpy)

    def _welding_distortion(self, step: Dict[str, Any], state: AssemblyState):
        target_info = step.get("target", {})
        model = step.get("model", {})
        inst_ids = self.flow.resolve_selector(target_info.get("selector", ""), self.geom)
        for inst_id in inst_ids:
            tr = state.get_transform(inst_id)
            o  = tr["origin"].copy()
            rpy = tr["rpy_deg"][:]
            dz  = self._sample(model.get("outplane_dz", 0.0))
            d_rx = self._sample(model.get("weak_bending_about_x", 0.0))
            o[2]  += dz
            rpy[0] += d_rx
            state.set_transform(inst_id, o, rpy)

    def _fitup_pair_chain(self, step: Dict[str, Any], state: AssemblyState):
        """
        ★2モード対応★

        [A] butt joint（突き合わせ）2点合わせモード
          - 条件：step.model に dx0_logn, dx1_logn, dy_norm がある
          - chain要素は以下のどちらでもOK：
              1) 新形式:
                 {"base":{"instance":"A1","p0":"points.B","p1":"points.C"},
                  "guest":{"instance":"A2","q0":"points.A","q1":"points.D"}}
              2) 旧形式のまま + refs を別キーで持たせるのは非推奨（混乱するので）

        [B] 従来モード（L+gapで並べる）
          - 条件：上記キーが無い
          - chain要素: {"base":[ "A1" ], "guest":[ "A2" ]} の形式
        """
        model = step.get("model", {})
        has_butt_keys = ("dx0_logn" in model) and ("dx1_logn" in model) and ("dy_norm" in model)

        chain = step.get("chain", [])
        constraints = step.get("constraints", {})

        # =========================================================
        # [A] butt joint モード（あなたの意図）
        # =========================================================
        if has_butt_keys:
            dx0_spec = model["dx0_logn"]   # lognormal（q0用）
            dx1_spec = model["dx1_logn"]   # lognormal（q1用・独立）
            dy_spec  = model["dy_norm"]    # normal（面内直交方向、q0のy）
            signed_logn = bool(model.get("signed_logn", True))
            keep_z = bool(model.get("keep_z", True))  # Trueならzは現状維持

            for pr in chain:
                base = pr.get("base", {})
                guest = pr.get("guest", {})

                base_id = base.get("instance")
                guest_id = guest.get("instance")
                if not base_id or not guest_id:
                    raise ValueError("butt joint chain item requires base.instance and guest.instance")

                base_p0 = base.get("p0", "points.B")
                base_p1 = base.get("p1", "points.C")
                guest_q0 = guest.get("q0", "points.A")
                guest_q1 = guest.get("q1", "points.D")

                # --- base側参照点（world） ---
                P0 = get_world_point(self.geom, state, base_id, base_p0)
                P1 = get_world_point(self.geom, state, base_id, base_p1)

                # --- ジョイント局所軸（world） ---
                u = self._unit(P1 - P0)  # 接合線方向（x相当）
                if float(np.linalg.norm(u)) < 1e-12:
                    raise ValueError(f"butt joint base axis is degenerate: {base_id} {base_p0}->{base_p1}")

                n = self._get_plane_normal_world(base_id, state)
                # v = 面内でuに直交（y相当）。crossが弱い場合に備えてfallbackを用意
                v = self._safe_unit_from_cross(n, u, fallback=np.array([0.0, 1.0, 0.0], dtype=float))

                # --- ばらつきサンプル（独立） ---
                dx0 = self._sample(dx0_spec)
                dx1 = self._sample(dx1_spec)
                if signed_logn:
                    dx0 *= (-1.0 if self.rng.random() < 0.5 else 1.0)
                    dx1 *= (-1.0 if self.rng.random() < 0.5 else 1.0)
                dy = self._sample(dy_spec)

                # =================================================
                # 1) q0 を p0 に合わせる（剛体並進）
                #    q0.x: P0 + dx0*u
                #    q0.y: P0 + dy*v
                # =================================================
                Q0_now = get_world_point(self.geom, state, guest_id, guest_q0)
                if keep_z:
                    Q0_target = (P0 + dx0 * u + dy * v)
                    Q0_target[2] = Q0_now[2]
                else:
                    Q0_target = (P0 + dx0 * u + dy * v)

                t = Q0_target - Q0_now
                gtr = state.get_transform(guest_id)
                state.set_transform(guest_id, gtr["origin"] + t, gtr["rpy_deg"])

                # =================================================
                # 2) q1 の xだけを独立に p1 に合わせる（点オフセット）
                #    q1.y は q0 に完全従属（追加y補正しない）
                # =================================================
                Q1_now = get_world_point(self.geom, state, guest_id, guest_q1)
                # 目標は「u方向成分のみ」：P1 + dx1*u
                # 実際には u 方向投影だけ合わせる
                delta_u = float(np.dot((P1 + dx1 * u) - Q1_now, u))
                delta_world = delta_u * u

                # world補正→guestローカル補正へ
                delta_local = self._world_vec_to_local(guest_id, state, delta_world)

                q1_name = self._parse_point_name(guest_q1)
                state.add_point_offset(guest_id, q1_name, delta_local)

            return  # butt joint で処理完了（従来モードへ行かない）

        # =========================================================
        # [B] 従来モード（L+gap）
        # =========================================================
        for pair in chain:
            base_id = pair["base"][0]
            guest_id = pair["guest"][0]
            base_tr  = state.get_transform(base_id)

            # 実現された寸法を使用
            base_dims = state.get_realized_dims(base_id)
            base_inst = self.geom.get_instance(base_id)
            proto = self.geom.get_prototype(base_inst["prototype"])
            L = base_dims.get("L", float(proto.get("dims", {}).get("L", 2000.0)))

            gap = 0.0
            if "coincident_1D" in constraints:
                gap = self._sample(constraints["coincident_1D"].get("gap_dist", 0.0))

            new_o = base_tr["origin"] + np.array([L + gap, 0.0, 0.0], dtype=float)
            state.set_transform(guest_id, new_o, base_tr["rpy_deg"])



# =============================================================================
# 検証
# =============================================================================
class Validator:
    @staticmethod
    def validate(geom: GeometryModel, flow: FlowModel) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        for inst_id, inst in geom.instances.items():
            pid = inst.get("prototype", "")
            if pid not in geom.prototypes:
                issues.append({"level":"error","message":f"Instance '{inst_id}' references unknown prototype '{pid}'"})
        for step in flow.steps:
            if step.get("op") == "fitup_array_attach":
                sel = step.get("guests", {}).get("selector", "")
                if sel and sel not in flow.selectors:
                    issues.append({"level":"warning","message":f"Step '{step.get('id')}' references unknown selector '{sel}'"})
        return issues


# =============================================================================
# Monte Carlo
# =============================================================================
class MonteCarloSimulator:
    def __init__(self, geom: GeometryModel, flow: FlowModel):
        self.geom = geom
        self.flow = flow

    def run(self, n_trials: int, steps_mask: List[bool], seed: int=42) -> pd.DataFrame:
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
        # edge_gap_x: A1右端とA2左端の x ギャップ
        if "A1" in state.transforms and "A2" in state.transforms:
            a1 = state.get_transform("A1"); a2 = state.get_transform("A2")
            a1_dims = state.get_realized_dims("A1")
            a1_inst = self.geom.get_instance("A1")
            proto = self.geom.get_prototype(a1_inst["prototype"])
            L = a1_dims.get("L", float(proto.get("dims", {}).get("L", 2000.0)))
            
            gap_x = float(a2["origin"][0] - (a1["origin"][0] + L))
            metrics["edge_gap_x"] = gap_x
            metrics["flush_z"] = float(abs(a2["origin"][2] - a1["origin"][2]))

        # 任意計測（measurements）
        for m in self.flow.measurements:
            name = m.get("metric_name", m.get("id", "dist"))
            p1 = m["p1"]; p2 = m["p2"]
            P1 = get_world_point(self.geom, state, p1["instance"], p1["ref"])
            P2 = get_world_point(self.geom, state, p2["instance"], p2["ref"])
            metrics[name] = float(np.linalg.norm(P2 - P1))

        return metrics


# =============================================================================
# 3D 可視化（機能1&2対応: 偏差カラーマップ＋変形倍率）
# =============================================================================
class Open3DVisualizer:
    def __init__(self):
        self.geometries: List[Any] = []

    def build_scene(self, geom: GeometryModel, state: AssemblyState, show_groups: Dict[str, bool],
                    deviation_mode: bool = False, tol_mm: float = 5.0, deform_scale: float = 1.0):
        """
        機能1: deviation_mode=True のとき、偏差に応じた色を付ける
        機能2: deform_scale で変位・回転を拡大表示
        """
        self.geometries = []
        for inst_id, inst in geom.instances.items():
            tags = inst.get("tags", [])
            show = (("A" in tags and show_groups.get("A", True)) or
                    ("C" in tags and show_groups.get("C", True)))
            if not show:
                continue
            proto = geom.get_prototype(inst["prototype"])
            
            # 実現された寸法を使用
            dims = state.get_realized_dims(inst_id)
            if not dims:
                dims = proto.get("dims", {})
            
            L, H, t = float(dims.get("L",1000)), float(dims.get("H",1000)), float(dims.get("t",10))
            
            # 【機能2】変形倍率を適用した描画用 origin/rpy を計算
            nominal_o = np.array(inst["frame"]["origin"], dtype=float)
            current_o = state.get_transform(inst_id)["origin"]
            draw_o = nominal_o + (current_o - nominal_o) * deform_scale
            
            nominal_rpy = np.array(inst["frame"]["rpy_deg"], dtype=float)
            current_rpy = np.array(state.get_transform(inst_id)["rpy_deg"], dtype=float)
            draw_rpy = nominal_rpy + (current_rpy - nominal_rpy) * deform_scale

            box = o3d.geometry.TriangleMesh.create_box(L, H, t)
            box.compute_vertex_normals()
            R = rpy_to_rotation_matrix(*draw_rpy)
            box.rotate(R, center=(0,0,0))
            box.translate(draw_o)

            # 【機能1】偏差カラーマップ
            if deviation_mode:
                deviation = float(np.linalg.norm(current_o - nominal_o))
                color = deviation_color_map(deviation, tol_mm)
                box.paint_uniform_color(color.tolist())
            else:
                # 従来の固定色
                if "A" in tags:
                    box.paint_uniform_color([0.7, 0.8, 1.0])
                elif "C" in tags:
                    box.paint_uniform_color([0.4, 0.6, 0.9])

            self.geometries.append(box)
        self.geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0,0,0]))

    def get_geometries(self) -> List[Any]:
        return self.geometries


class MatplotlibVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8,6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def build_scene(self, geom: GeometryModel, state: AssemblyState, show_groups: Dict[str, bool],
                    deviation_mode: bool = False, tol_mm: float = 5.0, deform_scale: float = 1.0):
        """
        機能1: deviation_mode=True のとき、偏差に応じた色を付ける
        機能2: deform_scale で変位・回転を拡大表示
        """
        self.ax.clear()
        all_x, all_y, all_z = [], [], []
        for inst_id, inst in geom.instances.items():
            tags = inst.get("tags", [])
            show = (("A" in tags and show_groups.get("A", True)) or
                    ("C" in tags and show_groups.get("C", True)))
            if not show:
                continue
            proto = geom.get_prototype(inst["prototype"])
            
            # 実現された寸法を使用
            dims = state.get_realized_dims(inst_id)
            if not dims:
                dims = proto.get("dims", {})
            
            L, H, t = float(dims.get("L",1000)), float(dims.get("H",1000)), float(dims.get("t",10))
            
            # 【機能2】変形倍率を適用した描画用 origin/rpy を計算
            nominal_o = np.array(inst["frame"]["origin"], dtype=float)
            current_o = state.get_transform(inst_id)["origin"]
            draw_o = nominal_o + (current_o - nominal_o) * deform_scale
            
            nominal_rpy = np.array(inst["frame"]["rpy_deg"], dtype=float)
            current_rpy = np.array(state.get_transform(inst_id)["rpy_deg"], dtype=float)
            draw_rpy = nominal_rpy + (current_rpy - nominal_rpy) * deform_scale
            
            R = rpy_to_rotation_matrix(*draw_rpy)
            
            # 【機能1】偏差カラーマップ
            if deviation_mode:
                deviation = float(np.linalg.norm(current_o - nominal_o))
                color_rgb = deviation_color_map(deviation, tol_mm)
                # matplotlib用に色を変換
                color = tuple(color_rgb)
            else:
                color = "lightblue" if "A" in tags else "steelblue"
            
            # 回転を考慮した頂点計算
            verts = self._draw_box_rotated(draw_o, L, H, t, R, color)
            xs, ys, zs = zip(*verts)
            all_x += xs; all_y += ys; all_z += zs

        if all_x and all_y and all_z:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            zmin, zmax = min(all_z), max(all_z)
            span_x, span_y, span_z = xmax-xmin, ymax-ymin, zmax-zmin
            max_span = max(span_x, span_y, span_z, 1.0)
            pad = 0.1 * max_span
            self.ax.set_xlim(xmin-pad, xmax+pad)
            self.ax.set_ylim(ymin-pad, ymax+pad)
            self.ax.set_zlim(zmin-pad, zmax+pad)
            try:
                self.ax.set_box_aspect((span_x or 1.0, span_y or 1.0, span_z or 1.0))
            except Exception:
                cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
                half = 0.5*max_span
                self.ax.set_xlim(cx-half, cx+half)
                self.ax.set_ylim(cy-half, cy+half)
                self.ax.set_zlim(cz-half, cz+half)

        self.ax.set_xlabel("X [mm]"); self.ax.set_ylabel("Y [mm]"); self.ax.set_zlabel("Z [mm]")
        title = "Assembly View"
        if deviation_mode:
            title += f" (偏差モード, 許容公差={tol_mm:.1f}mm)"
        if deform_scale != 1.0:
            title += f" (変形倍率 x{deform_scale:.0f})"
        self.ax.set_title(title)
        self.canvas.draw()

    def _draw_box_rotated(self, origin, L, H, t, R, color):
        """回転を考慮したボックス描画"""
        # ローカル座標系での8頂点
        local_v = np.array([
            [0, 0, 0], [L, 0, 0], [L, H, 0], [0, H, 0],
            [0, 0, t], [L, 0, t], [L, H, t], [0, H, t],
        ])
        
        # 回転と平行移動を適用
        world_v = []
        for lv in local_v:
            wv = R @ lv + origin
            world_v.append(wv)
        
        # エッジを描画
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        for a,b in edges:
            xs = [world_v[a][0], world_v[b][0]]
            ys = [world_v[a][1], world_v[b][1]]
            zs = [world_v[a][2], world_v[b][2]]
            self.ax.plot(xs, ys, zs, color=color, linewidth=1.5)
        
        return world_v


# =============================================================================
# 距離分布可視化ウィジェット（アプリ内埋め込み用）
# =============================================================================
class DistanceHistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_histogram(self, values: np.ndarray, name1: str, name2: str, n_trials: int):
        self.ax.clear()
        self.ax.hist(values, bins=30, alpha=0.85, edgecolor="black", color="steelblue")
        self.ax.set_xlabel("Distance [mm]")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title(f"{name1} ↔ {name2} (N={n_trials})")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


# =============================================================================
# インタラクティブな点選択ウィジェット（修正版）
# =============================================================================
class InteractivePointSelector(QWidget):
    """GUI上で点を選択するためのウィジェット"""
    
    selection_completed = Signal(dict)  # 選択完了時のシグナル
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.geom = None
        self.state = None
        self.selected_points = []  # [(inst_id, ref, world_coords), ...]
        self.all_selectable_points = []  # クリック可能な全ての点
        self.scatter_plot = None  # 散布図オブジェクトの参照を保持
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 説明ラベル
        info_label = QLabel(
            "3Dビュー上の点をクリックして2点を選択してください\n"
            "（点は赤い○で表示されます）"
        )
        layout.addWidget(info_label)
        
        # Matplotlib 3D キャンバス
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas)
        
        # イベント接続
        self.canvas.mpl_connect('pick_event', self._on_pick)
        
        # 選択状態表示
        status_group = QGroupBox("選択状態")
        status_layout = QVBoxLayout(status_group)
        
        self.point1_label = QLabel("点1: 未選択")
        self.point2_label = QLabel("点2: 未選択")
        status_layout.addWidget(self.point1_label)
        status_layout.addWidget(self.point2_label)
        
        layout.addWidget(status_group)
        
        # ボタン
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("選択をリセット")
        self.reset_btn.clicked.connect(self._reset_selection)
        button_layout.addWidget(self.reset_btn)
        
        self.confirm_btn = QPushButton("選択を確定")
        self.confirm_btn.clicked.connect(self._confirm_selection)
        self.confirm_btn.setEnabled(False)
        button_layout.addWidget(self.confirm_btn)
        
        layout.addLayout(button_layout)
        
    def set_geometry_and_state(self, geom, state, show_groups):
        """幾何モデルとアセンブリ状態を設定し、3Dビューを更新"""
        self.geom = geom
        self.state = state
        self.show_groups = show_groups
        self._update_3d_view()
        
    def _update_3d_view(self):
        """3Dビューを描画し、選択可能な点を表示"""
        self.ax.clear()
        self.all_selectable_points = []
        self.scatter_plot = None
        
        if not self.geom or not self.state:
            self.canvas.draw()
            return
        
        all_x, all_y, all_z = [], [], []
        
        # インスタンスの描画
        for inst_id, inst in self.geom.instances.items():
            tags = inst.get("tags", [])
            show = (("A" in tags and self.show_groups.get("A", True)) or
                    ("C" in tags and self.show_groups.get("C", True)))
            if not show:
                continue
                
            proto = self.geom.get_prototype(inst["prototype"])
            dims = self.state.get_realized_dims(inst_id)
            if not dims:
                dims = proto.get("dims", {})
            
            L = float(dims.get("L", 1000))
            H = float(dims.get("H", 1000))
            t = float(dims.get("t", 10))
            
            tr = self.state.get_transform(inst_id)
            R = rpy_to_rotation_matrix(*tr["rpy_deg"])
            
            # ボックスの描画（回転考慮）
            verts = self._draw_box_rotated(tr["origin"], L, H, t, R, "A" in tags)
            xs, ys, zs = zip(*verts)
            all_x += xs
            all_y += ys
            all_z += zs
            
            # 選択可能な点を追加
            self._add_selectable_points(inst_id, proto)
        
        # 選択可能な点を描画
        if self.all_selectable_points:
            pts_x = [p['coords'][0] for p in self.all_selectable_points]
            pts_y = [p['coords'][1] for p in self.all_selectable_points]
            pts_z = [p['coords'][2] for p in self.all_selectable_points]
            
            self.scatter_plot = self.ax.scatter(
                pts_x, pts_y, pts_z, 
                c='red', s=150, marker='o',
                alpha=0.7, 
                picker=True,
                depthshade=True
            )
        
        # 選択済みの点を強調表示
        for i, (inst_id, ref, coords) in enumerate(self.selected_points):
            color = 'green' if i == 0 else 'blue'
            self.ax.scatter([coords[0]], [coords[1]], [coords[2]], 
                          c=color, s=300, marker='*', 
                          edgecolors='black', linewidths=2,
                          depthshade=False)
        
        # 軸の設定
        if all_x and all_y and all_z:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            zmin, zmax = min(all_z), max(all_z)
            span_x = xmax - xmin or 1.0
            span_y = ymax - ymin or 1.0
            span_z = zmax - zmin or 1.0
            max_span = max(span_x, span_y, span_z)
            pad = 0.1 * max_span
            
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)
            self.ax.set_zlim(zmin - pad, zmax + pad)
        
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Z [mm]")
        self.ax.set_title("点を選択してください（赤○をクリック）")
        
        self.canvas.draw()
    
    def _draw_box_rotated(self, origin, L, H, t, R, is_A_group):
        """回転を考慮したボックスを描画"""
        color = 'lightblue' if is_A_group else 'steelblue'
        
        # ローカル座標系での8頂点
        local_v = np.array([
            [0, 0, 0], [L, 0, 0], [L, H, 0], [0, H, 0],
            [0, 0, t], [L, 0, t], [L, H, t], [0, H, t],
        ])
        
        # 回転と平行移動を適用
        world_v = []
        for lv in local_v:
            wv = R @ lv + origin
            world_v.append(wv)
        
        # エッジを描画
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        for a, b in edges:
            xs = [world_v[a][0], world_v[b][0]]
            ys = [world_v[a][1], world_v[b][1]]
            zs = [world_v[a][2], world_v[b][2]]
            self.ax.plot(xs, ys, zs, color=color, linewidth=1.5)
        
        return world_v
    
    def _add_selectable_points(self, inst_id, proto):
        """選択可能な点を登録"""
        refs = self.geom.get_available_refs_for_instance(inst_id)
        
        for ref in refs:
            try:
                # ワールド座標を計算
                coords = get_world_point(self.geom, self.state, inst_id, ref)
                
                self.all_selectable_points.append({
                    'inst_id': inst_id,
                    'ref': ref,
                    'coords': coords,
                    'label': f"{inst_id}:{ref}"
                })
            except Exception as e:
                print(f"Warning: Could not compute point {inst_id}:{ref}: {e}")
    
    def _on_pick(self, event):
        """ピックイベント処理"""
        if len(self.selected_points) >= 2:
            return
        
        if event.artist != self.scatter_plot:
            return
        
        ind = event.ind
        if len(ind) == 0:
            return
        
        point_index = ind[0]
        
        if point_index >= len(self.all_selectable_points):
            return
        
        closest_point = self.all_selectable_points[point_index]
        
        # 重複選択を防止
        for selected_inst, selected_ref, _ in self.selected_points:
            if (selected_inst == closest_point['inst_id'] and 
                selected_ref == closest_point['ref']):
                return
        
        self.selected_points.append((
            closest_point['inst_id'],
            closest_point['ref'],
            closest_point['coords']
        ))
        self._update_selection_labels()
        self._update_3d_view()
    
    def _update_selection_labels(self):
        """選択状態ラベルを更新"""
        if len(self.selected_points) >= 1:
            inst_id, ref, coords = self.selected_points[0]
            self.point1_label.setText(
                f"点1: {inst_id}:{ref}\n"
                f"  座標: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
            )
        else:
            self.point1_label.setText("点1: 未選択")
        
        if len(self.selected_points) >= 2:
            inst_id, ref, coords = self.selected_points[1]
            self.point2_label.setText(
                f"点2: {inst_id}:{ref}\n"
                f"  座標: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
            )
            self.confirm_btn.setEnabled(True)
        else:
            self.point2_label.setText("点2: 未選択")
            self.confirm_btn.setEnabled(False)
    
    def _reset_selection(self):
        """選択をリセット"""
        self.selected_points = []
        self._update_selection_labels()
        self._update_3d_view()
    
    def _confirm_selection(self):
        """選択を確定してシグナルを発行"""
        if len(self.selected_points) != 2:
            return
        
        result = {
            'point1': {
                'instance': self.selected_points[0][0],
                'ref': self.selected_points[0][1],
                'coords': self.selected_points[0][2]
            },
            'point2': {
                'instance': self.selected_points[1][0],
                'ref': self.selected_points[1][1],
                'coords': self.selected_points[1][2]
            }
        }
        
        self.selection_completed.emit(result)


# =============================================================================
# ファイル監視
# =============================================================================
if HAS_WATCHDOG:
    class FileChangeHandler(FileSystemEventHandler):
        def __init__(self, callback):
            super().__init__()
            self.callback = callback
            self.last = {}

        def on_modified(self, event):
            if event.is_directory: return
            now = time.time()
            if event.src_path in self.last and (now - self.last[event.src_path] < 0.8):
                return
            self.last[event.src_path] = now
            if event.src_path.endswith((".json", ".csv")):
                self.callback()

def print_all_edge_stds_after_cutting(csv_path: Path, n_trials: int = 5000, seed: int = 42) -> None:
    """
    model_onefile.csv を読み、
    step id="10_cutting" を適用した後の「全edge長さ」の標準偏差をコンソール出力する。

    edge長さ = endpoints 2点（points.*）のワールド距離
    標準偏差 = ddof=1（標本標準偏差）
    """
    geom_dict, flow_dict = load_data_from_csv(csv_path)
    geom = GeometryModel(geom_dict)
    flow = FlowModel(flow_dict)

    # 10_cutting を探す
    cutting_step = None
    for s in flow.steps:
        if s.get("id") == "10_cutting":
            cutting_step = s
            break
    if cutting_step is None:
        raise RuntimeError('Step id="10_cutting" not found in flow.steps')

    # edge定義（prototype.features.edges）を全インスタンス分集める
    edge_defs = []  # (inst_id, edge_name, ep0, ep1)
    for inst_id, inst in geom.instances.items():
        proto = geom.get_prototype(inst.get("prototype", ""))
        edges = proto.get("features", {}).get("edges", {})
        for edge_name, e in edges.items():
            endpoints = e.get("endpoints", [])
            if len(endpoints) >= 2:
                edge_defs.append((inst_id, edge_name, endpoints[0], endpoints[1]))

    if not edge_defs:
        print("[EDGE STD] No edges found in prototypes.features.edges")
        return

    # Monte Carlo
    values = { (inst_id, edge_name): [] for (inst_id, edge_name, _, _) in edge_defs }

    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        state = AssemblyState(geom)
        engine = ProcessEngine(geom, flow, rng)

        # 10_cutting だけ適用
        engine.apply_step(cutting_step, state)

        # 全edge長さを計算
        for inst_id, edge_name, ep0, ep1 in edge_defs:
            p0 = get_world_point(geom, state, inst_id, f"points.{ep0}")
            p1 = get_world_point(geom, state, inst_id, f"points.{ep1}")
            L = float(np.linalg.norm(p1 - p0))
            values[(inst_id, edge_name)].append(L)

    # 出力
    print(f"[EDGE STD] After 10_cutting  (n_trials={n_trials}, seed={seed})")
    for (inst_id, edge_name) in sorted(values.keys()):
        arr = np.asarray(values[(inst_id, edge_name)], dtype=float)
        std = float(arr.std(ddof=1))
        mean = float(arr.mean())
        print(f"  {inst_id}:{edge_name:<6}  mean={mean:10.6f} mm   std={std:10.6f} mm")

# =============================================================================
# メインウィンドウ（CSVファイル対応版 - flow.jsonもCSVから読み込み）
# =============================================================================

# MainWindow moved to src/sov_app/main_window.py



