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
# CSV → JSON 変換ユーティリティ
# =============================================================================
def csv_to_nested_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    CSVの doc, path, value_json 形式から階層的なJSON辞書を再構築
    
    例:
      doc='geometry', path='/units', value_json='"mm"'
      → result['units'] = 'mm'
    """
    result = {}
    
    for _, row in df.iterrows():
        doc = row['doc']
        path = row['path']
        value_json = row['value_json']
        
        # JSONパース
        try:
            value = json.loads(value_json)
        except:
            value = value_json  # パース失敗時は文字列のまま
        
        # パスを分解（先頭の / を除去）
        path = path.lstrip('/')
        keys = path.split('/')
        
        # 階層を辿って値を設定
        current = result
        for i, key in enumerate(keys[:-1]):
            # 配列インデックスか辞書キーか判定
            if key.isdigit():
                key = int(key)
                # 親がリストでない場合は初期化
                if not isinstance(current, list):
                    parent_key = keys[i-1] if i > 0 else None
                    if parent_key and parent_key in current:
                        current[parent_key] = []
                        current = current[parent_key]
                # リストを必要なサイズに拡張
                while len(current) <= key:
                    current.append({})
                current = current[key]
            else:
                if key not in current:
                    # 次のキーが数字ならリスト、そうでなければ辞書
                    next_key = keys[i+1] if i+1 < len(keys) else None
                    if next_key and next_key.isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
        
        # 最後のキーに値を設定
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            if not isinstance(current, list):
                current = []
            while len(current) <= final_key:
                current.append(None)
            current[final_key] = value
        else:
            current[final_key] = value
    
    return result


def load_data_from_csv(csv_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    model_onefile.csv から geometry と flow の両方のデータを読み込む
    （区切り: カンマ or タブ、ヘッダ有無も吸収）
    """
    # 1) 先頭行で区切り文字をざっくり判定
    first_line = ""
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()

    # タブが多いならTSV扱い、そうでなければCSV扱い
    sep = "\t" if (first_line.count("\t") > first_line.count(",")) else ","

    # 2) まずは「ヘッダあり」で読む
    df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding="utf-8-sig", engine="python")

    # 3) 列名の正規化（前後空白除去）
    df.columns = [str(c).strip() for c in df.columns]

    # 4) doc列が無いなら「ヘッダ無し3列」扱いで読み直す
    if "doc" not in df.columns:
        df = pd.read_csv(
            csv_path,
            sep=sep,
            header=None,
            names=["doc", "path", "value_json"],
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
        )

    # 5) 念のため必要列チェック
    required = {"doc", "path", "value_json"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV format error: required columns {required}, got {list(df.columns)}")

    # geometry / flow を抽出
    geom_df = df[df["doc"] == "geometry"].copy()
    geom_dict = csv_to_nested_dict(geom_df) if not geom_df.empty else {}

    flow_df = df[df["doc"] == "flow"].copy()
    flow_dict = csv_to_nested_dict(flow_df) if not flow_df.empty else {}

    return geom_dict, flow_dict



def nested_dict_to_csv_rows(data: Dict[str, Any], doc_name: str, parent_path: str = "") -> List[Dict[str, str]]:
    """
    階層的な辞書をCSV行形式に変換
    
    Args:
        data: 変換する辞書
        doc_name: ドキュメント名 ('geometry' or 'flow')
        parent_path: 親パス（再帰用）
    
    Returns:
        CSV行のリスト [{doc, path, value_json}, ...]
    """
    rows = []
    
    def flatten(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}/{key}"
                flatten(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}/{i}"
                flatten(item, new_path)
        else:
            rows.append({
                'doc': doc_name,
                'path': path,
                'value_json': json.dumps(obj, ensure_ascii=False)
            })
    
    flatten(data, parent_path)
    return rows


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
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("幾何・工程統合可視化アプリ（完全CSV対応版）")
        self.setGeometry(100, 80, 1700, 900)

        self.geom: Optional[GeometryModel] = None
        self.flow: Optional[FlowModel] = None
        self.state: Optional[AssemblyState] = None

        # main() で設定したCSVパスを読む（ここではハードコードしない）
        global MODEL_ONEFILE_CSV_PATH
        if MODEL_ONEFILE_CSV_PATH is None:
            raise RuntimeError("MODEL_ONEFILE_CSV_PATH is not set. main() でCSVパスを設定してください。")
        self.data_path = MODEL_ONEFILE_CSV_PATH

        self.steps_mask: List[bool] = []
        self.step_checkboxes: List[QCheckBox] = []

        self.visual_selected_points = None  # GUI選択した点の情報

        # Monte Carlo結果保存用
        self.mc_results: Optional[pd.DataFrame] = None

        self._setup_ui()
        self._setup_file_watcher()
        self.reload_all()

    # ---------- UI ----------
    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 800, 500])
        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        file_grp = QGroupBox("ファイル操作"); file_l = QVBoxLayout(file_grp)
        btn_reload = QPushButton("再読み込み"); btn_reload.clicked.connect(self.reload_all); file_l.addWidget(btn_reload)
        btn_save_data = QPushButton("model_onefile.csv 保存"); btn_save_data.clicked.connect(self.save_data); file_l.addWidget(btn_save_data)
        lay.addWidget(file_grp)

        self.editor_tabs = QTabWidget()
        self.geom_editor = QTextEdit(); self.geom_editor.setFont(QFont("Courier", 9))
        self.flow_editor = QTextEdit(); self.flow_editor.setFont(QFont("Courier", 9))
        self.editor_tabs.addTab(self.geom_editor, "geometry (JSON)")
        self.editor_tabs.addTab(self.flow_editor, "flow (JSON)")
        lay.addWidget(self.editor_tabs)

        tree_grp = QGroupBox("インスタンス一覧"); tree_l = QVBoxLayout(tree_grp)
        self.instance_tree = QTreeWidget(); self.instance_tree.setHeaderLabels(["ID", "Prototype", "Tags"])
        tree_l.addWidget(self.instance_tree); lay.addWidget(tree_grp)
        return w

    def _create_center_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        # ステップ制御（flow.steps から動的に生成）
        step_grp = QGroupBox("工程ステップ"); step_l = QVBoxLayout(step_grp)
        self.step_box_container = QWidget(); self.step_box_layout = QVBoxLayout(self.step_box_container)
        step_l.addWidget(self.step_box_container)
        btn_update = QPushButton("ビュー更新"); btn_update.clicked.connect(self.update_view); step_l.addWidget(btn_update)
        lay.addWidget(step_grp)

        # 【機能1&2】表示オプション拡張
        disp_grp = QGroupBox("表示オプション"); disp_l = QVBoxLayout(disp_grp)
        
        # グループ表示
        self.show_a_cb = QCheckBox("A群表示"); self.show_a_cb.setChecked(True); self.show_a_cb.stateChanged.connect(self.update_view); disp_l.addWidget(self.show_a_cb)
        self.show_c_cb = QCheckBox("C群表示"); self.show_c_cb.setChecked(True); self.show_c_cb.stateChanged.connect(self.update_view); disp_l.addWidget(self.show_c_cb)
        
        # 【機能1】偏差表示モード
        self.deviation_mode_cb = QCheckBox("偏差表示モード")
        self.deviation_mode_cb.setChecked(False)
        self.deviation_mode_cb.stateChanged.connect(self.update_view)
        disp_l.addWidget(self.deviation_mode_cb)
        
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("許容公差 [mm]:"))
        self.tol_edit = QDoubleSpinBox()
        self.tol_edit.setRange(0.1, 100.0)
        self.tol_edit.setValue(5.0)
        self.tol_edit.setSingleStep(0.5)
        self.tol_edit.editingFinished.connect(self.update_view)
        tol_layout.addWidget(self.tol_edit)
        disp_l.addLayout(tol_layout)
        
        # 【機能2】変形倍率スライダー
        deform_layout = QVBoxLayout()
        deform_label_layout = QHBoxLayout()
        deform_label_layout.addWidget(QLabel("変形倍率:"))
        self.deform_scale_label = QLabel("x1")
        deform_label_layout.addWidget(self.deform_scale_label)
        deform_label_layout.addStretch()
        deform_layout.addLayout(deform_label_layout)
        
        self.deform_scale_slider = QSlider(Qt.Horizontal)
        self.deform_scale_slider.setRange(1, 100)
        self.deform_scale_slider.setValue(1)
        self.deform_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.deform_scale_slider.setTickInterval(10)
        self.deform_scale_slider.valueChanged.connect(self._on_deform_scale_changed)
        deform_layout.addWidget(self.deform_scale_slider)
        
        disp_l.addLayout(deform_layout)
        
        lay.addWidget(disp_grp)

        if USE_OPEN3D:
            self.view_label = QLabel("Open3D ウィンドウで表示します（ボタン押下で更新）")
            self.view_label.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.view_label)
            self.visualizer = Open3DVisualizer()
        else:
            self.visualizer = MatplotlibVisualizer()
            lay.addWidget(self.visualizer)

        return w

    def _create_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # スクロール対応
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 検証ログ
        log_grp = QGroupBox("検証"); log_l = QVBoxLayout(log_grp)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(150); log_l.addWidget(self.log_text)
        btn_validate = QPushButton("検証実行"); btn_validate.clicked.connect(self.run_validation); log_l.addWidget(btn_validate)
        scroll_layout.addWidget(log_grp)

        # 【機能3】Monte Carlo（ワースト/ベストボタン追加）
        mc_grp = QGroupBox("Monte Carlo"); mc_l = QVBoxLayout(mc_grp)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("試行回数:")); self.mc_n_spin = QSpinBox(); self.mc_n_spin.setRange(10, 20000); self.mc_n_spin.setValue(200); row1.addWidget(self.mc_n_spin); mc_l.addLayout(row1)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("乱数シード:")); self.mc_seed_spin = QSpinBox(); self.mc_seed_spin.setRange(0, 999999); self.mc_seed_spin.setValue(42); row2.addWidget(self.mc_seed_spin); mc_l.addLayout(row2)
        btn_mc = QPushButton("Monte Carlo 実行"); btn_mc.clicked.connect(self.run_monte_carlo); mc_l.addWidget(btn_mc)
        btn_save = QPushButton("結果保存 (CSV/PNG)"); btn_save.clicked.connect(self.save_monte_carlo_results); mc_l.addWidget(btn_save)
        
        # 【機能3】ワースト/ベストケース再現ボタン
        case_layout = QHBoxLayout()
        self.btn_show_worst = QPushButton("Show Worst Case")
        self.btn_show_worst.clicked.connect(self.show_worst_case)
        case_layout.addWidget(self.btn_show_worst)
        
        self.btn_show_best = QPushButton("Show Best Case")
        self.btn_show_best.clicked.connect(self.show_best_case)
        case_layout.addWidget(self.btn_show_best)
        
        mc_l.addLayout(case_layout)
        
        self.mc_result_text = QTextEdit(); self.mc_result_text.setReadOnly(True); self.mc_result_text.setMaximumHeight(150); mc_l.addWidget(self.mc_result_text)
        scroll_layout.addWidget(mc_grp)

        # ========== 任意2点距離（モード選択付き） ==========
        pair_grp = QGroupBox("任意2点の距離（MC）"); pair_l = QVBoxLayout(pair_grp)
        
        # モード選択
        mode_layout = QHBoxLayout()
        self.point_select_mode_group = QButtonGroup()
        
        self.dropdown_mode_radio = QRadioButton("プルダウン選択")
        self.dropdown_mode_radio.setChecked(True)
        self.dropdown_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        self.point_select_mode_group.addButton(self.dropdown_mode_radio)
        mode_layout.addWidget(self.dropdown_mode_radio)
        
        self.visual_mode_radio = QRadioButton("GUI上で選択")
        self.visual_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        self.point_select_mode_group.addButton(self.visual_mode_radio)
        mode_layout.addWidget(self.visual_mode_radio)
        
        pair_l.addLayout(mode_layout)
        
        # プルダウン選択用ウィジェット
        self.dropdown_widget = QWidget()
        dropdown_layout = QVBoxLayout(self.dropdown_widget)
        
        dropdown_layout.addWidget(QLabel("名前1:"))
        self.instref_name1 = QLineEdit("instref1")
        dropdown_layout.addWidget(self.instref_name1)
        
        dropdown_layout.addWidget(QLabel("点1 - インスタンス:"))
        self.p1_inst_combo = QComboBox()
        self.p1_inst_combo.currentIndexChanged.connect(self._update_p1_ref_combo)
        dropdown_layout.addWidget(self.p1_inst_combo)
        
        dropdown_layout.addWidget(QLabel("点1 - 参照:"))
        self.p1_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p1_ref_combo)
        
        dropdown_layout.addWidget(QLabel("名前2:"))
        self.instref_name2 = QLineEdit("instref2")
        dropdown_layout.addWidget(self.instref_name2)
        
        dropdown_layout.addWidget(QLabel("点2 - インスタンス:"))
        self.p2_inst_combo = QComboBox()
        self.p2_inst_combo.currentIndexChanged.connect(self._update_p2_ref_combo)
        dropdown_layout.addWidget(self.p2_inst_combo)
        
        dropdown_layout.addWidget(QLabel("点2 - 参照:"))
        self.p2_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p2_ref_combo)
        
        pair_l.addWidget(self.dropdown_widget)
        
        # GUI選択用ウィジェット
        self.visual_select_widget = QWidget()
        visual_layout = QVBoxLayout(self.visual_select_widget)
        
        self.open_visual_selector_btn = QPushButton("3Dビューで点を選択")
        self.open_visual_selector_btn.clicked.connect(self._open_visual_point_selector)
        visual_layout.addWidget(self.open_visual_selector_btn)
        
        self.selected_points_label = QLabel("選択された点: なし")
        self.selected_points_label.setWordWrap(True)
        visual_layout.addWidget(self.selected_points_label)
        
        self.visual_select_widget.setVisible(False)
        pair_l.addWidget(self.visual_select_widget)

        btn_pair = QPushButton("距離分布を表示")
        btn_pair.clicked.connect(self.run_pair_distance_mc)
        pair_l.addWidget(btn_pair)

        # ヒストグラム埋め込みウィジェット
        self.distance_histogram = DistanceHistogramWidget()
        pair_l.addWidget(self.distance_histogram)

        # 統計情報テキスト
        self.pair_result_text = QTextEdit()
        self.pair_result_text.setReadOnly(True)
        self.pair_result_text.setMaximumHeight(100)
        pair_l.addWidget(self.pair_result_text)

        scroll_layout.addWidget(pair_grp)

        scroll.setWidget(scroll_content)
        lay.addWidget(scroll)

        return w

    # ---------- 【機能2】変形倍率スライダー ----------
    def _on_deform_scale_changed(self):
        """変形倍率スライダー変更時"""
        scale = self.deform_scale_slider.value()
        self.deform_scale_label.setText(f"x{scale}")
        self.update_view()

    # ---------- モード切り替え ----------
    def _toggle_point_selection_mode(self):
        """選択モードの切り替え"""
        is_visual = self.visual_mode_radio.isChecked()
        self.dropdown_widget.setVisible(not is_visual)
        self.visual_select_widget.setVisible(is_visual)
    
    def _open_visual_point_selector(self):
        """ビジュアル点選択ダイアログを開く"""
        if not (self.geom and self.state):
            self.log_message("データ未ロード", "error")
            return
        
        # 現在のステップを適用
        self.apply_steps()
        
        # 選択ダイアログを作成
        dialog = QDialog(self)
        dialog.setWindowTitle("点を選択")
        dialog.setGeometry(200, 100, 1000, 800)
        
        layout = QVBoxLayout(dialog)
        
        selector = InteractivePointSelector()
        selector.set_geometry_and_state(
            self.geom, 
            self.state, 
            {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
        )
        
        def on_selection_completed(result):
            self.visual_selected_points = result
            self.selected_points_label.setText(
                f"点1: {result['point1']['instance']}:{result['point1']['ref']}\n"
                f"点2: {result['point2']['instance']}:{result['point2']['ref']}"
            )
            dialog.accept()
        
        selector.selection_completed.connect(on_selection_completed)
        layout.addWidget(selector)
        
        dialog.exec()

    # ---------- プルダウン更新 ----------
    def _update_p1_ref_combo(self):
        self.p1_ref_combo.clear()
        if not self.geom:
            return
        inst_id = self.p1_inst_combo.currentText()
        if inst_id:
            refs = self.geom.get_available_refs_for_instance(inst_id)
            self.p1_ref_combo.addItems(refs)

    def _update_p2_ref_combo(self):
        self.p2_ref_combo.clear()
        if not self.geom:
            return
        inst_id = self.p2_inst_combo.currentText()
        if inst_id:
            refs = self.geom.get_available_refs_for_instance(inst_id)
            self.p2_ref_combo.addItems(refs)

    def _rebuild_instance_combos(self):
        """geometry 読み込み後、インスタンスリストをプルダウンに反映"""
        self.p1_inst_combo.clear()
        self.p2_inst_combo.clear()
        if not self.geom:
            return
        inst_ids = self.geom.get_instance_ids()
        self.p1_inst_combo.addItems(inst_ids)
        self.p2_inst_combo.addItems(inst_ids)

    # ---------- File watcher ----------
    def _setup_file_watcher(self):
        if not HAS_WATCHDOG: return
        self.observer = Observer()
        self.observer.schedule(FileChangeHandler(self.on_file_changed), str(self.data_path.parent), recursive=False)
        self.observer.start()

    def on_file_changed(self):
        QTimer.singleShot(400, self.reload_all)

    # ---------- Load/Save ----------
    def reload_all(self):
        try:
            # ★存在しない場合は必ずエラーにする
            if not self.data_path.exists():
                raise FileNotFoundError(f"model_onefile.csv not found: {self.data_path}")
    
            # CSVから両方のデータを読み込み
            geom_data, flow_data = load_data_from_csv(self.data_path)
    
            if geom_data:
                self.geom = GeometryModel(geom_data)
                # エディタにはJSON形式で表示
                self.geom_editor.setPlainText(json.dumps(geom_data, indent=2, ensure_ascii=False))
    
            if flow_data:
                self.flow = FlowModel(flow_data)
                self.flow_editor.setPlainText(json.dumps(flow_data, indent=2, ensure_ascii=False))
    
            if self.flow:
                self._rebuild_step_checkboxes()
            if self.geom:
                self.state = AssemblyState(self.geom)
                self._rebuild_instance_combos()
    
            self.update_instance_tree()
            self.update_view()
    
            # ★成功したときだけ「完了」を出す
            self.log_message(f"ファイル読み込み完了: {self.data_path}", "info")
    
        except Exception as e:
            self.log_message(f"読み込みエラー: {e}\n{traceback.format_exc()}", "error")
            # もし「必ず落としたい」なら、次の1行を有効化
            # raise

    def _rebuild_step_checkboxes(self):
        for i in reversed(range(self.step_box_layout.count())):
            item = self.step_box_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.step_checkboxes.clear()
        self.steps_mask = []
        for step in (self.flow.steps if self.flow else []):
            cb = QCheckBox(step.get("id", step.get("op","step")))
            cb.setChecked(False)
            cb.stateChanged.connect(self.update_steps_mask)
            self.step_box_layout.addWidget(cb)
            self.step_checkboxes.append(cb)
            self.steps_mask.append(False)

    def save_data(self):
        """
        エディタの内容をCSVファイルとして保存
        geometryとflowの両方を1つのCSVに保存
        """
        try:
            # エディタから取得したJSONデータ
            geom_data = json.loads(self.geom_editor.toPlainText())
            flow_data = json.loads(self.flow_editor.toPlainText())
            
            # JSON → CSV行に変換
            geom_rows = nested_dict_to_csv_rows(geom_data, 'geometry')
            flow_rows = nested_dict_to_csv_rows(flow_data, 'flow')
            
            # 両方を結合
            all_rows = geom_rows + flow_rows
            
            # DataFrameに変換して保存
            df = pd.DataFrame(all_rows)
            df.to_csv(self.data_path, index=False)
            
            self.log_message("model_onefile.csv 保存完了", "info")
            self.reload_all()
        except Exception as e:
            self.log_message(f"保存エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- View / Steps ----------
    def update_instance_tree(self):
        self.instance_tree.clear()
        if not self.geom: return
        for inst_id, inst in self.geom.instances.items():
            proto = inst.get("prototype",""); tags = ", ".join(inst.get("tags",[]))
            self.instance_tree.addTopLevelItem(QTreeWidgetItem([inst_id, proto, tags]))

    def update_steps_mask(self):
        self.steps_mask = [cb.isChecked() for cb in self.step_checkboxes]

    def apply_steps(self):
        if not (self.geom and self.flow): return
        self.state = AssemblyState(self.geom)
        rng = np.random.default_rng(42)
        engine = ProcessEngine(self.geom, self.flow, rng)
        for i, st in enumerate(self.flow.steps):
            if i < len(self.steps_mask) and self.steps_mask[i]:
                engine.apply_step(st, self.state)

    def update_view(self):
        """【機能1&2】偏差表示モード＋変形倍率を反映してビュー更新"""
        if not (self.geom and self.state): return
        try:
            self.apply_steps()
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            
            # 【機能1】偏差表示モード
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            
            # 【機能2】変形倍率
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, self.state, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(self.visualizer.get_geometries(),
                                                  window_name="Assembly View", width=900, height=640)
        except Exception as e:
            self.log_message(f"ビュー更新エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- Validation ----------
    def run_validation(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error")
            return
        issues = Validator.validate(self.geom, self.flow)
        if not issues:
            self.log_message("検証OK: 問題なし", "info")
        else:
            msg = "検証結果:\n" + "\n".join([f"[{i['level'].upper()}] {i['message']}" for i in issues])
            self.log_message(msg, "warning")

    # ---------- Monte Carlo ----------
    def run_monte_carlo(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error"); return
        try:
            n = self.mc_n_spin.value(); seed = self.mc_seed_spin.value()
            self.log_message(f"Monte Carlo 実行中… (N={n})", "info"); QApplication.processEvents()
            sim = MonteCarloSimulator(self.geom, self.flow)
            self.mc_results = sim.run(n, self.steps_mask, seed)
            lines = [f"Monte Carlo 結果 (N={n})"]
            for col in [c for c in self.mc_results.columns if c != "trial"][:6]:
                s = self.mc_results[col]
                lines.append(f"- {col}: mean={s.mean():.3f}, std={s.std():.3f}, 95%CI=[{s.quantile(0.025):.3f}, {s.quantile(0.975):.3f}]")
            self.mc_result_text.setPlainText("\n".join(lines))
            self.log_message("Monte Carlo 完了", "info")
        except Exception as e:
            self.log_message(f"Monte Carlo エラー: {e}\n{traceback.format_exc()}", "error")

    def save_monte_carlo_results(self):
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("実行結果がありません", "warning"); return
        try:
            csv_path = "monte_carlo_results.csv"
            self.mc_results.to_csv(csv_path, index=False)
            import matplotlib.pyplot as plt
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if cols:
                col = "edge_gap_x" if "edge_gap_x" in cols else cols[0]
                plt.figure(figsize=(8,6))
                plt.hist(self.mc_results[col], bins=30, alpha=0.8, edgecolor="black")
                plt.xlabel(col + " [mm]"); plt.ylabel("Freq"); plt.grid(True, alpha=0.3)
                plt.title("Monte Carlo Histogram")
                png_path = "monte_carlo_histogram.png"
                plt.savefig(png_path, dpi=150); plt.close()
                self.log_message(f"保存完了: {csv_path}, {png_path}", "info")
            else:
                self.log_message(f"保存完了: {csv_path}（可視化対象なし）", "info")
        except Exception as e:
            self.log_message(f"保存エラー: {e}", "error")

    # ---------- 【機能3】ワースト/ベストケース再現 ----------
    def build_state_for_trial(self, trial: int, seed_base: int) -> AssemblyState:
        """
        指定された試行番号のアセンブリ状態を再構築
        """
        rng = np.random.default_rng(seed_base + trial)
        state = AssemblyState(self.geom)
        engine = ProcessEngine(self.geom, self.flow, rng)
        for i, st in enumerate(self.flow.steps):
            if i < len(self.steps_mask) and self.steps_mask[i]:
                engine.apply_step(st, state)
        return state
    
    def show_worst_case(self):
        """ワーストケース（最大edge_gap_x）を表示"""
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("Monte Carlo結果がありません", "warning")
            return
        
        try:
            # 対象メトリクスを決定
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if "edge_gap_x" in cols:
                metric_col = "edge_gap_x"
            elif len(cols) > 0:
                metric_col = cols[0]
            else:
                self.log_message("計測メトリクスがありません", "warning")
                return
            
            # 最大値の試行を特定
            worst_trial = int(self.mc_results[metric_col].idxmax())
            worst_value = float(self.mc_results.loc[worst_trial, metric_col])
            
            # 状態を再構築
            seed_base = self.mc_seed_spin.value()
            state_worst = self.build_state_for_trial(worst_trial, seed_base)
            
            # 3Dビューを更新
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, state_worst, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(
                    self.visualizer.get_geometries(),
                    window_name=f"Worst Case (trial={worst_trial}, {metric_col}={worst_value:.3f}mm)",
                    width=900, height=640
                )
            
            self.log_message(
                f"Worst Case表示: trial={worst_trial}, {metric_col}={worst_value:.3f} mm",
                "info"
            )
            
        except Exception as e:
            self.log_message(f"Worst Case表示エラー: {e}\n{traceback.format_exc()}", "error")
    
    def show_best_case(self):
        """ベストケース（最小edge_gap_x）を表示"""
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("Monte Carlo結果がありません", "warning")
            return
        
        try:
            # 対象メトリクスを決定
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if "edge_gap_x" in cols:
                metric_col = "edge_gap_x"
            elif len(cols) > 0:
                metric_col = cols[0]
            else:
                self.log_message("計測メトリクスがありません", "warning")
                return
            
            # 最小値の試行を特定
            best_trial = int(self.mc_results[metric_col].idxmin())
            best_value = float(self.mc_results.loc[best_trial, metric_col])
            
            # 状態を再構築
            seed_base = self.mc_seed_spin.value()
            state_best = self.build_state_for_trial(best_trial, seed_base)
            
            # 3Dビューを更新
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, state_best, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(
                    self.visualizer.get_geometries(),
                    window_name=f"Best Case (trial={best_trial}, {metric_col}={best_value:.3f}mm)",
                    width=900, height=640
                )
            
            self.log_message(
                f"Best Case表示: trial={best_trial}, {metric_col}={best_value:.3f} mm",
                "info"
            )
            
        except Exception as e:
            self.log_message(f"Best Case表示エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- 任意2点距離（両モード対応） ----------
    def run_pair_distance_mc(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error"); return
        try:
            # モードに応じて点情報を取得
            if self.visual_mode_radio.isChecked():
                if not self.visual_selected_points:
                    self.log_message("まず3Dビューで点を選択してください", "warning")
                    return
                
                i1 = self.visual_selected_points['point1']['instance']
                r1 = self.visual_selected_points['point1']['ref']
                i2 = self.visual_selected_points['point2']['instance']
                r2 = self.visual_selected_points['point2']['ref']
                name1 = f"{i1}:{r1}"
                name2 = f"{i2}:{r2}"
            else:
                # プルダウンモード
                i1 = self.p1_inst_combo.currentText()
                r1 = self.p1_ref_combo.currentText()
                i2 = self.p2_inst_combo.currentText()
                r2 = self.p2_ref_combo.currentText()
                
                if not (i1 and r1 and i2 and r2):
                    self.log_message("インスタンスと参照を選択してください", "warning")
                    return
                
                name1 = self.instref_name1.text() or f"{i1}:{r1}"
                name2 = self.instref_name2.text() or f"{i2}:{r2}"

            n = self.mc_n_spin.value(); seed = self.mc_seed_spin.value()
            self.log_message(f"距離分布計算中… (N={n})", "info"); QApplication.processEvents()

            vals = []
            for t in range(n):
                rng = np.random.default_rng(seed + t)
                state = AssemblyState(self.geom)
                engine = ProcessEngine(self.geom, self.flow, rng)
                for i, st in enumerate(self.flow.steps):
                    if i < len(self.steps_mask) and self.steps_mask[i]:
                        engine.apply_step(st, state)
                P1 = get_world_point(self.geom, state, i1, r1)
                P2 = get_world_point(self.geom, state, i2, r2)
                vals.append(float(np.linalg.norm(P2 - P1)))

            vals = np.asarray(vals, dtype=float)
            
            # アプリ内にヒストグラム描画
            self.distance_histogram.plot_histogram(vals, name1, name2, n)

            # 統計情報表示（平均・標準偏差を明記）
            mean_val = vals.mean()
            std_val = vals.std()
            lo, hi = np.quantile(vals, [0.025, 0.975])
            
            self.pair_result_text.setPlainText(
                f"=== 任意2点間距離の統計 ===\n"
                f"試行回数: N={n}\n"
                f"点1: {name1}\n"
                f"点2: {name2}\n"
                f"\n"
                f"平均 (mean):      {mean_val:.3f} mm\n"
                f"標準偏差 (std):   {std_val:.3f} mm\n"
                f"95%信頼区間:      [{lo:.3f}, {hi:.3f}] mm"
            )
            self.log_message("距離分布計算完了", "info")

        except Exception as e:
            self.log_message(f"距離分布エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- Log ----------
    def log_message(self, message: str, level: str="info"):
        color = {"info":"black","warning":"orange","error":"red"}.get(level,"black")
        self.log_text.append(f'<span style="color:{color};">[{level.upper()}] {message}</span>')

    def closeEvent(self, event):
        if HAS_WATCHDOG and hasattr(self, "observer"):
            self.observer.stop(); self.observer.join()
        event.accept()


# =============================================================================
# エントリポイント
# =============================================================================
def main():
    global MODEL_ONEFILE_CSV_PATH
    global _APP_WINDOW  # %runfile 再実行時に前回ウィンドウを閉じる用

    log_env()

    # 既存の QApplication があれば再利用（Spyder/%runfile 対策）
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    app.setFont(setup_font())

    csv_path = Path(
        r"C:\Users\tsumura-s\OneDrive - 国立研究開発法人海上・港湾・航空技術研究所\1_current\python\ブロック精度\model_onefile_buttchain_fixed.csv"
    )

    # ★存在しない場合は「サンプル作成」せず、エラーで停止（fail fast）
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] CSVが見つかりません: {csv_path}")

    # MainWindow はこのグローバルを参照して読み込む
    MODEL_ONEFILE_CSV_PATH = csv_path

    # ★ここを追加：標準偏差を出して終了
    print_all_edge_stds_after_cutting(MODEL_ONEFILE_CSV_PATH, n_trials=5000, seed=42)

    # 前回のウィンドウが残っていれば閉じて破棄
    try:
        if _APP_WINDOW is not None:
            _APP_WINDOW.close()
            _APP_WINDOW.deleteLater()
            app.processEvents()
    except NameError:
        _APP_WINDOW = None

    w = MainWindow()
    _APP_WINDOW = w
    w.show()

    # 既にイベントループが走っている環境では exec() しない
    if created:
        sys.exit(app.exec())
    else:
        return w



if __name__ == "__main__":
    main()
