"""Visualization backends and interactive selection widgets."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..engine.core_models import AssemblyState, GeometryModel, get_world_point, rpy_to_rotation_matrix
from ..env import USE_OPEN3D

try:
    import matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    matplotlib = None
    FigureCanvasQTAgg = None
    Figure = None

if USE_OPEN3D:
    try:
        import open3d as o3d
    except Exception:  # pragma: no cover
        o3d = None
else:
    o3d = None


def deviation_color_map(deviation: float, tol_mm: float) -> np.ndarray:
    ratio = min(deviation / max(tol_mm, 0.001), 1.0)
    blue = np.array([0.2, 0.4, 1.0])
    red = np.array([1.0, 0.0, 0.0])
    return (1.0 - ratio) * blue + ratio * red


class Open3DVisualizer:
    def __init__(self):
        self.geometries: List[Any] = []

    def build_scene(self, geom: GeometryModel, state: AssemblyState, show_groups: Dict[str, bool], deviation_mode: bool = False, tol_mm: float = 5.0, deform_scale: float = 1.0):
        self.geometries = []
        if o3d is None:
            return
        for inst_id, inst in geom.instances.items():
            tags = inst.get("tags", [])
            if not (("A" in tags and show_groups.get("A", True)) or ("C" in tags and show_groups.get("C", True))):
                continue
            proto = geom.get_prototype(inst["prototype"])
            dims = state.get_realized_dims(inst_id) or proto.get("dims", {})
            l_size, h_size, t_size = float(dims.get("L", 1000)), float(dims.get("H", 1000)), float(dims.get("t", 10))

            nominal_o = np.array(inst["frame"]["origin"], dtype=float)
            current_o = state.get_transform(inst_id)["origin"]
            draw_o = nominal_o + (current_o - nominal_o) * deform_scale

            nominal_rpy = np.array(inst["frame"]["rpy_deg"], dtype=float)
            current_rpy = np.array(state.get_transform(inst_id)["rpy_deg"], dtype=float)
            draw_rpy = nominal_rpy + (current_rpy - nominal_rpy) * deform_scale

            box = o3d.geometry.TriangleMesh.create_box(l_size, h_size, t_size)
            box.compute_vertex_normals()
            box.rotate(rpy_to_rotation_matrix(*draw_rpy), center=(0, 0, 0))
            box.translate(draw_o)
            if deviation_mode:
                deviation = float(np.linalg.norm(current_o - nominal_o))
                box.paint_uniform_color(deviation_color_map(deviation, tol_mm).tolist())
            else:
                box.paint_uniform_color([0.7, 0.8, 1.0] if "A" in tags else [0.4, 0.6, 0.9])
            self.geometries.append(box)
        self.geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0]))

    def get_geometries(self) -> List[Any]:
        return self.geometries


class MatplotlibVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        if Figure is None or FigureCanvasQTAgg is None:
            self.figure = None
            self.canvas = None
            self.ax = None
            layout.addWidget(QLabel("matplotlib が利用できないため 3D 表示できません"))
            return
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas)

    def build_scene(self, geom: GeometryModel, state: AssemblyState, show_groups: Dict[str, bool], deviation_mode: bool = False, tol_mm: float = 5.0, deform_scale: float = 1.0):
        if self.ax is None:
            return
        self.ax.clear()
        all_x, all_y, all_z = [], [], []
        for inst_id, inst in geom.instances.items():
            tags = inst.get("tags", [])
            if not (("A" in tags and show_groups.get("A", True)) or ("C" in tags and show_groups.get("C", True))):
                continue
            proto = geom.get_prototype(inst["prototype"])
            dims = state.get_realized_dims(inst_id) or proto.get("dims", {})
            l_size, h_size, t_size = float(dims.get("L", 1000)), float(dims.get("H", 1000)), float(dims.get("t", 10))
            nominal_o = np.array(inst["frame"]["origin"], dtype=float)
            current_o = state.get_transform(inst_id)["origin"]
            draw_o = nominal_o + (current_o - nominal_o) * deform_scale
            nominal_rpy = np.array(inst["frame"]["rpy_deg"], dtype=float)
            current_rpy = np.array(state.get_transform(inst_id)["rpy_deg"], dtype=float)
            draw_rpy = nominal_rpy + (current_rpy - nominal_rpy) * deform_scale
            color = tuple(deviation_color_map(float(np.linalg.norm(current_o - nominal_o)), tol_mm)) if deviation_mode else ("lightblue" if "A" in tags else "steelblue")
            verts = self._draw_box_rotated(draw_o, l_size, h_size, t_size, rpy_to_rotation_matrix(*draw_rpy), color)
            xs, ys, zs = zip(*verts)
            all_x += xs
            all_y += ys
            all_z += zs
        if all_x:
            self.ax.set_xlim(min(all_x), max(all_x)); self.ax.set_ylim(min(all_y), max(all_y)); self.ax.set_zlim(min(all_z), max(all_z))
        self.canvas.draw()

    def _draw_box_rotated(self, origin, l_size, h_size, t_size, rot, color):
        local_v = np.array([[0, 0, 0], [l_size, 0, 0], [l_size, h_size, 0], [0, h_size, 0], [0, 0, t_size], [l_size, 0, t_size], [l_size, h_size, t_size], [0, h_size, t_size]])
        world_v = [rot @ lv + origin for lv in local_v]
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for a, b in edges:
            self.ax.plot([world_v[a][0], world_v[b][0]], [world_v[a][1], world_v[b][1]], [world_v[a][2], world_v[b][2]], color=color, linewidth=1.5)
        return world_v


class DistanceHistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        if Figure is None or FigureCanvasQTAgg is None:
            self.figure = None
            self.canvas = None
            self.ax = None
            layout.addWidget(QLabel("matplotlib が利用できないためヒストグラム表示できません"))
            return
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

    def plot_histogram(self, values: np.ndarray, name1: str, name2: str, n_trials: int):
        if self.ax is None:
            return
        self.ax.clear()
        self.ax.hist(values, bins=30, alpha=0.85, edgecolor="black", color="steelblue")
        self.ax.set_xlabel("Distance [mm]")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title(f"{name1} ↔ {name2} (N={n_trials})")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


class InteractivePointSelector(QWidget):
    selection_completed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.geom = None
        self.state = None
        self.show_groups = {}
        self.selected_points = []
        self.all_selectable_points = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("3Dビュー上の点をクリックして2点を選択してください\n（点は赤い○で表示されます）"))
        if Figure is None or FigureCanvasQTAgg is None:
            self.figure = None
            self.canvas = None
            self.ax = None
            layout.addWidget(QLabel("matplotlib が利用できないため選択ビュー表示できません"))
        else:
            self.figure = Figure(figsize=(10, 8))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.canvas.mpl_connect("pick_event", self._on_pick)
            layout.addWidget(self.canvas)
        status_group = QGroupBox("選択状態")
        status_layout = QVBoxLayout(status_group)
        self.point1_label = QLabel("点1: 未選択")
        self.point2_label = QLabel("点2: 未選択")
        status_layout.addWidget(self.point1_label); status_layout.addWidget(self.point2_label)
        layout.addWidget(status_group)
        row = QHBoxLayout()
        reset_btn = QPushButton("選択をリセット"); reset_btn.clicked.connect(self._reset_selection)
        self.confirm_btn = QPushButton("選択を確定"); self.confirm_btn.clicked.connect(self._confirm_selection); self.confirm_btn.setEnabled(False)
        row.addWidget(reset_btn); row.addWidget(self.confirm_btn)
        layout.addLayout(row)

    def set_geometry_and_state(self, geom, state, show_groups):
        self.geom = geom; self.state = state; self.show_groups = show_groups; self._update_3d_view()

    def _update_3d_view(self):
        if self.ax is None:
            return
        self.ax.clear(); self.all_selectable_points = []
        if not self.geom or not self.state:
            self.canvas.draw(); return
        for inst_id in self.geom.get_instance_ids():
            for ref in self.geom.get_available_refs_for_instance(inst_id):
                try:
                    coords = get_world_point(self.geom, self.state, inst_id, ref)
                    self.all_selectable_points.append({"inst_id": inst_id, "ref": ref, "coords": coords})
                except Exception:
                    pass
        if self.all_selectable_points:
            self.ax.scatter([p["coords"][0] for p in self.all_selectable_points], [p["coords"][1] for p in self.all_selectable_points], [p["coords"][2] for p in self.all_selectable_points], color="red", s=30, picker=True)
        self.canvas.draw()

    def _on_pick(self, event):
        if not getattr(event, "ind", None):
            return
        p = self.all_selectable_points[event.ind[0]]
        if any((a == p["inst_id"] and b == p["ref"]) for a, b, _ in self.selected_points):
            return
        if len(self.selected_points) >= 2:
            return
        self.selected_points.append((p["inst_id"], p["ref"], p["coords"]))
        self._update_selection_labels()

    def _update_selection_labels(self):
        self.point1_label.setText("点1: 未選択" if len(self.selected_points) < 1 else f"点1: {self.selected_points[0][0]}:{self.selected_points[0][1]}")
        self.point2_label.setText("点2: 未選択" if len(self.selected_points) < 2 else f"点2: {self.selected_points[1][0]}:{self.selected_points[1][1]}")
        self.confirm_btn.setEnabled(len(self.selected_points) == 2)

    def _reset_selection(self):
        self.selected_points = []
        self._update_selection_labels()
        self._update_3d_view()

    def _confirm_selection(self):
        if len(self.selected_points) != 2:
            return
        self.selection_completed.emit({
            "point1": {"instance": self.selected_points[0][0], "ref": self.selected_points[0][1], "coords": self.selected_points[0][2]},
            "point2": {"instance": self.selected_points[1][0], "ref": self.selected_points[1][1], "coords": self.selected_points[1][2]},
        })


__all__ = ["DistanceHistogramWidget", "InteractivePointSelector", "MatplotlibVisualizer", "Open3DVisualizer", "deviation_color_map"]
