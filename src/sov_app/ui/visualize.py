"""Visualization backends and interactive selection widgets."""

from __future__ import annotations

import logging
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
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except Exception:  # pragma: no cover
    matplotlib = None
    FigureCanvasQTAgg = None
    Figure = None
    Line3DCollection = None

if USE_OPEN3D:
    try:
        import open3d as o3d
    except Exception:  # pragma: no cover
        o3d = None
else:
    o3d = None


logger = logging.getLogger(__name__)


def deviation_color_map(deviation: float, tol_mm: float) -> np.ndarray:
    ratio = min(deviation / max(tol_mm, 0.001), 1.0)
    blue = np.array([0.2, 0.4, 1.0])
    red = np.array([1.0, 0.0, 0.0])
    return (1.0 - ratio) * blue + ratio * red


class Open3DVisualizer:
    def __init__(self):
        self.geometries: List[Any] = []
        self._vis = None
        self._window_created = False
        self._window_title = "Assembly View"

    def is_available(self) -> bool:
        return o3d is not None

    def ensure_window(self, title: str = "Assembly View", width: int = 900, height: int = 640) -> bool:
        if o3d is None:
            return False
        if self._window_created and self._vis is not None:
            return True

        self._vis = o3d.visualization.Visualizer()
        self._window_created = bool(self._vis.create_window(window_name=title, width=width, height=height))
        if not self._window_created:
            self._vis = None
            return False
        self._window_title = title
        return True

    def show_scene(self, title: str = "Assembly View", width: int = 900, height: int = 640) -> bool:
        if not self.ensure_window(title=title, width=width, height=height):
            return False

        if self._vis is None:
            return False

        self._vis.clear_geometries()
        for geometry in self.geometries:
            self._vis.add_geometry(geometry, reset_bounding_box=False)

        self._vis.poll_events()
        self._vis.update_renderer()
        return True

    def close_window(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
        self._vis = None
        self._window_created = False

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
        self._all_points_artist = None
        self._selected_point1_artist = None
        self._selected_point2_artist = None
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

        all_edge_segments = []
        for inst_id in self.geom.get_instance_ids():
            all_edge_segments.extend(self._build_wireframe_segments(inst_id))
            for ref in self.geom.get_available_refs_for_instance(inst_id):
                try:
                    coords = get_world_point(self.geom, self.state, inst_id, ref)
                    self.all_selectable_points.append({"inst_id": inst_id, "ref": ref, "coords": coords})
                except Exception:
                    pass

        # NOTE:
        # `Line3DCollection` is sorted as one 3D artist in mplot3d.  When the collection
        # contains both near/far edges, painter-order artifacts can hide parts of back-side
        # outlines.  Draw each edge as an individual Line3D artist so depth sorting is done
        # per edge and the rear contour remains stable.
        if all_edge_segments:
            for p0, p1 in all_edge_segments:
                self.ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color="tab:blue",
                    linewidth=2.8,
                    alpha=0.9,
                    picker=False,
                    clip_on=False,
                    antialiased=True,
                )

        bounds_points = [np.asarray(seg[0], dtype=float) for seg in all_edge_segments] + [
            np.asarray(seg[1], dtype=float) for seg in all_edge_segments
        ]

        if self.all_selectable_points:
            self._all_points_artist = self.ax.scatter(
                [p["coords"][0] for p in self.all_selectable_points],
                [p["coords"][1] for p in self.all_selectable_points],
                [p["coords"][2] for p in self.all_selectable_points],
                facecolors="red",
                edgecolors="none",
                s=35,
                picker=True,
            )
            self._selected_point1_artist = self.ax.scatter([], [], [], marker="o", picker=False, zorder=11)
            self._selected_point2_artist = self.ax.scatter([], [], [], marker="^", picker=False, zorder=12)
            self._redraw_selected_points()
            bounds_points.extend(np.asarray(p["coords"], dtype=float) for p in self.all_selectable_points)
        else:
            self._all_points_artist = None
            self._selected_point1_artist = None
            self._selected_point2_artist = None

        if bounds_points:
            bx = [pt[0] for pt in bounds_points]
            by = [pt[1] for pt in bounds_points]
            bz = [pt[2] for pt in bounds_points]
            self.ax.auto_scale_xyz(bx, by, bz)

        self.canvas.draw()

    def _redraw_selected_points(self):
        if self._selected_point1_artist is None or self._selected_point2_artist is None:
            return

        if len(self.selected_points) >= 1:
            c1 = np.asarray(self.selected_points[0][2], dtype=float)
            self._selected_point1_artist._offsets3d = ([c1[0]], [c1[1]], [c1[2]])
            self._selected_point1_artist.set_facecolor("gold")
            self._selected_point1_artist.set_edgecolor("black")
            self._selected_point1_artist.set_linewidths([1.5])
            self._selected_point1_artist.set_sizes([160])
        else:
            self._selected_point1_artist._offsets3d = ([], [], [])
            self._selected_point1_artist.set_sizes([])

        if len(self.selected_points) >= 2:
            c2 = np.asarray(self.selected_points[1][2], dtype=float)
            self._selected_point2_artist._offsets3d = ([c2[0]], [c2[1]], [c2[2]])
            self._selected_point2_artist.set_facecolor("lime")
            self._selected_point2_artist.set_edgecolor("black")
            self._selected_point2_artist.set_linewidths([1.5])
            self._selected_point2_artist.set_sizes([200])
        else:
            self._selected_point2_artist._offsets3d = ([], [], [])
            self._selected_point2_artist.set_sizes([])

    def _build_wireframe_segments(self, inst_id: str) -> List[List[np.ndarray]]:
        inst = self.geom.get_instance(inst_id)
        if not inst:
            return []
        proto = self.geom.get_prototype(inst.get("prototype", ""))
        feats = proto.get("features", {})
        edge_containers = [feats.get("edges", {})]
        for key, value in feats.items():
            if key == "edges":
                continue
            if isinstance(value, dict) and "edges" in value and isinstance(value["edges"], dict):
                edge_containers.append(value["edges"])

        total_edges = 0
        dropped_counts = {"invalid_endpoints": 0, "resolve_failed": 0, "nan_or_inf": 0}
        segments = []
        for edge_map in edge_containers:
            for edge in edge_map.values():
                total_edges += 1
                endpoints = edge.get("endpoints", []) if isinstance(edge, dict) else []
                if len(endpoints) < 2:
                    dropped_counts["invalid_endpoints"] += 1
                    continue

                try:
                    p0 = np.asarray(get_world_point(self.geom, self.state, inst_id, f"points.{endpoints[0]}"), dtype=float)
                    p1 = np.asarray(get_world_point(self.geom, self.state, inst_id, f"points.{endpoints[1]}"), dtype=float)
                except Exception:
                    dropped_counts["resolve_failed"] += 1
                    continue

                if p0.shape != (3,) or p1.shape != (3,) or not np.isfinite(p0).all() or not np.isfinite(p1).all():
                    dropped_counts["nan_or_inf"] += 1
                    continue
                segments.append([p0, p1])

        dropped_total = sum(dropped_counts.values())
        logger.debug(
            "Wireframe segments for %s: total_edges=%d created=%d dropped=%d details=%s",
            inst_id,
            total_edges,
            len(segments),
            dropped_total,
            dropped_counts,
        )
        return segments

    def _on_pick(self, event):
        inds = getattr(event, "ind", None)
        if inds is None:
            return
        # event.ind may be a numpy array and can contain multiple hit indices.
        try:
            if len(inds) == 0:
                return
            pick_i = int(inds[0])
        except TypeError:
            pick_i = int(inds)

        p = self.all_selectable_points[pick_i]
        if any((a == p["inst_id"] and b == p["ref"]) for a, b, _ in self.selected_points):
            return
        if len(self.selected_points) >= 2:
            return
        self.selected_points.append((p["inst_id"], p["ref"], p["coords"]))
        self._redraw_selected_points()
        self.canvas.draw_idle()
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
