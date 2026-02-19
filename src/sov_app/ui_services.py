from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .core_models import AssemblyState
from .env import USE_OPEN3D
from .services import AppState
from .visualize import DistanceHistogramWidget, InteractivePointSelector, MatplotlibVisualizer, Open3DVisualizer


@dataclass
class VisualizerConfig:
    use_open3d: bool = USE_OPEN3D


@dataclass
class RenderConfig:
    show_groups: Dict[str, bool]
    deviation_mode: bool = False
    tol_mm: float = 5.0
    deform_scale: float = 1.0


def build_visualizer(config: VisualizerConfig):
    if config.use_open3d:
        return Open3DVisualizer()
    return MatplotlibVisualizer()


def render(visualizer: Any, state: AppState, assembly_state: AssemblyState, render_config: RenderConfig) -> None:
    visualizer.build_scene(
        state.geom,
        assembly_state,
        render_config.show_groups,
        deviation_mode=render_config.deviation_mode,
        tol_mm=render_config.tol_mm,
        deform_scale=render_config.deform_scale,
    )


def show_rendered_scene(visualizer: Any, title: str = "Assembly View", width: int = 900, height: int = 640) -> bool:
    if not USE_OPEN3D:
        return False
    try:
        import open3d as o3d
    except Exception:
        return False

    o3d.visualization.draw_geometries(visualizer.get_geometries(), window_name=title, width=width, height=height)
    return True


def create_distance_histogram_widget() -> DistanceHistogramWidget:
    return DistanceHistogramWidget()


def create_point_selector() -> InteractivePointSelector:
    return InteractivePointSelector()


__all__ = [
    "RenderConfig",
    "VisualizerConfig",
    "build_visualizer",
    "create_distance_histogram_widget",
    "create_point_selector",
    "render",
    "show_rendered_scene",
]
