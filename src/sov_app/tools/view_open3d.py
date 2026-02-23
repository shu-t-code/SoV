from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Open3D viewer")
    parser.add_argument("scene", help="Path to mesh file (.ply/.obj)")
    parser.add_argument("--title", default="Assembly View", help="Window title")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scene_path = Path(args.scene).expanduser()
    if not scene_path.exists():
        print(f"[Open3D Viewer] file not found: {scene_path}")
        return 2

    try:
        import open3d as o3d
    except Exception as exc:
        print(f"[Open3D Viewer] Open3D unavailable: {exc}")
        return 3

    try:
        mesh = o3d.io.read_triangle_mesh(str(scene_path))
        if mesh.is_empty():
            print(f"[Open3D Viewer] mesh is empty: {scene_path}")
            return 4
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=args.title)
        return 0
    except Exception as exc:
        print(f"[Open3D Viewer] rendering failed: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
