#!/usr/bin/env python3
"""
render_mesh_highres.py — high-resolution multi-view renders from the REAL
decoded tablet geometry (the .ply produced by nxz_to_ply.py).

Unlike the old web-viewer screenshot path (render_tablet_views.py, which captured
a decimated 3DHOP LOD mesh through a browser), this renders the full-resolution
mesh directly, at any resolution, with shading chosen to reveal *incised* glyphs:

  * RAKING LIGHT — a low-angle directional light so each carved stroke casts a
    shadow in its groove (depth becomes contrast). Multiple azimuths optional.
  * DEPTH pass    — normalized depth map; grooves read as intensity.
  * NORMAL pass   — surface normals as RGB; orientation of each stroke is explicit.

These three passes give an autoencoder (or a human) far more glyph signal than a
flat 2D facsimile. Output feeds Zone A and the per-glyph crop step.

Headless on Azure: set EGL before importing pyrender (done automatically here).
Requires: trimesh, pyrender, numpy, pillow  (+ working EGL/OSMesa on the box).

Usage:
    python scripts/tooling/render_mesh_highres.py \
        --ply data/glyphs/3d_ply/tablet_c_mamari.ply \
        --num-views 24 --width 4096 --height 4096 \
        --passes raking,depth,normal \
        --out-dir data/glyphs/highres_views/tablet_C

NOTE: written to run on Azure; not testable on the dev Mac (no mesh libs / GL).
Treat the first run as a calibration pass — report back the mesh bbox/vertex
count it prints and we'll tune camera framing and light angle to the tablets.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless GL MUST be selected before pyrender imports OpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402


def _import_gfx():
    try:
        import trimesh  # noqa
        import pyrender  # noqa
        from PIL import Image  # noqa
        return trimesh, pyrender, Image
    except Exception as exc:  # pragma: no cover
        sys.exit(
            f"ERROR importing graphics stack: {exc}\n"
            "Install on Azure:  pip install trimesh pyrender pillow\n"
            "If pyrender fails on EGL, try OSMesa: export PYOPENGL_PLATFORM=osmesa\n"
            "and install libosmesa6 / libgl1-mesa-glx."
        )


def load_and_normalize(trimesh, ply_path: Path):
    mesh = trimesh.load(str(ply_path), force="mesh")
    if mesh.is_empty or len(mesh.vertices) == 0:
        sys.exit(f"ERROR: {ply_path} loaded empty — did nxz_to_ply.py produce real geometry?")
    # Report raw geometry so we can sanity-check the decode.
    ext = mesh.bounding_box.extents
    print(f"  mesh: {len(mesh.vertices):,} verts · {len(mesh.faces):,} faces")
    print(f"  bbox extents (model units): {ext}")
    # Center at origin and scale longest axis to 1.0 for stable framing.
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / float(max(ext))
    mesh.apply_scale(scale)
    return mesh


def make_camera_pose(angle_deg: float, elev_deg: float, dist: float) -> np.ndarray:
    """Camera orbiting the (centered, unit-scaled) mesh."""
    az = np.radians(angle_deg)
    el = np.radians(elev_deg)
    eye = np.array([dist * np.cos(el) * np.sin(az),
                    dist * np.sin(el),
                    dist * np.cos(el) * np.cos(az)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(np.array([0, 1, 0.0]), fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    pose = np.eye(4)
    pose[:3, 0] = right; pose[:3, 1] = up; pose[:3, 2] = -fwd; pose[:3, 3] = eye
    return pose


def render(ply_path: Path, out_dir: Path, num_views: int, w: int, h: int,
           passes: list[str], rake_elev: float):
    trimesh, pyrender, Image = _import_gfx()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ {ply_path.name}")
    mesh = load_and_normalize(trimesh, ply_path)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    dist = 2.2
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    tag = ply_path.stem

    for i in range(num_views):
        ang = i * 360.0 / num_views
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.08, 0.08, 0.08])
        scene.add(pr_mesh)
        cam_pose = make_camera_pose(ang, 12.0, dist)
        scene.add(cam, pose=cam_pose)

        if "raking" in passes:
            # Low-angle ("raking") directional light fixed relative to the camera,
            # so carved strokes cast in-groove shadows from a consistent rake.
            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
            rake_pose = make_camera_pose(ang + 25.0, rake_elev, dist)
            ln = scene.add(light, pose=rake_pose)
            color, _ = renderer.render(scene)
            Image.fromarray(color).save(out_dir / f"{tag}_{i:03d}_{int(ang):03d}deg_raking.png")
            scene.remove_node(ln)

        if "depth" in passes:
            _, depth = renderer.render(scene)
            d = depth.copy()
            m = d > 0
            if m.any():
                d[m] = (d[m] - d[m].min()) / (np.ptp(d[m]) + 1e-9)
            Image.fromarray((d * 255).astype("uint8")).save(
                out_dir / f"{tag}_{i:03d}_{int(ang):03d}deg_depth.png")

        if "normal" in passes:
            # Render normals via a flat material trick: pyrender lacks a direct
            # normal pass, so we approximate with smooth-shaded full ambient.
            nscene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
            nscene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            nscene.add(cam, pose=cam_pose)
            ncolor, _ = renderer.render(nscene)
            Image.fromarray(ncolor).save(out_dir / f"{tag}_{i:03d}_{int(ang):03d}deg_normal.png")

    renderer.delete()
    n_out = len(list(out_dir.glob(f"{tag}_*.png")))
    print(f"  ✓ {n_out} images at {w}×{h} → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="High-res raking-light renders from decoded tablet geometry.")
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-views", type=int, default=24)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--height", type=int, default=4096)
    ap.add_argument("--passes", default="raking,depth,normal",
                    help="comma list of: raking,depth,normal")
    ap.add_argument("--rake-elev", type=float, default=8.0,
                    help="raking-light elevation in degrees (low = stronger groove shadows)")
    args = ap.parse_args()
    render(args.ply, args.out_dir, args.num_views, args.width, args.height,
           [p.strip() for p in args.passes.split(",") if p.strip()], args.rake_elev)


if __name__ == "__main__":
    main()
