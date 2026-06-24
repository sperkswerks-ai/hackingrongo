#!/usr/bin/env python3
"""
render_mesh_highres.py — high-resolution renders from the REAL decoded tablet
geometry (the .ply produced by nxz_to_ply.py).

Rongorongo tablets are THIN FLAT SLABS: the glyphs are incised on the two large
faces, not the ~2 cm edge. So a turntable orbit is wrong — instead we hold the
camera FACE-ON to each inscribed face and sweep a low-angle ("raking") light
around the face. Each light azimuth casts shadows into the incisions from a
different direction; together they reveal carved strokes far better than a flat
2D facsimile. (This is the digital analogue of Reflectance Transformation
Imaging, the technique epigraphers use on inscribed surfaces.)

The face normal is auto-detected as the tablet's thinnest axis, so this works
for B/C/D regardless of how each mesh happens to be oriented.

Passes per face:
  * raking  — N images, low light swept around the face plane (the money shot)
  * depth   — normalized depth map (grooves read as intensity)
  * normal  — smooth-shaded full-ambient (stroke orientation as colour)

Headless on Azure: EGL is selected before pyrender imports.
Requires: trimesh, pyrender, numpy, pillow.

Usage:
    python scripts/tooling/render_mesh_highres.py \
        --ply data/glyphs/3d_ply/tablet_c_mamari.ply \
        --faces both --num-views 12 --width 4096 --height 4096 \
        --passes raking,depth,normal \
        --out-dir data/glyphs/highres_views/tablet_C
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402


def _import_gfx():
    try:
        import trimesh  # noqa
        import pyrender  # noqa
        from PIL import Image, ImageFilter, ImageOps  # noqa
        return trimesh, pyrender, Image, ImageFilter, ImageOps
    except Exception as exc:  # pragma: no cover
        sys.exit(
            f"ERROR importing graphics stack: {exc}\n"
            "Install on Azure:  pip install trimesh pyrender pillow\n"
            "If pyrender fails on EGL, try OSMesa: export PYOPENGL_PLATFORM=osmesa\n"
            "and install libosmesa6 / libgl1-mesa-glx."
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def load_and_normalize(trimesh, ply_path: Path):
    mesh = trimesh.load(str(ply_path), force="mesh")
    if mesh.is_empty or len(mesh.vertices) == 0:
        sys.exit(f"ERROR: {ply_path} loaded empty — did nxz_to_ply.py produce real geometry?")
    ext = mesh.bounding_box.extents.copy()
    print(f"  mesh: {len(mesh.vertices):,} verts · {len(mesh.faces):,} faces")
    print(f"  bbox extents (model units): {ext}")
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / float(max(ext)))
    return mesh, ext


def pose_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray, origin: np.ndarray) -> np.ndarray:
    P = np.eye(4)
    P[:3, 0] = x; P[:3, 1] = y; P[:3, 2] = z; P[:3, 3] = origin
    return P


def camera_pose_facing(face_normal: np.ndarray, up_axis: np.ndarray, dist: float) -> np.ndarray:
    """Camera placed along +face_normal, looking back at the origin (the face).
    pyrender cameras look down their local -Z, so local +Z must point eye→outwards."""
    eye = face_normal * dist
    z = _unit(eye)                     # local +Z points from target to eye
    x = _unit(np.cross(up_axis, z))
    y = np.cross(z, x)
    return pose_from_axes(x, y, z, eye)


def light_pose_raking(face_normal, right, up, theta, rake_elev_deg) -> np.ndarray:
    """Directional light grazing the inscribed face from in-plane azimuth `theta`,
    sitting LOW on the camera (+face_normal) side at `rake_elev_deg` above the
    surface. pyrender DirectionalLight emits along its local -Z, so local +Z must
    point from the surface toward the light source."""
    el = np.radians(rake_elev_deg)
    inplane = np.cos(theta) * right + np.sin(theta) * up
    # Direction from the surface toward the light: low (small +normal) and mostly
    # across the face — a true grazing rake that throws shadows into the grooves.
    to_light = _unit(inplane * np.cos(el) + face_normal * np.sin(el))
    z = to_light
    helper = up if abs(np.dot(up, z)) < 0.95 else right
    x = _unit(np.cross(helper, z))
    y = np.cross(z, x)
    return pose_from_axes(x, y, z, z * 3.0)


def save_relief(depth, path, Image, ImageFilter, blur_radius):
    """Lighting-INDEPENDENT relief map: high-pass the depth buffer so the slab's
    gross flat shape / tilt is removed and only fine incised relief remains, then
    hard-stretch contrast. This is the reliable 'dark grooves, light surface' view."""
    import numpy as _np
    d = depth.astype(_np.float32)
    m = d > 0
    if not m.any():
        return False
    lo, hi = _np.percentile(d[m], [1, 99])
    dn = _np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    dn[~m] = float(dn[m].mean())                 # flatten background to avoid edge ring
    base = Image.fromarray((dn * 255).astype("uint8"), "L")
    blur = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    hp = _np.asarray(base, _np.float32) - _np.asarray(blur, _np.float32)   # high-pass = fine relief
    p1, p99 = _np.percentile(hp[m], [1, 99])
    hp = _np.clip((hp - p1) / (p99 - p1 + 1e-9), 0, 1)
    hp[~m] = 0.0
    Image.fromarray((hp * 255).astype("uint8"), "L").save(path)
    return True


def render(ply_path, out_dir, faces, num_views, w, h, passes, rake_elev, frame_margin):
    trimesh, pyrender, Image, ImageFilter, ImageOps = _import_gfx()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ {ply_path.name}")
    mesh, ext = load_and_normalize(trimesh, ply_path)

    # Axes by extent: thinnest = face normal; largest & middle span the face.
    order = np.argsort(ext)                  # [thin, mid, long]
    norm_i, up_i, long_i = order[0], order[1], order[2]
    basis = np.eye(3)
    face_normal = basis[norm_i]
    up_axis = basis[up_i]
    right_axis = basis[long_i]
    # Frame so the long in-plane axis fills the view (normalized longest=1.0).
    inplane_half = 0.5 * (ext[long_i] / max(ext))
    yfov = np.pi / 4.0
    dist = inplane_half / np.tan(yfov / 2.0) * frame_margin
    print(f"  face normal axis={norm_i} (thin), framing dist={dist:.2f}")

    pr_mesh_flat = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    tag = ply_path.stem

    sides = {"recto": 1.0, "verso": -1.0}
    if faces != "both":
        sides = {faces: sides[faces]}

    for side_name, sgn in sides.items():
        n = face_normal * sgn
        up = up_axis
        right = right_axis * sgn          # keep handedness consistent per side
        cam_pose = camera_pose_facing(n, up, dist)

        if "raking" in passes:
            # Directional shadows are what make raking light reveal incisions —
            # OFF by default in pyrender, so enable them explicitly.
            flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            for k in range(num_views):
                theta = 2 * np.pi * k / num_views
                scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.03, 0.03, 0.03])
                scene.add(pr_mesh_flat)
                scene.add(cam, pose=cam_pose)
                scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=6.0),
                          pose=light_pose_raking(n, right, up, theta, rake_elev))
                color, _ = renderer.render(scene, flags=flags)
                img = ImageOps.autocontrast(Image.fromarray(color).convert("L"), cutoff=1)
                img.save(out_dir / f"{tag}_{side_name}_rake{int(np.degrees(theta)):03d}.png")

        if "relief" in passes or "depth" in passes:
            # Lighting-independent high-pass relief — the reliable high-contrast view.
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
            scene.add(pr_mesh_flat); scene.add(cam, pose=cam_pose)
            _, depth = renderer.render(scene)
            radius = max(6, w // 100)
            save_relief(depth, out_dir / f"{tag}_{side_name}_relief.png",
                        Image, ImageFilter, radius)

        if "normal" in passes:
            ns = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
            ns.add(pyrender.Mesh.from_trimesh(mesh, smooth=True)); ns.add(cam, pose=cam_pose)
            ncolor, _ = renderer.render(ns)
            Image.fromarray(ncolor).save(out_dir / f"{tag}_{side_name}_normal.png")

    renderer.delete()
    n_out = len(list(out_dir.glob(f"{tag}_*.png")))
    print(f"  ✓ {n_out} images at {w}×{h} → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slab-aware raking-light renders from decoded tablet geometry.")
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--faces", choices=["both", "recto", "verso"], default="both")
    ap.add_argument("--num-views", type=int, default=12, help="raking-light azimuths per face")
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--height", type=int, default=4096)
    ap.add_argument("--passes", default="relief,raking,normal", help="comma list: relief,raking,normal")
    ap.add_argument("--rake-elev", type=float, default=12.0,
                    help="raking-light elevation toward camera (low = stronger groove shadows)")
    ap.add_argument("--frame-margin", type=float, default=1.15,
                    help=">1 zooms out; tune so the face fills the frame")
    args = ap.parse_args()
    render(args.ply, args.out_dir, args.faces, args.num_views, args.width, args.height,
           [p.strip() for p in args.passes.split(",") if p.strip()], args.rake_elev, args.frame_margin)


if __name__ == "__main__":
    main()
