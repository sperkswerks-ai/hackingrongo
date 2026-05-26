#!/usr/bin/env python3
"""
Multi-view synthetic image renderer for INSCRIBE Nexus 3D tablet models.

Renders Tablets B, C, D from multiple viewing angles via the INSCRIBE web
viewer (3DHOP) using a headless browser. Output images feed into Zone A
autoencoder training as high-quality multi-view glyph data.

Usage:
    pip install playwright
    playwright install chromium

    python render_tablet_views.py --tablet D --num-views 12 --output-dir data/glyphs/synthetic_views_test/
    python render_tablet_views.py --tablet all --num-views 24 --output-dir data/glyphs/synthetic_views/
"""

import asyncio
import argparse
import math
from pathlib import Path
from typing import List
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


TABLET_URLS = {
    "B": "https://www.inscribercproject.com/Aruku_Kurenga_-_Tablet_B.php",
    "C": "https://www.inscribercproject.com/Mamari_-_Tablet_C.php",
    "D": "https://www.inscribercproject.com/%C3%89chancr%C3%A9e_-_Tablet_D.php",
}

# 3DHOP model load can take 30–90 s for 60–124 MB models over the network.
_PAGE_LOAD_TIMEOUT_MS = 90_000
_MODEL_LOAD_TIMEOUT_MS = 120_000
# Wait after sending a rotation command before screenshotting — lets WebGL finish a frame.
_RENDER_SETTLE_MS = 400


class TabletViewRenderer:
    """Renders multi-view synthetic images of rongorongo tablets via 3DHOP."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _wait_for_model(self, page) -> bool:
        """Wait for 3DHOP to finish loading the mesh. Returns True on success."""
        # INSCRIBE uses window.presenter (3DHOP), not THREEDHOP/scene/viewer globals.
        # The presenter.trackball property is created during scene parse and is the
        # most reliable readiness signal — it only appears after the mesh is loaded.
        js_ready = """
            () => {
                // INSCRIBE 3DHOP: presenter with loaded trackball
                if (window.presenter && window.presenter.trackball) return true;
                // Fallback: WebGL canvas context exists
                const c = document.querySelector('canvas');
                if (!c) return false;
                return !!(c.getContext('webgl') || c.getContext('webgl2'));
            }
        """
        deadline = asyncio.get_event_loop().time() + _MODEL_LOAD_TIMEOUT_MS / 1000
        while asyncio.get_event_loop().time() < deadline:
            ready = await page.evaluate(js_ready)
            if ready:
                return True
            await asyncio.sleep(2)
        return False

    async def _rotate_and_shoot(self, page, angle_deg: float, output_path: Path) -> None:
        """Rotate the 3DHOP viewer to azimuth angle_deg and take a screenshot.

        INSCRIBE uses ``window.presenter`` (3DHOP).  The trackball stores its
        orientation as a 4×4 column-major float array split across two fields:

          - ``_sphereMatrix`` — pure rotation (no translation)
          - ``_matrix`` — rotation + camera-distance translation (what the
                          renderer reads via the ``matrix`` getter)
          - ``_distance`` — scalar camera distance (normalised to scene radius)

        For a turntable Y-axis rotation by azimuth φ the column-major view
        matrix is::

            [ cos φ,  0,  sin φ,  0,      ← col 0
                 0,   1,  0,     0,      ← col 1
             -sin φ,  0,  cos φ,  0,      ← col 2
                 0,   0,  -dist, 1 ]     ← col 3  (camera distance)

        Verified: φ=0 → identity [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-2,1] ✓
        """
        angle_rad = math.radians(angle_deg)

        rotate_js = f"""
            () => {{
                const pr = window.presenter;
                if (!pr || !pr.trackball) return 'no_trackball';

                const phi  = {angle_rad:.8f};
                const cos  = Math.cos(phi);
                const sin  = Math.sin(phi);
                const tb   = pr.trackball;
                const dist = (tb._distance != null) ? tb._distance : 2;

                // 1. Set the pure-rotation sphere matrix (column-major, no translation)
                tb._sphereMatrix = [
                    cos,  0,  sin,  0,
                    0,    1,  0,    0,
                    -sin, 0,  cos,  0,
                    0,    0,  0,    1,
                ];

                // 2. Let the trackball recompute _matrix from _sphereMatrix + dist
                //    (falls back to writing _matrix directly if _computeMatrix absent)
                if (typeof tb._computeMatrix === 'function') {{
                    tb._computeMatrix();
                }} else {{
                    tb._matrix = [
                        cos,  0,  sin,  0,
                        0,    1,  0,    0,
                        -sin, 0,  cos,  0,
                        0,    0,  -dist, 1,
                    ];
                }}

                // 3. Trigger a WebGL redraw
                pr.repaint();
                return 'rotated';
            }}
        """
        result = await page.evaluate(rotate_js)
        if result != 'rotated':
            # Mouse-drag fallback: only reached if presenter/trackball is missing
            # (should not happen on INSCRIBE — log it as a warning).
            print(f"    WARNING: JS rotation failed ({result}), falling back to mouse drag")
            canvas = await page.query_selector("canvas")
            if canvas:
                box = await canvas.bounding_box()
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                drag_px = int(angle_deg / 360 * box["width"])
                await page.mouse.move(cx, cy)
                await page.mouse.down()
                await page.mouse.move(cx + drag_px, cy, steps=5)
                await page.mouse.up()

        # Let WebGL finish rendering the new frame before screenshotting.
        await asyncio.sleep(_RENDER_SETTLE_MS / 1000)
        # Screenshot just the WebGL canvas element to exclude page chrome
        # (header, sidebar, copyright footer which are in normal document flow).
        canvas = await page.query_selector("canvas")
        if canvas:
            await canvas.screenshot(path=str(output_path))
        else:
            await page.screenshot(path=str(output_path))

    async def _apply_raking_light(self, page) -> str:
        """
        Configure the INSCRIBE 3DHOP viewer for optimal glyph visibility:

        1. **Solid color mode** — strips the photogrammetric UV texture that bakes
           wood-grain colour into the mesh surface.  With texture removed, WebGL
           shades the mesh purely from geometry, making incised glyph strokes
           visually distinct from the surrounding flat surface.

           Implemented via direct property write on the instance data object
           (the tag-based API ``setInstanceSolidColor`` requires ``HOP_ALL``
           which is not exposed as a global on the INSCRIBE page).

        2. **Overhead-left light** — ``rotateLight(0.10, 0.15)`` sets the 3DHOP
           light direction to a position slightly left and moderately above
           horizontal (≈ 25° elevation, 15° azimuth), which was empirically found
           to maximise the glyph candidate count on Tablet B (110 vs. 10 without).

        Returns a short status string for logging.
        """
        raking_js = """
            (() => {
                const pr = window.presenter;
                if (!pr) return 'no_presenter';

                // ── 1. Enable solid colour (strips photographic texture) ──────
                const instances = pr._scene && pr._scene.modelInstances;
                if (!instances) return 'no_instances';
                let n_solid = 0;
                for (const key of Object.keys(instances)) {
                    const inst = instances[key];
                    inst.useSolidColor = true;
                    inst.color = [0.85, 0.85, 0.85];  // neutral light gray
                    n_solid++;
                }

                // ── 2. Set overhead-left light (best empirical angle) ─────────
                // rotateLight(x, y):  _lightDirection = [-2x, -2y, -sqrt(1-r²)]
                // (0.10, 0.15) → ≈ [-0.20, -0.30, -0.93]  ~25° elevation, 15° azimuth
                pr.rotateLight(0.10, 0.15);
                pr.repaint();

                const dir = pr._lightDirection.map(v => +v.toFixed(3));
                return 'solid=' + n_solid + ' dir=' + JSON.stringify(dir);
            })()
        """
        result = await page.evaluate(raking_js)
        return result

    async def render_tablet(
        self,
        tablet: str,
        num_views: int = 24,
        width: int = 512,
        height: int = 512,
        raking_light: bool = True,
    ) -> List[Path]:
        """
        Render a tablet from `num_views` evenly-spaced azimuth angles.

        Returns a list of saved image paths.
        """
        if tablet not in TABLET_URLS:
            raise ValueError(f"Unknown tablet: {tablet}. Must be one of {list(TABLET_URLS)}")

        url = TABLET_URLS[tablet]
        saved: List[Path] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": width, "height": height})
            page.set_default_timeout(_PAGE_LOAD_TIMEOUT_MS)

            print(f"Loading Tablet {tablet} from {url}")
            await page.goto(url, wait_until="networkidle", timeout=_PAGE_LOAD_TIMEOUT_MS)

            print("Waiting for 3D model to initialise…")
            ready = await self._wait_for_model(page)
            if not ready:
                print(f"  WARNING: model did not signal ready within "
                      f"{_MODEL_LOAD_TIMEOUT_MS // 1000}s — attempting renders anyway")

            # Resize the WebGL canvas to the requested output resolution.
            # The 3DHOP viewer HTML sets a fixed canvas size; overriding width/height
            # attributes forces WebGL to render at full resolution before we screenshot.
            resized = await page.evaluate(f"""
                () => {{
                    const canvas = document.querySelector('canvas');
                    if (!canvas) return false;
                    canvas.width  = {width};
                    canvas.height = {height};
                    canvas.style.width  = '{width}px';
                    canvas.style.height = '{height}px';
                    // Notify 3DHOP presenter renderer of the new size if possible.
                    // INSCRIBE uses window.presenter (not THREEDHOP/scene/viewer).
                    const pr = window.presenter;
                    if (pr && pr.renderer && typeof pr.renderer.setSize === 'function') {{
                        pr.renderer.setSize({width}, {height});
                    }}
                    return true;
                }}
            """)
            if resized:
                print(f"  Canvas resized to {width}×{height}.")
                await asyncio.sleep(0.5)  # allow one re-render frame
            else:
                print(f"  WARNING: could not resize canvas — renders may be at native resolution.")

            if raking_light:
                light_status = await self._apply_raking_light(page)
                print(f"  Raking light: {light_status}")
                await asyncio.sleep(0.4)  # allow one frame to re-render with new lighting

            tablet_dir = self.output_dir / f"tablet_{tablet}"
            tablet_dir.mkdir(exist_ok=True)

            print(f"Rendering {num_views} views of Tablet {tablet}…")
            for i in range(num_views):
                angle_deg = 360.0 / num_views * i
                out = tablet_dir / f"{tablet.lower()}_{i:03d}_{angle_deg:.0f}deg.png"
                await self._rotate_and_shoot(page, angle_deg, out)
                saved.append(out)
                print(f"  View {i + 1}/{num_views}: {angle_deg:.0f}° → {out.name}")

            await browser.close()

        return saved

    async def render_all_tablets(
        self,
        num_views: int = 24,
        width: int = 512,
        height: int = 512,
        raking_light: bool = True,
    ) -> dict:
        """Render all three tablets sequentially."""
        results = {}
        for tablet in TABLET_URLS:
            try:
                images = await self.render_tablet(
                    tablet, num_views=num_views, width=width, height=height,
                    raking_light=raking_light,
                )
                results[tablet] = {"success": True, "count": len(images),
                                   "images": [str(p) for p in images]}
            except Exception as e:
                results[tablet] = {"success": False, "error": str(e)}
        return results


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render multi-view synthetic images of INSCRIBE rongorongo tablets"
    )
    parser.add_argument("--tablet", choices=["B", "C", "D", "all"], default="all")
    parser.add_argument("--num-views", type=int, default=24,
                        help="Number of azimuth angles to render (default: 24 = 15° steps)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/glyphs/synthetic_views"))
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--raking-light", action="store_true", default=True,
                        help="Set 3DHOP lighting to a grazing angle to reveal incised glyphs (default: on)")
    parser.add_argument("--no-raking-light", dest="raking_light", action="store_false",
                        help="Disable raking light (restore original diffuse lighting)")
    args = parser.parse_args()

    renderer = TabletViewRenderer(args.output_dir)

    if args.tablet == "all":
        results = await renderer.render_all_tablets(
            num_views=args.num_views, width=args.width, height=args.height,
            raking_light=args.raking_light,
        )
        print("\n=== SUMMARY ===")
        for tablet, r in results.items():
            if r["success"]:
                print(f"  ✓ Tablet {tablet}: {r['count']} images → {args.output_dir}/tablet_{tablet}/")
            else:
                print(f"  ✗ Tablet {tablet}: {r['error']}")
    else:
        images = await renderer.render_tablet(
            args.tablet,
            num_views=args.num_views,
            width=args.width,
            height=args.height,
            raking_light=args.raking_light,
        )
        print(f"\n✓ {len(images)} images → {args.output_dir}/tablet_{args.tablet}/")


if __name__ == "__main__":
    asyncio.run(main())
