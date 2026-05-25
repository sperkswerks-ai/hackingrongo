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
        # 3DHOP signals readiness in several ways depending on version;
        # we poll for the most common indicators.
        js_ready = """
            () => {
                // 3DHOP v4+: THREEDHOP global with trackball
                if (window.THREEDHOP && window.THREEDHOP.trackball) return true;
                // Older builds expose `scene`
                if (window.scene && window.scene.trackball) return true;
                // Some builds use `viewer`
                if (window.viewer && window.viewer.trackball) return true;
                // Fallback: WebGL canvas exists and has a context
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
        """Rotate the 3DHOP viewer to azimuth angle_deg and take a screenshot."""
        angle_rad = math.radians(angle_deg)  # 3DHOP/Three.js uses radians, not degrees

        # Try multiple 3DHOP API variants in priority order.
        # The `phi` property controls azimuth on the trackball in 3DHOP.
        rotate_js = f"""
            () => {{
                const rad = {angle_rad:.8f};
                let rotated = false;

                // 3DHOP v4+
                if (window.THREEDHOP && window.THREEDHOP.trackball) {{
                    window.THREEDHOP.trackball.phi = rad;
                    rotated = true;
                }}
                // Older 3DHOP `scene` global
                else if (window.scene && window.scene.trackball) {{
                    window.scene.trackball.phi = rad;
                    rotated = true;
                }}
                // Some builds use `viewer`
                else if (window.viewer && window.viewer.trackball) {{
                    window.viewer.trackball.phi = rad;
                    rotated = true;
                }}

                if (!rotated) return false;

                // Request a re-render — 3DHOP uses RAF, but try explicit draw if available.
                const hub = window.THREEDHOP || window.scene || window.viewer;
                if (hub && typeof hub.draw === 'function') hub.draw();
                else if (hub && hub.renderer && hub.scene3js && hub.camera3js)
                    hub.renderer.render(hub.scene3js, hub.camera3js);

                return true;
            }}
        """
        rotated = await page.evaluate(rotate_js)
        if not rotated:
            # Last resort: synthetic mouse drag on the canvas to force rotation.
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

    async def render_tablet(
        self,
        tablet: str,
        num_views: int = 24,
        width: int = 512,
        height: int = 512,
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
                    // Notify Three.js / 3DHOP renderer of the new size if possible
                    const hub = window.THREEDHOP || window.scene || window.viewer;
                    if (hub && hub.renderer && typeof hub.renderer.setSize === 'function') {{
                        hub.renderer.setSize({width}, {height});
                        if (hub.camera3js) {{
                            hub.camera3js.aspect = {width} / {height};
                            hub.camera3js.updateProjectionMatrix();
                        }}
                    }}
                    return true;
                }}
            """)
            if resized:
                print(f"  Canvas resized to {width}×{height}.")
                await asyncio.sleep(0.5)  # allow one re-render frame
            else:
                print(f"  WARNING: could not resize canvas — renders may be at native resolution.")

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

    async def render_all_tablets(self, num_views: int = 24) -> dict:
        """Render all three tablets sequentially."""
        results = {}
        for tablet in TABLET_URLS:
            try:
                images = await self.render_tablet(tablet, num_views=num_views)
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
    args = parser.parse_args()

    renderer = TabletViewRenderer(args.output_dir)

    if args.tablet == "all":
        results = await renderer.render_all_tablets(num_views=args.num_views)
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
        )
        print(f"\n✓ {len(images)} images → {args.output_dir}/tablet_{args.tablet}/")


if __name__ == "__main__":
    asyncio.run(main())
