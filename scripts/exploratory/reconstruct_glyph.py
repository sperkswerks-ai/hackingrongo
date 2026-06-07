"""
EXPLORATORY — speculative / tangential analysis; not part of the reproducible analysis pipeline.

Reconstruct a damaged or masked rongorongo glyph via the trained autoencoder.

The "fill the gap" game: supply a glyph image and tell the model which region
is missing or damaged; the autoencoder fills it in from the learned latent space.
With --knn K the script also decodes the mean of the K nearest embeddings in the
training corpus as an independent second opinion.

Usage
-----
    python scripts/reconstruct_glyph.py \\
        --image data/glyphs/H_021_200.png \\
        --mask 0.25,0.25,0.5,0.5 \\
        --output outputs/reconstruction/

    python scripts/reconstruct_glyph.py \\
        --image data/glyphs/H_021_200.png \\
        --mask-ratio 0.35 \\
        --knn 8 \\
        --output outputs/reconstruction/

Flags
-----
    --image PATH         Input glyph PNG (grayscale).
    --mask X0,Y0,W,H     Rectangular mask as fractions of image width/height, all in [0,1].
                         Example: 0.1,0.1,0.8,0.4 → top 40% of the image.
    --mask-ratio FLOAT   Apply a centred square mask covering this fraction of the image.
                         Ignored when --mask is given.
    --mask-fill FLOAT    Value written into the masked region in [-1, 1] space.
                         Default 0.0 (neutral gray after normalisation).
    --checkpoint PATH    Autoencoder .pt checkpoint.  Auto-discovers latest if omitted.
    --config PATH        Path to config.yaml.  Defaults to conf/config.yaml.
    --knn K              Also decode the average of the K nearest-neighbour embeddings
                         from the corpus.  K=0 disables (default).
    --embeddings PATH    Embeddings cache (.pt).  Auto-discovers if omitted (required
                         when --knn > 0).
    --output DIR         Directory for output files.  Default: outputs/reconstruction/.
    --prefix STR         Filename prefix.  Defaults to the input image stem.

Outputs
-------
    {prefix}_reconstruction.png   — horizontal strip: original | masked | decoded | error
                                    (plus knn | knn_error columns when --knn > 0)
    {prefix}_metrics.json         — MSE and SSIM for full image and masked region
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from hackingrongo.zone_a.autoencoder import ConvAutoencoder, _ssim_loss


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_config(path: Path):
    return OmegaConf.load(path)


def _load_model(ckpt_path: Path, cfg, device: torch.device) -> ConvAutoencoder:
    model = ConvAutoencoder(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _autodiscover_checkpoint(checkpoints_dir: Path) -> Path | None:
    candidates = sorted(checkpoints_dir.glob("autoencoder_epoch*.pt"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_glyph_image(image_path: Path, cfg) -> torch.Tensor:
    """Load a PNG, normalise to [-1, 1], return shape (1, C, H, W)."""
    img_size = int(cfg.glyph.image_size)
    channels = int(cfg.glyph.image_channels)

    img = Image.open(image_path)
    img = img.convert("L") if channels == 1 else img.convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0   # [0,1] → [-1,1]
    if channels == 1:
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (1, C, H, W) tensor in [-1, 1] to a grayscale or RGB PIL image."""
    arr = t.squeeze(0).detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]                                   # (H, W)
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")


def error_map_pil(original: torch.Tensor, recon: torch.Tensor) -> Image.Image:
    """Absolute-error heatmap: high error = red, low = dark."""
    err = (original - recon).abs().squeeze(0).detach().cpu().numpy()
    if err.ndim == 3:
        err = err.mean(0)                              # (H, W)
    err = (err / max(err.max(), 1e-6) * 255).astype(np.uint8)
    rgb = np.zeros((*err.shape, 3), dtype=np.uint8)
    rgb[..., 0] = err                                 # red channel only
    return Image.fromarray(rgb, mode="RGB")


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(
    x: torch.Tensor,
    mask_spec: str | None,
    mask_ratio: float | None,
    fill: float,
) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
    """Apply a rectangular mask; return (masked_tensor, (y0, x0, h, w) in pixels)."""
    _, _C, H, W = x.shape
    out = x.clone()

    if mask_spec is not None:
        try:
            parts = [float(v) for v in mask_spec.split(",")]
            if len(parts) != 4:
                raise ValueError(f"expected 4 values, got {len(parts)}")
            x0_f, y0_f, w_f, h_f = parts
        except ValueError as exc:
            raise SystemExit(
                f"--mask must be x0,y0,w,h as four floats in [0,1] (e.g. 0.1,0.1,0.5,0.5): {exc}"
            ) from exc
        x0 = int(x0_f * W)
        y0 = int(y0_f * H)
        mw = max(1, int(w_f * W))
        mh = max(1, int(h_f * H))
    elif mask_ratio is not None:
        side = mask_ratio ** 0.5
        mh = max(1, int(side * H))
        mw = max(1, int(side * W))
        y0 = (H - mh) // 2
        x0 = (W - mw) // 2
    else:
        return out, None

    y0 = max(0, min(y0, H - 1))
    x0 = max(0, min(x0, W - 1))
    mh = min(mh, H - y0)
    mw = min(mw, W - x0)
    out[0, :, y0:y0 + mh, x0:x0 + mw] = fill
    return out, (y0, x0, mh, mw)


# ---------------------------------------------------------------------------
# KNN reconstruction
# ---------------------------------------------------------------------------

def knn_reconstruct(
    z_query: torch.Tensor,
    embeddings_path: Path,
    k: int,
    model: ConvAutoencoder,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Average the K nearest corpus embeddings and decode.

    Returns (decoded_image, list_of_neighbor_barthel_codes).
    """
    cache = torch.load(embeddings_path, map_location=device, weights_only=True)
    emb: torch.Tensor = cache["embeddings"].to(device)      # (N, D)
    codes: list[str] = cache["barthel_codes"]

    z_norm = F.normalize(z_query, dim=1)                    # (1, D)
    emb_norm = F.normalize(emb, dim=1)                      # (N, D)
    sims = (emb_norm @ z_norm.T).squeeze(1)                 # (N,)

    top_k = sims.topk(min(k, len(sims))).indices            # (k,)
    z_knn = emb[top_k].mean(dim=0, keepdim=True)            # (1, D)

    with torch.no_grad():
        x_knn = model.decoder(z_knn)

    return x_knn, [codes[i] for i in top_k.cpu().tolist()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    original: torch.Tensor,
    recon: torch.Tensor,
    mask_rect: tuple[int, int, int, int] | None,
    prefix: str = "",
) -> dict:
    """Return MSE and SSIM (full image + masked region if rect given)."""
    p = f"{prefix}_" if prefix else ""
    out: dict = {}
    out[f"{p}mse_full"]  = round(F.mse_loss(recon, original).item(), 6)
    out[f"{p}ssim_full"] = round(1.0 - float(_ssim_loss(recon, original)), 4)

    if mask_rect is not None:
        y0, x0, mh, mw = mask_rect
        # SSIM needs at least an 11×11 region; fall back to MSE-only for smaller.
        if min(mh, mw) >= 11:
            orig_c = original[:, :, y0:y0 + mh, x0:x0 + mw]
            recon_c = recon[:, :, y0:y0 + mh, x0:x0 + mw]
            out[f"{p}mse_masked"]  = round(F.mse_loss(recon_c, orig_c).item(), 6)
            out[f"{p}ssim_masked"] = round(1.0 - float(_ssim_loss(recon_c, orig_c)), 4)

    return out


# ---------------------------------------------------------------------------
# Composite strip image
# ---------------------------------------------------------------------------

def save_strip(panels: list[tuple[str, Image.Image]], out_path: Path) -> None:
    """Save a horizontal strip with panel labels in the gutter."""
    label_h = 18
    w = panels[0][1].width
    h = panels[0][1].height
    canvas = Image.new("RGB", (w * len(panels), h + label_h), color=(15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    for i, (label, img) in enumerate(panels):
        canvas.paste(img.convert("RGB"), (i * w, label_h))
        draw.text((i * w + 3, 2), label, fill=(190, 190, 190))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill-the-gap: reconstruct a masked rongorongo glyph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image",      required=True,  help="Input glyph PNG.")
    p.add_argument("--mask",       default=None,
                   help="x0,y0,w,h as fractions in [0,1].  E.g. 0.1,0.1,0.8,0.4")
    p.add_argument("--mask-ratio", type=float, default=None,
                   help="Centred square mask covering this fraction of the image.")
    p.add_argument("--mask-fill",  type=float, default=0.0,
                   help="Fill value in [-1,1] (default 0.0 = neutral gray).")
    p.add_argument("--checkpoint", default=None,
                   help="Autoencoder checkpoint .pt.  Auto-discovers latest if omitted.")
    p.add_argument("--config",     default=None,
                   help="Path to config.yaml.  Default: conf/config.yaml.")
    p.add_argument("--knn",        type=int, default=0,
                   help="Blend K nearest-neighbour embeddings and decode.  0 = off.")
    p.add_argument("--embeddings", default=None,
                   help="Embeddings cache .pt (required when --knn > 0).")
    p.add_argument("--output",     default=None,
                   help="Output directory.  Default: outputs/reconstruction/.")
    p.add_argument("--prefix",     default=None,
                   help="Output filename prefix.  Default: input image stem.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "conf" / "config.yaml"
    cfg = _load_config(cfg_path)

    checkpoints_dir = PROJECT_ROOT / cfg.paths.checkpoints_dir
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _autodiscover_checkpoint(checkpoints_dir)
        if ckpt_path is None:
            sys.exit(
                "No checkpoint found in outputs/checkpoints/.  "
                "Run train_autoencoder.py first or pass --checkpoint."
            )
        print(f"Checkpoint: {ckpt_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, cfg, device)

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    original = load_glyph_image(image_path, cfg).to(device)
    masked, mask_rect = apply_mask(original, args.mask, args.mask_ratio, args.mask_fill)

    with torch.no_grad():
        z, x_hat = model(masked)

    metrics: dict = {
        "image":      str(image_path),
        "checkpoint": str(ckpt_path),
    }
    if mask_rect:
        metrics["mask_rect"] = list(mask_rect)
    metrics.update(compute_metrics(original, x_hat, mask_rect))

    panels: list[tuple[str, Image.Image]] = [
        ("original", tensor_to_pil(original)),
        ("masked",   tensor_to_pil(masked)),
        ("decoded",  tensor_to_pil(x_hat)),
        ("error",    error_map_pil(original, x_hat)),
    ]

    if args.knn > 0:
        emb_path = (
            Path(args.embeddings) if args.embeddings
            else PROJECT_ROOT / cfg.paths.embeddings_cache
        )
        if not emb_path.exists():
            print(f"WARNING: embeddings cache not found at {emb_path} — skipping KNN.")
        else:
            x_knn, neighbors = knn_reconstruct(z, emb_path, args.knn, model, device)
            knn_m = compute_metrics(original, x_knn, mask_rect, prefix="knn")
            metrics.update(knn_m)
            metrics["knn_neighbors"] = neighbors
            panels.append(("knn",     tensor_to_pil(x_knn)))
            panels.append(("knn_err", error_map_pil(original, x_knn)))

    out_dir = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "reconstruction"
    prefix = args.prefix or image_path.stem

    strip_path = out_dir / f"{prefix}_reconstruction.png"
    save_strip(panels, strip_path)

    metrics_path = out_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Strip:   {strip_path}")
    print(f"Metrics: {metrics_path}")
    print()
    for k, v in metrics.items():
        if k not in ("image", "checkpoint", "knn_neighbors", "mask_rect"):
            print(f"  {k}: {v}")
    if "knn_neighbors" in metrics:
        print(f"  knn_neighbors: {metrics['knn_neighbors']}")


if __name__ == "__main__":
    main()
