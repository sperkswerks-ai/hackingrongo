"""
Train the Zone A convolutional autoencoder and save an embeddings cache.

Usage (local)
-------------
    conda run -n hackingrongo python scripts/train_autoencoder.py
    conda run -n hackingrongo python scripts/train_autoencoder.py \\
        zone_a.autoencoder.num_epochs=100 \\
        zone_a.autoencoder.batch_size=32

Usage (Colab — via subprocess)
-------------------------------
    subprocess.run([
        sys.executable, "scripts/train_autoencoder.py",
        "paths.glyphs_dir=/content/repo/data/glyphs",
        "paths.corpus_dir=/content/repo/data/corpus",
        "paths.catalog_dir=/content/repo/data/catalog",
        "paths.checkpoints_dir=/content/drive/MyDrive/hackingrongo_checkpoints",
        "paths.embeddings_cache=/content/drive/MyDrive/hackingrongo_checkpoints/embeddings_cache.pt",
        "zone_a.autoencoder.num_epochs=50",
        "zone_a.autoencoder.batch_size=64",
    ], cwd="/content/repo")

All training hyperparameters live in conf/config.yaml; override on the
command line using Hydra syntax (``key=value``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — makes `hackingrongo` importable when this file is run
# directly as a script (python scripts/train_autoencoder.py) even if the
# package was not installed via pip.  When installed, this insert is a no-op.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from hackingrongo.data.corpus import load_corpus  # noqa: E402
from hackingrongo.data.dataset import GlyphImageDataset  # noqa: E402
from hackingrongo.zone_a.autoencoder import (  # noqa: E402
    ConvAutoencoder,
    build_optimizer,
    build_scheduler,
    extract_embeddings,
    train_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Full autoencoder training loop with checkpointing."""
    import hydra.utils as hu

    project_root = Path(hu.get_original_cwd())
    ae_cfg = cfg.zone_a.autoencoder

    # ── Paths ────────────────────────────────────────────────────────────────
    glyphs_dir = project_root / cfg.paths.glyphs_dir
    checkpoints_dir = project_root / cfg.paths.checkpoints_dir
    embeddings_cache = project_root / cfg.paths.embeddings_cache
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    embeddings_cache.parent.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Loading corpus from %s", project_root / cfg.paths.corpus_dir)
    tablets = load_corpus(cfg, project_root)
    all_tokens = [tok for tablet in tablets for tok in tablet.tokens]
    log.info("Corpus: %d tablets, %d tokens", len(tablets), len(all_tokens))

    dataset = GlyphImageDataset(all_tokens, glyphs_dir, cfg, training=True)
    loader = DataLoader(
        dataset,
        batch_size=int(ae_cfg.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ConvAutoencoder(cfg).to(device)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)

    num_epochs = int(ae_cfg.num_epochs)
    checkpoint_interval = int(ae_cfg.checkpoint_interval_epochs)

    # ── Checkpoint resume ────────────────────────────────────────────────────
    start_epoch = 0
    resume_setting = str(ae_cfg.get("resume_checkpoint", "auto")).strip().lower()
    if resume_setting not in ("", "none", "false", "no"):
        if resume_setting == "auto":
            existing = sorted(checkpoints_dir.glob("autoencoder_epoch*.pt"))
            ckpt_path = existing[-1] if existing else None
        else:
            ckpt_path = project_root / resume_setting
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"resume_checkpoint path not found: {ckpt_path}"
                )

        if ckpt_path is not None:
            log.info("Resuming from checkpoint: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = int(ckpt["epoch"])
                log.info("Resumed: starting from epoch %d / %d", start_epoch, num_epochs)
                # Fast-forward the LR scheduler to match the resumed epoch so the
                # cosine / step schedule continues from the right position.
                if scheduler is not None:
                    for _ in range(start_epoch):
                        scheduler.step()
            except RuntimeError as exc:
                log.warning(
                    "Checkpoint %s is incompatible with the current model "
                    "architecture (%s). Training from scratch.",
                    ckpt_path.name, exc,
                )
                start_epoch = 0
        else:
            log.info(
                "resume_checkpoint=auto but no checkpoint found in %s — "
                "starting from scratch.",
                checkpoints_dir,
            )
    else:
        log.info("resume_checkpoint=%r — starting from scratch.", resume_setting)

    if start_epoch >= num_epochs:
        log.info(
            "Already trained for %d / %d epochs — skipping training loop.",
            start_epoch, num_epochs,
        )
    else:
        log.info("Training for epochs %d → %d", start_epoch + 1, num_epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        loss = train_epoch(model, loader, optimizer, cfg, device, epoch=epoch)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            ckpt_path = checkpoints_dir / f"autoencoder_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                ckpt_path,
            )
            log.info("Checkpoint saved: %s", ckpt_path)

    # ── Embeddings cache ─────────────────────────────────────────────────────
    log.info("Extracting embeddings for all %d tokens …", len(dataset))
    eval_dataset = GlyphImageDataset(all_tokens, glyphs_dir, cfg, training=False)
    emb_dict = extract_embeddings(model, eval_dataset, cfg, device)

    # Stack into (N, D) tensor in token order and collect barthel codes.
    vecs = np.stack(
        [emb_dict[(t.tablet_id, t.position)] for t in eval_dataset.tokens]
    ).astype(np.float32)
    barthel_codes = [str(t.barthel_code) for t in eval_dataset.tokens]
    torch.save(
        {
            "embeddings": torch.from_numpy(vecs),
            "barthel_codes": barthel_codes,
        },
        embeddings_cache,
    )
    log.info("Embeddings cache saved: %s  (%d entries)", embeddings_cache, len(vecs))


if __name__ == "__main__":
    main()
