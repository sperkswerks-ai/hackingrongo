"""
scripts/train_sequential_embeddings.py
=======================================

Train a 2-layer transformer as a contrastive learner over rongorongo sign
context windows.  Positive pairs are two context windows drawn from
different occurrences of the **same** Barthel sign code; negative pairs are
windows from different signs (in-batch negatives via NT-Xent).

The resulting per-sign embeddings capture sequential behaviour rather than
visual appearance.  A companion metric, **sequential entropy**, is computed
directly from corpus bigrams: H(X_{i+1} | X_i = S) for each sign S.

Outputs
-------
outputs/sequential_embeddings.pt
    torch.save'd dict with keys:
        "sign_codes"      list[str]  — ordered Barthel codes
        "embeddings"      Tensor[N, D]  — mean context-window embedding per sign
        "vocab"           dict[str, int]  — sign-code → integer ID
        "sign_entropy"    dict[str, float]  — sequential entropy per sign (nats)
        "context_window"  int  — context half-width (= 4)
        "embedding_dim"   int  — D

outputs/sequential_entropy.json
    JSON: { "sign_code": entropy_value, ... }  (nats, float)
    Low  → structural sign (stereotyped behaviour)
    High → phonemic sign (appears in many different contexts)

Usage
-----
    python scripts/train_sequential_embeddings.py
    python scripts/train_sequential_embeddings.py \\
        --corpus-dir data/corpus \\
        --epochs 30 \\
        --batch-size 256 \\
        --emb-dim 128 \\
        --output outputs/sequential_embeddings.pt

Dependency on the rest of the pipeline
---------------------------------------
After running this script, update Zone B classification by passing
``sequential_entropy`` into ``classify_inventory``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_HALF = 4          # signs on each side → window length = 2*CONTEXT_HALF + 1 = 9
PAD_TOKEN = "<PAD>"
TEMPERATURE = 0.07        # NT-Xent temperature (Chen et al. 2020)

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_sequences(corpus_dir: Path) -> list[list[str]]:
    """Load all tablet JSON files; return one flat list of Barthel code sequences.

    Each tablet is represented as a single sequence of sign codes.  Tablets
    are loaded in sorted filename order.
    """
    sequences: list[list[str]] = []
    for p in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", p.name, exc)
            continue
        glyphs = data.get("glyphs", [])
        if not glyphs:
            continue
        # Sort by position (1-based ordinal)
        glyphs_sorted = sorted(glyphs, key=lambda g: g.get("position", 0))
        seq = [g["barthel_code"] for g in glyphs_sorted if "barthel_code" in g]
        if seq:
            sequences.append(seq)
    log.info("Loaded %d tablet sequences from %s", len(sequences), corpus_dir)
    return sequences


def _build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """Map sign codes → integer IDs; reserve 0 for PAD."""
    codes: set[str] = set()
    for seq in sequences:
        codes.update(seq)
    vocab = {PAD_TOKEN: 0}
    for code in sorted(codes):
        vocab[code] = len(vocab)
    return vocab


def _extract_windows(
    sequences: list[list[str]],
    vocab: dict[str, int],
    context_half: int = CONTEXT_HALF,
) -> dict[str, list[list[int]]]:
    """For each sign occurrence extract an integer-encoded context window.

    Returns
    -------
    dict[str, list[list[int]]]
        Maps Barthel code → list of windows.  Each window is a list of
        2*context_half + 1 integer token IDs (PAD=0 at boundaries).
    """
    pad_id = vocab[PAD_TOKEN]
    windows_by_code: dict[str, list[list[int]]] = defaultdict(list)
    w = context_half

    for seq in sequences:
        ids = [vocab.get(c, pad_id) for c in seq]
        n = len(ids)
        for i, code in enumerate(seq):
            left = [ids[j] if j >= 0 else pad_id for j in range(i - w, i)]
            right = [ids[j] if j < n else pad_id for j in range(i + 1, i + w + 1)]
            window = left + [ids[i]] + right  # length = 2w+1
            windows_by_code[code].append(window)

    return windows_by_code

# ---------------------------------------------------------------------------
# Sequential entropy (bigram-based, independent of the transformer)
# ---------------------------------------------------------------------------


def compute_sequential_entropy(sequences: list[list[str]]) -> dict[str, float]:
    """Compute H(X_{i+1} | X_i = S) for each sign S.

    Parameters
    ----------
    sequences : list[list[str]]
        One sequence per tablet in corpus order.

    Returns
    -------
    dict[str, float]
        Maps Barthel code → conditional entropy (nats).
        Signs that never appear before another sign get entropy 0.
    """
    # Bigram counts: next_sign_counts[S][T] = #{(S,T) bigrams}
    next_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for i in range(len(seq) - 1):
            next_counts[seq[i]][seq[i + 1]] += 1

    entropy: dict[str, float] = {}
    for sign, successors in next_counts.items():
        total = sum(successors.values())
        if total == 0:
            entropy[sign] = 0.0
            continue
        h = 0.0
        for count in successors.values():
            p = count / total
            h -= p * math.log(p)
        entropy[sign] = h

    # Signs that only appear at the end of sequences get 0.
    all_signs: set[str] = set()
    for seq in sequences:
        all_signs.update(seq)
    for s in all_signs:
        if s not in entropy:
            entropy[s] = 0.0

    return entropy

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ContrastiveWindowDataset(Dataset):
    """Each item is a positive pair (anchor_window, positive_window, sign_id).

    For each sign code that appears ≥ 2 times, we sample 2 distinct windows
    without replacement.  Signs with only 1 occurrence are skipped.
    """

    def __init__(
        self,
        windows_by_code: dict[str, list[list[int]]],
        code_to_id: dict[str, int],
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        self._pairs: list[tuple[list[int], list[int], int]] = []

        eligible = {
            code: wins
            for code, wins in windows_by_code.items()
            if len(wins) >= 2
        }
        if not eligible:
            raise ValueError(
                "No sign codes with ≥ 2 context windows found. "
                "Corpus may be too small."
            )

        for code, wins in eligible.items():
            i, j = self._rng.sample(range(len(wins)), 2)
            self._pairs.append((wins[i], wins[j], code_to_id[code]))

        log.info(
            "ContrastiveWindowDataset: %d positive pairs from %d sign types",
            len(self._pairs),
            len(eligible),
        )

    def resample(self) -> None:
        """Re-draw pairs each epoch to avoid overfitting to a fixed partition."""
        self._pairs = [
            (
                self._rng.choice(self._pairs_by_id.get(sid, [a])),
                self._rng.choice(self._pairs_by_id.get(sid, [p])),
                sid,
            )
            for a, p, sid in self._pairs
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        anchor, positive, sign_id = self._pairs[idx]
        return (
            torch.tensor(anchor, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            sign_id,
        )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SequentialEncoder(nn.Module):
    """2-layer transformer encoder over a context window of sign IDs.

    Architecture
    ------------
    Token embedding  → positional encoding  → 2× TransformerEncoderLayer
    → mean-pool all positions  → projection head  → L2-normalised embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        emb_dim: int = 128,
        window_len: int = 2 * CONTEXT_HALF + 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(window_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for training stability on small data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Projection head: linear → ReLU → linear (following SimCLR)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of context windows.

        Parameters
        ----------
        x : Tensor[B, L]
            Integer token IDs.

        Returns
        -------
        Tensor[B, emb_dim]
            L2-normalised embeddings.
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        tok = self.token_emb(x)          # [B, L, d_model]
        pos = self.pos_emb(positions)    # [B, L, d_model]
        h = tok + pos

        # Mask PAD tokens from attention
        pad_mask = (x == 0)             # True where PAD
        h = self.transformer(h, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        # Mean-pool over non-PAD positions
        non_pad = (~pad_mask).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (h * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)

        z = self.proj(pooled)            # [B, emb_dim]
        return F.normalize(z, dim=-1)

# ---------------------------------------------------------------------------
# NT-Xent loss (SimCLR)
# ---------------------------------------------------------------------------


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """NT-Xent (normalised temperature-scaled cross-entropy) contrastive loss.

    Parameters
    ----------
    z_i, z_j : Tensor[B, D]
        L2-normalised embeddings of anchor and positive views.
    temperature : float

    Returns
    -------
    Tensor (scalar)
    """
    B = z_i.size(0)
    # Concatenate both views: [2B, D]
    z = torch.cat([z_i, z_j], dim=0)
    # Similarity matrix: [2B, 2B]
    sim = torch.mm(z, z.t()) / temperature
    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive indices: for row i ∈ [0,B), positive is i+B; for i ∈ [B,2B), it's i-B
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)

# ---------------------------------------------------------------------------
# Epoch resampler: rebuild pairs at each epoch
# ---------------------------------------------------------------------------


def _build_pairs_epoch(
    windows_by_code: dict[str, list[list[int]]],
    code_to_id: dict[str, int],
    rng: random.Random,
) -> list[tuple[list[int], list[int], int]]:
    pairs = []
    for code, wins in windows_by_code.items():
        if len(wins) < 2:
            continue
        i, j = rng.sample(range(len(wins)), 2)
        pairs.append((wins[i], wins[j], code_to_id[code]))
    return pairs


def _collate(batch):
    anchors, positives, sign_ids = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.tensor(sign_ids, dtype=torch.long),
    )

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: SequentialEncoder,
    windows_by_code: dict[str, list[list[int]]],
    code_to_id: dict[str, int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> None:
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )
    rng = random.Random(seed)
    model.train()

    for epoch in range(1, epochs + 1):
        pairs = _build_pairs_epoch(windows_by_code, code_to_id, rng)
        if not pairs:
            log.error("No eligible sign types for contrastive training.")
            break

        # Shuffle pairs
        rng.shuffle(pairs)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if len(batch) < 2:
                continue
            anchors_t = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            positives_t = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

            z_i = model(anchors_t)
            z_j = model(positives_t)
            loss = nt_xent_loss(z_i, z_j)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  loss=%.4f  lr=%.2e", epoch, epochs, avg,
                     scheduler.get_last_lr()[0])

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_sign_embeddings(
    model: SequentialEncoder,
    windows_by_code: dict[str, list[list[int]]],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Return mean context-window embedding per sign code.

    For each sign code, embed all its context windows and average them.
    This gives a single representative vector per sign that reflects its
    typical sequential neighbourhood.
    """
    model.eval()
    sign_embeddings: dict[str, torch.Tensor] = {}

    for code, wins in windows_by_code.items():
        all_z: list[torch.Tensor] = []
        for start in range(0, len(wins), batch_size):
            batch = torch.tensor(wins[start : start + batch_size],
                                  dtype=torch.long, device=device)
            z = model(batch)         # [B, D] — already L2-normalised
            all_z.append(z.cpu())
        stacked = torch.cat(all_z, dim=0)   # [N, D]
        mean_z = stacked.mean(dim=0)
        mean_z = F.normalize(mean_z, dim=0)
        sign_embeddings[code] = mean_z

    return sign_embeddings

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train contrastive sequential sign embeddings for rongorongo"
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory of per-tablet JSON corpus files.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Pairs per training batch (default: 256).",
    )
    p.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="Contrastive embedding dimension (default: 128).",
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Transformer hidden size (default: 64).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW (default: 3e-4).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sequential_embeddings.pt"),
        help="Path for the saved embeddings .pt file.",
    )
    p.add_argument(
        "--entropy-output",
        type=Path,
        default=Path("outputs/sequential_entropy.json"),
        help="Path for the per-sign entropy JSON file.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------ corpus
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        # Try relative to script location (running from project root)
        corpus_dir = Path(__file__).resolve().parents[1] / args.corpus_dir
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")

    sequences = _load_sequences(corpus_dir)
    if not sequences:
        raise ValueError(f"No valid corpus files found in {corpus_dir}")

    total_tokens = sum(len(s) for s in sequences)
    log.info("Total tokens: %d across %d tablets", total_tokens, len(sequences))

    # ------------------------------------------------------------------ vocab
    vocab = _build_vocab(sequences)
    vocab_size = len(vocab)
    log.info("Vocabulary size: %d sign types", vocab_size - 1)  # -1 for PAD

    # code → int id for loss label (not same as vocab id — we want contiguous IDs)
    all_codes = sorted(c for c in vocab if c != PAD_TOKEN)
    code_to_id: dict[str, int] = {code: i for i, code in enumerate(all_codes)}

    # ------------------------------------------------------------------ windows
    windows_by_code = _extract_windows(sequences, vocab, CONTEXT_HALF)
    n_eligible = sum(1 for w in windows_by_code.values() if len(w) >= 2)
    log.info(
        "Signs with ≥2 context windows (eligible for contrastive training): %d / %d",
        n_eligible,
        len(windows_by_code),
    )

    # ------------------------------------------------------------------ entropy
    log.info("Computing sequential entropy (bigram-based)…")
    seq_entropy = compute_sequential_entropy(sequences)
    log.info(
        "Sequential entropy  min=%.3f  max=%.3f  mean=%.3f  (nats)",
        min(seq_entropy.values()),
        max(seq_entropy.values()),
        sum(seq_entropy.values()) / max(len(seq_entropy), 1),
    )

    # ------------------------------------------------------------------ model
    window_len = 2 * CONTEXT_HALF + 1
    model = SequentialEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=max(1, args.d_model // 16),   # nhead = d_model/16, must divide d_model
        dim_feedforward=args.d_model * 4,
        dropout=0.1,
        emb_dim=args.emb_dim,
        window_len=window_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    # ------------------------------------------------------------------ train
    if n_eligible < 2:
        log.warning(
            "Only %d sign types have ≥2 windows — contrastive training skipped. "
            "Embeddings will be random projections.",
            n_eligible,
        )
    else:
        log.info("Training for %d epochs…", args.epochs)
        train(
            model,
            windows_by_code,
            code_to_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )

    # ------------------------------------------------------------------ extract
    log.info("Extracting per-sign embeddings…")
    sign_embeddings = extract_sign_embeddings(model, windows_by_code, device)

    # Stack into a single tensor in sorted sign-code order
    sign_codes = sorted(sign_embeddings.keys())
    emb_matrix = torch.stack([sign_embeddings[c] for c in sign_codes])  # [N, D]

    # ------------------------------------------------------------------ save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sign_codes": sign_codes,
        "embeddings": emb_matrix,
        "vocab": vocab,
        "sign_entropy": seq_entropy,
        "context_window": CONTEXT_HALF,
        "embedding_dim": args.emb_dim,
    }
    torch.save(payload, args.output)
    log.info("Embeddings saved → %s  (%d signs, dim=%d)", args.output, len(sign_codes), args.emb_dim)

    args.entropy_output.parent.mkdir(parents=True, exist_ok=True)
    args.entropy_output.write_text(
        json.dumps(seq_entropy, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("Sequential entropy saved → %s", args.entropy_output)

    # Summary table (top 10 lowest and highest entropy signs)
    sorted_by_entropy = sorted(seq_entropy.items(), key=lambda kv: kv[1])
    log.info("── Lowest sequential entropy (structural signs) ──")
    for code, h in sorted_by_entropy[:10]:
        log.info("  %-8s  H=%.3f nats", code, h)
    log.info("── Highest sequential entropy (phonemic signs) ──")
    for code, h in sorted_by_entropy[-10:]:
        log.info("  %-8s  H=%.3f nats", code, h)


if __name__ == "__main__":
    main()
