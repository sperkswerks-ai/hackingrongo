"""
hackingrongo.zone_a.autoencoder
==========================

Convolutional autoencoder for self-supervised glyph image representation
learning (Zone A, first branch).

Public API
----------
``SharedConvBackbone``
    Configurable stack of conv blocks whose parameters are tied across
    all three Zone A networks (autoencoder, Siamese, sequence model).

``ConvEncoder``
    Backbone + flatten + linear projection to ``bottleneck_dim``.

``ConvDecoder``
    Reverse of the encoder: linear projection + spatial reshape +
    configurable upsample blocks (bilinear | nearest | conv_transpose).

``ConvAutoencoder``
    Full encode-reconstruct model.  Exposes ``self.backbone`` for
    parameter tying.

``build_optimizer`` / ``build_scheduler``
    Factory functions that read directly from the Hydra config.

``train_epoch``
    Runs one full training epoch; returns mean reconstruction loss.

``extract_embeddings``
    Embeds every token in a ``GlyphImageDataset``, returning a dict
    keyed by ``(tablet_id, position)``.

All hyperparameters are read from ``cfg``; no literals appear in this
module.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hackingrongo.data.dataset import GlyphImageDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def build_activation(name: str) -> nn.Module:
    """Instantiate an activation module by the name string used in config.

    Parameters
    ----------
    name : str
        Activation name: ``"relu"`` | ``"leaky_relu"`` | ``"gelu"``.

    Returns
    -------
    nn.Module
        New activation module instance.

    Raises
    ------
    ValueError
        If ``name`` is not in the supported registry.
    """
    key = name.lower()
    if key not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {sorted(_ACTIVATION_REGISTRY)}."
        )
    return _ACTIVATION_REGISTRY[key]()


# ---------------------------------------------------------------------------
# Spatial-size helper
# ---------------------------------------------------------------------------


def _encoder_spatial_size(cfg: DictConfig) -> int:
    """Compute the spatial (H = W) dimension of the encoder feature map.

    With ``image_size = 64``, three MaxPool(2) operations → 8.
    Returns 1 when a pretrained (global-token) backbone is configured,
    because :class:`PretrainedBackbone` outputs a ``(B, D, 1, 1)`` tensor.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.glyph.image_size``,
        ``cfg.zone_a.shared_backbone.pool_kernel_size``, and the length
        of ``cfg.zone_a.shared_backbone.conv_channels``.

    Returns
    -------
    int
        Spatial side length of the encoder's final feature map.
    """
    # Top-level zone_a.backbone = "dinov2" shortcut — global CLS token, no spatial map.
    top_backbone = str(cfg.zone_a.get("backbone", "custom")).lower()
    if top_backbone == "dinov2":
        return 1
    bb = cfg.zone_a.shared_backbone
    pretrained = str(bb.get("pretrained_backbone", "null")).lower()
    if pretrained not in ("", "null", "none"):
        return 1  # pretrained backbone yields a global token — no spatial map
    n_pools: int = len(list(bb.conv_channels))
    pool_k: int = int(bb.pool_kernel_size)
    image_size: int = int(cfg.glyph.image_size)
    return image_size // (pool_k ** n_pools)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Single encoder convolutional block.

    Structure: ``Conv2d → [BatchNorm2d] → Activation → [Dropout2d] → MaxPool2d``.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_size : int
        Convolutional kernel size (same for H and W).
    padding : int
        Zero-padding added to both sides of each spatial dimension.
    pool_kernel_size : int
        Kernel size (and stride) for the max-pooling step.
    use_batch_norm : bool
        If ``True``, insert BatchNorm2d after the convolution.  When
        batch norm is active, the conv bias is disabled (redundant).
    activation_name : str
        Activation function name passed to :func:`build_activation`.
    dropout_rate : float
        Spatial dropout probability.  ``0.0`` disables the dropout
        layer entirely (no-op layer is not inserted).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        pool_kernel_size: int,
        use_batch_norm: bool,
        activation_name: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(build_activation(activation_name))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the conv block to ``x``."""
        return self.block(x)


class SharedConvBackbone(nn.Module):
    """Configurable stack of :class:`ConvBlock` modules shared across Zone A.

    Output shape: ``(B, conv_channels[-1], H // pool^n, W // pool^n)``
    where ``n = len(conv_channels)``.

    This module is constructed once and its reference passed to all three
    Zone A networks so that their early convolutional parameters are tied
    (identical weight objects, not copies).

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_a.shared_backbone`` and
        ``cfg.glyph.image_channels``.
    image_channels : int
        Number of channels in the input images (1 for grayscale).

    Attributes
    ----------
    out_channels : int
        Channel depth of the backbone output, equal to
        ``conv_channels[-1]``.
    """

    def __init__(self, cfg: DictConfig, image_channels: int) -> None:
        super().__init__()
        bb = cfg.zone_a.shared_backbone
        conv_channels: list[int] = list(bb.conv_channels)

        blocks: list[nn.Module] = []
        in_ch = image_channels
        for out_ch in conv_channels:
            blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=int(bb.conv_kernel_size),
                    padding=int(bb.conv_padding),
                    pool_kernel_size=int(bb.pool_kernel_size),
                    use_batch_norm=bool(bb.use_batch_norm),
                    activation_name=str(bb.activation),
                    dropout_rate=float(bb.dropout_rate),
                )
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.out_channels: int = conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all conv blocks over the input image batch."""
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Pretrained backbone (DINOv2 / EfficientNet)
# ---------------------------------------------------------------------------

class PretrainedBackbone(nn.Module):
    """Drop-in replacement for :class:`SharedConvBackbone` using a pretrained model.

    Wraps a DINOv2 ViT (loaded via ``torch.hub``) or an EfficientNet backbone
    (loaded via ``torchvision.models``) as the shared Zone A backbone.  All
    pretrained weights are frozen by default; a lightweight trainable adapter
    projects the pretrained feature dimension to ``out_channels``.

    The module produces a 4-D output tensor ``(B, out_channels, 1, 1)`` so it
    is API-compatible with :class:`SharedConvBackbone` when combined with
    :class:`ConvEncoder` (which flattens the spatial dimensions before the
    linear projection, yielding ``flat_dim = out_channels * 1 * 1``).

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads
        ``cfg.zone_a.shared_backbone.pretrained_backbone`` (the model key)
        and ``cfg.zone_a.shared_backbone.freeze_pretrained_backbone``.
    image_channels : int
        Number of input image channels (1 for grayscale glyphs).

    Attributes
    ----------
    out_channels : int
        Feature dimension of the adapter output.  Matches the last value
        of ``cfg.zone_a.shared_backbone.conv_channels`` so that downstream
        code is unchanged.
    """

    # name → (hub_repo_or_None, hub_model, pretrained_feat_dim, needs_rgb)
    _CONFIGS: dict[str, tuple[str | None, str, int, bool]] = {
        "dinov2_small": ("facebookresearch/dinov2", "dinov2_vits14", 384, True),
        "dinov2_base":  ("facebookresearch/dinov2", "dinov2_vitb14", 768, True),
        "efficientnet_b0": (None, "efficientnet_b0", 1280, True),
    }

    def __init__(self, cfg: DictConfig, image_channels: int) -> None:
        super().__init__()
        bb = cfg.zone_a.shared_backbone
        backbone_key = str(bb.pretrained_backbone).lower()
        freeze: bool = bool(bb.get("freeze_pretrained_backbone", True))
        # Match the out_channels that SharedConvBackbone would produce.
        target_ch: int = list(bb.conv_channels)[-1]

        if backbone_key not in self._CONFIGS:
            raise ValueError(
                f"Unknown pretrained_backbone '{backbone_key}'. "
                f"Supported: {sorted(self._CONFIGS)}."
            )

        hub_repo, hub_model, feat_dim, needs_rgb = self._CONFIGS[backbone_key]
        self._needs_rgb: bool = needs_rgb and (image_channels == 1)
        self._is_vit: bool = backbone_key.startswith("dinov2")

        if self._is_vit:
            encoder = torch.hub.load(hub_repo, hub_model, pretrained=True, verbose=False)
        else:
            import torchvision.models as tvm  # type: ignore
            weights_enum_name = hub_model.replace("_", "").capitalize() + "_Weights"
            weights_cls = getattr(tvm, weights_enum_name, None)
            encoder = getattr(tvm, hub_model)(
                weights=(weights_cls.DEFAULT if weights_cls is not None else None)
            )
            # Strip the classifier head; keep only the feature extractor.
            encoder = torch.nn.Sequential(*list(encoder.children())[:-2])

        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)

        self.encoder = encoder
        # Adapter: project pretrained feat_dim → target_ch
        self.adapter_pool = torch.nn.AdaptiveAvgPool2d(1)  # for EfficientNet
        self.proj = torch.nn.Linear(feat_dim, target_ch, bias=False)
        self.out_channels: int = target_ch

        logger.info(
            "PretrainedBackbone: %s  feat_dim=%d → out_channels=%d  freeze=%s",
            backbone_key, feat_dim, target_ch, freeze,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and return shape ``(B, out_channels, 1, 1)``."""
        if self._needs_rgb:
            x = x.expand(-1, 3, -1, -1)  # grayscale → pseudo-RGB
        if self._is_vit:
            # DINOv2: forward_features returns a dict; use the CLS token.
            feat = self.encoder.forward_features(x)["x_norm_clstoken"]  # (B, feat_dim)
        else:
            # EfficientNet: produces spatial feature map → global average pool.
            h = self.encoder(x)             # (B, feat_dim, H', W')
            feat = self.adapter_pool(h).flatten(start_dim=1)  # (B, feat_dim)
        out = self.proj(feat)               # (B, target_ch)
        return out.unsqueeze(-1).unsqueeze(-1)  # (B, target_ch, 1, 1)


class DINOv2GlyphEncoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 backbone for glyph embedding.

    Config-free standalone class for scripts that do not use the full
    Hydra config tree (e.g. ``cross_script_similarity.py``).  Backbone
    weights are frozen by default; only the lightweight projection head
    is trainable.  For use inside the main training pipeline see
    :class:`PretrainedBackbone` and the ``zone_a.backbone`` config key.

    Parameters
    ----------
    latent_dim : int
        Output embedding dimension (after projection).
    dinov2_model : str
        Torch-hub model identifier: ``"dinov2_vits14"`` (384-d, default)
        or ``"dinov2_vitb14"`` (768-d).
    freeze_backbone : bool
        If True, backbone parameters receive no gradients.
    """

    _FEAT_DIMS: dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
    }

    def __init__(
        self,
        latent_dim: int = 128,
        dinov2_model: str = "dinov2_vits14",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            dinov2_model,
            pretrained=True,
            verbose=False,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        feat_dim = self._FEAT_DIMS.get(dinov2_model, 384)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.latent_dim = latent_dim

        logger.info(
            "DINOv2GlyphEncoder: %s  feat_dim=%d → latent_dim=%d  frozen=%s",
            dinov2_model, feat_dim, latent_dim, freeze_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of glyph images.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, H, W)`` (grayscale) or ``(B, 3, H, W)`` (RGB).
            Any spatial size; resized to 224 × 224 if needed.

        Returns
        -------
        torch.Tensor
            Shape ``(B, latent_dim)``.
        """
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.backbone.forward_features(x)["x_norm_clstoken"]
        return self.projection(features)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings suitable for cosine similarity."""
        return F.normalize(self.forward(x), dim=1)


def build_backbone(
    cfg: DictConfig,
    image_channels: int,
) -> "SharedConvBackbone | PretrainedBackbone":
    """Factory: return a :class:`SharedConvBackbone` or :class:`PretrainedBackbone`.

    Resolution order:

    1. ``cfg.zone_a.backbone`` — new top-level shortcut:
       * ``"dinov2"`` → :class:`PretrainedBackbone` with DINOv2 ViT-S/14.
         Uses ``cfg.zone_a.dinov2_model`` and ``cfg.zone_a.freeze_backbone``.
       * ``"custom"`` or absent → fall through to (2).
    2. ``cfg.zone_a.shared_backbone.pretrained_backbone`` — legacy path:
       * non-null → :class:`PretrainedBackbone`.
       * null / none / "" → :class:`SharedConvBackbone`.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.
    image_channels : int
        Number of image channels (1 for grayscale).

    Returns
    -------
    SharedConvBackbone or PretrainedBackbone
    """
    from omegaconf import OmegaConf

    # ── 1. Top-level zone_a.backbone shortcut ────────────────────────────────
    top_backbone = str(cfg.zone_a.get("backbone", "custom")).lower()
    if top_backbone == "dinov2":
        dinov2_model = str(cfg.zone_a.get("dinov2_model", "dinov2_vits14")).lower()
        freeze = bool(cfg.zone_a.get("freeze_backbone", True))
        # Map user-facing model name to the key PretrainedBackbone._CONFIGS expects
        _MODEL_MAP = {
            "dinov2_vits14": "dinov2_small",
            "dinov2_vitb14": "dinov2_base",
        }
        pretrained_key = _MODEL_MAP.get(dinov2_model, "dinov2_small")
        logger.info(
            "build_backbone: zone_a.backbone='dinov2' → '%s'  freeze=%s",
            pretrained_key, freeze,
        )
        cfg2 = OmegaConf.merge(cfg, OmegaConf.create({
            "zone_a": {"shared_backbone": {
                "pretrained_backbone": pretrained_key,
                "freeze_pretrained_backbone": freeze,
            }}
        }))
        return PretrainedBackbone(cfg2, image_channels)

    # ── 2. Legacy shared_backbone.pretrained_backbone path ───────────────────
    bb = cfg.zone_a.shared_backbone
    pretrained = str(bb.get("pretrained_backbone", "null")).lower()
    if pretrained not in ("", "null", "none"):
        return PretrainedBackbone(cfg, image_channels)
    return SharedConvBackbone(cfg, image_channels)


class ConvEncoder(nn.Module):
    """Encoder: SharedConvBackbone → flatten → [two-stage Linear] → bottleneck.

    Parameters
    ----------
    backbone : SharedConvBackbone
        Shared conv backbone (parameter-tied with other Zone A modules).
    bottleneck_dim : int
        Dimensionality of the output embedding vector.
    spatial_size : int
        Spatial side length of the backbone output feature map, as
        computed by :func:`_encoder_spatial_size`.
    intermediate_dim : int, optional
        If > 0, inserts a two-stage projection
        ``flat → BatchNorm1d → ReLU → intermediate → bottleneck``
        for additional representational capacity.  ``0`` (default) uses
        a single ``Linear(flat, bottleneck)``.
    """

    def __init__(
        self,
        backbone: SharedConvBackbone,
        bottleneck_dim: int,
        spatial_size: int,
        intermediate_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        flat_dim = backbone.out_channels * spatial_size * spatial_size
        if intermediate_dim > 0:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(flat_dim, intermediate_dim, bias=False),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, bottleneck_dim),
            )
        else:
            self.proj = nn.Linear(flat_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to embedding vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor, shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embedding tensor, shape ``(B, bottleneck_dim)``.
        """
        h = self.backbone(x)        # (B, C_last, S, S)
        h = h.flatten(start_dim=1)  # (B, C_last * S * S)
        return self.proj(h)         # (B, bottleneck_dim)


class ConvDecoder(nn.Module):
    """Decoder: bottleneck → Linear → spatial reshape → upsample blocks.

    Reverses the encoder's channel and spatial progression.  The final
    block applies ``Tanh`` to produce outputs in ``[-1, 1]``, matching
    the normalised input range.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.zone_a.shared_backbone``,
        ``cfg.zone_a.autoencoder``, and ``cfg.glyph``.
    bottleneck_dim : int
        Input embedding dimension (must equal encoder's bottleneck).
    spatial_size : int
        Spatial side length to reshape to after the linear projection.
    image_channels : int
        Number of channels in the reconstructed output image.
    """

    def __init__(
        self,
        cfg: DictConfig,
        bottleneck_dim: int,
        spatial_size: int,
        image_channels: int,
    ) -> None:
        super().__init__()
        bb = cfg.zone_a.shared_backbone
        ae = cfg.zone_a.autoencoder

        conv_channels: list[int] = list(bb.conv_channels)
        upsample_mode: str = str(ae.decoder_upsample_mode)
        use_batch_norm: bool = bool(bb.use_batch_norm)
        activation_name: str = str(bb.activation)
        align_corners: bool | None = (
            bool(ae.get("align_corners", False)) if upsample_mode == "bilinear" else None
        )
        pool_k: int = int(bb.pool_kernel_size)
        kernel_size: int = int(bb.conv_kernel_size)
        padding: int = int(bb.conv_padding)
        last_enc_channels: int = conv_channels[-1]

        self._last_enc_channels = last_enc_channels
        self._spatial_size = spatial_size

        flat_dim = last_enc_channels * spatial_size * spatial_size
        intermediate_dim: int = int(ae.get("encoder_intermediate_dim", 0))
        if intermediate_dim > 0:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(bottleneck_dim, intermediate_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, flat_dim),
            )
        else:
            self.proj = nn.Linear(bottleneck_dim, flat_dim)

        # Channel transitions: reversed encoder channels, final out = image_channels.
        dec_in: list[int] = list(reversed(conv_channels))
        dec_out: list[int] = list(reversed(conv_channels))[1:] + [image_channels]

        decoder_blocks: list[nn.Module] = []
        for i, (in_ch, out_ch) in enumerate(zip(dec_in, dec_out)):
            is_last = i == len(dec_in) - 1
            decoder_blocks.append(
                self._make_decoder_block(
                    in_ch, out_ch, kernel_size, padding, pool_k,
                    upsample_mode, use_batch_norm, activation_name,
                    align_corners, is_last,
                )
            )
        self.blocks = nn.ModuleList(decoder_blocks)

    @staticmethod
    def _make_decoder_block(
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int,
        pool_k: int,
        upsample_mode: str,
        use_batch_norm: bool,
        activation_name: str,
        align_corners: bool | None,
        is_last: bool,
    ) -> nn.Sequential:
        """Construct one decoder block.

        For ``"conv_transpose"`` mode: ``ConvTranspose2d``.
        For ``"bilinear"`` / ``"nearest"``: ``Upsample + Conv2d``.
        All non-final blocks end with BN + activation; the final block
        ends with ``Tanh``.
        """
        layers: list[nn.Module] = []

        if upsample_mode == "conv_transpose":
            if pool_k % 2 != 0:
                raise ValueError(
                    f"decoder_upsample_mode='conv_transpose' requires an even "
                    f"pool_kernel_size; got pool_k={pool_k}. "
                    "Use 'bilinear' or 'nearest' for odd pool sizes."
                )
            # kernel = 2*pool_k, stride = pool_k gives exact 2× upsampling.
            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=pool_k * 2,
                    stride=pool_k,
                    padding=pool_k // 2,
                    output_padding=0,
                    bias=not use_batch_norm,
                )
            )
        else:
            upsample_kwargs: dict[str, Any] = {
                "scale_factor": pool_k,
                "mode": upsample_mode,
            }
            if upsample_mode == "bilinear":
                upsample_kwargs["align_corners"] = align_corners
            layers.append(nn.Upsample(**upsample_kwargs))
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            )

        if not is_last:
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(build_activation(activation_name))
        else:
            # Final block: Tanh maps output to [-1, 1] to match normalised input.
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode an embedding batch to image reconstructions.

        Parameters
        ----------
        z : torch.Tensor
            Embedding tensor, shape ``(B, bottleneck_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor, shape ``(B, C, H, W)`` with
            values in ``[-1, 1]``.
        """
        h = self.proj(z)
        h = h.view(
            h.size(0),
            self._last_enc_channels,
            self._spatial_size,
            self._spatial_size,
        )
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------


def _gaussian_kernel(
    window_size: int, sigma: float, channels: int
) -> torch.Tensor:
    """Build a normalised 2D Gaussian kernel for SSIM computation.

    Parameters
    ----------
    window_size : int
        Side length of the square kernel (odd recommended).
    sigma : float
        Standard deviation of the Gaussian.
    channels : int
        Number of image channels; the kernel is expanded to
        shape ``(channels, 1, window_size, window_size)`` for use with
        ``F.conv2d(..., groups=channels)``.

    Returns
    -------
    torch.Tensor
        Kernel tensor of shape ``(channels, 1, window_size, window_size)``.
    """
    if sigma <= 0.0:
        raise ValueError(f"_gaussian_kernel: sigma must be > 0, got {sigma}")
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


# Cache keyed by (window_size, sigma, channels) to avoid recomputing every forward pass.
_SSIM_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Differentiable single-scale SSIM loss: ``1 − mean_SSIM``.

    Parameters
    ----------
    pred : torch.Tensor
        Reconstructed image batch, shape ``(B, C, H, W)``, values in
        ``[-1, 1]``.
    target : torch.Tensor
        Ground-truth image batch, same shape and range.
    window_size : int
        Size of the Gaussian smoothing kernel.
    sigma : float
        Gaussian standard deviation.
    data_range : float
        Dynamic range of the input; ``2.0`` for ``[-1, 1]`` images.
    k1, k2 : float
        SSIM stability constants per Wang et al. (2004).

    Returns
    -------
    torch.Tensor
        Scalar loss value ``∈ [0, 2]``; ``0`` means perfect structural
        similarity.
    """
    channels = pred.size(1)
    _cache_key = (window_size, sigma, channels, str(pred.device), str(pred.dtype))
    if _cache_key not in _SSIM_KERNEL_CACHE:
        _SSIM_KERNEL_CACHE[_cache_key] = (
            _gaussian_kernel(window_size, sigma, channels)
            .to(device=pred.device, dtype=pred.dtype)
        )
    kernel = _SSIM_KERNEL_CACHE[_cache_key]

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator
    return 1.0 - ssim_map.mean()


def _ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_sizes: tuple[int, ...] = (5, 9, 11),
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Multi-scale SSIM loss: mean of single-scale losses at multiple window sizes.

    Averaging over several window sizes captures both fine texture similarity
    (small window) and coarse structural similarity (large window), making the
    loss more robust to the stroke-vs-background trade-off in glyph images.

    Parameters
    ----------
    pred : torch.Tensor
        Reconstructed image batch, shape ``(B, C, H, W)``, values in ``[-1, 1]``.
    target : torch.Tensor
        Ground-truth image batch, same shape and range.
    window_sizes : tuple[int, ...]
        Gaussian window sizes evaluated; must all be smaller than image H and W.
    sigma, data_range, k1, k2 :
        Forwarded to :func:`_ssim_loss`; see its docstring for semantics.

    Returns
    -------
    torch.Tensor
        Scalar loss ``∈ [0, 2]``; lower is better.
    """
    losses = torch.stack(
        [_ssim_loss(pred, target, ws, sigma, data_range, k1, k2) for ws in window_sizes]
    )
    return losses.mean()


# ---------------------------------------------------------------------------
# Anthropomorphic head-type metric learning
# ---------------------------------------------------------------------------


def _anthropomorphic_head_type_group(code: str) -> int | None:
    """Return the head-type group label for a Barthel code in 200–399.

    The label is ``n // 10``, so signs 200–209 → 20, 210–219 → 21, …,
    390–399 → 39.  Signs outside 200–399 return ``None``.
    """
    digits = "".join(c for c in code if c.isdigit())
    if not digits:
        return None
    n = int(digits)
    return n // 10 if 200 <= n <= 399 else None


def anthropomorphic_head_loss(
    embeddings: torch.Tensor,
    barthel_codes: list[str],
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon) over Barthel head-type groups.

    Restricts to signs in the 200–399 range (Barthel's anthropomorphic
    blocks).  Signs sharing the same tens-group digit within their century
    (``n // 10``) are treated as positives; all others are negatives.

    For each anchor *i* with positive set *P(i)*:

    .. math::

        L_i = \\log\\sum_{j \\ne i} e^{s_{ij}/\\tau}
              - \\frac{1}{|P(i)|} \\sum_{p \\in P(i)} s_{ip}/\\tau

    where :math:`s_{ij} = z_i^\\top z_j` on L2-normalised embeddings.
    Only anchors that have at least one positive contribute to the mean.

    Returns ``0.0`` (as a differentiable scalar) when the batch contains
    fewer than two distinct head-type groups among 200–399 signs.

    Parameters
    ----------
    embeddings : torch.Tensor
        Shape ``(B, D)``; **not** required to be normalised on entry.
    barthel_codes : list[str]
        Parallel Barthel code string for each row in ``embeddings``.
    temperature : float
        NT-Xent / SupCon temperature τ.  Lower = sharper contrast.

    Returns
    -------
    torch.Tensor
        Scalar loss (gradient-connected to ``embeddings``).

    References
    ----------
    Khosla, P. et al. (2020). Supervised Contrastive Learning.
    *Advances in Neural Information Processing Systems*, 33.
    """
    # Collect anthropomorphic indices and their group labels.
    indices: list[int] = []
    labels: list[int] = []
    for i, code in enumerate(barthel_codes):
        g = _anthropomorphic_head_type_group(code)
        if g is not None:
            indices.append(i)
            labels.append(g)

    # Need ≥ 2 distinct groups to form any positive pair.
    if len(set(labels)) < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings[indices], dim=1)  # (M, D), unit-norm
    M = z.size(0)
    device = z.device

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)  # (M,)

    # Cosine similarity matrix scaled by temperature.
    sim = (z @ z.T) / temperature  # (M, M)

    # Diagonal mask (self-similarity excluded from denominator and numerator).
    eye = torch.eye(M, dtype=torch.bool, device=device)
    sim_no_diag = sim.masked_fill(eye, -1e9)

    # Log-partition over all non-self entries (denominator).
    log_denom = torch.logsumexp(sim_no_diag, dim=1)  # (M,)

    # Positive mask: same label, not self.
    pos_mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1)) & ~eye  # (M, M)

    # Anchors with at least one positive.
    has_pos = pos_mask.any(dim=1)  # (M,)
    if not has_pos.any():
        return embeddings.sum() * 0.0

    # Mean similarity over positives for each anchor.
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)       # (M,)
    mean_pos_sim = (sim * pos_mask.float()).sum(dim=1) / n_pos  # (M,)

    # SupCon loss per anchor; average only over anchors with ≥ 1 positive.
    loss_per_anchor = log_denom - mean_pos_sim  # (M,)
    return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# Reconstruction loss dispatch
# ---------------------------------------------------------------------------


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    ssim_weight: float = 0.5,
) -> torch.Tensor:
    """Compute the reconstruction loss between predicted and target images.

    Parameters
    ----------
    pred : torch.Tensor
        Reconstructed image batch, shape ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth image batch, same shape.
    loss_type : str
        Loss function name from config:
        ``"mse"`` | ``"l1"`` | ``"ssim"`` | ``"mixed"``.
    ssim_weight : float
        For ``loss_type="mixed"``: weight of the MS-SSIM term.  The MSE
        weight is ``1 - ssim_weight``.  Ignored for other loss types.

    Returns
    -------
    torch.Tensor
        Scalar loss value.

    Raises
    ------
    ValueError
        If ``loss_type`` is not one of the four supported options.
    """
    lt = loss_type.lower()
    if lt == "mse":
        return F.mse_loss(pred, target)
    if lt == "l1":
        return F.l1_loss(pred, target)
    if lt == "ssim":
        return _ms_ssim_loss(pred, target)
    if lt == "mixed":
        mse = F.mse_loss(pred, target)
        ssim = _ms_ssim_loss(pred, target)
        return (1.0 - ssim_weight) * mse + ssim_weight * ssim
    raise ValueError(
        f"Unsupported reconstruction_loss '{loss_type}'. "
        "Choose mse | l1 | ssim | mixed."
    )


# ---------------------------------------------------------------------------
# Contrastive projection head
# ---------------------------------------------------------------------------


class MLPProjectionHead(nn.Module):
    """Two-layer MLP projection head for supervised contrastive learning.

    Per Khosla et al. (SupCon, 2020) and Chen et al. (SimCLR, 2020), applying
    the contrastive loss on a *separate* non-linear projection head rather than
    directly on the bottleneck allows the encoder to learn a representation
    jointly optimised for reconstruction while the head learns the metric space.
    The projection head is used only during training and discarded at inference.

    Architecture: ``Linear(in_dim, hidden_dim, bias=False) → BatchNorm1d → ReLU
    → Linear(hidden_dim, out_dim, bias=False) → L2-normalise``

    Parameters
    ----------
    in_dim : int
        Input dimension (= ``bottleneck_dim``).
    hidden_dim : int
        Hidden layer width.
    out_dim : int
        Output embedding dimension on which the contrastive loss is computed.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise the input embedding batch."""
        return F.normalize(self.net(z), dim=1)


# ---------------------------------------------------------------------------
# ConvAutoencoder
# ---------------------------------------------------------------------------


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for self-supervised glyph representation.

    Architecture::

        Encoder: SharedConvBackbone → flatten → [intermediate →] ReLU → bottleneck
        Decoder: bottleneck → [intermediate →] ReLU → spatial → reshape → upsample → Tanh
        ContrastiveHead: bottleneck → BN → ReLU → proj_out → L2-norm  (training only)

    The backbone is accessible via ``self.backbone`` for parameter tying
    with the Siamese network and sequence model.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.zone_a.shared_backbone``,
        ``cfg.zone_a.autoencoder``, and ``cfg.glyph``.
    backbone : SharedConvBackbone, optional
        Pre-constructed backbone whose weights should be shared with
        other Zone A modules.  If ``None``, a fresh backbone is
        instantiated from ``cfg``.

    Attributes
    ----------
    backbone : SharedConvBackbone
        The shared convolutional backbone.
    bottleneck_dim : int
        Dimensionality of each glyph embedding vector.
    """

    def __init__(
        self,
        cfg: DictConfig,
        backbone: SharedConvBackbone | None = None,
    ) -> None:
        super().__init__()
        ae_cfg = cfg.zone_a.autoencoder
        image_channels: int = int(cfg.glyph.image_channels)
        bottleneck_dim: int = int(ae_cfg.bottleneck_dim)
        spatial_size: int = _encoder_spatial_size(cfg)

        self.backbone: SharedConvBackbone | PretrainedBackbone = (
            backbone
            if backbone is not None
            else build_backbone(cfg, image_channels)
        )
        intermediate_dim: int = int(ae_cfg.get("encoder_intermediate_dim", 0))
        self.encoder = ConvEncoder(
            self.backbone, bottleneck_dim, spatial_size, intermediate_dim
        )
        self.decoder = ConvDecoder(cfg, bottleneck_dim, spatial_size, image_channels)
        proj_hidden: int = int(ae_cfg.get("proj_head_hidden_dim", bottleneck_dim))
        proj_out: int = int(ae_cfg.get("proj_head_out_dim", max(bottleneck_dim // 2, 32)))
        self.proj_head = MLPProjectionHead(bottleneck_dim, proj_hidden, proj_out)
        self._loss_type: str = str(ae_cfg.reconstruction_loss)
        self._ssim_weight: float = float(ae_cfg.get("mixed_loss_ssim_weight", 0.5))
        self.bottleneck_dim: int = bottleneck_dim

        logger.debug(
            "ConvAutoencoder: bottleneck_dim=%d, spatial_size=%d, intermediate_dim=%d, "
            "loss=%s (ssim_weight=%.2f), upsample=%s.",
            bottleneck_dim,
            spatial_size,
            intermediate_dim,
            ae_cfg.reconstruction_loss,
            self._ssim_weight,
            ae_cfg.decoder_upsample_mode,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and reconstruct an image batch.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor, shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(embedding, reconstruction)`` where:

            * ``embedding``: shape ``(B, bottleneck_dim)``
            * ``reconstruction``: shape ``(B, C, H, W)``, values in
              ``[-1, 1]``
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch to embedding vectors (no reconstruction).

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor, shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embedding tensor, shape ``(B, bottleneck_dim)``.
        """
        return self.encoder(x)

    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and L2-normalise to the unit hypersphere.

        Use this when downstream consumers rely on cosine similarity
        (UMAP with ``metric='cosine'``, compound detector geometry).
        The SupCon projection head also normalises, so these two spaces
        are consistent in direction though not identical.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor, shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Unit-norm embedding tensor, shape ``(B, bottleneck_dim)``.
        """
        return F.normalize(self.encoder(x), dim=1)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the contrastive projection head to bottleneck embeddings.

        The projection head is a 2-layer MLP with L2-normalised output,
        used only during training for the SupCon auxiliary loss.  Calling
        this at inference time is valid but the output is not the primary
        embedding representation — use :meth:`encode` or
        :meth:`encode_normalized` instead.

        Parameters
        ----------
        z : torch.Tensor
            Bottleneck embedding tensor, shape ``(B, bottleneck_dim)``.

        Returns
        -------
        torch.Tensor
            L2-normalised projection, shape ``(B, proj_head_out_dim)``.
        """
        return self.proj_head(z)

    def reconstruct_masked(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a batch with masked (occluded) regions inpainted.

        The masked input regions are zeroed before encoding so the decoder
        must infer their content from visible context — the MAE / spatial
        inpainting inference mode.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch, shape ``(B, C, H, W)``, values in ``[-1, 1]``.
        mask : torch.Tensor
            Boolean or float mask broadcastable to ``(B, C, H, W)``.
            ``True`` / ``1`` = masked (occluded); ``False`` / ``0`` = visible.

        Returns
        -------
        torch.Tensor
            Reconstructed image batch, shape ``(B, C, H, W)``, values in
            ``[-1, 1]``.  Visible regions closely match the input; masked
            regions are inpainted by the decoder.
        """
        mask_f = mask.float().to(x.device)
        x_masked = x * (1.0 - mask_f)
        z = self.encoder(x_masked)
        return self.decoder(z)

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        """Compute the configured reconstruction loss.

        Parameters
        ----------
        x : torch.Tensor
            Original image tensor, shape ``(B, C, H, W)``.
        x_hat : torch.Tensor
            Reconstructed image tensor, same shape.

        Returns
        -------
        torch.Tensor
            Scalar loss value computed by :func:`reconstruction_loss`.
        """
        return reconstruction_loss(x_hat, x, self._loss_type, self._ssim_weight)


# ---------------------------------------------------------------------------
# Optimizer and scheduler builders
# ---------------------------------------------------------------------------


def build_optimizer(
    params: Any,
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Build the autoencoder optimizer from config.

    Parameters
    ----------
    params : iterable
        Model parameters (e.g. ``model.parameters()``).
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_a.autoencoder.optimizer``,
        ``cfg.zone_a.autoencoder.lr``, and
        ``cfg.zone_a.autoencoder.weight_decay``.

    Returns
    -------
    torch.optim.Optimizer

    Raises
    ------
    ValueError
        If the optimizer name in config is not supported.
    """
    ae_cfg = cfg.zone_a.autoencoder
    name = str(ae_cfg.optimizer).lower()
    lr = float(ae_cfg.lr)
    wd = float(ae_cfg.weight_decay)

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(
        f"Unsupported optimizer '{ae_cfg.optimizer}'. "
        "Choose adam | adamw | sgd."
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    cosine_T_max: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Cosine annealing preceded by a linear warmup phase.

    Uses ``SequentialLR``: ``LinearLR`` ramps from near-zero to full LR
    over ``warmup_epochs``, then ``CosineAnnealingLR`` decays for the
    remaining ``cosine_T_max`` epochs.  Starting with a tiny LR prevents
    unstable gradient steps in the first few epochs when the randomly
    initialised decoder receives large reconstruction errors.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to wrap.
    warmup_epochs : int
        Number of epochs for the linear ramp (``start_factor=1e-3 → 1.0``).
    cosine_T_max : int
        Epochs for the cosine annealing phase (typically
        ``num_epochs - warmup_epochs``).

    Returns
    -------
    torch.optim.lr_scheduler.SequentialLR
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(warmup_epochs, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cosine_T_max, 1),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the learning rate scheduler from config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to wrap.
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_a.autoencoder.scheduler``
        and related fields.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or None
        ``None`` if ``cfg.zone_a.autoencoder.scheduler == "none"``.

    Raises
    ------
    ValueError
        If the scheduler name is not supported.
    """
    ae_cfg = cfg.zone_a.autoencoder
    name = str(ae_cfg.scheduler).lower()

    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(ae_cfg.scheduler_T_max)
        )
    if name == "cosine_warmup":
        warmup_epochs = int(ae_cfg.get("warmup_epochs", 5))
        total_epochs = int(ae_cfg.num_epochs)
        cosine_T_max = max(total_epochs - warmup_epochs, 1)
        return build_warmup_cosine_scheduler(optimizer, warmup_epochs, cosine_T_max)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(ae_cfg.scheduler_step_size),
            gamma=float(ae_cfg.scheduler_gamma),
        )
    raise ValueError(
        f"Unsupported scheduler '{ae_cfg.scheduler}'. "
        "Choose cosine | cosine_warmup | step | none."
    )


# ---------------------------------------------------------------------------
# Spatial mask corruption (inpainting training)
# ---------------------------------------------------------------------------


def _apply_random_masks(
    x: torch.Tensor,
    mask_prob: float,
    ratio_min: float,
    ratio_max: float,
    fill: float,
) -> torch.Tensor:
    """Apply independent random rectangular occlusions to a batch of images.

    Each image is masked with probability *mask_prob*.  The mask rectangle
    covers a fraction of total pixels sampled uniformly from
    [ratio_min, ratio_max]; its top-left corner is chosen uniformly at random.
    The encoder sees the corrupted image; reconstruction loss is still measured
    against the original clean target (in ``train_epoch``).

    Parameters
    ----------
    x : torch.Tensor
        Batch of images, shape ``(B, C, H, W)``, values in ``[-1, 1]``.
    mask_prob : float
        Per-image masking probability ``∈ [0, 1]``.
    ratio_min : float
        Minimum fraction of ``H × W`` to occlude.
    ratio_max : float
        Maximum fraction of ``H × W`` to occlude.
    fill : float
        Scalar fill value placed inside the masked rectangle.

    Returns
    -------
    torch.Tensor
        Corrupted batch, same shape and dtype as ``x`` (clone, not in-place).
    """
    B, _C, H, W = x.shape
    out = x.clone()
    for b in range(B):
        if torch.rand(1).item() >= mask_prob:
            continue
        ratio = ratio_min + torch.rand(1).item() * (ratio_max - ratio_min)
        # Square-root keeps the mask roughly square while hitting the target area.
        mh = max(1, int(ratio ** 0.5 * H))
        mw = max(1, int(ratio ** 0.5 * W))
        y0 = int(torch.randint(0, max(1, H - mh + 1), (1,)).item())
        x0 = int(torch.randint(0, max(1, W - mw + 1), (1,)).item())
        out[b, :, y0:y0 + mh, x0:x0 + mw] = fill
    return out


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


def train_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    epoch: int = 0,
) -> float:
    """Run one full training epoch for the convolutional autoencoder.

    Parameters
    ----------
    model : ConvAutoencoder
        The autoencoder.  Set to ``train()`` mode internally.
    loader : torch.utils.data.DataLoader
        DataLoader over a ``GlyphImageDataset`` (or any dataset whose
        batches expose an ``"image"`` key with tensors of shape
        ``(B, C, H, W)``).
    optimizer : torch.optim.Optimizer
        Optimizer already constructed for ``model.parameters()``.
    cfg : DictConfig
        Root Hydra config.  Reads
        ``cfg.zone_a.autoencoder.grad_clip_norm`` and
        ``cfg.zone_a.autoencoder.log_interval_steps``.
    device : torch.device
        Computation device.  Tensors are moved per batch.
    epoch : int, optional
        0-based epoch index used for log messages only.

    Returns
    -------
    float
        Mean reconstruction loss across all batches in this epoch.

    Notes
    -----
    * Gradient clipping is applied when ``grad_clip_norm > 0.0``.
    * Checkpointing is the caller's responsibility (``pipeline.py``).
    * The scheduler step (if any) must also be called by the caller
      after ``train_epoch`` returns.
    """
    ae_cfg = cfg.zone_a.autoencoder
    grad_clip: float = float(ae_cfg.grad_clip_norm)
    log_interval: int = int(ae_cfg.log_interval_steps)
    head_loss_weight: float = float(ae_cfg.get("anthropomorphic_head_loss_weight", 0.1))
    head_temperature: float = float(ae_cfg.get("anthropomorphic_head_temperature", 0.1))
    denoising_std: float = float(ae_cfg.get("denoising_noise_std", 0.0))
    mask_prob: float = float(ae_cfg.get("denoising_mask_prob", 0.0))
    mask_ratio_min: float = float(ae_cfg.get("denoising_mask_ratio_min", 0.10))
    mask_ratio_max: float = float(ae_cfg.get("denoising_mask_ratio_max", 0.50))
    mask_fill: float = float(ae_cfg.get("denoising_mask_fill", 0.0))

    model.train()
    model.to(device)

    total_loss = 0.0
    total_recon_loss = 0.0
    total_head_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        images: torch.Tensor = batch["image"].to(device)
        codes: list[str] = list(batch["barthel_code"])

        optimizer.zero_grad()

        # Denoising: corrupt encoder input, reconstruct against clean target.
        # Two independent corruption types compose: Gaussian noise + spatial masks.
        if denoising_std > 0.0:
            encoder_input = (
                images + denoising_std * torch.randn_like(images)
            ).clamp(-1.0, 1.0)
        else:
            encoder_input = images
        if mask_prob > 0.0:
            encoder_input = _apply_random_masks(
                encoder_input, mask_prob, mask_ratio_min, mask_ratio_max, mask_fill
            )

        z, reconstructions = model(encoder_input)
        recon_loss = model.loss(images, reconstructions)

        if head_loss_weight > 0.0:
            # Apply the contrastive loss on the non-linear projection head output
            # rather than directly on the bottleneck, following Khosla et al. (2020).
            # The bottleneck remains free to optimise for reconstruction quality.
            z_proj = model.project(z)
            head_loss = anthropomorphic_head_loss(z_proj, codes, head_temperature)
            loss = recon_loss + head_loss_weight * head_loss
            total_head_loss += head_loss.item()
        else:
            loss = recon_loss

        loss.backward()

        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        total_recon_loss += recon_loss.item()
        n_batches += 1

        if (step + 1) % log_interval == 0:
            if head_loss_weight > 0.0:
                logger.info(
                    "Epoch %d  step %d/%d  loss=%.6f  (recon=%.6f  head=%.6f)",
                    epoch, step + 1, len(loader), batch_loss,
                    recon_loss.item(),
                    total_head_loss / n_batches,
                )
            else:
                logger.info(
                    "Epoch %d  step %d/%d  loss=%.6f",
                    epoch, step + 1, len(loader), batch_loss,
                )

    mean_loss = total_loss / max(n_batches, 1)
    mean_recon = total_recon_loss / max(n_batches, 1)
    mean_head = total_head_loss / max(n_batches, 1)
    if head_loss_weight > 0.0:
        logger.info(
            "Epoch %d complete  loss=%.6f  recon=%.6f  head=%.6f",
            epoch, mean_loss, mean_recon, mean_head,
        )
    else:
        logger.info("Epoch %d complete  mean_loss=%.6f", epoch, mean_loss)
    return mean_loss


# ---------------------------------------------------------------------------
# extract_embeddings
# ---------------------------------------------------------------------------


def extract_embeddings(
    model: ConvAutoencoder,
    dataset: GlyphImageDataset,
    cfg: DictConfig,
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    """Extract encoder embeddings for every glyph token in a dataset.

    Parameters
    ----------
    model : ConvAutoencoder
        Trained (or partially-trained) autoencoder.  Temporarily set to
        ``eval()`` mode; original training/eval state is restored on
        return.
    dataset : GlyphImageDataset
        Dataset whose full token list will be embedded.
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.zone_a.autoencoder.batch_size``
        as the inference batch size.
    device : torch.device
        Computation device.

    Returns
    -------
    dict[tuple[str, int], numpy.ndarray]
        Maps ``(tablet_id, position)`` → 1-D float32 NumPy array of
        length ``cfg.zone_a.autoencoder.bottleneck_dim``.

    Notes
    -----
    * ``shuffle=False`` and ``drop_last=False`` ensure every token is
      embedded exactly once.
    * No gradients are computed during this call.
    * This function is Zone-A-only; downstream Zone B code consumes the
      returned dict directly or via the embeddings cache written by
      ``pipeline.py``.
    """
    batch_size: int = int(cfg.zone_a.autoencoder.batch_size)
    was_training = model.training
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    embeddings: dict[tuple[str, int], np.ndarray] = {}

    with torch.no_grad():
        for batch in loader:
            images: torch.Tensor = batch["image"].to(device)
            tablet_ids: list[str] = list(batch["tablet_id"])
            positions: list[int] = [int(p) for p in batch["position"]]

            z: torch.Tensor = model.encode(images)
            z_np: np.ndarray = z.cpu().numpy().astype(np.float32)

            for i, (tid, pos) in enumerate(zip(tablet_ids, positions)):
                embeddings[(tid, pos)] = z_np[i]

    if was_training:
        model.train()

    logger.info("extract_embeddings: embedded %d glyph(s).", len(embeddings))
    return embeddings
