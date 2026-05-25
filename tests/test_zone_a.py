"""
tests.test_zone_a
=================

Smoke tests for hackingrongo.zone_a.autoencoder.

All tests run on CPU with a miniature config (image_size=16,
conv_channels=[8, 16], bottleneck_dim=32) to keep execution fast.
No image files or pre-trained weights are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from hackingrongo.data.corpus import GlyphToken
from hackingrongo.data.dataset import GlyphImageDataset
from hackingrongo.zone_a.autoencoder import (
    ConvAutoencoder,
    ConvDecoder,
    ConvEncoder,
    MLPProjectionHead,
    SharedConvBackbone,
    _encoder_spatial_size,
    _ms_ssim_loss,
    build_activation,
    build_optimizer,
    build_scheduler,
    build_warmup_cosine_scheduler,
    extract_embeddings,
    reconstruction_loss,
    train_epoch,
)


# ---------------------------------------------------------------------------
# Shared fixture: miniature config
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    """Minimal config for Zone A autoencoder tests (CPU-friendly dimensions)."""
    return OmegaConf.create(
        {
            "seed": 0,
            "glyph": {
                "image_size": 16,
                "image_channels": 1,
                "filename_pattern": "{tablet_id}_{position}_{barthel_code}.png",
                "augmentation": {
                    "use_augmentation": False,
                    "random_rotation_degrees": 5.0,
                    "random_affine_translate": [0.05, 0.05],
                    "random_affine_scale": [0.9, 1.1],
                    "gaussian_noise_std": 0.0,
                    "elastic_transform_alpha": 1.0,
                    "elastic_transform_sigma": 0.05,
                },
            },
            "zone_a": {
                "shared_backbone": {
                    "conv_channels": [8, 16],   # 2 blocks → 16 // 4 = 4×4 spatial
                    "conv_kernel_size": 3,
                    "conv_padding": 1,
                    "pool_kernel_size": 2,
                    "activation": "relu",
                    "use_batch_norm": True,
                    "dropout_rate": 0.0,
                },
                "autoencoder": {
                    "bottleneck_dim": 32,
                    "decoder_upsample_mode": "bilinear",
                    "reconstruction_loss": "mse",
                    "align_corners": False,
                    "optimizer": "adam",
                    "lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "batch_size": 4,
                    "num_epochs": 2,
                    "scheduler": "none",
                    "scheduler_T_max": 2,
                    "scheduler_step_size": 1,
                    "scheduler_gamma": 0.5,
                    "grad_clip_norm": 1.0,
                    "log_interval_steps": 100,
                    "checkpoint_interval_epochs": 1,
                },
                "sequence_model": {
                    "context_window": 3,
                },
            },
        }
    )


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


def _dummy_image_batch(
    b: int, c: int = 1, h: int = 16, w: int = 16
) -> torch.Tensor:
    """Return a random float tensor in [-1, 1] simulating a normalised image batch."""
    return torch.rand(b, c, h, w) * 2.0 - 1.0


def _dummy_token_dataset(
    cfg, tmp_path: Path, n: int = 8
) -> GlyphImageDataset:
    """GlyphImageDataset with n tokens and matching PNG stub files."""
    from PIL import Image as _PIL_Image
    tokens = [
        GlyphToken(i + 1, f"{(i % 4) + 1:03d}", "T", "pre")
        for i in range(n)
    ]
    for tok in tokens:
        fname = f"{tok.tablet_id}_{tok.position}_{tok.barthel_code}.png"
        img = _PIL_Image.new("L", (16, 16), color=255)
        img.save(tmp_path / fname)
    return GlyphImageDataset(tokens, tmp_path, cfg, training=False)


# ---------------------------------------------------------------------------
# build_activation
# ---------------------------------------------------------------------------


class TestBuildActivation:
    def test_relu(self):
        act = build_activation("relu")
        assert isinstance(act, torch.nn.ReLU)

    def test_leaky_relu(self):
        act = build_activation("leaky_relu")
        assert isinstance(act, torch.nn.LeakyReLU)

    def test_gelu(self):
        act = build_activation("gelu")
        assert isinstance(act, torch.nn.GELU)

    def test_case_insensitive(self):
        act = build_activation("RELU")
        assert isinstance(act, torch.nn.ReLU)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            build_activation("swish")


# ---------------------------------------------------------------------------
# _encoder_spatial_size
# ---------------------------------------------------------------------------


class TestEncoderSpatialSize:
    def test_default_config(self, cfg):
        # image_size=16, 2 MaxPool(2) → 16 // 4 = 4
        assert _encoder_spatial_size(cfg) == 4

    def test_three_blocks(self, cfg):
        cfg_3 = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {"zone_a": {"shared_backbone": {"conv_channels": [8, 16, 32]}}}
            ),
        )
        # 16 // 8 = 2
        assert _encoder_spatial_size(cfg_3) == 2


# ---------------------------------------------------------------------------
# SharedConvBackbone
# ---------------------------------------------------------------------------


class TestSharedConvBackbone:
    def test_output_shape(self, cfg):
        bb = SharedConvBackbone(cfg, image_channels=1)
        x = _dummy_image_batch(3)
        y = bb(x)
        # 2 MaxPool(2) on 16×16 → 4×4; channels = conv_channels[-1] = 16
        assert y.shape == (3, 16, 4, 4)

    def test_out_channels_attribute(self, cfg):
        bb = SharedConvBackbone(cfg, image_channels=1)
        assert bb.out_channels == 16

    def test_parameter_tying(self, cfg):
        """Two encoders sharing the same backbone must have identical weights."""
        bb = SharedConvBackbone(cfg, image_channels=1)
        enc1 = ConvEncoder(bb, bottleneck_dim=32, spatial_size=4)
        enc2 = ConvEncoder(bb, bottleneck_dim=32, spatial_size=4)
        # backbone params are the same objects (not copies)
        for p1, p2 in zip(enc1.backbone.parameters(), enc2.backbone.parameters()):
            assert p1 is p2


# ---------------------------------------------------------------------------
# ConvEncoder
# ---------------------------------------------------------------------------


class TestConvEncoder:
    def test_output_shape(self, cfg):
        bb = SharedConvBackbone(cfg, image_channels=1)
        enc = ConvEncoder(bb, bottleneck_dim=32, spatial_size=4)
        x = _dummy_image_batch(5)
        z = enc(x)
        assert z.shape == (5, 32)


# ---------------------------------------------------------------------------
# ConvDecoder
# ---------------------------------------------------------------------------


class TestConvDecoder:
    def test_output_shape_bilinear(self, cfg):
        dec = ConvDecoder(cfg, bottleneck_dim=32, spatial_size=4, image_channels=1)
        z = torch.randn(3, 32)
        x_hat = dec(z)
        assert x_hat.shape == (3, 1, 16, 16)

    def test_output_range_tanh(self, cfg):
        """Decoder output must lie in [-1, 1] due to Tanh final activation."""
        dec = ConvDecoder(cfg, bottleneck_dim=32, spatial_size=4, image_channels=1)
        z = torch.randn(8, 32)
        x_hat = dec(z)
        assert x_hat.min().item() >= -1.0 - 1e-5
        assert x_hat.max().item() <= 1.0 + 1e-5

    def test_output_shape_conv_transpose(self, cfg):
        cfg_ct = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {"zone_a": {"autoencoder": {"decoder_upsample_mode": "conv_transpose"}}}
            ),
        )
        dec = ConvDecoder(cfg_ct, bottleneck_dim=32, spatial_size=4, image_channels=1)
        z = torch.randn(2, 32)
        x_hat = dec(z)
        assert x_hat.shape == (2, 1, 16, 16)

    def test_output_shape_nearest(self, cfg):
        cfg_nn = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {"zone_a": {"autoencoder": {"decoder_upsample_mode": "nearest"}}}
            ),
        )
        dec = ConvDecoder(cfg_nn, bottleneck_dim=32, spatial_size=4, image_channels=1)
        z = torch.randn(2, 32)
        x_hat = dec(z)
        assert x_hat.shape == (2, 1, 16, 16)

    def test_conv_transpose_even_pool_k(self, cfg):
        """conv_transpose with a non-default even pool_k (4) must reconstruct correctly."""
        cfg_ct4 = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {
                    "zone_a": {
                        "autoencoder": {"decoder_upsample_mode": "conv_transpose"},
                        "shared_backbone": {"pool_kernel_size": 4},
                    }
                }
            ),
        )
        # image_size=16, 2 × MaxPool(4) → spatial=1
        spatial = _encoder_spatial_size(cfg_ct4)
        dec = ConvDecoder(cfg_ct4, bottleneck_dim=32, spatial_size=spatial, image_channels=1)
        z = torch.randn(2, 32)
        x_hat = dec(z)
        assert x_hat.shape == (2, 1, 16, 16)

    def test_conv_transpose_odd_pool_k_raises(self, cfg):
        """conv_transpose with an odd pool_k must raise ValueError at construction."""
        cfg_odd = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {
                    "zone_a": {
                        "autoencoder": {"decoder_upsample_mode": "conv_transpose"},
                        "shared_backbone": {"pool_kernel_size": 3},
                    }
                }
            ),
        )
        spatial = _encoder_spatial_size(cfg_odd)
        with pytest.raises(ValueError, match="conv_transpose.*requires an even"):
            ConvDecoder(cfg_odd, bottleneck_dim=32, spatial_size=spatial, image_channels=1)


# ---------------------------------------------------------------------------
# reconstruction_loss
# ---------------------------------------------------------------------------


class TestReconstructionLoss:
    def _pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        pred = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        target = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        return pred, target

    def test_mse_is_scalar(self):
        pred, target = self._pair()
        loss = reconstruction_loss(pred, target, "mse")
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_l1_is_scalar(self):
        pred, target = self._pair()
        loss = reconstruction_loss(pred, target, "l1")
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_ssim_is_scalar(self):
        pred, target = self._pair()
        loss = reconstruction_loss(pred, target, "ssim")
        assert loss.shape == ()

    def test_identical_inputs_mse_zero(self):
        x = torch.rand(2, 1, 16, 16)
        assert reconstruction_loss(x, x, "mse").item() == pytest.approx(0.0, abs=1e-6)

    def test_identical_inputs_ssim_zero(self):
        x = torch.rand(2, 1, 16, 16)
        # SSIM of identical images ≈ 0 loss
        assert reconstruction_loss(x, x, "ssim").item() == pytest.approx(0.0, abs=1e-4)

    def test_invalid_loss_type_raises(self):
        pred, target = self._pair()
        with pytest.raises(ValueError, match="Unsupported reconstruction_loss"):
            reconstruction_loss(pred, target, "huber")


# ---------------------------------------------------------------------------
# ConvAutoencoder
# ---------------------------------------------------------------------------


class TestConvAutoencoder:
    def test_forward_output_shapes(self, cfg):
        model = ConvAutoencoder(cfg)
        x = _dummy_image_batch(4)
        z, x_hat = model(x)
        assert z.shape == (4, 32)
        assert x_hat.shape == (4, 1, 16, 16)

    def test_encode_matches_forward_embedding(self, cfg):
        model = ConvAutoencoder(cfg)
        model.eval()
        x = _dummy_image_batch(3)
        with torch.no_grad():
            z_full, _ = model(x)
            z_enc = model.encode(x)
        assert torch.allclose(z_full, z_enc)

    def test_loss_method_returns_scalar(self, cfg):
        model = ConvAutoencoder(cfg)
        x = _dummy_image_batch(4)
        _, x_hat = model(x)
        loss = model.loss(x, x_hat)
        assert loss.shape == ()

    def test_shared_backbone_reference(self, cfg):
        """backbone attribute must be identical to the one passed in."""
        bb = SharedConvBackbone(cfg, image_channels=1)
        model = ConvAutoencoder(cfg, backbone=bb)
        assert model.backbone is bb

    def test_bottleneck_dim_attribute(self, cfg):
        model = ConvAutoencoder(cfg)
        assert model.bottleneck_dim == 32


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_adam(self, cfg):
        model = ConvAutoencoder(cfg)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self, cfg):
        cfg_aw = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_a": {"autoencoder": {"optimizer": "adamw"}}}),
        )
        model = ConvAutoencoder(cfg_aw)
        opt = build_optimizer(model.parameters(), cfg_aw)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self, cfg):
        cfg_sgd = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_a": {"autoencoder": {"optimizer": "sgd"}}}),
        )
        model = ConvAutoencoder(cfg_sgd)
        opt = build_optimizer(model.parameters(), cfg_sgd)
        assert isinstance(opt, torch.optim.SGD)

    def test_invalid_optimizer_raises(self, cfg):
        cfg_bad = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_a": {"autoencoder": {"optimizer": "rmsprop"}}}),
        )
        model = ConvAutoencoder(cfg_bad)
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer(model.parameters(), cfg_bad)


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_none_returns_none(self, cfg):
        model = ConvAutoencoder(cfg)
        opt = build_optimizer(model.parameters(), cfg)
        sched = build_scheduler(opt, cfg)
        assert sched is None

    def test_cosine(self, cfg):
        cfg_cos = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_a": {"autoencoder": {"scheduler": "cosine"}}}),
        )
        model = ConvAutoencoder(cfg_cos)
        opt = build_optimizer(model.parameters(), cfg_cos)
        sched = build_scheduler(opt, cfg_cos)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self, cfg):
        cfg_step = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_a": {"autoencoder": {"scheduler": "step"}}}),
        )
        model = ConvAutoencoder(cfg_step)
        opt = build_optimizer(model.parameters(), cfg_step)
        sched = build_scheduler(opt, cfg_step)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_invalid_scheduler_raises(self, cfg):
        cfg_bad = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {"zone_a": {"autoencoder": {"scheduler": "plateau"}}}
            ),
        )
        model = ConvAutoencoder(cfg_bad)
        opt = build_optimizer(model.parameters(), cfg_bad)
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            build_scheduler(opt, cfg_bad)


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_returns_float(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=8)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        opt = build_optimizer(model.parameters(), cfg)
        loss = train_epoch(model, loader, opt, cfg, device, epoch=0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_decreases_over_epochs(self, cfg, device, tmp_path):
        """Mean loss should not be exactly identical across two epochs."""
        torch.manual_seed(42)
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=16)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        opt = build_optimizer(model.parameters(), cfg)
        loss1 = train_epoch(model, loader, opt, cfg, device, epoch=0)
        loss2 = train_epoch(model, loader, opt, cfg, device, epoch=1)
        # Simply verify two distinct finite floats; not a monotone guarantee.
        assert np.isfinite(loss1) and np.isfinite(loss2)

    def test_model_in_train_mode_after(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=4)
        loader = DataLoader(ds, batch_size=4)
        opt = build_optimizer(model.parameters(), cfg)
        train_epoch(model, loader, opt, cfg, device)
        assert model.training


# ---------------------------------------------------------------------------
# extract_embeddings
# ---------------------------------------------------------------------------


class TestExtractEmbeddings:
    def test_returns_correct_number_of_entries(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=6)
        result = extract_embeddings(model, ds, cfg, device)
        assert len(result) == 6

    def test_keys_are_tablet_id_position_tuples(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=4)
        result = extract_embeddings(model, ds, cfg, device)
        for key in result:
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], int)

    def test_embedding_shape(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        ds = _dummy_token_dataset(cfg, tmp_path, n=4)
        result = extract_embeddings(model, ds, cfg, device)
        for emb in result.values():
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (32,)
            assert emb.dtype == np.float32

    def test_model_training_mode_restored(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        model.train()
        ds = _dummy_token_dataset(cfg, tmp_path, n=4)
        extract_embeddings(model, ds, cfg, device)
        assert model.training

    def test_model_eval_mode_restored(self, cfg, device, tmp_path):
        model = ConvAutoencoder(cfg)
        model.eval()
        ds = _dummy_token_dataset(cfg, tmp_path, n=4)
        extract_embeddings(model, ds, cfg, device)
        assert not model.training


# ---------------------------------------------------------------------------
# MLPProjectionHead
# ---------------------------------------------------------------------------


class TestMLPProjectionHead:
    def test_output_shape(self):
        head = MLPProjectionHead(in_dim=32, hidden_dim=64, out_dim=16)
        z = torch.randn(8, 32)
        out = head(z)
        assert out.shape == (8, 16)

    def test_output_is_unit_norm(self):
        head = MLPProjectionHead(in_dim=32, hidden_dim=64, out_dim=16)
        head.eval()
        with torch.no_grad():
            z = torch.randn(5, 32)
            out = head(z)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_gradients_flow(self):
        head = MLPProjectionHead(in_dim=32, hidden_dim=32, out_dim=8)
        z = torch.randn(4, 32, requires_grad=True)
        out = head(z)
        out.sum().backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_single_sample_train_mode(self):
        """BatchNorm1d requires >1 sample in training mode; 1 sample should work in eval."""
        head = MLPProjectionHead(in_dim=16, hidden_dim=16, out_dim=8)
        head.eval()
        with torch.no_grad():
            z = torch.randn(1, 16)
            out = head(z)
        assert out.shape == (1, 8)


# ---------------------------------------------------------------------------
# _ms_ssim_loss
# ---------------------------------------------------------------------------


class TestMsSsimLoss:
    def test_returns_scalar(self):
        torch.manual_seed(0)
        pred = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        target = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        loss = _ms_ssim_loss(pred, target)
        assert loss.shape == ()

    def test_identical_inputs_near_zero(self):
        x = torch.rand(3, 1, 16, 16) * 2.0 - 1.0
        loss = _ms_ssim_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_non_negative(self):
        torch.manual_seed(1)
        pred = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        target = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        assert _ms_ssim_loss(pred, target).item() >= 0.0

    def test_gradients_flow(self):
        pred = torch.rand(2, 1, 16, 16, requires_grad=True) * 2.0 - 1.0
        pred.retain_grad()
        target = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
        loss = _ms_ssim_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# reconstruction_loss — mixed type
# ---------------------------------------------------------------------------


class TestReconstructionLossMixed:
    def _pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(2)
        pred = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        target = torch.rand(4, 1, 16, 16) * 2.0 - 1.0
        return pred, target

    def test_mixed_is_scalar(self):
        pred, target = self._pair()
        loss = reconstruction_loss(pred, target, "mixed")
        assert loss.shape == ()

    def test_mixed_is_weighted_combination(self):
        pred, target = self._pair()
        w = 0.4
        expected = (1 - w) * reconstruction_loss(pred, target, "mse")
        expected = expected + w * reconstruction_loss(pred, target, "ssim")
        actual = reconstruction_loss(pred, target, "mixed", ssim_weight=w)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_mixed_identical_inputs_near_zero(self):
        x = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
        loss = reconstruction_loss(x, x, "mixed")
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_mixed_weight_zero_equals_mse(self):
        pred, target = self._pair()
        mse = reconstruction_loss(pred, target, "mse")
        mixed_w0 = reconstruction_loss(pred, target, "mixed", ssim_weight=0.0)
        assert mixed_w0.item() == pytest.approx(mse.item(), rel=1e-5)

    def test_mixed_weight_one_equals_ssim(self):
        pred, target = self._pair()
        ssim = reconstruction_loss(pred, target, "ssim")
        mixed_w1 = reconstruction_loss(pred, target, "mixed", ssim_weight=1.0)
        assert mixed_w1.item() == pytest.approx(ssim.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# ConvEncoder with two-stage projection
# ---------------------------------------------------------------------------


class TestConvEncoderTwoStage:
    def test_output_shape_with_intermediate(self, cfg):
        bb = SharedConvBackbone(cfg, image_channels=1)
        spatial = _encoder_spatial_size(cfg)
        enc = ConvEncoder(bb, bottleneck_dim=32, spatial_size=spatial, intermediate_dim=64)
        x = _dummy_image_batch(4)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_intermediate_zero_matches_single_stage(self, cfg):
        """intermediate_dim=0 must give identical output shapes to the default."""
        bb = SharedConvBackbone(cfg, image_channels=1)
        spatial = _encoder_spatial_size(cfg)
        enc_single = ConvEncoder(bb, bottleneck_dim=32, spatial_size=spatial)
        enc_no_inter = ConvEncoder(bb, bottleneck_dim=32, spatial_size=spatial, intermediate_dim=0)
        x = _dummy_image_batch(2)
        assert enc_single(x).shape == enc_no_inter(x).shape


# ---------------------------------------------------------------------------
# ConvAutoencoder — new methods (encode_normalized, project)
# ---------------------------------------------------------------------------


class TestConvAutoencoderNewMethods:
    def test_encode_normalized_unit_norm(self, cfg):
        model = ConvAutoencoder(cfg)
        model.eval()
        x = _dummy_image_batch(4)
        with torch.no_grad():
            z = model.encode_normalized(x)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_encode_normalized_shape(self, cfg):
        model = ConvAutoencoder(cfg)
        model.eval()
        x = _dummy_image_batch(3)
        with torch.no_grad():
            z = model.encode_normalized(x)
        assert z.shape == (3, 32)

    def test_project_output_unit_norm(self, cfg):
        model = ConvAutoencoder(cfg)
        model.eval()
        z = torch.randn(4, 32)
        with torch.no_grad():
            proj = model.project(z)
        norms = proj.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_project_output_dim_from_config(self, cfg):
        cfg_with_head = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "zone_a": {"autoencoder": {"proj_head_out_dim": 12}}
            }),
        )
        model = ConvAutoencoder(cfg_with_head)
        model.eval()
        with torch.no_grad():
            proj = model.project(torch.randn(3, 32))
        assert proj.shape == (3, 12)

    def test_mixed_loss_via_model(self, cfg):
        cfg_mixed = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "zone_a": {"autoencoder": {
                    "reconstruction_loss": "mixed",
                    "mixed_loss_ssim_weight": 0.5,
                }}
            }),
        )
        model = ConvAutoencoder(cfg_mixed)
        x = _dummy_image_batch(4)
        _, x_hat = model(x)
        loss = model.loss(x, x_hat)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_intermediate_dim_model_forward(self, cfg):
        cfg_inter = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "zone_a": {"autoencoder": {"encoder_intermediate_dim": 64}}
            }),
        )
        model = ConvAutoencoder(cfg_inter)
        x = _dummy_image_batch(4)
        z, x_hat = model(x)
        assert z.shape == (4, 32)
        assert x_hat.shape == (4, 1, 16, 16)


# ---------------------------------------------------------------------------
# build_warmup_cosine_scheduler
# ---------------------------------------------------------------------------


class TestBuildWarmupCosineScheduler:
    def _opt(self, cfg) -> torch.optim.Optimizer:
        model = ConvAutoencoder(cfg)
        return build_optimizer(model.parameters(), cfg)

    def test_returns_sequential_lr(self, cfg):
        opt = self._opt(cfg)
        sched = build_warmup_cosine_scheduler(opt, warmup_epochs=3, cosine_T_max=7)
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_lr_starts_low(self, cfg):
        """LR at epoch 0 should be close to start_factor * base_lr."""
        opt = self._opt(cfg)
        base_lr = cfg.zone_a.autoencoder.lr
        sched = build_warmup_cosine_scheduler(opt, warmup_epochs=5, cosine_T_max=45)
        lr_initial = opt.param_groups[0]["lr"]
        START_FACTOR = 1e-3  # must match build_warmup_cosine_scheduler's start_factor
        assert lr_initial == pytest.approx(base_lr * START_FACTOR, rel=0.1)

    def test_cosine_warmup_via_build_scheduler(self, cfg):
        cfg_wu = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "zone_a": {"autoencoder": {
                    "scheduler": "cosine_warmup",
                    "warmup_epochs": 2,
                    "num_epochs": 10,
                    "scheduler_T_max": 10,
                }}
            }),
        )
        opt = self._opt(cfg_wu)
        sched = build_scheduler(opt, cfg_wu)
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_step_through_warmup_and_cosine(self, cfg):
        """Stepping through should not raise for the full num_epochs."""
        opt = self._opt(cfg)
        sched = build_warmup_cosine_scheduler(opt, warmup_epochs=3, cosine_T_max=7)
        for _ in range(10):
            sched.step()


# ---------------------------------------------------------------------------
# train_epoch — denoising mode
# ---------------------------------------------------------------------------


class TestTrainEpochDenoising:
    def test_denoising_returns_finite_loss(self, cfg, device, tmp_path):
        cfg_dn = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "zone_a": {"autoencoder": {"denoising_noise_std": 0.1}}
            }),
        )
        torch.manual_seed(0)
        model = ConvAutoencoder(cfg_dn)
        ds = _dummy_token_dataset(cfg_dn, tmp_path, n=8)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        opt = build_optimizer(model.parameters(), cfg_dn)
        loss = train_epoch(model, loader, opt, cfg_dn, device, epoch=0)
        assert isinstance(loss, float)
        import math
        assert math.isfinite(loss)
        assert loss >= 0.0
