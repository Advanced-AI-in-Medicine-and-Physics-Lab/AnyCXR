"""ConvNeXtV2 encoder and AnyCXR decoder, retained from the segmentation experiments.

The decoder body is fixed.  Only the two deep encoder skips change between
``naive``, ``tiny_attn``, the original AnyCXR ``CASModule``, and a bottlenecked
CAS derivative.  This keeps the experiment about skip refinement rather than
decoder depth or the number of gated feature levels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ConvNextV2Config, ConvNextV2Model

from cxas.losses.compound_losses import DC_and_BCE_loss  # noqa: E402
from cxas.models.UNet.attention_modules import AMFSModule, CASModule  # noqa: E402
from cxas.models.UNet.decoder import AngleInjectionModule  # noqa: E402
from cxas.models.UNet.jtr_modules import JTR_Decoder, JTR_Encoder  # noqa: E402
from cxas.models.UNet.unet_blocks import OutConv, ResidualConvBlock  # noqa: E402


class NaiveSkip(nn.Module):
    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        del decoder
        return encoder


class TinyAttentionSkip(nn.Module):
    """A minimal Attention-UNet-style spatial gate."""

    def __init__(self, encoder_channels: int, decoder_channels: int) -> None:
        super().__init__()
        hidden = min(64, max(16, encoder_channels // 8))
        self.encoder_projection = nn.Conv2d(encoder_channels, hidden, 1, bias=False)
        self.decoder_projection = nn.Conv2d(decoder_channels, hidden, 1, bias=False)
        self.attention = nn.Sequential(nn.GELU(), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        weight = self.attention(
            self.encoder_projection(encoder) + self.decoder_projection(decoder)
        )
        return encoder * weight


class LiteCASSkip(nn.Module):
    """AnyCXR CAS topology with a small shared bottleneck.

    Original CAS projects both inputs to all decoder channels.  This version
    keeps the conditioned fusion, squeeze/excitation, and separable spatial
    gate, but caps that interaction space at 64 channels.
    """

    def __init__(self, encoder_channels: int, decoder_channels: int) -> None:
        super().__init__()
        hidden = min(64, max(16, encoder_channels // 8))
        squeeze = max(4, hidden // 4)
        self.encoder_projection = nn.Conv2d(encoder_channels, hidden, 1)
        self.decoder_projection = nn.Conv2d(decoder_channels, hidden, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, hidden, 1),
            nn.Sigmoid(),
        )
        self.horizontal = nn.Conv2d(hidden, 1, (1, 3), padding=(0, 1), bias=False)
        self.vertical = nn.Conv2d(hidden, 1, (3, 1), padding=(1, 0), bias=False)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        fused = self.encoder_projection(encoder) + self.decoder_projection(decoder)
        fused = fused * self.channel(fused)
        weight = torch.sigmoid(self.horizontal(fused) + self.vertical(fused))
        return encoder * weight


def build_skip(kind: str, encoder_channels: int, decoder_channels: int) -> nn.Module:
    if kind == "naive":
        return NaiveSkip()
    if kind == "tiny_attn":
        return TinyAttentionSkip(encoder_channels, decoder_channels)
    if kind == "anychest_cas":
        return CASModule(encoder_channels, decoder_channels)
    if kind == "anychest_cas_lite":
        return LiteCASSkip(encoder_channels, decoder_channels)
    raise ValueError(f"unknown skip kind: {kind}")


class ConvNextV2AnyCXREncoder(nn.Module):
    def __init__(self, bundle: Path | dict, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        saved = bundle if isinstance(bundle, dict) else torch.load(bundle, map_location="cpu", weights_only=False)
        config = ConvNextV2Config.from_dict(saved["config"])
        if config.num_channels != 1 or len(config.hidden_sizes) != 4:
            raise RuntimeError("expected a four-stage, one-channel ConvNeXtV2 bundle")
        self.model = ConvNextV2Model(config)
        if "model_state_dict" in saved:
            self.model.load_state_dict(saved["model_state_dict"], strict=True)
        if gradient_checkpointing and self.model.supports_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.channels = tuple(map(int, config.hidden_sizes))
        self.provenance = saved.get("provenance", {})
        self.angle = nn.ModuleList(AngleInjectionModule(value, 2) for value in self.channels)

    def forward(self, image: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(image, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states
        if hidden is None or len(hidden) < 5:
            raise RuntimeError("ConvNeXtV2 did not return four stage features")
        # HF returns stem plus the four stage outputs; use the final four.
        stages = list(hidden[-4:])
        if tuple(value.shape[1] for value in stages) != self.channels:
            raise RuntimeError(
                f"unexpected stage channels: {[value.shape[1] for value in stages]} != {self.channels}"
            )
        # The HF pooler has a trainable LayerNorm outside the spatial feature
        # path. Keep it in DDP with an exact zero gradient.
        stages[-1] = stages[-1] + output.pooler_output.sum() * 0.0
        return tuple(module(value, angle) for module, value in zip(self.angle, stages))


class AnyCXRSkipAblationDecoder(nn.Module):
    """AnyCXR CAS/AMFS decoder adapted to ConvNeXtV2's four native stages."""

    def __init__(self, channels: tuple[int, int, int, int], n_classes: int,
                 skip_kind: str, enable_amfs: bool) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        rates = (1, 2, 6)
        self.skip3 = build_skip(skip_kind, c3, c4)
        self.skip2 = build_skip(skip_kind, c2, c3)
        self.block3 = ResidualConvBlock(c4 + c3, c3)
        self.block2 = ResidualConvBlock(c3 + c2, c2)
        self.block1 = ResidualConvBlock(c2 + c1, c1)
        self.amfs3 = AMFSModule(c3, c3, rates) if enable_amfs else nn.Identity()
        self.amfs2 = AMFSModule(c2, c2, rates) if enable_amfs else nn.Identity()
        self.amfs1 = AMFSModule(c1, c1, rates) if enable_amfs else nn.Identity()
        self.angle3 = AngleInjectionModule(c3, 2)
        self.angle2 = AngleInjectionModule(c2, 2)
        self.angle1 = AngleInjectionModule(c1, 2)
        self.output = OutConv(c1, n_classes)

    @staticmethod
    def up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, features: tuple[torch.Tensor, ...], output_shape: tuple[int, int],
                angle: torch.Tensor) -> torch.Tensor:
        feat1, feat2, feat3, feat4 = features
        value = self.up(feat4, feat3)
        value = self.amfs3(self.angle3(
            self.block3(torch.cat((self.skip3(feat3, value), value), 1)), angle
        ))
        value = self.up(value, feat2)
        value = self.amfs2(self.angle2(
            self.block2(torch.cat((self.skip2(feat2, value), value), 1)), angle
        ))
        value = self.up(value, feat1)
        value = self.amfs1(self.angle1(self.block1(torch.cat((feat1, value), 1)), angle))
        value = self.output(value)
        if value.shape[-2:] != output_shape:
            value = F.interpolate(value, size=output_shape, mode="bilinear", align_corners=True)
        return value


class AnyCXRConvNextV2SkipAblation(nn.Module):
    def __init__(self, n_classes: int, backbone_bundle: Path | dict, skip_kind: str,
                 gradient_checkpointing: bool = True, enable_amfs: bool = True,
                 enable_jtr: bool = False, jtr_dist_weight: float = 0.10,
                 jtr_recon_weight: float = 0.10) -> None:
        super().__init__()
        self.encoder = ConvNextV2AnyCXREncoder(backbone_bundle, gradient_checkpointing)
        self.decoder = AnyCXRSkipAblationDecoder(
            self.encoder.channels, n_classes, skip_kind, enable_amfs
        )
        self.skip_kind = skip_kind
        self.enable_amfs = enable_amfs
        self.enable_jtr = enable_jtr
        self.jtr_dist_weight = float(jtr_dist_weight)
        self.jtr_recon_weight = float(jtr_recon_weight)
        self.jtr_encoder = JTR_Encoder(n_classes, 128) if enable_jtr else None
        self.jtr_decoder = JTR_Decoder(n_classes, 128) if enable_jtr else None
        self.n_classes = n_classes
        self.architecture_name = (
            f"convnextv2_anycxr_{skip_kind}_amfs{int(enable_amfs)}_jtr{int(enable_jtr)}"
        )
        self.seg_loss_fn = DC_and_BCE_loss(
            bce_kwargs={"reduction": "none"},
            soft_dice_kwargs={
                "do_bg": True, "batch_dice": False, "smooth": 1e-5, "ddp": False,
            },
            weight_ce=1, weight_dice=1, use_ignore_label=False,
        )

    def forward(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = inputs["data"]
        radians = torch.deg2rad(inputs["view_angle"].float())
        angle = torch.stack((torch.sin(radians), torch.cos(radians)), dim=1)
        output_shape = tuple(inputs["label"].shape[-2:]) if "label" in inputs else (96, 96)
        logits = self.decoder(self.encoder(image, angle), output_shape, angle)
        result = {"logits": logits, "probabilities": logits.sigmoid()}
        if "label" in inputs:
            loss, bce, dice = self.seg_loss_fn(
                logits, inputs["label"], inputs.get("class_weights")
            )
            result.update({"loss": loss, "ce_loss": bce, "dc_loss": dice})
            if self.enable_jtr:
                assert self.jtr_encoder is not None and self.jtr_decoder is not None
                weights = inputs["class_weights"][:, :, None, None] > 0
                target_probability = torch.where(
                    weights, inputs["label"], result["probabilities"].detach()
                )
                image = F.interpolate(
                    inputs["data"], size=logits.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted_latent = self.jtr_encoder(result["probabilities"], image)
                target_latent = self.jtr_encoder(target_probability, image)
                distance = (
                    1 - F.cosine_similarity(
                        predicted_latent.flatten(1), target_latent.flatten(1), dim=1, eps=1e-8
                    )
                ).mean() / 2
                predicted_reconstruction = self.jtr_decoder(
                    predicted_latent, tuple(logits.shape[-2:])
                )
                target_reconstruction = self.jtr_decoder(
                    target_latent, tuple(logits.shape[-2:])
                )
                reconstruction = (
                    F.mse_loss(predicted_reconstruction, result["probabilities"].detach())
                    + F.mse_loss(target_reconstruction, target_probability.detach())
                )
                result.update({"jtr_dist_loss": distance, "jtr_recon_loss": reconstruction})
                result["loss"] = (
                    result["loss"] + self.jtr_dist_weight * distance
                    + self.jtr_recon_weight * reconstruction
                )
        return result
