"""Explicit AP/PA inference for experimental ConvNeXtV2 checkpoints.

Inputs are single-channel uint8 images in canonical viewer orientation.
Outputs retain 54 overlapping probability channels on the trained 96 grid.
The acquisition view must come from metadata; this API does not infer it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

FORMAT = "anycxr_convnextv2_frontal_v1"


class FrontalSegmentor:
    def __init__(self, checkpoint: str | Path, device="cpu"):
        from cxas.models.convnextv2 import AnyCXRConvNextV2SkipAblation

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT:
            raise ValueError("Expected an exported AnyCXR frontal checkpoint")
        self.metadata = saved["metadata"]
        self.class_names = self.metadata["class_names"]
        if len(self.class_names) != 54 or len(set(self.class_names)) != 54:
            raise ValueError("Expected 54 unique channel names")
        self.device = torch.device(device)
        self.model = AnyCXRConvNextV2SkipAblation(
            54, {"config": saved["backbone_config"]}, "anychest_cas_lite",
            gradient_checkpointing=False, enable_amfs=True, enable_jtr=False,
        )
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, images: list[np.ndarray], view: str) -> np.ndarray:
        if view not in self.metadata["supported_views"]:
            raise ValueError(f"Unsupported acquisition view: {view}")
        if not images:
            raise ValueError("Provide at least one image")
        arrays = []
        for image in images:
            if image.dtype != np.uint8 or image.ndim != 2:
                raise ValueError("Expected a 2D uint8 grayscale image")
            arrays.append(np.asarray(Image.fromarray(image).resize(
                (384, 384), Image.Resampling.BILINEAR)))
        array = np.stack(arrays).astype(np.float32) / 255
        raw_pa = self.metadata["pa_training_camera"] == "legacy_mirrored"
        if view == "PA" and raw_pa:
            array = array[..., ::-1].copy()
        x = torch.from_numpy(array[:, None]).to(self.device)
        x = (x - self.metadata["mean"]) / self.metadata["std"]
        angle = torch.full((len(x),), 0.0 if view == "AP" else 180.0,
                           device=self.device)
        output = self.model({"data": x, "view_angle": angle})["probabilities"]
        result = output.float().cpu().numpy()
        return result[..., ::-1].copy() if view == "PA" and raw_pa else result
