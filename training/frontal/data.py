"""Datasets for AnyCXR v3: filtered D3 supervision and CT-RATE MSDR pairs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VIEW_ANGLES = {"AP": 0.0, "PA": 180.0, "LA": 90.0, "OB": 45.0}


class FilteredD3Dataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, seed: int = 20260825,
                 use_legacy_flavors: bool = False,
                 canonicalize_pa_left_right: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.meta = json.loads((self.cache_dir / "meta.json").read_text())
        if not self.meta.get("complete"):
            raise RuntimeError(f"incomplete D3 cache: {cache_dir}")
        with (self.cache_dir / "records.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.training = split == "train"
        self.seed = seed
        self.use_legacy_flavors = use_legacy_flavors
        self.canonicalize_pa_left_right = canonicalize_pa_left_right
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.images: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self.availability: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)

    def _open(self) -> None:
        if self.images is None:
            self.images = np.load(self.cache_dir / "images.uint8.npy", mmap_mode="r")
            self.targets = np.load(self.cache_dir / "targets.soft.uint8.npy", mmap_mode="r")
            self.availability = np.load(
                self.cache_dir / "target_availability.uint8.npy", mmap_mode="r"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        assert self.images is not None and self.targets is not None
        assert self.availability is not None
        row = self.rows[index]
        geometry = int(row["geometry_index"])
        rng = random.Random(self.seed + int(self.epoch) * 1_000_003 + geometry)
        image_index = int(row["base_image_index"])
        flavor_count = int(row["flavor_count"])
        if self.training and self.use_legacy_flavors and flavor_count and rng.random() < 0.5:
            image_index = int(row["flavor_start_index"]) + rng.randrange(flavor_count)
        image = torch.from_numpy(np.array(self.images[image_index], copy=True)).float()[None] / 255
        if self.training:
            image = image.clamp(0, 1).pow(rng.uniform(0.9, 1.1))
            image = (image - image.mean()) * rng.uniform(0.9, 1.1) + image.mean()
            image = image.add(rng.uniform(-0.03, 0.03)).clamp(0, 1)
        target = torch.from_numpy(np.array(self.targets[geometry], copy=True)).float() / 255
        # The legacy D3 camera contract mirrors PA relative to AP, whereas
        # real radiographs and the CT/MSDR renderer use one viewer-oriented
        # laterality convention for both views.  Flip the paired PA image and
        # all target channels together; channel identities remain anatomical.
        if self.canonicalize_pa_left_right and row["view"] == "PA":
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        available = torch.from_numpy(np.array(self.availability[geometry], copy=True)).float()
        return {
            "image": image,
            "target": target,
            "availability": available,
            "angle": torch.tensor(VIEW_ANGLES[row["view"]]),
        }

