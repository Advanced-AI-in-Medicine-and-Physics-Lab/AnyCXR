"""Joint D3 and geometry-matched CT-RATE/MSDR supervision datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import FilteredD3Dataset, VIEW_ANGLES


VARIANTS = (
    "base", "regular_0", "regular_1", "regular_2",
    "drastic_0", "drastic_1", "drastic_2", "portable_0", "portable_1",
)
VARIANT_INDEX = {name: index for index, name in enumerate(VARIANTS)}
VIEW_INDEX = {name: index for index, name in enumerate(("AP", "PA", "LA", "OB"))}
STANDARD_VARIANTS = VARIANTS[:7]


class ProjectedCTMSDRDataset(Dataset):
    """All successful CT cases with one item per geometry-matched target slot.

    Standard-camera targets are shared by base/regular/drastic images and one
    of those seven source images is chosen deterministically per epoch.  Each
    portable image uses its own ray-cast target.  A target is supervised only
    when both the projection QC and the view-independent 3D source QC pass.
    """

    def __init__(
        self,
        target_root: Path,
        source_exclusion_manifest: Path,
        seed: int = 20260828,
        require_finalized: bool = True,
    ) -> None:
        self.target_root = Path(target_root)
        self.seed = seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        source_path = Path(source_exclusion_manifest)
        if require_finalized and not (source_path.parent / "COMPLETE").is_file():
            raise RuntimeError(f"source exclusion manifest is not complete: {source_path.parent}")
        source = json.loads(source_path.read_text())
        if source.get("schema") != "ct_rate_source_mask_organ_exclusion_v1":
            raise RuntimeError(f"unsupported source exclusion schema: {source.get('schema')}")
        self.class_names = list(source["class_names"])
        self.source_available: dict[str, np.ndarray] = {}
        for case in source["cases"]:
            excluded = set(case["excluded_organs"])
            self.source_available[str(case["case_id"])] = np.asarray(
                [name not in excluded for name in self.class_names], dtype=np.uint8
            )

        self.cases: list[dict[str, Any]] = []
        self.slot_names: tuple[str, ...] | None = None
        target_identity: list[str] = []
        candidates = sorted(self.target_root.glob("shard_*_of_*"))
        if (self.target_root / "meta.json").is_file():
            candidates = [self.target_root]
        for shard in candidates:
            meta = json.loads((shard / "meta.json").read_text())
            if not meta.get("complete"):
                raise RuntimeError(f"incomplete target shard: {shard}")
            if require_finalized and not meta.get("qc_finalized"):
                raise RuntimeError(f"target QC not finalized: {shard}")
            if meta.get("schema") != "anychest_ct_msdr_all_geometry_targets_v1":
                raise RuntimeError(f"unsupported target schema in {shard}: {meta.get('schema')}")
            if int(meta["target_size"]) != 96:
                raise RuntimeError(f"expected 96 target in {shard}, got {meta['target_size']}")
            if list(meta["class_names"]) != self.class_names:
                raise RuntimeError(f"class schema mismatch: {shard}")
            slots = tuple(meta["views"])
            if self.slot_names is None:
                self.slot_names = slots
            elif slots != self.slot_names:
                raise RuntimeError(f"target slot mismatch: {shard}")
            target_identity.append(str(meta["identity"]))
            state = np.load(shard / "case_state.uint8.npy", mmap_mode="r")
            records = [
                json.loads(line) for line in (shard / "records.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if len(records) != len(state):
                raise RuntimeError(f"record/state mismatch: {shard}")
            for record in records:
                target_index = int(record["target_index"])
                if int(state[target_index]) != 1:
                    continue
                case_id = str(record["case_id"])
                if case_id not in self.source_available:
                    raise RuntimeError(f"case missing from source exclusion manifest: {case_id}")
                self.cases.append({
                    "case_id": case_id,
                    "source_image_path": str(record["source_image_path"]),
                    "source_local_index": int(record["source_local_index"]),
                    "target_slots": record["target_slots"],
                    "target_index": target_index,
                    "target_path": str(shard / "targets.packbits.npy"),
                    "availability_path": str(shard / "target_availability.uint8.npy"),
                })
        if not self.cases or self.slot_names is None:
            raise RuntimeError(f"no successful CT targets under {self.target_root}")
        if len(self.slot_names) != 12:
            raise RuntimeError(f"expected 12 target slots, got {self.slot_names}")
        if len(self.source_available) != len(source["cases"]):
            raise RuntimeError("duplicate case ids in source exclusion manifest")
        self.target_size = 96
        self.meta = {
            "schema": "anychest_joint_projected_ct_msdr_dataset_v1",
            "class_names": self.class_names,
            "target_identities": target_identity,
            "source_exclusion_schema": source["schema"],
            "successful_cases": len(self.cases),
            "slots": list(self.slot_names),
        }
        self.source_arrays: dict[str, np.ndarray] = {}
        self.target_arrays: dict[str, np.ndarray] = {}
        self.availability_arrays: dict[str, np.ndarray] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)

    def __len__(self) -> int:
        assert self.slot_names is not None
        return len(self.cases) * len(self.slot_names)

    @staticmethod
    def _array(cache: dict[str, np.ndarray], path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = np.load(path, mmap_mode="r")
        return cache[path]

    @staticmethod
    def _image(value: np.ndarray, quantile: list[float]) -> torch.Tensor:
        low, high = map(float, quantile)
        image = torch.from_numpy(np.array(value, dtype=np.float32, copy=True))[None]
        image = ((image - low) / max(high - low, 1e-6)).clamp(0, 1)
        return F.interpolate(image[None], size=(384, 384), mode="area")[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert self.slot_names is not None
        case_index, slot_index = divmod(index, len(self.slot_names))
        case = self.cases[case_index]
        slot_name = self.slot_names[slot_index]
        slot = case["target_slots"][slot_name]
        view = str(slot["source_view"])
        flavor = str(slot["flavor"])
        if flavor == "standard":
            rng = random.Random(
                self.seed + int(self.epoch) * 1_000_003 + case_index * 17 + slot_index
            )
            variant = STANDARD_VARIANTS[rng.randrange(len(STANDARD_VARIANTS))]
            quantile = slot["quantiles"][variant]
        else:
            variant = flavor
            quantile = slot["quantile"]
        source = self._array(self.source_arrays, str(case["source_image_path"]))
        image = self._image(
            source[
                int(case["source_local_index"]), VARIANT_INDEX[variant], VIEW_INDEX[view]
            ],
            quantile,
        )
        packed = np.asarray(
            self._array(self.target_arrays, str(case["target_path"]))[
                int(case["target_index"]), slot_index
            ]
        )
        target = np.unpackbits(
            packed, axis=-1, count=self.target_size, bitorder="big"
        ).astype(np.float32)
        projected = np.asarray(
            self._array(self.availability_arrays, str(case["availability_path"]))[
                int(case["target_index"]), slot_index
            ],
            dtype=np.uint8,
        )
        availability = projected * self.source_available[str(case["case_id"])]
        return {
            "image": image,
            "target": torch.from_numpy(target),
            "availability": torch.from_numpy(availability.copy()).float(),
            "angle": torch.tensor(VIEW_ANGLES[view]),
            "source_id": torch.tensor(1, dtype=torch.int64),
        }


class BalancedProjectionTrainingDataset(Dataset):
    """Deterministic epoch blocks that cover CT while retaining a D3 anchor."""

    def __init__(
        self,
        d3: FilteredD3Dataset,
        ct: ProjectedCTMSDRDataset,
        d3_samples_per_epoch: int,
        ct_samples_per_epoch: int,
        seed: int = 20260828,
    ) -> None:
        if min(d3_samples_per_epoch, ct_samples_per_epoch) <= 0:
            raise ValueError("both D3 and CT samples per epoch must be positive")
        self.d3 = d3
        self.ct = ct
        self.d3_samples = d3_samples_per_epoch
        self.ct_samples = ct_samples_per_epoch
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        generator = np.random.default_rng(seed)
        self.d3_order = generator.permutation(len(d3)).astype(np.int64)
        self.ct_order = generator.permutation(len(ct)).astype(np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)
        self.d3.set_epoch(epoch)
        self.ct.set_epoch(epoch)

    def __len__(self) -> int:
        return self.d3_samples + self.ct_samples

    @staticmethod
    def _cycled(order: np.ndarray, position: int) -> int:
        return int(order[position % len(order)])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        epoch = int(self.epoch)
        if index < self.d3_samples:
            child = self._cycled(self.d3_order, epoch * self.d3_samples + index)
            item = self.d3[child]
            item["source_id"] = torch.tensor(0, dtype=torch.int64)
            return item
        local = index - self.d3_samples
        child = self._cycled(self.ct_order, epoch * self.ct_samples + local)
        return self.ct[child]
