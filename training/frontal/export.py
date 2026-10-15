"""Export verified full training checkpoints without optimizer or external backbone."""
import argparse
import hashlib
from pathlib import Path

import torch

from cxas.frontal import FORMAT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(source, backbone, destination):
    saved = torch.load(source, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    expected = "convnextv2_anycxr_anychest_cas_lite_amfs1_jtr0"
    if cfg["architecture"] != expected:
        raise ValueError("Only the verified CAS-lite/AMFS/no-JTR architecture is supported")
    names = cfg["class_names"]
    if len(names) != 54 or len(set(names)) != 54:
        raise ValueError("Expected exact 54-channel schema")
    if Path(destination).exists():
        raise FileExistsError(destination)
    config = torch.load(backbone, map_location="cpu", weights_only=False)["config"]
    metadata = {
        "architecture": expected, "class_names": names,
        "supported_views": ["AP", "PA"], "input_size": 384, "output_size": 96,
        "mean": cfg.get("mean", cfg.get("args", {}).get("mean", .4897996446751163)),
        "std": cfg.get("std", cfg.get("args", {}).get("std", .2952534283781206)),
        "input_camera": "canonical_viewer_orientation",
        "pa_training_camera": "canonical" if cfg.get("d3_pa_canonicalized", False) else "legacy_mirrored",
        "source_sha256": sha256(source), "backbone_sha256": sha256(backbone),
        "epoch_zero_based": int(saved["epoch"]),
        "release_status": "unpublished_candidate",
    }
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(destination) + ".tmp")
    torch.save({"format": FORMAT, "metadata": metadata, "backbone_config": config,
                "model_state_dict": saved["model"]}, tmp)
    tmp.replace(destination)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.backbone, a.output)
