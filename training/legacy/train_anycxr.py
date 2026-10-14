from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cxas.trainer import CXAS_Trainer
from shared_training_config import DEFAULT_SHARED_CONFIG, build_augmentation_configs, load_training_config, parse_inf, resolve_model_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnyCXR from a shared YAML config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_SHARED_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_model_config(load_training_config(args.config), "anycxr")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    if args.dry_run:
        print("Resolved AnyCXR config:")
        for key in sorted(config.keys()):
            print(f"{key}: {config[key]}")
        return

    if bool(config.get("reset_augmentation_seeds", False)):
        seed_path = Path(str(config["data_dir"])) / "augmentation_seeds.pkl"
        if seed_path.exists():
            seed_path.unlink()
            print(f"Removed {seed_path}")

    trainer = CXAS_Trainer(
        base_dir=str(config["data_dir"]),
        fold_id=int(config.get("fold", 4)),
        gpus=str(config.get("gpus", "0")),
        lr=float(config.get("lr", 1e-4)),
        batch_size=int(config.get("batch_size", 8)),
        output_dir=str(config["output_dir"]),
        num_epochs=int(config.get("epochs", 200)),
        resume_training=bool(config.get("resume_training", config.get("resume", False))),
        selected_views=list(config.get("selected_views", ["LA"])),
        selected_classes=config.get("selected_classes"),
        angle_encoding_type=str(config.get("angle_encoding_type", "sin_cos")),
        progressive_training_resolutions=list(config.get("progressive_training_resolutions", [int(config.get("resolution", 512))])),
        jtr_start_epoch=parse_inf(config.get("car", {}).get("start_epoch"), default=float("inf")),
        jtr_c_dist=float(config.get("car", {}).get("dist_weight", 0.0)),
        jtr_c_recon=float(config.get("car", {}).get("recon_weight", 0.0)),
        further_training=bool(config.get("further_training", False)),
        augmentation_configs=build_augmentation_configs(str(config.get("augmentation_profile", "full"))),
        accumulation_steps=int(config.get("accumulation_steps", 1)),
        use_variations=bool(config.get("use_variations", False)),
    )

    start_time = time.time()
    trainer.train_model()
    elapsed_time = (time.time() - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} mins ({elapsed_time/60:.2f} hours)")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
