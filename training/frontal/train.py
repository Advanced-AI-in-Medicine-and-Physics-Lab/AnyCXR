"""Bounded real-CXR encoder -> frontal DRR segmentation fine-tuning.

Run with torchrun. Private hospital images are not accepted as training inputs.
The exclusion manifest and encoder provenance must be frozen before launch.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from cxas.models.convnextv2 import AnyCXRConvNextV2SkipAblation
from training.frontal.data import FilteredD3Dataset
from training.frontal.data_joint_projection import BalancedProjectionTrainingDataset, ProjectedCTMSDRDataset
from training.frontal.export import export, sha256
from training.frontal.splits import restrict
from training.frontal.hospital_training_policy import HELD_OUT_HOSPITALS


class FrontalCT(torch.utils.data.Dataset):
    def __init__(self, source):
        self.source = source
        self.slots = [i for i, name in enumerate(source.slot_names)
                      if name.split("_")[0] in {"AP", "PA"}]
        if len(self.slots) != 6:
            raise ValueError("Expected six frontal standard/portable target slots")

    def __len__(self):
        return len(self.source.cases) * len(self.slots)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __getitem__(self, index):
        case, slot = divmod(index, len(self.slots))
        return self.source[case * len(self.source.slot_names) + self.slots[slot]]


def atomic_save(value, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["d3-cache", "ct-target-root", "source-exclusion-manifest",
                 "evaluation-exclusions", "backbone-bundle", "output-dir", "recipe"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--resume", type=Path, help="Explicit full checkpoint; never inferred from output files")
    a = p.parse_args()
    recipe = json.loads(a.recipe.read_text())
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    device = torch.device("cuda", local)
    seed = int(recipe["seed"])
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if sha256(a.backbone_bundle) != recipe["encoder_bundle_sha256"]:
        raise ValueError("Encoder bundle hash changed")
    if recipe.get("training_hospitals") != [] or recipe.get("clinical_images_used_by_optimizer") is not False:
        raise ValueError(f"This runner accepts synthetic supervision only; {sorted(HELD_OUT_HOSPITALS)} stay evaluation-only")
    provenance = torch.load(a.backbone_bundle, map_location="cpu", weights_only=True)["provenance"]
    if (provenance.get("private_hospital_optimizer_updates") is not False
            or provenance.get("source_kind") != "public17_F1_real_CXR_only"
            or provenance.get("source_sha256") != recipe["source_f1_sha256"]):
        raise ValueError("Public real-CXR encoder provenance is missing or differs from the frozen recipe")
    d3 = FilteredD3Dataset(a.d3_cache, "train", seed, canonicalize_pa_left_right=True)
    val = FilteredD3Dataset(a.d3_cache, "validation", seed, canonicalize_pa_left_right=True)
    ct = ProjectedCTMSDRDataset(a.ct_target_root, a.source_exclusion_manifest, seed)
    if ct.class_names != d3.meta["class_names"]:
        raise ValueError("Channel schema mismatch")
    membership, exclusion_hash = restrict(d3, val, ct, a.evaluation_exclusions)
    identity = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    if identity != recipe["membership_sha256"]:
        raise ValueError("Training/evaluation membership changed")
    train = BalancedProjectionTrainingDataset(
        d3, FrontalCT(ct), recipe["d3_samples_per_epoch"], recipe["ct_samples_per_epoch"], seed)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=seed)
    loader_kw = dict(batch_size=recipe["batch_size_per_gpu"], num_workers=recipe["workers_per_gpu"],
                     pin_memory=True, persistent_workers=recipe["workers_per_gpu"] > 0)
    loader = DataLoader(train, sampler=sampler, **loader_kw)
    # Exact striding avoids DistributedSampler's duplicated validation records.
    validation = DataLoader(Subset(val, range(rank, len(val), world)), **loader_kw)
    model = AnyCXRConvNextV2SkipAblation(54, a.backbone_bundle, "anychest_cas_lite",
                gradient_checkpointing=False, enable_amfs=True, enable_jtr=False).to(device)
    base = model
    if world > 1:
        model = DDP(model, device_ids=[local])
    encoder = list(base.encoder.parameters()); ids = {id(x) for x in encoder}
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": recipe["encoder_lr"]},
        {"params": [x for x in base.parameters() if id(x) not in ids], "lr": recipe["decoder_lr"]},
    ], weight_decay=recipe["weight_decay"], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, recipe["epochs"], eta_min=1e-6)
    config = {"architecture": base.architecture_name, "class_names": d3.meta["class_names"],
        "mean": recipe["mean"], "std": recipe["std"], "d3_pa_canonicalized": True,
        "recipe": recipe, "membership_sha256": identity, "exclusion_sha256": exclusion_hash,
        "d3_cache_identity": d3.meta["identity"], "ct_dataset": ct.meta,
        "selection_metric": "exact_421_D3_development_macro_dice",
        "world_size": world, "initialization": "real_CXR_encoder_plus_fresh_segmentation_decoder"}
    if rank == 0 and not a.resume and a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty; implicit resume is disabled")
    start_epoch, best = 0, -math.inf
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume contract differs from frozen configuration")
        base.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, best = int(saved["epoch"]) + 1, float(saved["best"])
        rng = saved["rng_by_rank"][rank]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
    if rank == 0:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        (a.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        (a.output_dir / "membership.private.json").write_text(json.dumps(membership, indent=2))
    if world > 1:
        dist.barrier()

    def move(raw):
        return {"data": ((raw["image"].to(device) - recipe["mean"]) / recipe["std"]),
                "label": raw["target"].to(device), "class_weights": raw["availability"].to(device),
                "view_angle": raw["angle"].to(device)}

    for epoch in range(start_epoch, recipe["epochs"]):
        started = time.monotonic(); model.train(); train.set_epoch(epoch); sampler.set_epoch(epoch)
        total = torch.zeros(2, device=device, dtype=torch.float64)
        for step, raw in enumerate(loader):
            if a.smoke_only and step >= 2:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(move(raw))["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += torch.tensor([loss.item(), 1], device=device)
            if rank == 0 and step % 100 == 0:
                progress = {"epoch": epoch + 1, "step": step + 1, "steps": len(loader),
                            "loss": loss.item(), "elapsed_seconds": time.monotonic() - started}
                (a.output_dir / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        scheduler.step(); base.eval()
        # Uneven validation shards must bypass DDP forward collectives. Broadcast
        # BN statistics once so every rank evaluates the checkpoint saved by rank 0.
        if world > 1:
            for buffer in base.buffers():
                dist.broadcast(buffer, src=0)
        num = torch.zeros(54, dtype=torch.float64, device=device); den = torch.zeros_like(num)
        count = torch.zeros(1, dtype=torch.float64, device=device)
        with torch.inference_mode():
            for step, raw in enumerate(validation):
                batch = move(raw)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prob = base({"data": batch["data"], "view_angle": batch["view_angle"]})["probabilities"]
                pred = (prob.float() >= .5).double(); truth = batch["label"].double()
                valid = batch["class_weights"][:, :, None, None].double()
                num += (2 * pred * truth * valid).sum((0, 2, 3))
                den += ((pred + truth) * valid).sum((0, 2, 3)); count += len(pred)
        if world > 1:
            for value in [total, num, den, count]:
                dist.all_reduce(value)
        if int(count.item()) != len(val):
            raise ValueError("Validation coverage is not exact")
        macro = float((num[den > 0] / den[den > 0]).mean())
        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, rng)
        else:
            rngs = [rng]
        if rank == 0:
            record = {"epoch": epoch + 1, "train_loss": float(total[0] / total[1]),
                      "D3_validation_macro_dice": macro, "validation_images": int(count.item()),
                      "per_class_dice": dict(zip(config["class_names"], (num / den.clamp_min(1)).tolist())),
                      "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(record), flush=True)
            with (a.output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if not a.smoke_only:
                payload = {"epoch": epoch, "best": max(best, macro), "model": base.state_dict(),
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "rng_by_rank": rngs, "config": config}
                atomic_save(payload, a.output_dir / "last.pt")
                if macro > best:
                    best = macro; atomic_save(payload, a.output_dir / "best.pt")
        if world > 1:
            dist.barrier()
        if a.smoke_only:
            break
    if rank == 0 and not a.smoke_only:
        export(a.output_dir / "best.pt", a.backbone_bundle, a.output_dir / "inference.pt")
        files = {name: sha256(a.output_dir / name) for name in ["last.pt", "best.pt", "inference.pt"]}
        (a.output_dir / "COMPLETE.json").write_text(json.dumps(
            {"status": "complete", "epochs": recipe["epochs"], "best_development_macro": best,
             "sha256": files, "published": False, "adopted": False}, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
