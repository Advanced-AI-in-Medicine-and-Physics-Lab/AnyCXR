"""Inspect a training template without importing Torch or accessing image data.

The retained trainer lives in legacy/. Executable training integration awaits
a frozen split adapter; this preparation entry point deliberately only plans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"pa", "la", "oblique"}


def inspect_plan(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1 or config.get("model_profile") not in PROFILES:
        raise ValueError("Expected schema_version 1 and model_profile pa, la or oblique.")
    if config.get("status") != "template":
        raise ValueError("This preparation entry point accepts template configurations only.")
    profile = config["model_profile"]
    reference_path = ROOT / config["dataset"]["class_reference"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if len(reference["class_names"]) != 54:
        raise ValueError("The training template must refer to the 54-channel release metadata.")
    selected = config["training"]["selected_views"]
    if not selected or any(view not in reference["view_configs"] for view in selected):
        raise ValueError("Selected views are missing or not in the packaged reference.")
    registry = json.loads((ROOT / "checkpoints/registry.json").read_text(encoding="utf-8"))
    reserved = [slot["id"] for slot in registry["slots"] if slot["model"] == profile]
    required = {"dataset.root": config["dataset"].get("root"),
                "dataset.split_manifest": config["dataset"].get("split_manifest"),
                "dataset.split_sha256": config["dataset"].get("split_sha256"),
                "output_dir": config.get("output_dir")}
    return {"mode": "preparation_only", "model_profile": profile,
            "selected_views": selected, "output_channels": 54,
            "reserved_checkpoints": reserved, "training_started": False,
            "ready_to_train": False,
            "unfilled_fields": [key for key, value in required.items() if not value],
            "pending_integration": ["frozen train/validation split adapter",
                                    "training-to-inference checkpoint compatibility",
                                    "dataset provenance and configuration validation"],
            "config": config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the prepared AnyCXR training configuration; no training is run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the preparation plan without loading model dependencies.")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Use --dry-run. Executable training is pending frozen split integration; see training/README.md.")
    try:
        plan = inspect_plan(args.config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
