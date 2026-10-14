"""Validate reserved checkpoint slots without downloading or loading weights."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_registry(registry: dict, root: Path = ROOT, require_ready: bool = False) -> dict:
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint registry schema.")
    slots = registry.get("slots", [])
    if not slots:
        raise ValueError("No checkpoint slots were declared.")
    ids, destinations, placeholders = set(), set(), []
    for slot in slots:
        slot_id = slot["id"]
        if slot_id in ids:
            raise ValueError(f"Duplicate checkpoint ID: {slot_id}")
        ids.add(slot_id)
        kind = slot["kind"]
        if kind not in {"full", "inference", "inference_bundle"}:
            raise ValueError(f"Unknown checkpoint kind for {slot_id}")
        filename = Path(slot["filename"])
        if filename.is_absolute() or ".." in filename.parts or filename.suffix not in {".pt", ".pth"}:
            raise ValueError(f"Invalid artifact filename for {slot_id}")
        destination = (kind == "full", str(filename))
        if destination in destinations:
            raise ValueError(f"Duplicate artifact destination: {filename}")
        destinations.add(destination)
        config = slot.get("training_config")
        if config and not (root / config).is_file():
            raise ValueError(f"Missing training configuration for {slot_id}")
        status = slot["status"]
        if status == "placeholder":
            placeholders.append(slot_id)
            for field in ("hf_repo_id", "hf_revision", "sha256", "size_bytes", "source_checkpoint", "epoch", "metrics"):
                if slot.get(field) is not None:
                    raise ValueError(f"Placeholder {slot_id} has a populated availability field: {field}")
        elif status in {"ready", "published"}:
            if not re.fullmatch(r"[0-9a-f]{64}", slot.get("sha256") or ""):
                raise ValueError(f"{slot_id} requires a SHA-256 before becoming {status}")
            size = slot.get("size_bytes")
            if type(size) is not int or size <= 0 or not slot.get("source_checkpoint"):
                raise ValueError(f"{slot_id} requires byte size and source identity")
            if status == "published" and (not slot.get("hf_repo_id") or not slot.get("hf_revision")):
                raise ValueError(f"Published artifact {slot_id} requires HF repository and pinned revision")
        else:
            raise ValueError(f"Unknown checkpoint status for {slot_id}: {status}")
    for repo in registry["planned_repositories"].values():
        if not (root / repo["model_card"]).is_file():
            raise ValueError(f"Missing model-card template: {repo['model_card']}")
    if require_ready and placeholders:
        raise ValueError(f"Release is not ready: {len(placeholders)} checkpoint placeholders remain.")
    return {"structure_valid": True, "reserved_artifacts": len(slots),
            "placeholders": len(placeholders), "all_artifacts_ready": not placeholders,
            "weights_loaded": False, "hf_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "checkpoints/registry.json")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")), require_ready=args.require_ready)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
