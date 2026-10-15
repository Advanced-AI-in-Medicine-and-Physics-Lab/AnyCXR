"""Fail-closed patient-group filtering for the synthetic segmentation sources."""
import hashlib
import json
from pathlib import Path


def group(case_id):
    parts = str(case_id).split("_")
    if len(parts) < 2 or parts[0] not in {"train", "t", "valid"} or not parts[1].isdigit():
        raise ValueError(f"Unrecognized CT patient identity: {case_id}")
    return ("t" if parts[0] == "valid" else parts[0]) + "_" + parts[1]


def restrict(d3, validation, ct, excluded_file):
    payload = json.loads(Path(excluded_file).read_text())
    excluded = set(payload["excluded_patient_groups"])
    if not excluded or payload.get("schema") != "anycxr_eval_group_exclusion_v1":
        raise ValueError("A frozen evaluation-group exclusion manifest is required")
    validation_groups = {group(r["case_id"]) for r in validation.rows}
    if not validation_groups <= excluded:
        raise ValueError("Exclusion manifest omits D3 validation groups")
    if set(payload["d3_validation_groups"]) != validation_groups:
        raise ValueError("Validation membership changed")
    if not set(payload["nw_groups"]) <= excluded:
        raise ValueError("Exclusion manifest omits NW evaluation groups")
    d3.rows = [r for r in d3.rows if group(r["case_id"]).startswith("train_")
               and group(r["case_id"]) not in excluded]
    ct.cases = [r for r in ct.cases if group(r["case_id"]).startswith("train_")
                and group(r["case_id"]) not in excluded]
    if not d3.rows or not ct.cases:
        raise ValueError("No eligible training cases")
    ct.meta["eligible_training_cases"] = len(ct.cases)
    ct.meta["training_slots"] = [name for name in ct.slot_names if name.split("_")[0] in {"AP", "PA"}]
    membership = {
        "d3_case_ids": sorted(r["case_id"] + ":" + r["view"] for r in d3.rows),
        "ct_case_ids": sorted(r["case_id"] for r in ct.cases),
        "excluded_patient_groups": sorted(excluded),
        "validation_groups": sorted(validation_groups),
    }
    return membership, hashlib.sha256(Path(excluded_file).read_bytes()).hexdigest()
