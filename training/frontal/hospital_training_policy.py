"""Mandatory private-hospital training boundary, confirmed by user 2026-10-05."""
from collections import Counter
import json
import math
from pathlib import Path

POLICY_ID = 'xray_f2_source_hospitals_only_v20261005'
ALLOWED_TRAIN_HOSPITALS = frozenset(('安贞', '北大西直门'))
HELD_OUT_HOSPITALS = frozenset(('安徽', '北大通州'))


def validate_training_hospitals(hospitals):
    counts = Counter(str(hospital).strip() for hospital in hospitals)
    if not counts:
        raise ValueError('F2 training hospital provenance is empty')
    invalid = sorted(set(counts) - ALLOWED_TRAIN_HOSPITALS)
    if invalid:
        raise ValueError(
            'F2 hospital training boundary violated: only 安贞 and 北大西直门 may train; '
            f'forbidden/unknown train hospitals: {invalid}. 安徽/北大通州 are evaluation-only.'
        )
    return {'policy_id': POLICY_ID, 'train_hospitals': sorted(counts),
            'held_out_hospitals': sorted(HELD_OUT_HOSPITALS)}


def validate_training_rows(rows):
    return validate_training_hospitals(
        row['hospital_id'] for row in rows if str(row['split']).strip().lower() == 'train'
    )


def validate_f2_checkpoint_provenance(path):
    """Fail before loading/resuming F2 weights when source hospitals are invalid or unknown."""
    directory = Path(path).parent
    candidates = [('run_config.json', 'train_patient_site_counts'), ('preflight.json', 'train_patient_counts')]
    found = []
    for name, key in candidates:
        source = directory / name
        if source.exists():
            record = json.loads(source.read_text())
            if key in record:
                counts = record[key]
                if not isinstance(counts, dict) or any(not math.isfinite(float(n)) or float(n) < 0 for n in counts.values()):
                    raise ValueError('invalid F2 checkpoint hospital provenance counts')
                active = [hospital for hospital, count in counts.items() if float(count) > 0]
                found.append(validate_training_hospitals(active))
    if not found:
        raise ValueError('F2 checkpoint lacks auditable training-hospital provenance; load refused')
    return found[0]
