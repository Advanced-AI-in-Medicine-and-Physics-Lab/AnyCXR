---
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - chest-xray
  - anycxr
  - training-checkpoint
---

# AnyCXR — Full training checkpoints

**LOCAL MODEL-CARD TEMPLATE · CHECKPOINTS NOT YET PROVIDED · NOT PUBLISHED**

This planned release will contain full AnyCXR training checkpoints for PA, lateral and shared oblique models. The HF repository name, final checkpoint selection and license remain unassigned. There are no downloadable weights in this preparation.

## Reserved files

| Model | Latest | Best without CAR | Best with CAR |
| --- | --- | --- | --- |
| PA | `pa/latest.pth` | `pa/best_no_jtr.pth` | `pa/best_jtr.pth` |
| Lateral | `la/latest.pth` | `la/best_no_jtr.pth` | `la/best_jtr.pth` |
| Oblique | `oblique/latest.pth` | `oblique/best_no_jtr.pth` | `oblique/best_jtr.pth` |

Availability and source identity are recorded in the lab repository's checkpoint registry. All nine entries are placeholders. Inclusion and checkpoint selection will be based on the actual supplied weights.

## Expected training state

The retained trainer serializes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, phase-specific best losses, view/class selection, normalization, progressive resolution and CAR settings. Complete the actual configuration, dataset/split provenance, runtime version, source checkpoint and verification results before release.

The retained source does not save RNG or AMP scaler state. Exact resumed continuation is not established by the presence of optimizer and scheduler tensors alone. The next release must document which state was restored and any intentional resets.

## Intended use

Research training, continued development and experiment inspection. The checkpoint files require matching training/model code and configuration. They are distinct from the slim inference-only release.

## Training data and evaluation

Dataset version: pending. Frozen split: pending. Training configuration: pending. Checkpoint selection: pending. Metrics and evaluation provenance: pending. These fields must be supplied for the actual release; no new training or evaluation is claimed here.

## License and citation

Model-weight terms are pending and must be established separately from inherited source-code licenses. Citation details will follow the AnyCXR paper metadata when available.
