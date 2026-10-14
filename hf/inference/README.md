---
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - chest-xray
  - anycxr
  - inference
---

# AnyCXR — Inference-only checkpoints

**LOCAL MODEL-CARD TEMPLATE · CHECKPOINTS NOT YET PROVIDED · NOT PUBLISHED**

This planned release will contain slim AnyCXR weights for PA, lateral and shared oblique inference, plus a combined bundle. The future HF repository name and model-weight terms remain unassigned. This template does not replace the existing `agaresd/anychest-inference` release.

## Reserved files

| Model | Reserved inference artifact |
| --- | --- |
| PA | `pa/inference.pt` |
| Lateral | `la/inference.pt` |
| Oblique | `oblique/inference.pt` |
| All three models | `anychest_inference_bundle.pt` |

Each slot is a placeholder. The oblique model supplies six angle-specific inference profiles through shared weights and view metadata.

## Export contents

Inference artifacts will retain model weights and metadata required for architecture construction, class order, view selection, normalization and input resolution. Optimizer and scheduler state will be omitted. A matching `anychest_reference.json` and verified full-checkpoint source identity must accompany the release.

Validate export/import compatibility with the lab inference package and compare predictions with the selected full checkpoints before marking weights available. Do not infer compatibility from filenames or from a 54-channel count alone.

## Usage

Download commands will be added after the actual repository and pinned revision exist. Installation and the current released inference workflow are documented in the lab AnyCXR README. No command pointing at an unassigned repository is provided here.

## Evaluation and limitations

Actual checkpoint selection, dataset/split provenance, validation outputs, source hashes and export parity results are pending. The public example montage is a qualitative synthetic illustration and does not validate these future weights.

## License and citation

Model-weight terms are pending. Citation details will follow the AnyCXR paper metadata when available.
