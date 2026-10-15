# Frontal ConvNeXtV2 training and inference

This experimental route reuses a verified **real-CXR-pretrained ConvNeXtV2 encoder**, creates a fresh AnyCXR CAS-lite/AMFS segmentation decoder, and fine-tunes with D3 and geometry-matched CT/DRR supervision. It supports AP and PA. Existing PA/lateral/oblique ResNet50 release profiles remain available through the original CLI.

The retained Large models use 384 × 384 inputs and **54 independent channels on a 96 × 96 target grid**. Upsampling does not establish finer anatomical accuracy. No new checkpoint is published or selected as the released default by this integration.

## Environment

From a repository checkout:

```bash
python -m pip install -e '.[train,frontal]'
```

The integration is exercised with PyTorch 2.8.0+cu128 and Transformers 4.56.2. The Transformers version is pinned for checkpoint compatibility.

## Required inputs

- A completed filtered D3 cache: metadata with exact class names, records, uint8 images, soft uint8 targets and per-organ availability.
- Completed CT/MSDR target shards with packed 96-grid masks, per-organ availability and source-image references. Standard/portable target slots retain their own camera geometry.
- A completed source-mask exclusion manifest. Invalid organs are masked out of the loss; images are retained.
- An encoder bundle with HF `config`, encoder `model_state_dict`, and audited real-CXR initialization `provenance`.
- A frozen patient-group exclusion manifest and recipe. Data and checkpoint files are supplied separately; they are not included in Git.

The exclusion manifest has schema `anycxr_eval_group_exclusion_v1` and lists `d3_validation_groups`, `nw_groups`, and their union as `excluded_patient_groups`. CT `valid_N` and D3 `t_N` resolve to the same patient group. Every reconstruction and projection of an excluded patient is removed from training. Untraceable evaluation inputs cannot serve as a clean held-out cohort.

The recipe fixes the seed, epochs, D3/CT samples per epoch, batch/workers per GPU, encoder/decoder learning rates, weight decay, normalization, encoder/source hashes and membership hash. The runner verifies all validation patients are excluded. It accepts synthetic segmentation supervision and an audited public real-CXR encoder; clinical evaluation images never enter its optimizer. Original training-hospital policy checks are retained in `hospital_training_policy.py`.

## Train

```bash
torchrun --standalone --nproc-per-node=4 -m training.frontal.train \
  --d3-cache /path/to/d3_cache \
  --ct-target-root /path/to/ct_targets \
  --source-exclusion-manifest /path/to/mask_exclusion_manifest.json \
  --evaluation-exclusions /path/to/evaluation_group_exclusions.json \
  --backbone-bundle /path/to/real_cxr_encoder.pt \
  --recipe /path/to/frozen_recipe.json \
  --output-dir /path/to/new_run
```

Add `--smoke-only` for two optimizer steps and a complete validation pass, without saving candidate weights. Validation uses each image exactly once across ranks, with rank-0 BatchNorm buffers broadcast before evaluation. Selection uses the frozen D3 development set; exposed NW and clinical comparisons are descriptive.

The output directory must be empty unless `--resume /path/to/last.pt` is explicit. Resume verifies the frozen configuration, restores optimizer/scheduler and per-rank RNG states, and requires the same world size. The dataset's per-epoch sampling is deterministic. CUDA kernels may remain nondeterministic; bitwise training replay is not claimed.

Successful training saves:

| Artifact | Contents |
| --- | --- |
| `last.pt`, `best.pt` | Model, optimizer, scheduler, epoch, development score, config and RNG state for every rank |
| `inference.pt` | Selected model, backbone configuration, channel/view/normalization/camera metadata, source hashes; no optimizer or separate encoder download |
| `metrics.jsonl`, `progress.json` | Development measurements and current progress |
| `COMPLETE.json` | Completion and checkpoint hashes; publication/adoption remain false |

BF16 autocast is used without a gradient scaler. Completion means the prescribed run and export finished; scientific acceptance still requires comparison and inspection. The original inference release is not changed automatically.

## Export existing full checkpoints

Only the verified `convnextv2_anycxr_anychest_cas_lite_amfs1_jtr0` architecture is accepted:

```bash
python -m training.frontal.export \
  --source /path/to/best.pt \
  --backbone /path/to/original_encoder_bundle.pt \
  --output /path/to/inference.pt
```

The exporter refuses to overwrite an existing file. Retained legacy-camera Phase A and canonical-camera Phase B preserve their different PA preprocessing contracts in metadata.

## Inference

```python
import numpy as np
from PIL import Image
from cxas.frontal import FrontalSegmentor

segmentor = FrontalSegmentor("inference.pt", device="cuda:0")
image = np.asarray(Image.open("chest.png").convert("L"))
probabilities = segmentor.predict([image], view="PA")  # (1, 54, 96, 96)
masks = probabilities >= 0.5
```

Supply the actual acquisition view and a 2D uint8 grayscale image in canonical viewer orientation. There is no automatic AP/PA inference. The loader applies the recorded PA camera adapter when required and returns probabilities in the input's orientation. It does not apply the legacy U-Net control-point/min-max preprocessing. Keep overlapping channels; do not replace them with an exclusive argmax label map.

Source lineage: the retained ConvNeXtV2 segmentation implementation, D3 loader and CT/MSDR target loader were integrated on 2026-10-15. CAS/AMFS, angle injection, loss and channel meanings reuse the existing AnyCXR components. The new split guard, exact validation, explicit resume and paired export form the executable training route; the legacy templates remain preparation-only.
