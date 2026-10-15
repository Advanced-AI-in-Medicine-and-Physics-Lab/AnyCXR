# Training preparation

The repository now retains the **AnyCXR U-Net / CAR training source** alongside the public inference package. The legacy source and templates remain preparation-only. An executable experimental [frontal ConvNeXtV2 route](frontal/README.md) now adds real-CXR initialization, DRR fine-tuning, patient-group exclusion, exact validation, explicit resume, and paired full/inference export.

## Included source

`legacy/` contains the original trainer, dataset loader, augmentation code, U-Net/CAR components, loss, shared configuration helper and training entry point. These 16 files are identical in the retained ARES and EROS snapshots at commit `daee4030227a3ffe5b298f3a564550951f6a225f`. Exact source paths and SHA-256 values are recorded in [source_manifest.json](source_manifest.json).

The training source is kept in its own directory because both the training and public inference implementations use the Python package name `cxas`. The regular installed package continues to resolve to the public inference implementation. Baseline vendors, machine-specific configuration, notebooks and image data are not part of this training import. The original [CC BY-NC-SA 4.0 license](legacy/LICENSE.txt) is retained for the imported training source.

## Configuration templates

| Model | Template | Selected projections |
| --- | --- | --- |
| PA | [configs/pa.json](configs/pa.json) | PA |
| Lateral | [configs/la.json](configs/la.json) | LA |
| Oblique | [configs/oblique.json](configs/oblique.json) | Six oblique views from the inference reference |

Hyperparameters start from the retained shared configuration; these templates are **not verified recipes for the released checkpoints**. Dataset paths, fixed split manifest/hash and output paths remain `null`. The inference class reference documents the intended 54-channel order; the actual training dataset must be checked against it before export.

Inspect a template using only the Python standard library:

```bash
python training/train.py --config training/configs/pa.json --dry-run
python training/train.py --config training/configs/la.json --dry-run
python training/train.py --config training/configs/oblique.json --dry-run
```

The result reports `training_started: false`, `ready_to_train: false`, unfilled fields and reserved checkpoint IDs. A successful dry run checks the preparation structure; it does not validate data, train a model or establish reproducibility. Running this preparation entry point without `--dry-run` exits with an explanation.

The optional dependencies for the retained training source can be installed from a checkout:

```bash
python -m pip install -e '.[train]'
```

The original `legacy/train_anycxr.py` is preserved as source, including its original configuration interface. It is not the supported preparation entry point.

## Remaining integration

The legacy `get_5fold_dataloaders` constructs five variation-priority train/validation samples dynamically. Its function name does not imply an immutable, patient-grouped fold contract. The next executable training integration must use the existing approved split or a separately defined frozen manifest, preserve group separation across views, and keep locked test data outside training and model selection.

The legacy trainer writes `latest.pth`, `best_no_jtr.pth` and `best_jtr.pth`. It includes model, optimizer and scheduler state, epoch, phase-related best losses, view/class selection, normalization and resolution metadata. RNG and AMP scaler states are not saved by this source, so those files alone do not establish exact continuation.

Before a training release, complete the fixed split adapter, data/class-order checks, training/inference architecture compatibility and checkpoint export validation. Record the actual configuration, source weights, metrics and hashes with the selected checkpoint. The original legacy import involved no training or image-data access. The separate frontal integration is validated against retained checkpoints and a bounded segmentation experiment; see its own documentation for scope.

See the [checkpoint registry](../checkpoints/README.md) and [HF model-card preparation](../hf/README.md) for the paired full/inference release layout.
