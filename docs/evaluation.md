# Evaluate your own data

Organize radiographs by acquisition view, with one reference PNG per available anatomical structure:

```text
evaluation_data/
├── PA/
│   ├── imagesTr/
│   │   └── example.nii.gz
│   └── labelsTr/
│       └── example_total/
│           ├── heart.png
│           ├── rib_left_1.png
│           └── ...
├── LA/
│   ├── imagesTr/
│   └── labelsTr/
└── 45/
    ├── imagesTr/
    └── labelsTr/
```

Images can be PNG, JPEG, DICOM or 2D NIfTI (`.nii`, `.nii.gz`). The example name is arbitrary: `example.png`, `example.nii` and `example.nii.gz` each pair with `labelsTr/example_total/`. Put images directly in `imagesTr/`; reference masks must match the radiograph's pixel dimensions and orientation. Reference filenames use the class names in the [anatomy catalog](anatomy.md).

View folder names are `PA`, `LA`, and `225`, `45`, `675`, `1125`, `135`, `1575` for the six oblique profiles. Select a model/view with `--profile`:

```bash
python scripts/evaluate.py \
  --dataset-dir /path/to/evaluation_data \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile pa \
  --output-dir outputs/evaluation/pa \
  --device cuda:0
```

For CPU, use `--device cpu`. Repeat with `--profile la` or `--profile oblique` to evaluate lateral or all available oblique views. `--skip-inference` scores predictions already present in the output directory. With no saved predictions, run inference first.

## Reports and metrics

Predictions use separate anatomical PNGs under `<output>/<view>/labelsTr/<image_name>/`. Reports are written to `<output>/reports/`:

- `per_image_dice.csv`: Dice for each image/class pair and the image's mean over scored classes.
- `per_class_summary.csv`: mean, standard deviation and sample count for each view/class pair.
- `summary.json`: selected views, number of scored view/class groups, and their equally weighted mean Dice.

Dice is `2 × intersection / (prediction area + reference area)`. PNG mask values above 127 are foreground; two empty masks score 1. Missing reference classes are skipped rather than treated as negative labels. The aggregate is a mean of view/class means, not a patient-weighted or pooled-pixel score. Inspect sample counts when comparing datasets with incomplete references.

The bundled example cases demonstrate the interface; evaluate performance using a dataset appropriate to your research question and maintain independent training and evaluation patients.
