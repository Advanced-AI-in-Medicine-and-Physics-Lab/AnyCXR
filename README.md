<p align="center">
  <img src="docs/assets/wordmark.svg" alt="AnyCXR" width="370">
</p>
<h2 align="center">Anatomical segmentation of chest X-rays<br>with synthetic supervision</h2>
<p align="center">
  <b>54 anatomical structures · PA, lateral &amp; oblique views</b><br>
  Multi-stage Domain Randomization &amp; Conditional Joint Annotation Regularization
</p>
<p align="center">
  Zifei Dong*, Wenjie Wu*, Jinkui Hao, Tianqi Chen, Ziqiao Weng, Bo Zhou<br>
  <sub>* Equal contribution</sub>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2512.17263"><img src="https://img.shields.io/badge/arXiv-2512.17263-b31b1b?style=flat-square" alt="AnyCXR paper on arXiv"></a>
  <a href="https://huggingface.co/agaresd/anychest-inference"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Model_weights-7752a4?style=flat-square" alt="Hugging Face model weights"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Get_started-Installation-336c91?style=flat-square" alt="Installation"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/CLI-Quick_start-336c91?style=flat-square" alt="CLI quick start"></a>
  <a href="#anatomical-structures"><img src="https://img.shields.io/badge/Anatomy-54_structures-7752a4?style=flat-square" alt="54 anatomical structures"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-Documentation-7752a4?style=flat-square" alt="Training documentation"></a>
</p>
<p align="center">
  <img src="docs/assets/anycxr-overview.png" alt="AnyCXR overview: CT anatomy labels, multi-stage randomized synthetic radiographs, partial-label learning with CAR, and real chest X-ray anatomy prediction" width="100%">
</p>

**AnyCXR** learns chest X-ray anatomy from CT-derived synthetic radiographs and partial anatomical labels. Multi-stage Domain Randomization (MSDR) varies projection geometry and image appearance, while Conditional Joint Annotation Regularization (CAR) models joint anatomy under incomplete supervision. The inference package exports 54 anatomical channels as masks and color overlays.

This repository provides the AnyCXR inference package, pretrained model weights, example inputs and outputs, Docker support, evaluation tools, and [training documentation](training/README.md).

[Installation](#installation) · [Quick start](#quick-start) · [Model weights](#model-weights) · [Structures](#anatomical-structures) · [Method](#method) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

## Anatomy on real chest X-rays

<p align="center">
  <a href="docs/assets/anycxr-real-anatomy.png"><img src="docs/assets/anycxr-real-anatomy.png" alt="Square-panel chest X-ray anatomy gallery: physician-refined PA and lateral annotations, plus oblique model predictions, grouped as lungs, ribs and bones, vertebrae, and heart and vessels" width="100%"></a>
</p>

**Lungs → ribs → vertebrae → heart and vessels.** PA and lateral rows show the manuscript's **physician-refined annotations**; the oblique (OB) row shows **model predictions**. Each row includes the original radiograph.

Browse all **54 supported output channels** in the [anatomy catalog](docs/anatomy.md), or use the [machine-readable JSON](docs/anatomy.json) for group names, class names and exact channel indices. [Figure sources and reproduction](docs/assets/README.md) document the original panels and composition.

## Installation

Python **3.9 or newer** is required. Create an isolated environment, then install from this repository:

```bash
git clone https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/AnyCXR.git
cd AnyCXR
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
anycxr --help
```

The installed command is **`anycxr`**. The previous `anychest-infer` command remains available as a compatibility alias. The package installs PyTorch and the image-processing dependencies declared in [pyproject.toml](pyproject.toml). For GPU inference, use a PyTorch build compatible with your CUDA environment. CPU inference is available with `--device cpu`.

## Model weights

Download the inference bundle from [Hugging Face → agaresd/anychest-inference](https://huggingface.co/agaresd/anychest-inference):

```bash
python -m pip install huggingface_hub
hf download agaresd/anychest-inference anychest_inference_bundle.pt --local-dir ./weights
```

The bundle contains **three model checkpoints**: PA, lateral and oblique. Six angle-specific oblique profiles select the shared oblique checkpoint and the matching view metadata.

| Projection | CLI profile | Checkpoint in the bundle |
| --- | --- | --- |
| PA | `pa` | `pa` |
| Lateral | `la` | `la` |
| Oblique, 22.5° | `oblique_22_5` | `oblique` |
| Oblique, 45° | `oblique_45` | `oblique` |
| Oblique, 67.5° | `oblique_67_5` | `oblique` |
| Oblique, 112.5° | `oblique_112_5` | `oblique` |
| Oblique, 135° | `oblique_135` | `oblique` |
| Oblique, 157.5° | `oblique_157_5` | `oblique` |

Use `--profile oblique` for a dataset-style directory with view-specific subfolders. For a single image or a flat folder, choose an angle-specific profile. The release does not provide a separate AP checkpoint or continuous angle estimation.

This bundle is **inference-only**: optimizer and scheduler states were removed.

## Quick start

Run the bundled PA example:

```bash
anycxr \
  --input-path examples/inputs/pa \
  --output-dir outputs/pa \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile pa \
  --device cpu
```

For a lateral radiograph, use `--profile la`. For a 45° oblique radiograph, use `--profile oblique_45`:

```bash
anycxr \
  --input-path /path/to/image.png \
  --output-dir outputs/oblique \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile oblique_45 \
  --device cuda:0
```

`--input-path` accepts **PNG, JPEG, DICOM and NIfTI (`.nii`, `.nii.gz`)** radiographs, a folder of images, or a view-organized directory such as `PA/imagesTr`, `LA/imagesTr` and `45/imagesTr`. NIfTI inputs must contain a 2D radiograph; singleton slice dimensions are accepted. A profile selects the model and acquisition view.

### Outputs and settings

The default output contains a separate PNG mask for every class and a color overlay, organized by view:

```text
outputs/pa/PA/
├── labelsTr/
│   └── image_name/
│       ├── heart.png
│       ├── lung_upper_lobe_left.png
│       ├── rib_left_1.png
│       └── ...
└── overlays/
    └── image_name_overlay.png
```

| Option | Purpose | Default |
| --- | --- | --- |
| `--device` | PyTorch device: `cpu`, `cuda:0`, etc. | `cuda:0` |
| `--batch-size` | Number of images per inference batch | `8` |
| `--save-format` | PNG masks (`img`) or NumPy arrays (`npy`) | `img` |
| `--save-option` | Per-image mask folder (`sep`), flat class files (`one`), or `_total` folder (`total`) | `sep` |
| `--disable-overlays` | Skip color overlay export | Off |
| `--view-name` | Explicit view metadata for ambiguous inputs | Inferred from profile/layout |
| `--metadata-json` | Override the packaged reference JSON | Packaged reference |

All save modes retain separate anatomical channels. Projected structures can overlap, so the output is not a single mutually exclusive label map. See `anycxr --help` for all options.

### Docker

The Docker image supports **NVIDIA GPU and CPU inference** and uses a CUDA-enabled PyTorch base. Build it with:

```bash
docker build -t anycxr .
docker run --rm anycxr --help
```

For GPU inference, install the NVIDIA Container Toolkit on the host and expose the GPU with `--gpus all`:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/AnyCXR" \
  anycxr \
  --input-path examples/inputs/pa \
  --output-dir outputs/pa \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile pa --device cuda:0
```

For CPU inference, omit `--gpus all` and select `--device cpu`:

```bash
docker run --rm \
  -v "$PWD:/workspace/AnyCXR" \
  anycxr \
  --input-path examples/inputs/pa \
  --output-dir outputs/pa \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile pa --device cpu
```

The base image can be selected at build time with `--build-arg PYTORCH_IMAGE=<pytorch-image>` to match your CUDA environment.

## Anatomical structures

The [anatomy catalog](docs/anatomy.md) lists every supported structure, and the [machine-readable anatomy JSON](docs/anatomy.json) provides its group and **zero-based channel index**. Both are generated from the packaged [reference JSON](cxas/data/anychest_reference.json), which remains the source of truth for the exact **54-channel output order**, view names and dataset-folder mapping. These counts describe the output taxonomy, not per-structure accuracy.

| Group | Channels | Structures |
| --- | ---: | --- |
| Lungs | 5 | Left upper/lower lobes; right upper/middle/lower lobes |
| Ribs | 24 | Left and right ribs 1–12 |
| Thoracic vertebrae | 11 | T2–T12 |
| Heart and vessels | 9 | Whole heart, left/right ventricles, left/right atria, myocardium; pulmonary artery, `aorta`, `aorta_long` |
| Other bones | 5 | Left/right clavicles, left/right humeri, sternum |

The `aorta` and `aorta_long` outputs are separate channels; the packaged reference does not specify a finer anatomical distinction. Left/right names refer to the patient's anatomy. Independent channels preserve overlapping projected structures.

<details>
<summary><b>Show all 54 class names in inference order</b></summary>

```text
heart_ventricle_left, heart_atrium_left, heart_ventricle_right,
heart_atrium_right, heart_myocardium, pulmonary_artery, heart,
rib_left_1, rib_left_2, rib_left_3, rib_left_4, rib_left_5, rib_left_6,
rib_left_7, rib_left_8, rib_left_9, rib_left_10, rib_left_11, rib_left_12,
rib_right_1, rib_right_2, rib_right_3, rib_right_4, rib_right_5, rib_right_6,
rib_right_7, rib_right_8, rib_right_9, rib_right_10, rib_right_11, rib_right_12,
aorta, vertebrae_T2, vertebrae_T3, vertebrae_T4, vertebrae_T5,
vertebrae_T6, vertebrae_T7, vertebrae_T8, vertebrae_T9, vertebrae_T10,
vertebrae_T11, vertebrae_T12, aorta_long,
lung_upper_lobe_left, lung_upper_lobe_right, lung_middle_lobe_right,
lung_lower_lobe_left, lung_lower_lobe_right,
clavicula_left, clavicula_right, humerus_left, humerus_right, sternum
```

</details>

## Method

The overview at the top adapts the manuscript's CT and 3D anatomy illustrations into the four-stage workflow below.

1. **Synthetic supervision.** Project CT volumes and anatomical labels into matched radiographs and segmentation targets. Quality control determines which class labels are available.
2. **Multi-stage Domain Randomization (MSDR).** Perturb the CT appearance, projection geometry, and detector/display appearance to diversify the synthetic training domain.
3. **Partial-label learning with CAR.** Supervise available classes and regularize the joint anatomical representation conditional on the image, allowing training with incomplete annotations.
4. **Anatomical inference.** Apply the trained model to chest radiographs using the appropriate view profile; export masks for analysis and overlays for inspection.

The manuscript's real-CXR reference annotations were refined from model predictions. Consider this when interpreting segmentation accuracy, particularly for ambiguous soft-tissue boundaries.

## Examples

[examples/](examples/README.md) contains inputs and existing outputs for PA, lateral and 45° oblique profiles. To regenerate them after downloading the weights:

```bash
DEVICE=cpu bash scripts/build_example_cases.sh
```

These synthetic examples complement the real-CXR gallery above. See [examples](examples/README.md) for the included cases and [figure documentation](docs/assets/README.md) for the gallery sources.

## Training

The [training guide](training/README.md) documents the U-Net/CAR source and PA, lateral and oblique configuration templates. The [frontal training guide](training/frontal/README.md) covers executable AP/PA training, patient-level splits, checkpoint recovery, and full/inference-only export.

Install the optional training dependencies with:

```bash
python -m pip install -e '.[train,frontal]'
```

## Evaluation

Evaluate your own radiographs against per-structure reference masks:

```bash
python scripts/evaluate.py \
  --dataset-dir /path/to/evaluation_data \
  --checkpoint weights/anychest_inference_bundle.pt \
  --profile pa \
  --output-dir outputs/evaluation/pa \
  --device cuda:0
```

Use `--profile la` for lateral images or `--profile oblique` for multiple oblique views. Results include `per_image_dice.csv`, `per_class_summary.csv` and `summary.json` in the output's `reports/` directory. See the [evaluation guide](docs/evaluation.md) for the expected image/mask layout and metric definitions.

## Citation

If you use AnyCXR in your research, please cite our [arXiv paper](https://arxiv.org/abs/2512.17263):

```bibtex
@misc{dong2025anycxr,
  title = {AnyCXR: Human Anatomy Segmentation of Chest X-ray at Any Acquisition Position using Multi-stage Domain Randomized Synthetic Data with Imperfect Annotations and Conditional Joint Annotation Regularization Learning},
  author = {Zifei Dong and Wenjie Wu and Jinkui Hao and Tianqi Chen and Ziqiao Weng and Bo Zhou},
  year = {2025},
  eprint = {2512.17263},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2512.17263},
  url = {https://arxiv.org/abs/2512.17263}
}
```

## License and acknowledgments

Source licenses are documented in the repository's [Apache-2.0 license](LICENSE), the inference components' [MIT notice](LICENSES/AnyCXR-inference-MIT.txt), and the retained training source's [CC BY-NC-SA 4.0 license](training/legacy/LICENSE.txt). Additional attribution and inherited licensing declarations are recorded in [NOTICE](NOTICE). Checkpoint terms are provided with the [Hugging Face model release](https://huggingface.co/agaresd/anychest-inference).

The initial inference release was maintained at [Agaresd47/AnyCXR](https://github.com/Agaresd47/AnyCXR). README organization and presentation were inspired by [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and [FleXray](https://flexray.csail.mit.edu/); all displayed segmentation examples are from AnyCXR.
