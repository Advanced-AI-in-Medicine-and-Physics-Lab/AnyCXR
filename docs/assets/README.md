# README figures

Two figures introduce the method and show the existing real-CXR examples:

- [anycxr-overview.png](anycxr-overview.png): a redesigned overview using the manuscript's CT and 3D anatomy illustrations, bundled synthetic inputs, a simplified learning schematic, and the original real PA input/predicted lung panel.
- [anycxr-real-anatomy.png](anycxr-real-anatomy.png): inputs and predictions for the `Real_compare.png` examples, reordered as lungs, ribs/bones, vertebrae, and heart/vessels. The three panels containing white presentation circles (PA ribs/bones, PA heart/vessels, lateral ribs/bones) are rendered from the matching retained per-class prediction PNGs. Other panels retain the original manuscript crops. Only anatomical group and projection labels appear in the figure.

The real examples come from the AnyCXR manuscript, rather than the synthetic inputs bundled with the inference CLI. They illustrate the manuscript model; they are not fresh predictions from the current frontal candidate or a new benchmark. The views are separate examples; the source does not establish that all three belong to one patient. Overlapping structures are retained. The expert-refined references in the manuscript started from predictions and are not independent blind ground truth.

## Sources and faithful composition

The source is the retained manuscript repository [Agaresd47/Anycxr_overleaf](https://github.com/Agaresd47/Anycxr_overleaf), commit `c79c1383774aa766b5172f49e305dcbd3a3bb9f6`:

- `figure/Figure_Method_Data.png`: CT volume and segmented 3D anatomy illustrations.
- `figure/Real_compare.png`: real PA, lateral and oblique input/prediction panels.
- `figure/model.png` and the accompanying method text inform the simplified partial-label/CAR schematic. This schematic is conceptual, not a layer-by-layer network specification.

[sources.json](sources.json) records the parent-image SHA-256, exact crop boxes, retained manuscript panel paths and panel hashes. [native-panels.json](native-panels.json) records the source image/mask hashes and the three rendered panel hashes. The PA/lateral inputs were matched to the manuscript input panels with image correlation above 0.99. The affected groups' original per-class prediction masks are displayed with translucent colors and boundary outlines, without adding the manuscript's discrepancy circles. No mask boundary was edited, and no inpainting, new inference or training was used. Other panels retain their original cropped pixels. The CT/3D illustration and synthetic thumbnails are conceptual examples, not a registered chain of one patient across CT, DRR and real CXR.

The original manuscript rasters set the resolution of the retained crops. Source crops and rendered native panels are retained in [source/](source/), allowing the figures to be rebuilt without access to a private paper repository or a clinical dataset. Original crop files, including the presentation circles, remain available as source history; the gallery uses separate native prediction renders for the three affected panels.

## Rebuild

With the inference package's Pillow dependency and either FreeSans or DejaVu Sans installed:

```bash
python scripts/build_readme_assets.py
```

This verifies all retained panel hashes, rebuilds both figures, and generates [anatomy.json](../anatomy.json) and [anatomy.md](../anatomy.md) from the packaged 54-channel reference. It does not load weights or run inference. To recreate the original crops from a manuscript checkout:

```bash
python scripts/build_readme_assets.py --paper-figures /path/to/Anycxr_overleaf/figure
```

Use the recorded manuscript revision when recreating crops; a different raster may require new crop coordinates and provenance.

To rerender the PA/lateral overlays from retained native predictions, pass `--native-predictions /path/to/matched-predictions`. That directory must contain `pa/` and `lateral/`, each with `input.png` (or `input.jpg`) and the 54 class-name PNG masks. Native image and mask files are not distributed with these documentation assets; the rendered panels and anonymous source hashes are retained here.
