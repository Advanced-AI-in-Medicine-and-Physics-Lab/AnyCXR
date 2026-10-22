# README figures

Two figures introduce the method and show the existing real-CXR examples:

- [anycxr-overview.png](anycxr-overview.png): a redesigned overview using the manuscript's CT and 3D anatomy illustrations, bundled synthetic inputs, a simplified learning schematic, and the original real PA input/predicted lung panel.
- [anycxr-real-anatomy.png](anycxr-real-anatomy.png): the input and **prediction** panels from `Real_compare.png`, reordered as lungs, ribs/bones, vertebrae, and heart/vessels for PA, lateral and oblique examples. Expert-refined annotation rows are omitted.

The real examples come from the AnyCXR manuscript, rather than the synthetic inputs bundled with the inference CLI. They illustrate the manuscript model; they are not fresh predictions from the current frontal candidate or a new benchmark. The views are separate examples; the source does not establish that all three belong to one patient. Lung colors follow the original overlays, and overlapping structures are retained. White circles remain visible where the manuscript highlighted discrepancies requiring expert editing. The expert-refined references in the manuscript started from predictions and are not independent blind ground truth.

## Sources and faithful composition

The source is the retained manuscript repository [Agaresd47/Anycxr_overleaf](https://github.com/Agaresd47/Anycxr_overleaf), commit `c79c1383774aa766b5172f49e305dcbd3a3bb9f6`:

- `figure/Figure_Method_Data.png`: CT volume and segmented 3D anatomy illustrations.
- `figure/Real_compare.png`: real PA, lateral and oblique input/prediction panels.
- `figure/model.png` and the accompanying method text inform the simplified partial-label/CAR schematic. This schematic is conceptual, not a layer-by-layer network specification.

[sources.json](sources.json) records the parent-image SHA-256, exact crop boxes, retained panel paths and panel hashes. The original clinical image/overlay pixels are cropped, resized with their aspect ratio preserved, and composed into a new layout. There is no anatomical repainting, mask correction, new inference or training. Original discrepancy circles are not removed. The CT/3D illustration and synthetic thumbnails are conceptual examples, not a registered chain of one patient across CT, DRR and real CXR.

The original manuscript rasters set the available resolution. Source crops are retained in [source/](source/), allowing the figures to be rebuilt without access to a private paper repository or a clinical dataset.

## Rebuild

With the inference package's Pillow dependency and either FreeSans or DejaVu Sans installed:

```bash
python scripts/build_readme_assets.py
```

This verifies all retained panel hashes, rebuilds both figures, and generates [anatomy.json](../anatomy.json) and [anatomy.md](../anatomy.md) from the packaged 54-channel reference. It does not load weights or run inference. To recreate the crops from an original manuscript checkout:

```bash
python scripts/build_readme_assets.py --paper-figures /path/to/Anycxr_overleaf/figure
```

Use the recorded manuscript revision when recreating crops; a different raster may require new crop coordinates and provenance.
