# README figures

Two figures introduce the method and show the existing real-CXR examples:

- [anycxr-overview.png](anycxr-overview.png): a redesigned overview using the manuscript's CT and 3D anatomy illustrations, bundled synthetic inputs, a simplified learning schematic, and the original real PA input/predicted lung panel. Four square stages use a blue background with white and pale-blue typography; source thumbnails fit square frames while retaining their proportions.
- [anycxr-real-anatomy.png](anycxr-real-anatomy.png): the original inputs and **physician-refined annotations** from the PA and lateral **Doctor Annotation** rows of `Real_compare.png`, followed by the manuscript's **oblique model predictions**. Columns show lungs, ribs/bones, vertebrae, and heart/vessels. All 15 panels fit identical square tiles in a tight blue grid with white column headings and vertical projection labels. Blue space accommodates the original aspect ratios without stretching or cropping anatomy.

PA and lateral panels show the manuscript's physician-refined annotations; the oblique (OB) panels show existing model predictions. These are separate examples, with no asserted cross-view patient correspondence. Overlapping structures are retained. The physicians refined model predictions; these annotations are not independent blind ground truth.

## Sources and faithful composition

The source is the retained manuscript repository [Agaresd47/Anycxr_overleaf](https://github.com/Agaresd47/Anycxr_overleaf), commit `c79c1383774aa766b5172f49e305dcbd3a3bb9f6`:

- `figure/Figure_Method_Data.png`: CT volume and segmented 3D anatomy illustrations.
- `figure/Real_compare.png`: real PA and lateral inputs and physician-refined annotation panels, plus the oblique input and prediction panels.
- `figure/model.png` and the accompanying method text inform the simplified partial-label/CAR schematic. This schematic is conceptual, not a layer-by-layer network specification.

[sources.json](sources.json) records the parent-image SHA-256, exact crop boxes, panel types and retained panel hashes. The eight `doctor-*` panels are direct crops from the manuscript's Doctor Annotation rows, and `real-oblique-*` panels retain the manuscript's oblique inference example. [gallery-layout.json](gallery-layout.json) records the displayed source, annotation type, crop, square tile bounds and image placement of every panel. Only flat monochrome edge padding (at most six source pixels per edge) is trimmed; native radiographic backgrounds and anatomical content are retained. Display resizing preserves aspect ratios. No new inference, inpainting or mask editing was used to compose this gallery. The CT/3D illustration and synthetic thumbnails in the overview are conceptual examples, not a registered chain of one patient across CT, DRR and real CXR.

The original manuscript rasters set the resolution of the retained crops. Panels are retained in [source/](source/), allowing the figures to be rebuilt without access to a private paper repository or a clinical dataset. Earlier PA/lateral prediction crops and the three native renders recorded in [native-panels.json](native-panels.json) remain as historical source assets; they are not used by the current gallery.

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

The optional `--native-predictions` argument rebuilds historical PA/lateral prediction overlays only; it does not alter the current gallery.
