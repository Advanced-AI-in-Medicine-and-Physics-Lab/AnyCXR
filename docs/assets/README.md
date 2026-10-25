# README figures

Two figures introduce the method and show the existing real-CXR examples:

- [anycxr-overview.png](anycxr-overview.png): a redesigned overview using the manuscript's CT and 3D anatomy illustrations, bundled synthetic inputs, a simplified learning schematic, and the original real PA input/predicted lung panel.
- [anycxr-real-anatomy.png](anycxr-real-anatomy.png): the original inputs and **physician-refined annotations** from the PA and lateral **Doctor Annotation** rows of `Real_compare.png`, reordered as lungs, ribs/bones, vertebrae, and heart/vessels. Only anatomical group and projection labels appear in the figure.

The gallery shows the manuscript's physician-refined anatomical annotations. The PA and lateral views are separate examples, with no asserted cross-view patient correspondence. The manuscript's oblique row contains predictions only, so it is omitted from this physician-refined gallery. Overlapping structures are retained. The physicians refined model predictions; these annotations are not independent blind ground truth.

## Sources and faithful composition

The source is the retained manuscript repository [Agaresd47/Anycxr_overleaf](https://github.com/Agaresd47/Anycxr_overleaf), commit `c79c1383774aa766b5172f49e305dcbd3a3bb9f6`:

- `figure/Figure_Method_Data.png`: CT volume and segmented 3D anatomy illustrations.
- `figure/Real_compare.png`: real PA and lateral inputs and physician-refined annotation panels; original prediction crops are retained separately as source history.
- `figure/model.png` and the accompanying method text inform the simplified partial-label/CAR schematic. This schematic is conceptual, not a layer-by-layer network specification.

[sources.json](sources.json) records the parent-image SHA-256, exact crop boxes, panel types and retained panel hashes. The eight `doctor-*` panels are direct crops from the manuscript's Doctor Annotation rows. Their anatomical content is preserved, with only cropping, aspect-preserving resizing and layout. No new inference, inpainting or mask editing was used to compose this gallery. The CT/3D illustration and synthetic thumbnails in the overview are conceptual examples, not a registered chain of one patient across CT, DRR and real CXR.

The original manuscript rasters set the resolution of the retained crops. Panels are retained in [source/](source/), allowing the figures to be rebuilt without access to a private paper repository or a clinical dataset. Earlier prediction crops and the three native renders recorded in [native-panels.json](native-panels.json) remain as historical source assets; they are not used by the current physician-refined gallery.

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

The optional `--native-predictions` argument rebuilds historical prediction overlays only; it does not alter the physician-refined gallery.
