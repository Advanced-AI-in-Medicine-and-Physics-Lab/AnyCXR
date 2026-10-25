#!/usr/bin/env python3
"""Compose README figures from retained manuscript panels; export the anatomy catalog.

Requires Pillow (already an inference dependency). Default builds use retained
presentation panels; optional rendering reads existing prediction PNGs. No model
or weights are loaded. See docs/assets/README.md for source and interpretation.
"""

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
SOURCE = ASSETS / "source"
PAPER_COMMIT = "c79c1383774aa766b5172f49e305dcbd3a3bb9f6"
PAPER_REPO = "https://github.com/Agaresd47/Anycxr_overleaf"
PAPER_HASHES = {
    "Real_compare.png": "02bc15e463fef0d60e0cb779b1158c2b350115d409cf6819ff6f25c81bc52457",
    "Figure_Method_Data.png": "9ea3987528a9faf0286d2bc4b11a897480dddec38121284730cc871112bdc5d4",
}
# Boxes are left, top, right, bottom in the original manuscript raster.
# Prediction rows only; expert-refined annotations are not relabelled as outputs.
CROPS = {
    "real-pa-input": ("Real_compare.png", (9, 59, 547, 557)),
    "real-pa-lungs": ("Real_compare.png", (1141, 297, 1390, 550)),
    "real-pa-ribs": ("Real_compare.png", (596, 297, 845, 550)),
    "real-pa-vertebrae": ("Real_compare.png", (868, 297, 1117, 550)),
    "real-pa-heart-vessels": ("Real_compare.png", (1414, 297, 1663, 550)),
    "real-lateral-input": ("Real_compare.png", (9, 617, 547, 1103)),
    "real-lateral-lungs": ("Real_compare.png", (1131, 869, 1396, 1106)),
    "real-lateral-ribs": ("Real_compare.png", (596, 869, 862, 1106)),
    "real-lateral-vertebrae": ("Real_compare.png", (864, 869, 1129, 1106)),
    "real-lateral-heart-vessels": ("Real_compare.png", (1398, 869, 1663, 1106)),
    "real-oblique-input": ("Real_compare.png", (7, 1142, 399, 1509)),
    "real-oblique-lungs": ("Real_compare.png", (1059, 1180, 1349, 1496)),
    "real-oblique-ribs": ("Real_compare.png", (474, 1180, 765, 1496)),
    "real-oblique-vertebrae": ("Real_compare.png", (766, 1180, 1057, 1496)),
    "real-oblique-heart-vessels": ("Real_compare.png", (1351, 1180, 1642, 1496)),
    "paper-ct": ("Figure_Method_Data.png", (28, 90, 247, 310)),
    "paper-anatomy": ("Figure_Method_Data.png", (766, 482, 998, 786)),
}
DOCTOR_CROPS = {
    "doctor-pa-lungs": ("Real_compare.png", (1141, 32, 1390, 285)),
    "doctor-pa-ribs": ("Real_compare.png", (596, 32, 845, 285)),
    "doctor-pa-vertebrae": ("Real_compare.png", (869, 32, 1117, 285)),
    "doctor-pa-heart-vessels": ("Real_compare.png", (1414, 32, 1663, 285)),
    "doctor-lateral-lungs": ("Real_compare.png", (1131, 592, 1396, 829)),
    "doctor-lateral-ribs": ("Real_compare.png", (596, 592, 862, 829)),
    "doctor-lateral-vertebrae": ("Real_compare.png", (864, 592, 1129, 829)),
    "doctor-lateral-heart-vessels": ("Real_compare.png", (1398, 592, 1663, 829)),
}
BACKGROUND = "#0b2447"
SURFACE = "#14375e"
INK = "#ffffff"
MUTED = "#ccdcf0"
PURPLE = "#c2a6ff"
BLUE = "#7ec7ff"
TEAL = "#67e1d0"
PINK = "#ffabd7"
LINE = "#335b85"
NATIVE_GROUPS = {("pa", "ribs"), ("pa", "heart-vessels"), ("lateral", "ribs")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    name = "FreeSansBold.ttf" if bold else "FreeSans.ttf"
    candidates = [Path("/usr/share/fonts/truetype/freefont") / name]
    candidates += [Path("/usr/share/fonts/truetype/dejavu") /
                   ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("Install FreeSans or DejaVu Sans to render these figures.")


def text(draw, pos, label, size=26, fill=INK, bold=False, anchor=None):
    draw.text(pos, label, font=font(size, bold), fill=fill, anchor=anchor)


def image_box(canvas, path, box, background=None):
    x, y, w, h = box
    if background:
        ImageDraw.Draw(canvas).rectangle((x, y, x + w - 1, y + h - 1), fill=background)
    im = Image.open(path).convert("RGB")
    scale = min(w / im.width, h / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    canvas.paste(im, (int(x + (w - im.width) / 2), int(y + (h - im.height) / 2)))


def arrow(draw, start, end, color=MUTED, width=3):
    x, y = start
    xx, yy = end
    draw.line((x, y, xx, yy), fill=color, width=width)
    if yy == y:
        draw.polygon(((xx, yy), (xx - 10, yy - 6), (xx - 10, yy + 6)), fill=color)
    else:
        draw.polygon(((xx, yy), (xx - 6, yy - 10), (xx + 6, yy - 10)), fill=color)


def prepare_sources(paper_dir):
    for filename, expected in PAPER_HASHES.items():
        if sha(paper_dir / filename) != expected:
            raise ValueError("Manuscript raster differs from recorded revision: " + filename)
    SOURCE.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (filename, box) in {**CROPS, **DOCTOR_CROPS}.items():
        parent = paper_dir / filename
        dest = SOURCE / (name + ".png")
        with Image.open(parent) as im:
            im.convert("RGB").crop(box).save(dest, optimize=True)
        records.append({"asset": str(dest.relative_to(ROOT)),
                        "sha256": sha(dest), "source_file": "figure/" + filename,
                        "source_sha256": sha(parent), "crop_xyxy": list(box),
                        "panel_type": "physician_refined_annotation" if name in DOCTOR_CROPS else "retained_manuscript_panel"})
    manifest = {"source_repository": PAPER_REPO, "source_commit": PAPER_COMMIT,
                "pixel_operations": ["crop", "aspect-preserving resize", "layout"],
                "clinical_panels": "Doctor-prefixed panels are the manuscript's physician-refined annotation rows; other panels retain the original inputs and predictions.",
                "view_relationship": "Separate examples; cross-view patient identity is not established.",
                "assets": records}
    (ASSETS / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_sources():
    manifest = json.loads((ASSETS / "sources.json").read_text())
    for record in manifest["assets"]:
        if sha(ROOT / record["asset"]) != record["sha256"]:
            raise ValueError("Source panel differs from manifest: " + record["asset"])
    clean_manifest = json.loads((ASSETS / "native-panels.json").read_text())
    for record in clean_manifest["panels"]:
        if sha(ROOT / record["asset"]) != record["sha256"]:
            raise ValueError("Native prediction panel differs from manifest: " + record["asset"])


def render_native_panels(input_dir):
    """Render existing masks without the manuscript's presentation circles."""
    names = json.loads((ROOT / "cxas/data/anychest_reference.json").read_text())["class_names"]
    groups = {
        "lungs": [n for n in names if n.startswith("lung_")],
        "ribs": [n for n in names if n.startswith(("rib_", "clavicula_", "humerus_")) or n == "sternum"],
        "vertebrae": [n for n in names if n.startswith("vertebrae_")],
        "heart-vessels": ["heart"] + [n for n in names if n != "heart" and (n.startswith("heart_") or n in {"aorta", "aorta_long", "pulmonary_artery"})],
    }
    lung_colors = {"lung_upper_lobe_left": (238, 55, 63), "lung_lower_lobe_left": (245, 215, 44),
                   "lung_upper_lobe_right": (53, 209, 80), "lung_middle_lobe_right": (177, 75, 224),
                   "lung_lower_lobe_right": (34, 174, 173)}
    panels, source_hashes = [], {}
    for view in ("pa", "lateral"):
        folder = input_dir / view
        image_path, = folder.glob("input.*")
        source_hashes[view] = {"image_sha256": sha(image_path),
                               "mask_sha256": {n: sha(folder / (n + ".png")) for n in names}}
        with Image.open(image_path) as im:
            base = im.convert("RGB")
            base.thumbnail((768, 768), Image.Resampling.LANCZOS)
        for group, selected in groups.items():
            if (view, group) not in NATIVE_GROUPS:
                continue
            overlay = base.copy()
            for i, name in enumerate(selected):
                color = lung_colors.get(name)
                if color is None:
                    rgb = colorsys.hsv_to_rgb((0.57 + i * 0.61803398875) % 1, 0.56, 0.93)
                    color = tuple(round(v * 255) for v in rgb)
                with Image.open(folder / (name + ".png")) as im:
                    mask = im.convert("L").point(lambda v: 255 if v > 127 else 0)
                    mask = mask.resize(base.size, Image.Resampling.NEAREST)
                alpha = 32 if name == "heart" else 85
                tint = Image.blend(overlay, Image.new("RGB", base.size, color), alpha / 255)
                overlay.paste(tint, (0, 0), mask)
                contour = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
                edge_color = color if group == "lungs" else (235, 239, 245)
                overlay.paste(Image.new("RGB", base.size, edge_color), (0, 0), contour)
            dest = SOURCE / f"native-{view}-{group}.png"
            overlay.save(dest, optimize=True)
            panels.append({"asset": str(dest.relative_to(ROOT)), "sha256": sha(dest),
                           "view": view, "group": group, "class_names": selected})
    manifest = {"source": "Retained manuscript per-class prediction PNGs for the matched PA and lateral examples",
                "input_matching": "Image correlation against retained manuscript input panels exceeds 0.99 for both examples",
                "operations": ["threshold existing mask PNGs at 127", "nearest-neighbor mask display resize",
                               "translucent color overlay", "mask boundary outline"],
                "mask_editing": False, "new_inference": False,
                "source_hashes": source_hashes, "panels": panels}
    (ASSETS / "native-panels.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_overview():
    canvas = Image.new("RGB", (1800, 686), BACKGROUND)
    d = ImageDraw.Draw(canvas)
    text(d, (54, 32), "ANYCXR / SYNTHETIC SUPERVISION", 23, PURPLE, True)
    text(d, (54, 78), "From 3D anatomy to chest X-ray segmentation", 49, INK, True)
    text(d, (56, 145), "CT-derived labels. Diverse projections. Joint anatomical learning.", 28, MUTED)
    xs = [54, 487, 920, 1353]
    colors = [TEAL, BLUE, PURPLE, PINK]
    badges = ["#127c80", "#1b66b5", "#6b4cb3", "#a94983"]
    titles = ["CT + anatomical labels", "Multi-stage randomization",
              "Partial labels + CAR", "Radiographic anatomy"]
    for i, x in enumerate(xs):
        d.rectangle((x, 208, x + 392, 600), fill=SURFACE, outline=LINE, width=1)
        d.rectangle((x + 20, 228, x + 61, 269), fill=badges[i])
        text(d, (x + 41, 249), str(i + 1).zfill(2), 22, INK, True, "mm")
        text(d, (x + 78, 235), ["SUPERVISION", "MSDR", "LEARNING", "INFERENCE"][i], 23, colors[i], True)
        text(d, (x + 20, 495), titles[i], 27, INK, True)
        if i < 3:
            arrow(d, (x + 402, 385), (x + 424, 385), BLUE)

    x = xs[0]
    image_box(canvas, SOURCE / "paper-ct.png", (x + 20, 302, 157, 157), "white")
    arrow(d, (x + 183, 380), (x + 207, 380), TEAL)
    image_box(canvas, SOURCE / "paper-anatomy.png", (x + 215, 302, 157, 157), "white")
    text(d, (x + 20, 539), "Matched image and label projections", 23, MUTED)
    text(d, (x + 20, 571), "Quality-controlled label availability", 23, MUTED)

    x = xs[1]
    examples = [ROOT / "examples/inputs/pa/t_10955_a_1.png",
                ROOT / "examples/inputs/oblique_45/train_12435_a_2.png",
                ROOT / "examples/inputs/la/train_743_a_1.png"]
    for i, path in enumerate(examples):
        xx = x + 18 + i * 121
        image_box(canvas, path, (xx, 323, 114, 114), BACKGROUND)
        text(d, (xx + 57, 451), ["PA", "Oblique", "Lateral"][i], 20, MUTED, anchor="mt")
    text(d, (x + 20, 539), "Attenuation · geometry · appearance", 23, MUTED)
    text(d, (x + 20, 571), "Synthetic radiographs with labels", 23, MUTED)

    x = xs[2]
    # A compact conceptual U-Net follows the manuscript's model diagram.
    heights = [92, 68, 45, 28, 45, 68, 92]
    for i, h in enumerate(heights):
        xx = x + 61 + i * 39
        yy = 350 - h // 2
        d.rounded_rectangle((xx, yy, xx + 27, yy + h), 3, fill="#8e80c6", outline=PURPLE, width=2)
        if i < 6:
            d.line((xx + 28, 350, xx + 38, 350), fill=PURPLE, width=2)
    text(d, (x + 196, 407), "Available-label supervision", 24, INK, anchor="mt")
    d.rectangle((x + 20, 444, x + 372, 480), fill="#31466f", outline="#7768ac", width=1)
    text(d, (x + 196, 462), "Conditional joint regularization", 21, INK, True, "mm")
    arrow(d, (x + 196, 435), (x + 196, 444), PURPLE, 2)
    text(d, (x + 20, 539), "Learn with incomplete annotations", 23, MUTED)
    text(d, (x + 20, 571), "Regularize the joint representation", 23, MUTED)

    x = xs[3]
    image_box(canvas, SOURCE / "real-pa-input.png", (x + 20, 302, 157, 157), BACKGROUND)
    arrow(d, (x + 183, 380), (x + 207, 380), PINK)
    image_box(canvas, SOURCE / "real-pa-lungs.png", (x + 215, 302, 157, 157), BACKGROUND)
    text(d, (x + 20, 539), "54 overlapping anatomy channels", 23, MUTED)
    text(d, (x + 20, 571), "Lungs · ribs · vertebrae · heart", 23, MUTED)
    d.line((54, 636, 1746, 636), fill=LINE, width=1)
    text(d, (54, 653), "SYNTHETIC TRAINING", 20, TEAL, True)
    text(d, (900, 653), "PA · LATERAL · OBLIQUE PROFILES", 20, MUTED, anchor="mt")
    text(d, (1746, 653), "REAL-CXR ILLUSTRATION", 20, PINK, True, "rt")
    canvas.save(ASSETS / "anycxr-overview.png", optimize=True)


def build_real_gallery():
    # Keep full anatomical content; strip only flat monochrome source padding.
    def trim_padding(im):
        def is_padding(box):
            extrema = im.crop(box).getextrema()
            low = min(v[0] for v in extrema)
            high = max(v[1] for v in extrema)
            return high - low <= 2 and (high <= 80 or low >= 248)

        left, top, right, bottom = 0, 0, im.width, im.height
        limit = 6
        while top < limit and is_padding((left, top, right, top + 1)):
            top += 1
        while im.height - bottom < limit and is_padding((left, bottom - 1, right, bottom)):
            bottom -= 1
        while left < limit and is_padding((left, top, left + 1, bottom)):
            left += 1
        while im.width - right < limit and is_padding((right - 1, top, right, bottom)):
            right -= 1
        return im.crop((left, top, right, bottom)), (left, top, right, bottom)

    width, margin, label_width, gap = 1800, 8, 32, 5
    header_height, row_gap = 50, 8
    column_width = (width - 2 * margin - label_width - 4 * gap) // 5
    xs = [margin + label_width + i * (column_width + gap) for i in range(5)]
    groups = ["input", "lungs", "ribs", "vertebrae", "heart-vessels"]
    rows = []
    for view, label in [("pa", "PA"), ("lateral", "LATERAL"), ("oblique", "OBLIQUE")]:
        panels = []
        for group in groups:
            prefix = "real" if group == "input" or view == "oblique" else "doctor"
            path = SOURCE / f"{prefix}-{view}-{group}.png"
            with Image.open(path) as original:
                panel, crop_box = trim_padding(original.convert("RGB"))
            scale = column_width / max(panel.size)
            panel = panel.resize((round(panel.width * scale), round(panel.height * scale)),
                                 Image.Resampling.LANCZOS)
            panels.append((panel, path, crop_box))
        rows.append((view, label, panels, column_width))

    height = header_height + sum(row[3] for row in rows) + row_gap * 2 + margin
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    d = ImageDraw.Draw(canvas)
    headers = ["Real CXR", "Lungs", "Ribs & bones", "Vertebrae", "Heart & vessels"]
    for x, title in zip(xs, headers):
        text(d, (x, 9), title, 28, INK, True)
    y = header_height
    layout = []
    for view, label, panels, row_height in rows:
        label_font = font(25, True)
        bounds = label_font.getbbox(label)
        label_image = Image.new("RGBA", (bounds[2] - bounds[0] + 4, bounds[3] - bounds[1] + 4))
        ImageDraw.Draw(label_image).text((2 - bounds[0], 2 - bounds[1]), label,
                                       font=label_font, fill=INK)
        label_image = label_image.transpose(Image.Transpose.ROTATE_90)
        canvas.paste(label_image, (margin + (label_width - label_image.width) // 2,
                                  y + (row_height - label_image.height) // 2), label_image)
        for x, group, (panel, path, crop_box) in zip(xs, groups, panels):
            d.rectangle((x, y, x + column_width - 1, y + column_width - 1), fill=SURFACE)
            panel_x = x + (column_width - panel.width) // 2
            panel_y = y + (row_height - panel.height) // 2
            canvas.paste(panel, (panel_x, panel_y))
            layout.append({"view": view, "group": group, "source": str(path.relative_to(ROOT)),
                           "source_crop_xyxy": list(crop_box),
                           "tile_xywh": [x, y, column_width, column_width],
                           "display_xywh": [panel_x, panel_y, panel.width, panel.height],
                           "annotation_type": "input" if group == "input" else
                           ("model_prediction" if view == "oblique" else "physician_refined")})
        y += row_height + row_gap
    canvas.save(ASSETS / "anycxr-real-anatomy.png", optimize=True)
    (ASSETS / "gallery-layout.json").write_text(json.dumps({"size": list(canvas.size),
        "background": BACKGROUND, "tile_background": SURFACE, "text_color": INK,
        "tile_size": column_width, "column_gap": gap, "row_gap": row_gap,
        "panels": layout}, indent=2) + "\n")


def display_name(name):
    special = {"heart": "Whole heart", "heart_myocardium": "Myocardium",
               "pulmonary_artery": "Pulmonary artery", "aorta": "Aorta (aorta channel)",
               "aorta_long": "Aorta (aorta_long channel)", "sternum": "Sternum"}
    if name in special:
        return special[name]
    parts = name.split("_")
    if parts[0] == "rib":
        return f"{parts[1].capitalize()} rib {parts[2]}"
    if parts[0] == "vertebrae":
        return "Thoracic vertebra " + parts[1]
    if parts[0] == "lung":
        return f"{parts[-1].capitalize()} {parts[1]} lung lobe"
    if parts[0] == "heart":
        return f"{parts[-1].capitalize()} " + {"ventricle": "ventricle", "atrium": "atrium"}[parts[1]]
    return f"{parts[-1].capitalize()} " + {"clavicula": "clavicle", "humerus": "humerus"}[parts[0]]


def build_catalog():
    reference = ROOT / "cxas/data/anychest_reference.json"
    names = json.loads(reference.read_text())["class_names"]
    definitions = [("lungs", "Lungs", lambda n: n.startswith("lung_")),
                   ("ribs", "Ribs", lambda n: n.startswith("rib_")),
                   ("vertebrae", "Thoracic vertebrae", lambda n: n.startswith("vertebrae_")),
                   ("heart_vessels", "Heart and vessels", lambda n: n.startswith("heart") or n in {"aorta", "aorta_long", "pulmonary_artery"}),
                   ("other_bones", "Other bones", lambda n: n.startswith(("clavicula_", "humerus_")) or n == "sternum")]
    groups, records = [], []
    for key, title, predicate in definitions:
        selected = [n for n in names if predicate(n)]
        groups.append({"id": key, "name": title, "count": len(selected), "class_names": selected})
        for n in selected:
            records.append({"channel_index": names.index(n), "class_name": n,
                            "display_name": display_name(n), "group": key})
    assert len(names) == 54 and len(records) == 54
    assert len({r["class_name"] for r in records}) == 54
    records.sort(key=lambda r: r["channel_index"])
    catalog = {"schema_version": 1, "scope": "Packaged legacy 54-channel inference taxonomy",
               "channel_index_base": 0, "channel_count": 54, "overlapping_channels": True,
               "reference": "cxas/data/anychest_reference.json", "reference_sha256": sha(reference),
               "notes": ["Left and right refer to patient anatomy.",
                         "aorta and aorta_long are distinct legacy output names; no further anatomical distinction is inferred here.",
                         "Structure counts describe model outputs, not validated per-structure accuracy.",
                         "Display grouping does not change inference channel order."],
               "groups": groups, "structures": records}
    (ROOT / "docs/anatomy.json").write_text(json.dumps(catalog, indent=2) + "\n")
    lines = ["# Anatomical structures", "", "AnyCXR's packaged inference taxonomy contains **54 overlapping anatomical channels**. "
             "This catalog is generated from the [inference reference](../cxas/data/anychest_reference.json); "
             "the [JSON catalog](anatomy.json) includes the exact zero-based channel index for every output.", "",
             "Left and right refer to the patient's anatomy. Grouping below is for browsing; the inference channel order is unchanged. "
             "These are output capabilities, not per-structure accuracy claims.", "",
             "| Group | Channels |", "| --- | ---: |"]
    lines += [f"| {g['name']} | {g['count']} |" for g in groups]
    for g in groups:
        lines += ["", "## " + g["name"], "", "| Channel (zero-based) | Structure | Output name |", "| ---: | --- | --- |"]
        lines += [f"| {r['channel_index']} | {r['display_name']} | `{r['class_name']}` |"
                  for r in records if r["group"] == g["id"]]
    lines += ["", "`aorta` and `aorta_long` are separate legacy output names. "
              "The packaged reference does not specify a finer distinction, so neither is renamed as a specific aortic segment.", "",
              "Projected anatomies overlap: keep independent channels rather than collapsing masks with argmax. "
              "Class names are the per-class PNG filenames written by the legacy CLI. "
              "A structure may be cropped, occluded, ambiguous or poorly predicted in a particular image.", "",
              "Regenerate this page and the JSON with `python scripts/build_readme_assets.py`.", ""]
    (ROOT / "docs/anatomy.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-figures", type=Path, help="Optional original manuscript figure directory to recrop retained panels")
    parser.add_argument("--native-predictions", type=Path, help="Optional matched PA/lateral input and existing class-mask directories to render overlays")
    args = parser.parse_args()
    if args.paper_figures:
        prepare_sources(args.paper_figures)
    if args.native_predictions:
        render_native_panels(args.native_predictions)
    verify_sources()
    build_overview()
    build_real_gallery()
    build_catalog()
    print("Built two README figures and the 54-channel JSON/Markdown catalog.")


if __name__ == "__main__":
    main()
