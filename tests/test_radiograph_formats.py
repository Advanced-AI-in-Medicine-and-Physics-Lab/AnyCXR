"""Real NIfTI round trips, public input routes, and reference-mask scoring."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import SimpleITK as sitk

from cxas.input_io import image_stem, is_supported_file, load_grayscale_array
from cxas.segmentor import CXAS_Segmentor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anycxr_evaluation", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class RadiographFormats(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pixels = np.arange(48, dtype=np.float32).reshape(6, 8) - 20
        self.segmentor = CXAS_Segmentor.__new__(CXAS_Segmentor)
        self.segmentor.selected_views = ["PA"]
        self.segmentor.valid_views = {"PA"}
        self.segmentor.view_configs = {"PA": {"angle": 180, "data_folder_name": "PA"}}

    def tearDown(self):
        self.temp.cleanup()

    def write_nifti(self, path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))

    def test_2d_and_singleton_nifti_preserve_pixel_order_and_values(self):
        for suffix in (".nii", ".nii.gz"):
            for index, array in enumerate((self.pixels, self.pixels[None], self.pixels[..., None])):
                path = self.root / f"sample_{index}{suffix}"
                self.write_nifti(path, array)
                with self.subTest(suffix=suffix, shape=array.shape):
                    self.assertTrue(is_supported_file(path))
                    np.testing.assert_array_equal(self.segmentor._load_grayscale_array(path), self.pixels)
                    self.assertEqual(image_stem(path), f"sample_{index}")

    def test_multislice_volume_is_not_silently_reduced_to_first_slice(self):
        path = self.root / "volume.nii.gz"
        self.write_nifti(path, np.stack([self.pixels, self.pixels + 1]))
        with self.assertRaisesRegex(ValueError, "2D radiograph"):
            load_grayscale_array(path)

    def test_single_flat_and_view_directory_routes_include_nifti(self):
        folder = self.root / "PA/imagesTr"
        for name in ("plain.nii", "compressed.nii.gz"):
            self.write_nifti(folder / name, self.pixels)
        Image.fromarray((self.pixels + 20).astype(np.uint8)).save(folder / "raster.png")
        (folder / "notes.gz").write_bytes(b"not a radiograph")
        for name in ("plain.nii", "compressed.nii.gz", "raster.png"):
            self.assertTrue(self.segmentor._is_supported_file(folder / name))
        self.assertFalse(self.segmentor._is_supported_file(folder / "notes.gz"))
        self.assertFalse(self.segmentor._is_supported_file(folder))
        grouped = self.segmentor._collect_dataset_like_inputs(self.root)
        flat = self.segmentor._collect_flat_inputs(folder, "PA")
        expected = {"plain.nii", "compressed.nii.gz", "raster.png"}
        self.assertEqual({p.name for p, _, _ in grouped}, expected)
        self.assertEqual({p.name for p, _, _ in flat}, expected)

    def test_existing_png_and_dicom_pixel_loading(self):
        pixels = (self.pixels + 20).astype(np.uint16)
        png = self.root / "sample.png"
        Image.fromarray(pixels.astype(np.uint8)).save(png)
        dicom = self.root / "sample.dcm"
        sitk.WriteImage(sitk.GetImageFromArray(pixels), str(dicom))
        for path in (png, dicom):
            with self.subTest(path=path):
                self.assertTrue(is_supported_file(path))
                np.testing.assert_array_equal(load_grayscale_array(path), pixels.astype(np.float32))

    def test_compound_suffix_mask_and_overlay_outputs(self):
        source = self.root / "example.nii.gz"
        self.write_nifti(source, self.pixels)
        self.segmentor.save_option = "sep"
        self.segmentor.save_format = "img"
        self.segmentor.save_overlay = True
        self.segmentor.class_names = ["heart"]
        self.segmentor.class_colors = [(1., 0., 0.)]
        masks = np.ones((1, 6, 8), dtype=np.uint8)
        out = self.root / "predictions"
        relative = Path(source.name)
        self.segmentor._save_mask(masks, image_stem(source), out, "PA", relative)
        self.segmentor._save_overlay(source, masks, out, "PA", relative)
        self.assertTrue((out / "PA/labelsTr/example/heart.png").is_file())
        overlay = out / "PA/overlays/example_overlay.png"
        self.assertTrue(overlay.is_file())
        with Image.open(overlay) as image:
            self.assertEqual(image.size, (8, 6))

    def test_nifti_evaluation_pairs_with_matching_reference_and_reports_count(self):
        self.write_nifti(self.root / "PA/imagesTr/example.nii.gz", self.pixels)
        reference = self.root / "PA/labelsTr/example_total"
        prediction = self.root / "predictions/PA/labelsTr/example"
        reference.mkdir(parents=True); prediction.mkdir(parents=True)
        mask = np.zeros((6, 8), dtype=np.uint8); mask[1:4, 2:5] = 255
        for folder in (reference, prediction):Image.fromarray(mask).save(folder / "heart.png")
        _, classes, summary = evaluation.collect_metrics(self.root, self.root / "predictions", ["PA"], ["heart", "unknown"])
        self.assertEqual(summary["overall_mean_dice"], 1.)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["n_samples"], 1)
        (prediction / "heart.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "predicted mask"):
            evaluation.collect_metrics(self.root, self.root / "predictions", ["PA"], ["heart"])

    def test_evaluation_rejects_broadcastable_but_misaligned_masks(self):
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            evaluation.calc_dice(np.ones((6, 8)), np.ones((1, 8)))


if __name__ == "__main__":
    unittest.main()
