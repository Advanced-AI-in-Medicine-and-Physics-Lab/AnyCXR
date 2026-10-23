"""File discovery and scalar radiograph loading for the public inference API."""
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

RASTER_SUFFIXES = {".png", ".jpg", ".jpeg"}
DICOM_SUFFIXES = {".dcm", ".dicom"}


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def is_supported_file(path: Path) -> bool:
    return path.is_file() and (
        path.suffix.lower() in RASTER_SUFFIXES | DICOM_SUFFIXES or is_nifti(path)
    )


def image_stem(path: Path) -> str:
    """Strip the complete NIfTI suffix for mask and overlay filenames."""
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def load_grayscale_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in RASTER_SUFFIXES:
        with Image.open(path) as image:
            return np.array(image.convert("L"), dtype=np.float32)
    if is_nifti(path):
        image = sitk.ReadImage(str(path), imageIO="NiftiImageIO")
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("NIfTI input must contain a scalar grayscale radiograph.")
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        while array.ndim > 2 and 1 in array.shape:
            array = np.squeeze(array, axis=array.shape.index(1))
        if array.ndim != 2:
            raise ValueError(
                f"NIfTI input must be a 2D radiograph (singleton dimensions are accepted); got {image.GetSize()}."
            )
        if not np.isfinite(array).all():
            raise ValueError("NIfTI radiograph contains non-finite pixel values.")
        return array
    if path.suffix.lower() in DICOM_SUFFIXES:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        return array[0] if array.ndim == 3 else array
    raise ValueError(f"Unsupported input file type: {path.name}")
