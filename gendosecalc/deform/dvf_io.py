"""Load and save displacement vector fields (DVF) via SimpleITK.

Supported formats:
    - MetaImage (``.mha``) — project default
    - NIfTI (``.nii``, ``.nii.gz``) — research interchange

The DVF is stored on disk as a 3-D vector image with per-pixel order
``(dx, dy, dz)`` (SimpleITK convention). This module converts to/from
the project's ``DeformationField`` dataclass transparently.
"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from gendosecalc.deform.models import DeformationField
from gendosecalc.deform.sitk_bridge import dvf_to_sitk, sitk_to_dvf

_SUPPORTED_EXTENSIONS = {".mha", ".nii", ".nii.gz"}


def _normalise_suffix(path: Path) -> str:
    """Return the file extension, collapsing ``.nii.gz``."""
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def load_dvf(path: str | Path) -> DeformationField:
    """Load a DVF from disk.

    Parameters:
        path: File path (``.mha``, ``.nii``, or ``.nii.gz``).

    Returns:
        A ``DeformationField`` with vectors in ``(dz, dy, dx)`` order.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported or the image does not
            contain exactly 3 vector components.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DVF file not found: {path}")

    ext = _normalise_suffix(path)
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported DVF format '{ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    image = sitk.ReadImage(str(path))

    n_comp = image.GetNumberOfComponentsPerPixel()
    if n_comp != 3:
        raise ValueError(
            f"Expected a 3-component vector image, got {n_comp} components"
        )

    return sitk_to_dvf(image, source_description=str(path))


def save_dvf(dvf: DeformationField, path: str | Path) -> None:
    """Save a DVF to disk.

    Parameters:
        dvf: The displacement vector field to save.
        path: Destination file path (``.mha``, ``.nii``, or ``.nii.gz``).

    Raises:
        ValueError: If the file extension is unsupported.
    """
    path = Path(path)
    ext = _normalise_suffix(path)
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported DVF format '{ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    image = dvf_to_sitk(dvf)
    sitk.WriteImage(image, str(path))
