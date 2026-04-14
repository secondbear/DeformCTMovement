"""Bidirectional conversion between numpy arrays and SimpleITK images.

All conversions preserve physical metadata (spacing, origin, direction cosines)
and use LPS coordinates throughout.

CT convention:
    numpy array — ``int16``, shape ``(nz, ny, nx)``, values in HU.
    SimpleITK   — ``sitkInt16``, 3-D scalar image.

DVF convention:
    numpy array — ``float32``, shape ``(3, nz, ny, nx)``, displacement in mm.
                  Component order ``(dz, dy, dx)``.
    SimpleITK   — ``sitkVectorFloat64``, 3-D vector image, per-pixel order
                  ``(dx, dy, dz)`` (SimpleITK stores x-y-z).
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gendosecalc.deform.models import DeformationField


# ---------------------------------------------------------------------------
# CT helpers
# ---------------------------------------------------------------------------

def ct_to_sitk(
    array: np.ndarray,
    spacing_mm: np.ndarray,
    origin_mm: np.ndarray,
    direction: np.ndarray,
) -> sitk.Image:
    """Convert a numpy CT volume to a SimpleITK image.

    Parameters:
        array: HU volume, shape ``(nz, ny, nx)``, dtype int16.
        spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        origin_mm: Volume origin ``(oz, oy, ox)`` in LPS mm.
        direction: Direction cosine matrix ``(3, 3)``.

    Returns:
        A ``sitkInt16`` 3-D scalar image with matching spatial metadata.

    Note:
        SimpleITK uses **(x, y, z)** ordering for spacing/origin, while
        our project uses **(z, y, x)**. This function handles the reversal.
    """
    array = np.asarray(array, dtype=np.int16)
    if array.ndim != 3:
        raise ValueError(f"CT array must be 3-D, got {array.ndim}-D")

    image = sitk.GetImageFromArray(array)
    # SimpleITK expects (x, y, z) ordering
    image.SetSpacing(spacing_mm[::-1].tolist())
    image.SetOrigin(origin_mm[::-1].tolist())
    # Direction: our (3,3) is stored row-major in (z,y,x) order.
    # SimpleITK wants a 9-element tuple in (x,y,z) row-major order.
    dir_zyx = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    # Reverse both row and column order to go from (z,y,x) to (x,y,z)
    dir_xyz = dir_zyx[::-1, ::-1]
    image.SetDirection(dir_xyz.flatten().tolist())
    return image


def sitk_to_ct(
    image: sitk.Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a SimpleITK scalar image back to numpy arrays.

    Returns:
        ``(array, spacing_mm, origin_mm, direction)`` where:
        - array: int16 ``(nz, ny, nx)``
        - spacing_mm: float64 ``(3,)`` in ``(sz, sy, sx)`` order
        - origin_mm: float64 ``(3,)`` in ``(sz, sy, sx)`` order
        - direction: float64 ``(3, 3)`` in ``(z, y, x)`` row/col order
    """
    array = sitk.GetArrayFromImage(image).astype(np.int16)
    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
    origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
    dir_xyz = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    # Reverse from (x, y, z) → (z, y, x)
    spacing_mm = spacing_xyz[::-1]
    origin_mm = origin_xyz[::-1]
    direction = dir_xyz[::-1, ::-1]
    return array, spacing_mm.copy(), origin_mm.copy(), direction.copy()


# ---------------------------------------------------------------------------
# DVF helpers
# ---------------------------------------------------------------------------

def dvf_to_sitk(dvf: DeformationField) -> sitk.Image:
    """Convert a DeformationField to a SimpleITK vector image.

    The internal array layout is ``(3, nz, ny, nx)`` with component order
    ``(dz, dy, dx)``. SimpleITK vector images store per-pixel vectors as
    ``(dx, dy, dz)``, so this function transposes accordingly.

    Returns:
        A ``sitkVectorFloat64`` 3-D vector image.
    """
    # dvf.vectors shape: (3, nz, ny, nx), order (dz, dy, dx)
    # SimpleITK wants (nz, ny, nx, 3), order (dx, dy, dz)
    vectors_zyx = dvf.vectors  # (dz, dy, dx)
    vectors_xyz = vectors_zyx[::-1]  # (dx, dy, dz)
    # Transpose from (3, nz, ny, nx) → (nz, ny, nx, 3)
    arr = np.ascontiguousarray(vectors_xyz.transpose(1, 2, 3, 0), dtype=np.float64)

    image = sitk.GetImageFromArray(arr, isVector=True)
    image.SetSpacing(dvf.spacing_mm[::-1].tolist())
    image.SetOrigin(dvf.origin_mm[::-1].tolist())
    dir_zyx = dvf.direction.reshape(3, 3)
    dir_xyz = dir_zyx[::-1, ::-1]
    image.SetDirection(dir_xyz.flatten().tolist())
    return image


def sitk_to_dvf(
    image: sitk.Image,
    source_description: str = "",
) -> DeformationField:
    """Convert a SimpleITK vector image to a DeformationField.

    Parameters:
        image: A 3-D vector image with 3 components per pixel (dx, dy, dz).
        source_description: Provenance string stored in the result.

    Returns:
        A ``DeformationField`` with vectors in ``(dz, dy, dx)`` order.
    """
    arr = sitk.GetArrayFromImage(image)  # (nz, ny, nx, 3) order (dx, dy, dz)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected a 3-component vector image, got shape {arr.shape}"
        )

    # Transpose to (3, nz, ny, nx) and reverse component order → (dz, dy, dx)
    vectors_xyz = arr.transpose(3, 0, 1, 2)  # (3, nz, ny, nx), order (dx, dy, dz)
    vectors_zyx = vectors_xyz[::-1]  # (dz, dy, dx)
    vectors = np.ascontiguousarray(vectors_zyx, dtype=np.float32)

    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
    origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
    dir_xyz = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    return DeformationField(
        vectors=vectors,
        spacing_mm=spacing_xyz[::-1].copy(),
        origin_mm=origin_xyz[::-1].copy(),
        direction=dir_xyz[::-1, ::-1].copy(),
        source_description=source_description,
    )
