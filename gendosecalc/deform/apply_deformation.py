"""Apply a displacement vector field to a CT volume with bone rigidity masking.

Workflow:
    1. Attenuate DVF by tissue weight map (zero displacement in bone)
    2. Convert masked DVF to SimpleITK displacement field
    3. Invert forward DVF to inverse mapping (SimpleITK convention)
    4. Resample CT using ``sitk.DisplacementFieldTransform`` + ``sitk.Resample``
    5. Clamp HU to ``[-1024, 3071]`` and cast to int16

The function accepts a **forward** DVF (original → deformed position) and
inverts it internally for use with SimpleITK's inverse-mapping resampler.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gendosecalc.deform.bone_mask import compute_bone_mask, compute_tissue_weight
from gendosecalc.deform.models import DeformationConfig, DeformationField
from gendosecalc.deform.sitk_bridge import ct_to_sitk, dvf_to_sitk, sitk_to_ct

# HU clamp range
_HU_MIN = -1024
_HU_MAX = 3071

_INTERPOLATION_MAP = {
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "nearest": sitk.sitkNearestNeighbor,
}


def _mask_dvf(
    dvf: DeformationField,
    tissue_weight: np.ndarray,
) -> DeformationField:
    """Attenuate DVF displacements by the tissue weight map.

    Bone regions (weight ≈ 0) get zero displacement; soft tissue (weight ≈ 1)
    keeps its original displacement.
    """
    # tissue_weight: (nz, ny, nx) → broadcast to (1, nz, ny, nx)
    weight = tissue_weight[np.newaxis, :, :, :]
    masked_vectors = dvf.vectors * weight
    return DeformationField(
        vectors=masked_vectors,
        spacing_mm=dvf.spacing_mm.copy(),
        origin_mm=dvf.origin_mm.copy(),
        direction=dvf.direction.copy(),
        source_description=dvf.source_description + "_masked",
    )


def deform_ct(
    ct_array: np.ndarray,
    ct_spacing_mm: np.ndarray,
    ct_origin_mm: np.ndarray,
    ct_direction: np.ndarray,
    dvf: DeformationField,
    config: DeformationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deform a CT volume using a forward DVF with bone rigidity masking.

    Parameters:
        ct_array: HU volume ``(nz, ny, nx)`` int16.
        ct_spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        ct_origin_mm: Origin ``(oz, oy, ox)`` in LPS mm.
        ct_direction: Direction cosine matrix ``(3, 3)``.
        dvf: Forward displacement vector field (original → deformed).
            Inverted internally for SimpleITK's inverse-mapping resampler.
        config: Deformation configuration. Defaults to ``DeformationConfig()``.

    Returns:
        ``(array, spacing_mm, origin_mm, direction)`` of the deformed CT.
        HU values are clamped to ``[-1024, 3071]`` and cast to int16.
    """
    if config is None:
        config = DeformationConfig()

    # 1. Compute bone mask and tissue weight
    bone_mask = compute_bone_mask(ct_array, config)
    tissue_weight = compute_tissue_weight(bone_mask, ct_spacing_mm, config)

    # 2. Mask the DVF
    masked_dvf = _mask_dvf(dvf, tissue_weight)

    # 3. Convert to SimpleITK
    ct_sitk = ct_to_sitk(ct_array, ct_spacing_mm, ct_origin_mm, ct_direction)
    dvf_sitk = dvf_to_sitk(masked_dvf)

    # 4. Invert forward DVF → inverse mapping
    inverse_dvf = sitk.InvertDisplacementField(dvf_sitk)

    # 5. Create displacement field transform and resample
    transform = sitk.DisplacementFieldTransform(inverse_dvf)
    interpolator = _INTERPOLATION_MAP.get(config.interpolation, sitk.sitkLinear)

    resampled = sitk.Resample(
        ct_sitk,
        ct_sitk,  # reference image (same grid)
        transform,
        interpolator,
        -1000.0,  # default pixel value (air)
        ct_sitk.GetPixelID(),
    )

    # 6. Convert back and clamp HU
    out_array, out_spacing, out_origin, out_direction = sitk_to_ct(resampled)

    if config.preserve_hounsfield:
        out_array = np.clip(out_array, _HU_MIN, _HU_MAX).astype(np.int16)

    return out_array, out_spacing, out_origin, out_direction
