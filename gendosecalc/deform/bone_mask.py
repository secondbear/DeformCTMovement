"""Bone segmentation and tissue weight map via HU thresholding.

Strategy:
    1. Threshold CT at ``bone_threshold_hu`` → binary bone mask (cortical bone)
    2. Fill enclosed cavities per axial slice with ``binary_fill_holes`` to include
       bone marrow (fatty marrow HU ≈ −100 to +100 is below the threshold but must
       also be kept rigid so it doesn't get displaced into the cortical shell)
    3. Gaussian-smooth the filled mask with sigma = ``transition_width_mm / spacing_mm``
    4. Hard-set all actual bone voxels to weight=0, leaving a smooth 0→1 gradient
       only in the soft tissue outside bone
    5. Invert to get tissue weight: 0 in bone/marrow, 1 in soft tissue

The tissue weight is used to attenuate DVF displacements so bone remains rigid.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

from gendosecalc.deform.models import DeformationConfig


def compute_bone_mask(
    ct_array: np.ndarray,
    config: DeformationConfig | None = None,
) -> np.ndarray:
    """Segment bone from a CT volume by HU thresholding.

    Parameters:
        ct_array: HU volume, shape ``(nz, ny, nx)``.
        config: Deformation configuration (uses ``bone_threshold_hu``).
            Defaults to ``DeformationConfig()`` if not provided.

    Returns:
        Boolean array ``(nz, ny, nx)`` — True where HU > threshold or inside a
        closed cortical bone cavity (marrow).
    """
    if config is None:
        config = DeformationConfig()
    mask = ct_array > config.bone_threshold_hu
    # Fill bone marrow cavities: cortical bone forms a closed shell in axial
    # slices; marrow inside is at soft-tissue HU but must move rigidly.
    filled = np.zeros_like(mask)
    for iz in range(mask.shape[0]):
        filled[iz] = binary_fill_holes(mask[iz])
    return filled


def compute_tissue_weight(
    bone_mask: np.ndarray,
    spacing_mm: np.ndarray,
    config: DeformationConfig | None = None,
) -> np.ndarray:
    """Compute a smooth tissue weight map from a bone mask.

    The weight is 1.0 in soft tissue and 0.0 inside bone, with a smooth
    Gaussian transition at bone–tissue interfaces.

    Parameters:
        bone_mask: Boolean array ``(nz, ny, nx)`` from ``compute_bone_mask``.
        spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        config: Deformation configuration (uses ``transition_width_mm``).

    Returns:
        Float32 array ``(nz, ny, nx)`` in ``[0, 1]``.
    """
    if config is None:
        config = DeformationConfig()

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    # Per-axis sigma in voxel units
    sigma_voxels = config.transition_width_mm / spacing

    smoothed = gaussian_filter(
        bone_mask.astype(np.float32), sigma=sigma_voxels, mode="nearest",
    )
    weight = 1.0 - smoothed
    # Hard-set all actual bone voxels to 0 regardless of bone thickness.
    # Without this, thin cortical bone (1-2 voxels) is barely suppressed by
    # the Gaussian (smoothed ≈ 0.1-0.2 for a 1-voxel sliver at sigma≈3),
    # leaving tissue_weight ≈ 0.8 and passing most of the displacement through.
    weight[bone_mask] = 0.0
    return np.clip(weight, 0.0, 1.0).astype(np.float32)
