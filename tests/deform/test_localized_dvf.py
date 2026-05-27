"""Tests for localized_rigid_to_dvf in dvf_generate.py."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.bone_mask import compute_bone_mask, compute_tissue_weight
from gendosecalc.deform.dvf_generate import localized_rigid_to_dvf
from gendosecalc.deform.models import DeformationConfig


# ---------------------------------------------------------------------------
# Shared phantom
# ---------------------------------------------------------------------------

@pytest.fixture()
def phantom():
    """20×20×20 CT with central soft-tissue cube and bone corners."""
    arr = np.full((20, 20, 20), -1000, dtype=np.int16)
    arr[4:16, 4:16, 4:16] = 0   # soft tissue
    arr[0:4, 0:4, 0:4] = 700    # bone insert at corner
    spacing = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64)
    direction = np.eye(3, dtype=np.float64)
    return arr, spacing, origin, direction


@pytest.fixture()
def ctv_mask():
    """CTV covers the central 6×6×6 cube inside the soft tissue region."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[7:13, 7:13, 7:13] = True
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalizedRigidToDvf:
    def test_zero_motion_produces_zero_dvf(self, phantom, ctv_mask) -> None:
        arr, spacing, origin, direction = phantom
        config = DeformationConfig(bone_threshold_hu=300, falloff_mm=10.0)
        bone_mask = compute_bone_mask(arr, config)
        bone_weight = compute_tissue_weight(bone_mask, spacing, config)

        dvf = localized_rigid_to_dvf(
            0, 0, 0, 0, 0, 0,
            arr.shape, spacing, origin, direction,
            ctv_mask, bone_weight, config,
        )
        np.testing.assert_allclose(dvf.vectors, 0.0, atol=1e-6)

    def test_bone_voxels_have_zero_displacement(self, phantom, ctv_mask) -> None:
        arr, spacing, origin, direction = phantom
        config = DeformationConfig(bone_threshold_hu=300, falloff_mm=20.0)
        bone_mask = compute_bone_mask(arr, config)
        bone_weight = compute_tissue_weight(bone_mask, spacing, config)

        dvf = localized_rigid_to_dvf(
            5.0, 0, 0, 0, 0, 0,
            arr.shape, spacing, origin, direction,
            ctv_mask, bone_weight, config,
        )

        # Displacement magnitude at bone voxels
        disp_mag = np.linalg.norm(dvf.vectors, axis=0)  # (nz, ny, nx)
        # strict bone core (0:2, 0:2, 0:2) — well inside bone
        bone_core_disp = disp_mag[0:2, 0:2, 0:2]
        assert float(bone_core_disp.max()) < 0.5, (
            f"Bone core has non-zero displacement: max={bone_core_disp.max():.4f}"
        )

    def test_ctv_core_full_displacement(self, phantom, ctv_mask) -> None:
        arr, spacing, origin, direction = phantom
        config = DeformationConfig(bone_threshold_hu=300, falloff_mm=20.0)
        bone_mask = compute_bone_mask(arr, config)
        bone_weight = compute_tissue_weight(bone_mask, spacing, config)

        tx = 3.0
        dvf = localized_rigid_to_dvf(
            tx, 0, 0, 0, 0, 0,
            arr.shape, spacing, origin, direction,
            ctv_mask, bone_weight, config,
        )

        disp_mag = np.linalg.norm(dvf.vectors, axis=0)
        ctv_core_disp = disp_mag[ctv_mask]
        # Inside CTV, displacement should be close to tx (in soft tissue)
        assert float(ctv_core_disp.max()) > tx * 0.5, (
            "CTV voxels should have substantial displacement"
        )

    def test_falloff_monotonicity(self, phantom, ctv_mask) -> None:
        """Displacement magnitude should not increase with distance from CTV."""
        arr, spacing, origin, direction = phantom
        config = DeformationConfig(bone_threshold_hu=300, falloff_mm=15.0)
        bone_mask = compute_bone_mask(arr, config)
        bone_weight = compute_tissue_weight(bone_mask, spacing, config)

        dvf = localized_rigid_to_dvf(
            4.0, 0, 0, 0, 0, 0,
            arr.shape, spacing, origin, direction,
            ctv_mask, bone_weight, config,
        )

        from scipy.ndimage import distance_transform_edt
        ctv_dist = distance_transform_edt(~ctv_mask) * float(spacing[0])
        disp_mag = np.linalg.norm(dvf.vectors, axis=0)

        # Sample points at increasing distance from CTV centre, in soft tissue
        # (avoid bone corner)
        dist_bands = [(0, 1), (1, 5), (5, 12), (12, 25)]
        prev_mean = float("inf")
        for d0, d1 in dist_bands:
            in_band = (ctv_dist >= d0) & (ctv_dist < d1) & (~bone_mask)
            if not np.any(in_band):
                continue
            band_mean = float(disp_mag[in_band].mean())
            assert band_mean <= prev_mean + 0.5, (
                f"Displacement increased from d∈[{d0},{d1}): {band_mean:.3f} > {prev_mean:.3f}"
            )
            prev_mean = band_mean

    def test_returns_deformation_field(self, phantom, ctv_mask) -> None:
        from gendosecalc.deform.models import DeformationField
        arr, spacing, origin, direction = phantom
        config = DeformationConfig()
        bone_weight = np.ones(arr.shape, dtype=np.float32)
        dvf = localized_rigid_to_dvf(
            1, 0, 0, 0, 0, 0,
            arr.shape, spacing, origin, direction,
            ctv_mask, bone_weight, config,
        )
        assert isinstance(dvf, DeformationField)
        assert dvf.vectors.shape == (3, *arr.shape)
