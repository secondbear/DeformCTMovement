"""Tests for apply_deformation — CT deformation with bone rigidity."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.apply_deformation import deform_ct
from gendosecalc.deform.models import DeformationConfig, DeformationField


class TestDeformCTZeroDVF:
    """A zero DVF should return the original CT unchanged."""

    def test_array_unchanged(
        self, ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
    ):
        out, _, _, _ = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
        )
        np.testing.assert_array_equal(out, ct_array)

    def test_metadata_unchanged(
        self, ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
    ):
        _, sp, orig, d = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
        )
        np.testing.assert_allclose(sp, ct_spacing, atol=1e-10)
        np.testing.assert_allclose(orig, ct_origin, atol=1e-10)
        np.testing.assert_allclose(d, ct_direction, atol=1e-10)


class TestDeformCTTranslation:
    """A uniform translation DVF should shift the CT content."""

    def test_translation_shifts_content(
        self, ct_array, ct_spacing, ct_origin, ct_direction,
        uniform_translation_dvf,
    ):
        out, _, _, _ = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction,
            uniform_translation_dvf,
        )
        # The output should differ from input (content shifted)
        assert not np.array_equal(out, ct_array)

    def test_hu_range_preserved(
        self, ct_array, ct_spacing, ct_origin, ct_direction,
        uniform_translation_dvf,
    ):
        out, _, _, _ = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction,
            uniform_translation_dvf,
        )
        assert out.min() >= -1024
        assert out.max() <= 3071
        assert out.dtype == np.int16


class TestDeformCTBoneRigidity:
    """Bone voxels should remain unchanged after masked deformation."""

    def test_bone_voxels_preserved(self):
        """Create a phantom with bone shell; verify bone region is unchanged."""
        # 20x20x20 phantom
        arr = np.full((20, 20, 20), -1000, dtype=np.int16)
        arr[4:16, 4:16, 4:16] = 0  # soft tissue
        # Bone shell
        arr[4, 4:16, 4:16] = 600
        arr[15, 4:16, 4:16] = 600
        arr[4:16, 4, 4:16] = 600
        arr[4:16, 15, 4:16] = 600
        arr[4:16, 4:16, 4] = 600
        arr[4:16, 4:16, 15] = 600

        spacing = np.array([1.5, 1.5, 1.5])
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.eye(3)

        # Small translation DVF
        vectors = np.zeros((3, 20, 20, 20), dtype=np.float32)
        vectors[2, :, :, :] = 3.0  # dx = 3 mm
        dvf = DeformationField(
            vectors=vectors,
            spacing_mm=spacing.copy(),
            origin_mm=origin.copy(),
            direction=direction.copy(),
            source_description="bone_test",
        )

        config = DeformationConfig(bone_threshold_hu=300.0, transition_width_mm=1.5)
        out, _, _, _ = deform_ct(arr, spacing, origin, direction, dvf, config)

        # Deep bone interior should be very close to original HU
        # Check a slice well inside bone
        bone_original = arr[4, 8:12, 8:12]
        bone_deformed = out[4, 8:12, 8:12]
        np.testing.assert_allclose(
            bone_deformed.astype(float), bone_original.astype(float),
            atol=50,  # allow small interpolation artefact
        )


class TestDeformCTWaterPhantom:
    """A uniform water phantom with no bone — deformation should be unrestricted."""

    def test_water_phantom_translates_freely(self):
        arr = np.full((16, 16, 16), 0, dtype=np.int16)
        spacing = np.array([2.0, 2.0, 2.0])
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.eye(3)

        vectors = np.zeros((3, 16, 16, 16), dtype=np.float32)
        vectors[2, :, :, :] = 4.0  # dx = 4 mm = 2 voxels
        dvf = DeformationField(
            vectors=vectors,
            spacing_mm=spacing.copy(),
            origin_mm=origin.copy(),
            direction=direction.copy(),
        )

        config = DeformationConfig(bone_threshold_hu=300.0)
        out, _, _, _ = deform_ct(arr, spacing, origin, direction, dvf, config)

        # Uniform water phantom stays uniform HU=0 (no content variation)
        # The key check is that no error occurs and output is valid int16
        assert out.dtype == np.int16
        assert out.min() >= -1024


class TestDeformCTInterpolation:
    """Test different interpolation modes."""

    def test_bspline_mode(
        self, ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
    ):
        config = DeformationConfig(interpolation="bspline")
        out, _, _, _ = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf, config,
        )
        assert out.dtype == np.int16

    def test_nearest_mode(
        self, ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf,
    ):
        config = DeformationConfig(interpolation="nearest")
        out, _, _, _ = deform_ct(
            ct_array, ct_spacing, ct_origin, ct_direction, zero_dvf, config,
        )
        np.testing.assert_array_equal(out, ct_array)
