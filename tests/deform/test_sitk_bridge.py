"""Tests for sitk_bridge — CT and DVF round-trip conversions."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.models import DeformationField
from gendosecalc.deform.sitk_bridge import (
    ct_to_sitk,
    dvf_to_sitk,
    sitk_to_ct,
    sitk_to_dvf,
)


class TestCTRoundTrip:
    """CT array → SimpleITK → CT array preserves all metadata."""

    def test_hu_values_preserved(self, ct_array, ct_spacing, ct_origin, ct_direction):
        image = ct_to_sitk(ct_array, ct_spacing, ct_origin, ct_direction)
        out_arr, _, _, _ = sitk_to_ct(image)
        np.testing.assert_array_equal(out_arr, ct_array)

    def test_spacing_preserved(self, ct_array, ct_spacing, ct_origin, ct_direction):
        image = ct_to_sitk(ct_array, ct_spacing, ct_origin, ct_direction)
        _, out_sp, _, _ = sitk_to_ct(image)
        np.testing.assert_allclose(out_sp, ct_spacing, atol=1e-10)

    def test_origin_preserved(self, ct_array, ct_spacing, ct_origin, ct_direction):
        image = ct_to_sitk(ct_array, ct_spacing, ct_origin, ct_direction)
        _, _, out_orig, _ = sitk_to_ct(image)
        np.testing.assert_allclose(out_orig, ct_origin, atol=1e-10)

    def test_direction_preserved(self, ct_array, ct_spacing, ct_origin, ct_direction):
        image = ct_to_sitk(ct_array, ct_spacing, ct_origin, ct_direction)
        _, _, _, out_dir = sitk_to_ct(image)
        np.testing.assert_allclose(out_dir, ct_direction, atol=1e-10)

    def test_non_identity_direction(self, ct_array, ct_spacing, ct_origin):
        """Round-trip with a non-trivial direction matrix."""
        direction = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float64)
        image = ct_to_sitk(ct_array, ct_spacing, ct_origin, direction)
        _, _, _, out_dir = sitk_to_ct(image)
        np.testing.assert_allclose(out_dir, direction, atol=1e-10)

    def test_rejects_non_3d(self, ct_spacing, ct_origin, ct_direction):
        with pytest.raises(ValueError, match="3-D"):
            ct_to_sitk(
                np.zeros((4, 4), dtype=np.int16),
                ct_spacing, ct_origin, ct_direction,
            )


class TestDVFRoundTrip:
    """DVF → SimpleITK vector image → DVF preserves all metadata."""

    def test_vectors_preserved(self, zero_dvf):
        image = dvf_to_sitk(zero_dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(out.vectors, zero_dvf.vectors, atol=1e-5)

    def test_nonzero_vectors_preserved(self, uniform_translation_dvf):
        image = dvf_to_sitk(uniform_translation_dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(
            out.vectors, uniform_translation_dvf.vectors, atol=1e-5,
        )

    def test_spacing_preserved(self, zero_dvf):
        image = dvf_to_sitk(zero_dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(out.spacing_mm, zero_dvf.spacing_mm, atol=1e-10)

    def test_origin_preserved(self, zero_dvf):
        image = dvf_to_sitk(zero_dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(out.origin_mm, zero_dvf.origin_mm, atol=1e-10)

    def test_direction_preserved(self, zero_dvf):
        image = dvf_to_sitk(zero_dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(out.direction, zero_dvf.direction, atol=1e-10)

    def test_multi_component_displacement(self, ct_array, ct_spacing, ct_origin, ct_direction):
        """Round-trip a DVF with non-zero displacement in all three axes."""
        nz, ny, nx = ct_array.shape
        rng = np.random.default_rng(42)
        vectors = rng.uniform(-5, 5, (3, nz, ny, nx)).astype(np.float32)
        dvf = DeformationField(
            vectors=vectors,
            spacing_mm=ct_spacing,
            origin_mm=ct_origin,
            direction=ct_direction,
            source_description="random_test",
        )
        image = dvf_to_sitk(dvf)
        out = sitk_to_dvf(image)
        np.testing.assert_allclose(out.vectors, dvf.vectors, atol=1e-4)
