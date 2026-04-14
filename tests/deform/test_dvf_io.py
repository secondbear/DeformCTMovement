"""Tests for dvf_io — DVF save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gendosecalc.deform.dvf_io import load_dvf, save_dvf
from gendosecalc.deform.models import DeformationField


class TestDVFSaveLoad:
    """Round-trip DVF through MHA and NIfTI formats."""

    def test_mha_round_trip(self, zero_dvf, tmp_path):
        path = tmp_path / "test.mha"
        save_dvf(zero_dvf, path)
        loaded = load_dvf(path)
        np.testing.assert_allclose(loaded.vectors, zero_dvf.vectors, atol=1e-5)
        np.testing.assert_allclose(loaded.spacing_mm, zero_dvf.spacing_mm, atol=1e-10)
        np.testing.assert_allclose(loaded.origin_mm, zero_dvf.origin_mm, atol=1e-10)
        np.testing.assert_allclose(loaded.direction, zero_dvf.direction, atol=1e-10)

    def test_nifti_round_trip(self, zero_dvf, tmp_path):
        path = tmp_path / "test.nii.gz"
        save_dvf(zero_dvf, path)
        loaded = load_dvf(path)
        np.testing.assert_allclose(loaded.vectors, zero_dvf.vectors, atol=1e-5)
        np.testing.assert_allclose(loaded.spacing_mm, zero_dvf.spacing_mm, atol=1e-10)

    def test_nonzero_dvf_mha_round_trip(self, uniform_translation_dvf, tmp_path):
        path = tmp_path / "translate.mha"
        save_dvf(uniform_translation_dvf, path)
        loaded = load_dvf(path)
        np.testing.assert_allclose(
            loaded.vectors, uniform_translation_dvf.vectors, atol=1e-5,
        )

    def test_random_dvf_mha_round_trip(
        self, ct_array, ct_spacing, ct_origin, ct_direction, tmp_path,
    ):
        rng = np.random.default_rng(99)
        nz, ny, nx = ct_array.shape
        vectors = rng.uniform(-10, 10, (3, nz, ny, nx)).astype(np.float32)
        dvf = DeformationField(
            vectors=vectors,
            spacing_mm=ct_spacing,
            origin_mm=ct_origin,
            direction=ct_direction,
            source_description="random_io_test",
        )
        path = tmp_path / "random.mha"
        save_dvf(dvf, path)
        loaded = load_dvf(path)
        np.testing.assert_allclose(loaded.vectors, dvf.vectors, atol=1e-4)

    def test_source_description_set(self, zero_dvf, tmp_path):
        path = tmp_path / "desc.mha"
        save_dvf(zero_dvf, path)
        loaded = load_dvf(path)
        assert str(path) in loaded.source_description


class TestDVFIOErrors:
    """Error handling in load/save."""

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dvf(tmp_path / "nonexistent.mha")

    def test_load_unsupported_format(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dvf(p)

    def test_save_unsupported_format(self, zero_dvf, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            save_dvf(zero_dvf, tmp_path / "test.xyz")
