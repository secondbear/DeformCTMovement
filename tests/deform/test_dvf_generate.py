"""Tests for dvf_generate — DVF from rigid parameters and motion log entries."""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.dvf_generate import (
    motion_log_entry_to_dvf,
    rigid_to_dvf,
    _rotation_matrix,
)


# Common reference geometry
SHAPE = (10, 10, 10)
SPACING = np.array([2.0, 2.0, 2.0], dtype=np.float64)
ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float64)
DIRECTION = np.eye(3, dtype=np.float64)


class TestRotationMatrix:
    """Euler angle rotation matrix correctness."""

    def test_identity(self):
        R = _rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_degree_z_rotation(self):
        R = _rotation_matrix(0, 0, 90)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_orthogonality(self):
        R = _rotation_matrix(30, 45, 60)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestRigidToDVF:
    """DVF generation from 6DOF rigid parameters."""

    def test_identity_produces_zero_dvf(self):
        dvf = rigid_to_dvf(0, 0, 0, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        np.testing.assert_allclose(dvf.vectors, 0, atol=1e-5)

    def test_pure_translation_uniform_dvf(self):
        """A pure translation should produce uniform displacement everywhere."""
        dvf = rigid_to_dvf(5.0, -3.0, 1.0, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        # tx=5 → dx component (index 2 in dz,dy,dx order) should be 5.0
        np.testing.assert_allclose(dvf.vectors[2], 5.0, atol=1e-5)
        # ty=-3 → dy component (index 1)
        np.testing.assert_allclose(dvf.vectors[1], -3.0, atol=1e-5)
        # tz=1 → dz component (index 0)
        np.testing.assert_allclose(dvf.vectors[0], 1.0, atol=1e-5)

    def test_translation_dvf_shape(self):
        dvf = rigid_to_dvf(1, 0, 0, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        assert dvf.vectors.shape == (3, 10, 10, 10)
        assert dvf.vectors.dtype == np.float32

    def test_metadata_matches_reference(self):
        dvf = rigid_to_dvf(0, 0, 0, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        np.testing.assert_array_equal(dvf.spacing_mm, SPACING)
        np.testing.assert_array_equal(dvf.origin_mm, ORIGIN)
        np.testing.assert_array_equal(dvf.direction, DIRECTION)

    def test_rotation_non_uniform_dvf(self):
        """Rotation about centre should produce varying displacement."""
        dvf = rigid_to_dvf(0, 0, 0, 0, 0, 5.0, SHAPE, SPACING, ORIGIN, DIRECTION)
        # Displacement should vary with distance from centre
        # Check that not all displacement values are the same
        assert dvf.vectors.std() > 0.01

    def test_rotation_zero_at_centre(self):
        """Rotation about volume centre should produce zero displacement there."""
        centre = ORIGIN + DIRECTION @ (SPACING * np.array([4.5, 4.5, 4.5]))
        dvf = rigid_to_dvf(
            0, 0, 0, 0, 0, 10.0, SHAPE, SPACING, ORIGIN, DIRECTION,
            centre_of_rotation_mm=centre,
        )
        # At the centre of rotation, displacement should be ~0
        # Centre is at voxel (4.5, 4.5, 4.5), check (5,5,5) which is closest
        disp_at_centre = dvf.vectors[:, 5, 5, 5]
        # Should be small (not exactly zero since (5,5,5) != (4.5,4.5,4.5))
        assert np.linalg.norm(disp_at_centre) < 2.0

    def test_custom_centre_of_rotation(self):
        """Custom centre of rotation changes the displacement pattern."""
        centre1 = np.array([0.0, 0.0, 0.0])
        centre2 = np.array([10.0, 10.0, 10.0])
        dvf1 = rigid_to_dvf(0, 0, 0, 0, 0, 5.0, SHAPE, SPACING, ORIGIN, DIRECTION, centre1)
        dvf2 = rigid_to_dvf(0, 0, 0, 0, 0, 5.0, SHAPE, SPACING, ORIGIN, DIRECTION, centre2)
        # Different centres → different displacement fields
        assert not np.allclose(dvf1.vectors, dvf2.vectors, atol=1e-3)

    def test_source_description_translation(self):
        dvf = rigid_to_dvf(5, -3, 1, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        assert "tx5" in dvf.source_description

    def test_source_description_identity(self):
        dvf = rigid_to_dvf(0, 0, 0, 0, 0, 0, SHAPE, SPACING, ORIGIN, DIRECTION)
        assert "identity" in dvf.source_description


class TestMotionLogEntryToDVF:
    """DVF from Synchrony motion log entries."""

    def test_zero_motion(self):
        entry = {"SI": 0, "LR": 0, "AP": 0, "Pitch": 0, "Roll": 0, "Yaw": 0}
        dvf = motion_log_entry_to_dvf(entry, SHAPE, SPACING, ORIGIN, DIRECTION)
        np.testing.assert_allclose(dvf.vectors, 0, atol=1e-5)

    def test_translation_only(self):
        entry = {"SI": 2.0, "LR": -1.0, "AP": 0.5}
        dvf = motion_log_entry_to_dvf(entry, SHAPE, SPACING, ORIGIN, DIRECTION)
        # SI → tz → dz (index 0), LR → tx → dx (index 2), AP → ty → dy (index 1)
        np.testing.assert_allclose(dvf.vectors[0], 2.0, atol=1e-5)  # dz = SI
        np.testing.assert_allclose(dvf.vectors[2], -1.0, atol=1e-5)  # dx = LR
        np.testing.assert_allclose(dvf.vectors[1], 0.5, atol=1e-5)  # dy = AP

    def test_missing_keys_default_to_zero(self):
        entry = {"SI": 3.0}  # only SI, rest default to 0
        dvf = motion_log_entry_to_dvf(entry, SHAPE, SPACING, ORIGIN, DIRECTION)
        np.testing.assert_allclose(dvf.vectors[0], 3.0, atol=1e-5)
        np.testing.assert_allclose(dvf.vectors[1], 0.0, atol=1e-5)
        np.testing.assert_allclose(dvf.vectors[2], 0.0, atol=1e-5)

    def test_with_rotation(self):
        entry = {"SI": 0, "LR": 0, "AP": 0, "Yaw": 5.0}
        dvf = motion_log_entry_to_dvf(entry, SHAPE, SPACING, ORIGIN, DIRECTION)
        # Non-uniform displacement from rotation
        assert dvf.vectors.std() > 0.01

    def test_source_description_contains_motion_log(self):
        entry = {"SI": 1.0, "LR": 2.0}
        dvf = motion_log_entry_to_dvf(entry, SHAPE, SPACING, ORIGIN, DIRECTION)
        assert "motion_log" in dvf.source_description
