"""Shared test fixtures for deform tests.

All fixtures use synthetic phantoms — no patient data.
"""

from __future__ import annotations

import numpy as np
import pytest

from gendosecalc.deform.models import DeformationConfig, DeformationField


# ---------------------------------------------------------------------------
# Synthetic CT fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ct_array() -> np.ndarray:
    """A small 16×16×16 synthetic CT volume (int16 HU).

    Layout:
        - Background: -1000 HU (air)
        - Central 8×8×8 cube: 0 HU (water/soft tissue)
        - 4×4×4 bone insert in one corner: 800 HU
    """
    arr = np.full((16, 16, 16), -1000, dtype=np.int16)
    arr[4:12, 4:12, 4:12] = 0  # soft tissue
    arr[1:5, 1:5, 1:5] = 800  # bone insert
    return arr


@pytest.fixture()
def ct_spacing() -> np.ndarray:
    """Isotropic 2 mm spacing in (z, y, x) order."""
    return np.array([2.0, 2.0, 2.0], dtype=np.float64)


@pytest.fixture()
def ct_origin() -> np.ndarray:
    """Origin at (-15, -15, -15) mm in (z, y, x) order."""
    return np.array([-15.0, -15.0, -15.0], dtype=np.float64)


@pytest.fixture()
def ct_direction() -> np.ndarray:
    """Identity direction cosine matrix."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture()
def ct_with_bone() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 20×20×20 CT with a distinct bone shell and soft tissue centre.

    Returns (array, spacing, origin, direction).
    """
    arr = np.full((20, 20, 20), -1000, dtype=np.int16)
    arr[4:16, 4:16, 4:16] = 0  # soft tissue
    # Bone shell: voxels at the outer edge of the soft tissue cube
    arr[4, 4:16, 4:16] = 600
    arr[15, 4:16, 4:16] = 600
    arr[4:16, 4, 4:16] = 600
    arr[4:16, 15, 4:16] = 600
    arr[4:16, 4:16, 4] = 600
    arr[4:16, 4:16, 15] = 600
    spacing = np.array([1.5, 1.5, 1.5], dtype=np.float64)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction = np.eye(3, dtype=np.float64)
    return arr, spacing, origin, direction


# ---------------------------------------------------------------------------
# DVF fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def zero_dvf(ct_array, ct_spacing, ct_origin, ct_direction) -> DeformationField:
    """A zero displacement field matching the ct_array geometry."""
    nz, ny, nx = ct_array.shape
    return DeformationField(
        vectors=np.zeros((3, nz, ny, nx), dtype=np.float32),
        spacing_mm=ct_spacing.copy(),
        origin_mm=ct_origin.copy(),
        direction=ct_direction.copy(),
        source_description="zero_test",
    )


@pytest.fixture()
def uniform_translation_dvf(
    ct_array, ct_spacing, ct_origin, ct_direction,
) -> DeformationField:
    """A uniform 4 mm translation in the x-direction (dx=4, dy=0, dz=0).

    In our (dz, dy, dx) layout, component index 2 holds dx.
    """
    nz, ny, nx = ct_array.shape
    vectors = np.zeros((3, nz, ny, nx), dtype=np.float32)
    vectors[2, :, :, :] = 4.0  # dx = 4 mm
    return DeformationField(
        vectors=vectors,
        spacing_mm=ct_spacing.copy(),
        origin_mm=ct_origin.copy(),
        direction=ct_direction.copy(),
        source_description="uniform_translate_dx4",
    )


@pytest.fixture()
def default_config() -> DeformationConfig:
    """Default deformation configuration."""
    return DeformationConfig()
