"""Data models for deformable CT generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DeformationField:
    """3D displacement vector field in LPS coordinates.

    Attributes:
        vectors: Displacement vectors in mm, shape ``(3, nz, ny, nx)`` float32.
            Component order is ``(dz, dy, dx)`` matching LPS axis order.
        spacing_mm: Voxel spacing in mm, shape ``(3,)`` — ``(sz, sy, sx)``.
        origin_mm: Volume origin in LPS mm, shape ``(3,)``.
        direction: Direction cosine matrix, shape ``(3, 3)``.
        source_description: Human-readable provenance string,
            e.g. ``"rigid_shift_tx5_ty-2_tz1"`` or ``"dvf_file.mha"``.
    """

    vectors: np.ndarray
    spacing_mm: np.ndarray
    origin_mm: np.ndarray
    direction: np.ndarray
    source_description: str = ""

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        self.spacing_mm = np.asarray(self.spacing_mm, dtype=np.float64)
        self.origin_mm = np.asarray(self.origin_mm, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)

        if self.vectors.ndim != 4 or self.vectors.shape[0] != 3:
            raise ValueError(
                f"vectors must have shape (3, nz, ny, nx), got {self.vectors.shape}"
            )
        if self.spacing_mm.shape != (3,):
            raise ValueError(
                f"spacing_mm must have shape (3,), got {self.spacing_mm.shape}"
            )
        if self.origin_mm.shape != (3,):
            raise ValueError(
                f"origin_mm must have shape (3,), got {self.origin_mm.shape}"
            )
        if self.direction.shape != (3, 3):
            raise ValueError(
                f"direction must have shape (3, 3), got {self.direction.shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial volume shape ``(nz, ny, nx)``."""
        return self.vectors.shape[1], self.vectors.shape[2], self.vectors.shape[3]


@dataclass
class DeformationConfig:
    """Parameters controlling CT deformation behaviour.

    Attributes:
        bone_threshold_hu: HU value above which voxels are classified as bone.
        transition_width_mm: Gaussian sigma (in mm) for smooth bone–tissue
            boundary transition.
        interpolation: Resampling interpolation mode —
            ``"linear"``, ``"bspline"``, or ``"nearest"``.
        preserve_hounsfield: If True, clamp result to ``[-1024, 3071]`` and
            cast to int16 after resampling.
    """

    bone_threshold_hu: float = 300.0
    transition_width_mm: float = 3.0
    interpolation: str = "linear"
    preserve_hounsfield: bool = True
