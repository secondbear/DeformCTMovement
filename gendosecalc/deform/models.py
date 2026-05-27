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

    # CTV-localised deformation settings
    falloff_mm: float = 25.0
    ctv_roi_names: list[str] = field(
        default_factory=lambda: ["CTV", "CTV_prostate", "CTV_Prostate", "ctv"]
    )

    # Motion-state selection settings
    rotation_weight_mm_per_deg: float = 10.0
    n_states: int = 20
    motion_tolerance_mode: str = "warn"  # "warn" | "drop" | "keep"

    # GPU acceleration
    use_gpu: bool = False


# ---------------------------------------------------------------------------
# Motion ingestion
# ---------------------------------------------------------------------------

@dataclass
class MotionSamples:
    """Collection of motion samples parsed from a Synchrony XML or CSV file.

    Attributes:
        timestamps_ms: Array of epoch millisecond timestamps, shape ``(N,)``.
        offsets_mm: LPS displacements in mm, shape ``(N, 3)`` — columns
            ``(dx_lr, dy_ap, dz_si)`` matching LPS x, y, z axes.
        rotations_deg: Euler angles in degrees, shape ``(N, 3)`` — columns
            ``(rx, ry, rz)``. Zero-filled when the source does not supply them.
        has_rotations: True if the source supplied non-zero rotation data.
        source_path: Path or description of the originating file.
    """

    timestamps_ms: np.ndarray  # int64 (N,)
    offsets_mm: np.ndarray     # float32 (N, 3)
    rotations_deg: np.ndarray  # float32 (N, 3)
    has_rotations: bool = False
    source_path: str = ""

    def __post_init__(self) -> None:
        self.timestamps_ms = np.asarray(self.timestamps_ms, dtype=np.int64)
        self.offsets_mm = np.asarray(self.offsets_mm, dtype=np.float32)
        self.rotations_deg = np.asarray(self.rotations_deg, dtype=np.float32)
        n = len(self.timestamps_ms)
        if self.offsets_mm.shape != (n, 3):
            raise ValueError(
                f"offsets_mm must have shape (N, 3), got {self.offsets_mm.shape}"
            )
        if self.rotations_deg.shape != (n, 3):
            raise ValueError(
                f"rotations_deg must have shape (N, 3), got {self.rotations_deg.shape}"
            )

    def __len__(self) -> int:
        return len(self.timestamps_ms)


# ---------------------------------------------------------------------------
# Motion-state selection result
# ---------------------------------------------------------------------------

@dataclass
class StateSelection:
    """Result of weighted k-medoids motion-state selection.

    Attributes:
        medoid_indices: Indices into the original ``MotionSamples`` for the
            chosen representative states, shape ``(k,)``.
        assignments: Cluster assignment for every sample, shape ``(N,)``.
        cluster_weights: Number of samples in each cluster, shape ``(k,)``.
        total_cost: Sum of within-cluster distances at convergence.
        cluster_mean_dist_mm: Mean 6-D feature distance from each medoid to
            its cluster members, shape ``(k,)``.  Units are mm-equivalent
            (rotations scaled by ``rotation_weight_mm_per_deg``).
        cluster_p95_dist_mm: 95th-percentile within-cluster distance,
            shape ``(k,)``.
    """

    medoid_indices: np.ndarray   # int64 (k,)
    assignments: np.ndarray      # int64 (N,)
    cluster_weights: np.ndarray  # int64 (k,)
    total_cost: float
    cluster_mean_dist_mm: np.ndarray  # float64 (k,)
    cluster_p95_dist_mm: np.ndarray   # float64 (k,)


# ---------------------------------------------------------------------------
# Ensemble manifest
# ---------------------------------------------------------------------------

@dataclass
class EnsembleManifestEntry:
    """Per-state record written into manifest.json.

    Attributes:
        state_index: Zero-based index within the ensemble.
        epoch_ms: Motion timestamp in epoch milliseconds.
        iso_timestamp: ISO-8601 string representation of ``epoch_ms``.
        cluster_weight: Number of source samples represented by this state.
        tx, ty, tz: Applied translation in mm (LPS x, y, z).
        rx, ry, rz: Applied rotation in degrees.
        ct_dir: Relative path to the exported DICOM CT directory.
        dvf_path: Relative path to the DVF ``.mha`` file.
        deformed_series_instance_uid: SeriesInstanceUID of the exported CT.
        source_ct_series_instance_uid: SeriesInstanceUID of the reference CT.
    """

    state_index: int
    epoch_ms: int
    iso_timestamp: str
    cluster_weight: int
    cluster_mean_dist_mm: float
    cluster_p95_dist_mm: float
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float
    ct_dir: str
    dvf_path: str
    deformed_series_instance_uid: str
    source_ct_series_instance_uid: str
    rtstruct_path: str = ""   # relative path to deformed RTSTRUCT, empty if not generated

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "state_index": self.state_index,
            "epoch_ms": self.epoch_ms,
            "iso_timestamp": self.iso_timestamp,
            "cluster_weight": self.cluster_weight,
            "cluster_mean_dist_mm": round(self.cluster_mean_dist_mm, 4),
            "cluster_p95_dist_mm": round(self.cluster_p95_dist_mm, 4),
            "tx_mm": self.tx,
            "ty_mm": self.ty,
            "tz_mm": self.tz,
            "rx_deg": self.rx,
            "ry_deg": self.ry,
            "rz_deg": self.rz,
            "ct_dir": self.ct_dir,
            "dvf_path": self.dvf_path,
            "deformed_series_instance_uid": self.deformed_series_instance_uid,
            "source_ct_series_instance_uid": self.source_ct_series_instance_uid,
            "rtstruct_path": self.rtstruct_path,
        }
