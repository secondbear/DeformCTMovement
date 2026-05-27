"""Apply a forward displacement vector field to an RTSTRUCT DICOM.

Each contour point (x, y, z) in LPS mm is displaced by trilinear
interpolation of the forward DVF at that location, producing an anatomically
consistent RTSTRUCT for the corresponding deformed CT state.

Convention notes:
    - DVF ``vectors`` shape: ``(3, nz, ny, nx)`` — components ``(dz, dy, dx)`` in mm.
    - DVF ``origin_mm`` / ``spacing_mm`` / ``direction``: project ``(z, y, x)`` convention.
    - DICOM ContourData triplets are ``(x, y, z)`` in LPS mm — reversed on ingestion.
    - The direction matrix is assumed to be a rotation (det = ±1). For
      axis-aligned CTs (identity direction, the common case), the voxel
      index calculation reduces to a simple shift-and-scale.
"""

from __future__ import annotations

import copy
import datetime
import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

from .models import DeformationConfig, DeformationField

logger = logging.getLogger(__name__)

_PRIVATE_BLOCK   = 0x6363
_PRIVATE_CREATOR = "DEFORMCT"
_ENGINE_VERSION  = "0.1.0"


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _build_voxel_coords(
    points_xyz_mm: np.ndarray,  # (N, 3)  DICOM LPS (x, y, z) in mm
    origin_mm: np.ndarray,      # (3,)    project (oz, oy, ox)
    spacing_mm: np.ndarray,     # (3,)    project (sz, sy, sx)
    direction: np.ndarray,      # (3, 3)  row-major, project convention
) -> np.ndarray:                # (3, N)  fractional voxel (fz, fy, fx)
    """Convert LPS world coordinates to fractional voxel indices in the DVF grid.

    Handles non-identity direction cosines via the full affine inverse.
    """
    # Flip DICOM (x, y, z) → project (z, y, x)
    pts_zyx = points_xyz_mm[:, ::-1].copy()   # (N, 3)

    # Affine: p = origin + direction @ diag(spacing) @ index
    # → A = direction @ diag(spacing),   index = A⁻¹ @ (p - origin)
    A = direction @ np.diag(spacing_mm)        # (3, 3)
    A_inv = np.linalg.inv(A)                   # (3, 3)

    rel = pts_zyx - origin_mm                  # (N, 3)
    coords = (A_inv @ rel.T)                   # (3, N) — (fz, fy, fx)
    return coords


# ---------------------------------------------------------------------------
# DVF interpolation
# ---------------------------------------------------------------------------

def _interp_dvf(
    dvf: DeformationField,
    points_xyz_mm: np.ndarray,  # (N, 3)  DICOM LPS (x, y, z) in mm
) -> np.ndarray:                # (N, 3)  displacements (dx, dy, dz) in mm
    """Trilinear interpolation of a forward DVF at arbitrary LPS points.

    Returns displacement in DICOM (dx, dy, dz) convention so they can be
    added directly to the input ``points_xyz_mm``.
    """
    nz, ny, nx = dvf.shape

    coords = _build_voxel_coords(
        points_xyz_mm, dvf.origin_mm, dvf.spacing_mm, dvf.direction
    )

    # Clamp to grid bounds (extrapolation = nearest edge value)
    coords[0] = np.clip(coords[0], 0, nz - 1)
    coords[1] = np.clip(coords[1], 0, ny - 1)
    coords[2] = np.clip(coords[2], 0, nx - 1)

    # DVF component order: vectors[0]=dz, vectors[1]=dy, vectors[2]=dx
    dz = map_coordinates(dvf.vectors[0], coords, order=1, mode="nearest")
    dy = map_coordinates(dvf.vectors[1], coords, order=1, mode="nearest")
    dx = map_coordinates(dvf.vectors[2], coords, order=1, mode="nearest")

    # Return in DICOM (dx, dy, dz) order so caller can do: pts + disps
    return np.column_stack([dx, dy, dz])   # (N, 3)


# ---------------------------------------------------------------------------
# RTSTRUCT deformation
# ---------------------------------------------------------------------------

def deform_rtstruct(
    rtstruct_path: str | Path,
    dvf: DeformationField,
    out_path: str | Path,
    state_index: int,
    epoch_ms: int,
    source_series_uid: str = "",
    deformed_series_uid: str = "",
    config: DeformationConfig | None = None,
) -> None:
    """Apply a forward DVF to all contour points and save a new RTSTRUCT DICOM.

    Each ``ContourData`` triplet ``(x, y, z)`` is displaced by the DVF vector
    interpolated at that LPS location.  The output RTSTRUCT gets new
    ``SOPInstanceUID`` / ``SeriesInstanceUID``, updated ``ContentDate`` /
    ``ContentTime``, a ``ReferencedSeriesSequence`` pointing at the deformed
    CT, and the ``DEFORMCT`` private tag block.

    Parameters:
        rtstruct_path: Original RTSTRUCT DICOM file.
        dvf: Forward displacement field (original → deformed space).
        out_path: Destination ``.dcm`` file path (directories are created).
        state_index: Zero-based ensemble state index (for naming / tags).
        epoch_ms: Motion epoch in milliseconds (sets ContentDate/Time).
        source_series_uid: SeriesInstanceUID of the original CT.
        deformed_series_uid: SeriesInstanceUID of the corresponding deformed CT.
        config: Deformation config for provenance tags.
    """
    if config is None:
        config = DeformationConfig()

    ds = copy.deepcopy(pydicom.dcmread(str(rtstruct_path)))

    # ── New UIDs ──────────────────────────────────────────────────────────────
    ds.SOPInstanceUID = generate_uid()
    if hasattr(ds, "file_meta") and ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = (
        getattr(ds, "SeriesDescription", "RT Structure Set")
        + f" | DeformCT state {state_index:03d}"
    )

    # ── Timestamps ───────────────────────────────────────────────────────────
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000.0, tz=datetime.timezone.utc)
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")[:13]

    # ── Reference deformed CT series ─────────────────────────────────────────
    if deformed_series_uid:
        _update_referenced_series(ds, deformed_series_uid)

    # ── Deform all contour points ─────────────────────────────────────────────
    n_contours = 0
    n_points   = 0

    for roi in getattr(ds, "ROIContourSequence", []):
        for contour in getattr(roi, "ContourSequence", []):
            flat = list(getattr(contour, "ContourData", []))
            if len(flat) % 3 != 0 or len(flat) == 0:
                continue

            pts   = np.array(flat, dtype=np.float64).reshape(-1, 3)  # (N, 3) x,y,z
            disps = _interp_dvf(dvf, pts)                             # (N, 3) dx,dy,dz
            contour.ContourData = (pts + disps).flatten().tolist()

            n_contours += 1
            n_points   += len(pts)

    # ── Private tags ─────────────────────────────────────────────────────────
    _set_private_block(
        ds,
        state_index=state_index,
        epoch_ms=epoch_ms,
        source_series_uid=source_series_uid,
        deformed_series_uid=deformed_series_uid,
        config=config,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), enforce_file_format=True)

    logger.info(
        "Saved deformed RTSTRUCT state %03d → %s  (%d contours, %d points)",
        state_index, out_path, n_contours, n_points,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_referenced_series(ds: pydicom.Dataset, new_series_uid: str) -> None:
    """Walk ReferencedFrameOfReferenceSequence and update series UIDs."""
    try:
        for fref in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            for study in getattr(fref, "RTReferencedStudySequence", []):
                for ser in getattr(study, "RTReferencedSeriesSequence", []):
                    ser.SeriesInstanceUID = new_series_uid
    except Exception as exc:
        logger.debug("Could not update ReferencedSeriesSequence: %s", exc)


def _set_private_block(
    ds: pydicom.Dataset,
    state_index: int,
    epoch_ms: int,
    source_series_uid: str,
    deformed_series_uid: str,
    config: DeformationConfig,
) -> None:
    g = _PRIVATE_BLOCK
    creator_tag = pydicom.tag.Tag(g, 0x0010)
    if creator_tag not in ds:
        ds.add_new(creator_tag, "LO", _PRIVATE_CREATOR)

    def _add(offset: int, vr: str, value: str) -> None:
        ds.add_new(pydicom.tag.Tag(g, 0x1000 + offset), vr, value)

    _add(0x00, "IS", str(state_index))
    _add(0x01, "LO", str(epoch_ms))
    _add(0x02, "UI", source_series_uid)
    _add(0x03, "UI", deformed_series_uid)
    _add(0x06, "DS", str(config.bone_threshold_hu))
    _add(0x07, "DS", str(config.transition_width_mm))
    _add(0x08, "DS", str(config.falloff_mm))
    _add(0x0A, "LO", _ENGINE_VERSION)
