"""Load RTSTRUCT DICOM and rasterise a named ROI onto a CT grid.

Usage::

    ctv_mask, centroid_lps = load_ctv_mask(
        rtstruct_path="RS.dcm",
        ct_shape=(nz, ny, nx),
        ct_spacing_mm=spacing,
        ct_origin_mm=origin,
        ct_direction=direction,
        roi_names=["CTV", "CTV_prostate"],
    )

The returned mask is aligned to the supplied CT grid (same shape / spacing /
origin). ``centroid_lps`` is the mass-weighted centroid in LPS mm.

Implementation notes:
    - Contours are rasterised slice-by-slice using
      ``skimage.draw.polygon`` (2-D convex or non-convex poly fill).
    - Multiple contour segments on the same slice are ORed together.
    - ROI matching is case-insensitive and searches the supplied ``roi_names``
      in order; the first match wins.
    - Only the ``(x, y)`` plane of each contour point is used; the ``z``
      coordinate is used to identify the CT slice index.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pydicom
from skimage.draw import polygon

logger = logging.getLogger(__name__)

_DEFAULT_ROI_NAMES = ["CTV", "CTV_prostate", "CTV_Prostate", "ctv"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_roi_sequence(rtstruct: pydicom.Dataset, roi_names: list[str]) -> pydicom.Dataset | None:
    """Return the first ROI contour sequence whose name matches any of ``roi_names``."""
    name_lower = [n.lower() for n in roi_names]

    # Build a mapping from ROI number → ROI name
    roi_number_to_name: dict[int, str] = {}
    if hasattr(rtstruct, "StructureSetROISequence"):
        for roi in rtstruct.StructureSetROISequence:
            roi_number_to_name[int(roi.ROINumber)] = getattr(roi, "ROIName", "")

    if not hasattr(rtstruct, "ROIContourSequence"):
        return None

    for contour_seq in rtstruct.ROIContourSequence:
        roi_num = int(getattr(contour_seq, "ReferencedROINumber", -1))
        roi_name = roi_number_to_name.get(roi_num, "")
        if roi_name.lower() in name_lower:
            logger.debug("Found ROI '%s' (number %d)", roi_name, roi_num)
            return contour_seq

    # Fallback: try matching directly on ContourSequence description (non-standard)
    logger.warning(
        "No ROI found matching names %s. Available: %s",
        roi_names,
        list(roi_number_to_name.values()),
    )
    return None


def _lps_to_voxel(
    lps_coords: np.ndarray,
    origin_mm: np.ndarray,
    spacing_mm: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Map LPS mm coordinates to continuous voxel indices.

    Parameters:
        lps_coords: shape ``(N, 3)`` in LPS (z, y, x) order.
        origin_mm: ``(3,)`` volume origin in LPS (z, y, x).
        spacing_mm: ``(3,)`` voxel spacing in LPS (z, y, x).
        direction: ``(3, 3)`` direction cosine matrix — each row is a LPS
            (x, y, z) unit vector along increasing (iz, iy, ix).

    Returns:
        Continuous voxel indices ``(N, 3)`` in (iz, iy, ix) order.
    """
    # Documented inverse transform (copilot-instructions §RTSTRUCT↔CT):
    #   M_inv = diag(1/spacing) @ direction   — acts on LPS (x,y,z)
    #   vox   = (M_inv @ (pts_xyz - origin_xyz).T).T
    # lps_coords and origin_mm arrive in project (z,y,x) order; flip to (x,y,z).
    pts_xyz = lps_coords[:, ::-1]                     # (N,3) LPS (x,y,z)
    origin_xyz = origin_mm[::-1]                      # (3,)  LPS (x,y,z)
    M_inv = np.diag(1.0 / spacing_mm) @ direction     # (3,3)
    rel = pts_xyz - origin_xyz                        # (N,3) LPS (x,y,z)
    return (M_inv @ rel.T).T                          # (N,3) → (iz, iy, ix)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ctv_mask(
    rtstruct_path: str | Path,
    ct_shape: tuple[int, int, int],
    ct_spacing_mm: np.ndarray,
    ct_origin_mm: np.ndarray,
    ct_direction: np.ndarray,
    roi_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise an RTSTRUCT ROI onto a CT grid.

    Parameters:
        rtstruct_path: Path to the RTSTRUCT DICOM file.
        ct_shape: Volume shape ``(nz, ny, nx)``.
        ct_spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        ct_origin_mm: Volume origin ``(oz, oy, ox)`` in LPS mm.
        ct_direction: Direction cosine matrix ``(3, 3)`` in (z, y, x) convention.
        roi_names: Ordered list of ROI names to search for (case-insensitive).
            Defaults to ``["CTV", "CTV_prostate", "CTV_Prostate", "ctv"]``.

    Returns:
        ``(mask, centroid_lps_mm)`` where:
        - mask: Boolean array ``(nz, ny, nx)`` — True inside the ROI.
        - centroid_lps_mm: LPS mm coordinates of the mask centroid,
          shape ``(3,)`` in ``(z, y, x)`` order.

    Raises:
        FileNotFoundError: If the RTSTRUCT file does not exist.
        ValueError: If no matching ROI is found or it has no contour data.
    """
    if roi_names is None:
        roi_names = _DEFAULT_ROI_NAMES

    rtstruct_path = Path(rtstruct_path)
    if not rtstruct_path.exists():
        raise FileNotFoundError(f"RTSTRUCT file not found: {rtstruct_path}")

    rtstruct = pydicom.dcmread(str(rtstruct_path))
    contour_seq = _find_roi_sequence(rtstruct, roi_names)
    if contour_seq is None:
        raise ValueError(
            f"No ROI matching names {roi_names} found in {rtstruct_path}"
        )

    if not hasattr(contour_seq, "ContourSequence") or not contour_seq.ContourSequence:
        raise ValueError(
            f"ROI found but contains no ContourSequence in {rtstruct_path}"
        )

    nz, ny, nx = ct_shape
    mask = np.zeros(ct_shape, dtype=bool)
    spacing = np.asarray(ct_spacing_mm, dtype=np.float64)
    origin = np.asarray(ct_origin_mm, dtype=np.float64)
    direction = np.asarray(ct_direction, dtype=np.float64).reshape(3, 3)

    for contour in contour_seq.ContourSequence:
        contour_type = getattr(contour, "ContourGeometricType", "CLOSED_PLANAR")
        if contour_type not in ("CLOSED_PLANAR", "POINT"):
            logger.debug("Skipping contour type '%s'", contour_type)
            continue

        raw = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        # ContourData is in DICOM LPS (x, y, z) mm. Convert to our (z, y, x) order.
        lps_zyx = raw[:, [2, 1, 0]]

        voxel = _lps_to_voxel(lps_zyx, origin, spacing, direction)

        # Determine slice index from z coordinate (round to nearest integer)
        iz = int(np.round(np.mean(voxel[:, 0])))
        if not (0 <= iz < nz):
            continue

        # Rasterise (iy, ix) polygon on this slice
        row_coords = voxel[:, 1]
        col_coords = voxel[:, 2]

        rr, cc = polygon(row_coords, col_coords, shape=(ny, nx))
        # Clamp to bounds
        valid = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        mask[iz, rr[valid], cc[valid]] = True

    if not np.any(mask):
        warnings.warn(
            f"Rasterised CTV mask is empty for ROI names {roi_names} in "
            f"{rtstruct_path}. Check ROI name and CT geometry alignment.",
            stacklevel=2,
        )

    # Compute centroid in LPS mm (z, y, x order)
    nz_i, ny_i, nx_i = np.where(mask)
    if len(nz_i) > 0:
        voxel_centroid = np.array([
            np.mean(nz_i),
            np.mean(ny_i),
            np.mean(nx_i),
        ])
        # Forward transform voxel (z,y,x) → LPS (x,y,z) → flip back to (z,y,x)
        # physical_xyz = origin_xyz + direction.T @ (spacing * vox_zyx)
        centroid_xyz = origin[::-1] + direction.T @ (voxel_centroid * spacing)
        centroid_lps = centroid_xyz[::-1]             # (x,y,z) → (z,y,x)
    else:
        centroid_lps = origin.copy()

    return mask, centroid_lps
