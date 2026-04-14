"""Generate DVFs from rigid-body parameters or Synchrony motion log entries.

Convention:
    All DVFs are **forward** (original → deformed position). The downstream
    ``deform_ct()`` function inverts them for SimpleITK's inverse-mapping
    resampler.

Euler rotation order: Rz · Ry · Rx (intrinsic XYZ / extrinsic ZYX).
Angles are in **degrees** at the API boundary, converted to radians internally.
Centre of rotation defaults to the image centre in LPS mm.
"""

from __future__ import annotations

import numpy as np

from gendosecalc.deform.models import DeformationField


def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build a 3×3 rotation matrix from Euler angles (degrees).

    Order: Rz · Ry · Rx (extrinsic ZYX convention).
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Rz · Ry · Rx
    R = np.array([
        [cz * cy,  cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx],
        [sz * cy,  sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx],
        [-sy,      cy * sx,                  cy * cx],
    ], dtype=np.float64)
    return R


def rigid_to_dvf(
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    reference_shape: tuple[int, int, int],
    spacing_mm: np.ndarray,
    origin_mm: np.ndarray,
    direction: np.ndarray,
    centre_of_rotation_mm: np.ndarray | None = None,
) -> DeformationField:
    """Generate a forward DVF from 6DOF rigid-body parameters.

    Parameters:
        tx, ty, tz: Translation in mm along LPS axes (z, y, x order in our
            convention but named after physical axes: x=LR, y=AP, z=SI).
        rx, ry, rz: Rotation in degrees about LPS axes.
        reference_shape: Volume shape ``(nz, ny, nx)``.
        spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        origin_mm: Volume origin ``(oz, oy, ox)`` in LPS mm.
        direction: Direction cosine matrix ``(3, 3)``.
        centre_of_rotation_mm: Rotation centre in LPS mm ``(cz, cy, cx)``.
            Defaults to volume centre.

    Returns:
        A forward ``DeformationField`` where each voxel stores the displacement
        from its original to deformed position.
    """
    nz, ny, nx = reference_shape
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64)
    dirmat = np.asarray(direction, dtype=np.float64).reshape(3, 3)

    # Compute volume centre in LPS if no centre_of_rotation given
    if centre_of_rotation_mm is None:
        half_extent = spacing * np.array([nz - 1, ny - 1, nx - 1]) / 2.0
        centre_of_rotation_mm = origin + dirmat @ half_extent
    else:
        centre_of_rotation_mm = np.asarray(centre_of_rotation_mm, dtype=np.float64)

    # Build rotation matrix
    R = _rotation_matrix(rx, ry, rz)
    translation = np.array([tz, ty, tx], dtype=np.float64)

    # Create voxel index grids
    iz, iy, ix = np.mgrid[0:nz, 0:ny, 0:nx]
    # Stack into (3, nz*ny*nx) — voxel indices in (z, y, x) order
    voxel_indices = np.stack([
        iz.ravel().astype(np.float64),
        iy.ravel().astype(np.float64),
        ix.ravel().astype(np.float64),
    ], axis=0)

    # Convert voxel indices to physical LPS coordinates
    # physical = origin + direction @ (indices * spacing)
    scaled = voxel_indices * spacing[:, np.newaxis]
    physical = origin[:, np.newaxis] + dirmat @ scaled

    # Apply rigid transform about centre of rotation
    centred = physical - centre_of_rotation_mm[:, np.newaxis]
    rotated = R @ centred
    deformed = rotated + centre_of_rotation_mm[:, np.newaxis] + translation[:, np.newaxis]

    # Displacement = deformed - original
    displacement = (deformed - physical).astype(np.float32)
    vectors = displacement.reshape(3, nz, ny, nx)

    desc_parts = []
    if any(v != 0 for v in (tx, ty, tz)):
        desc_parts.append(f"tx{tx}_ty{ty}_tz{tz}")
    if any(v != 0 for v in (rx, ry, rz)):
        desc_parts.append(f"rx{rx}_ry{ry}_rz{rz}")
    source_desc = "rigid_" + "_".join(desc_parts) if desc_parts else "rigid_identity"

    return DeformationField(
        vectors=vectors,
        spacing_mm=spacing.copy(),
        origin_mm=origin.copy(),
        direction=dirmat.copy(),
        source_description=source_desc,
    )


# ---------------------------------------------------------------------------
# Synchrony motion log mapping
# ---------------------------------------------------------------------------

# Mapping from Synchrony motion log keys to our (tx, ty, tz, rx, ry, rz)
# Synchrony convention: SI (superior-inferior), LR (left-right), AP (anterior-posterior)
_MOTION_KEY_MAP = {
    "LR": "tx",   # left-right → x
    "AP": "ty",   # anterior-posterior → y
    "SI": "tz",   # superior-inferior → z
    "Roll": "rx",
    "Pitch": "ry",
    "Yaw": "rz",
}


def motion_log_entry_to_dvf(
    motion_entry: dict,
    reference_shape: tuple[int, int, int],
    spacing_mm: np.ndarray,
    origin_mm: np.ndarray,
    direction: np.ndarray,
    centre_of_rotation_mm: np.ndarray | None = None,
) -> DeformationField:
    """Generate a forward DVF from a Synchrony motion log entry.

    Parameters:
        motion_entry: Dict with keys ``SI``, ``LR``, ``AP``, ``Pitch``,
            ``Roll``, ``Yaw``. Missing keys default to 0.
        reference_shape: Volume shape ``(nz, ny, nx)``.
        spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm.
        origin_mm: Volume origin ``(oz, oy, ox)`` in LPS mm.
        direction: Direction cosine matrix ``(3, 3)``.
        centre_of_rotation_mm: Optional rotation centre in LPS mm.

    Returns:
        A forward ``DeformationField``.
    """
    tx = float(motion_entry.get("LR", 0.0))
    ty = float(motion_entry.get("AP", 0.0))
    tz = float(motion_entry.get("SI", 0.0))
    rx = float(motion_entry.get("Roll", 0.0))
    ry = float(motion_entry.get("Pitch", 0.0))
    rz = float(motion_entry.get("Yaw", 0.0))

    dvf = rigid_to_dvf(
        tx, ty, tz, rx, ry, rz,
        reference_shape, spacing_mm, origin_mm, direction,
        centre_of_rotation_mm,
    )
    dvf.source_description = f"motion_log_{motion_entry}"
    return dvf
