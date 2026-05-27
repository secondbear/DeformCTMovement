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

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

from gendosecalc.deform.models import DeformationConfig, DeformationField

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# CTV-localised DVF generation
# ---------------------------------------------------------------------------

def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Smoothstep function clamped to [0, 1] between edge0 and edge1.

    Returns 1 at x <= edge0, 0 at x >= edge1, smooth cubic in between.
    The returned array has the same shape as ``x``.
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    # reverse: 1 near CTV (small distance), 0 far from CTV
    t = 1.0 - t
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def localized_rigid_to_dvf(
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
    ctv_mask: np.ndarray,
    bone_weight: np.ndarray,
    config: DeformationConfig | None = None,
    centre_of_rotation_mm: np.ndarray | None = None,
) -> DeformationField:
    """Generate a CTV-localised forward DVF with bone rigidity.

    The displacement field is:

    .. code-block:: text

        weight = bone_weight · ctv_falloff
        dvf.vectors *= weight[None, ...]

    where ``bone_weight`` (from ``compute_tissue_weight``) is 0 in bone and
    1 in soft tissue, and ``ctv_falloff`` is 1 inside the CTV and decays to 0
    at ``falloff_mm`` distance from the CTV boundary (smoothstep).

    The centre of rotation for the rigid transform defaults to the CTV
    centroid unless ``centre_of_rotation_mm`` is provided explicitly.

    Parameters:
        tx, ty, tz: Translation in mm (LPS axes).
        rx, ry, rz: Rotation in degrees.
        reference_shape: ``(nz, ny, nx)``.
        spacing_mm: ``(sz, sy, sx)`` in mm.
        origin_mm: ``(oz, oy, ox)`` in LPS mm.
        direction: Direction cosine matrix ``(3, 3)``.
        ctv_mask: Boolean array ``(nz, ny, nx)``, True inside the CTV.
        bone_weight: Float32 array ``(nz, ny, nx)``, 0 in bone, 1 in soft tissue.
        config: Deformation configuration; uses ``falloff_mm``.
        centre_of_rotation_mm: Rotation centre in LPS mm.  Defaults to the
            centroid of ``ctv_mask``.

    Returns:
        A forward ``DeformationField`` with spatially attenuated displacements.
    """
    if config is None:
        config = DeformationConfig()

    spacing = np.asarray(spacing_mm, dtype=np.float64)

    # Default centre of rotation = CTV centroid
    if centre_of_rotation_mm is None:
        nz_i, ny_i, nx_i = np.where(ctv_mask)
        if len(nz_i) > 0:
            voxel_centroid = np.array([
                np.mean(nz_i), np.mean(ny_i), np.mean(nx_i),
            ])
            dirmat = np.asarray(direction, dtype=np.float64).reshape(3, 3)
            origin = np.asarray(origin_mm, dtype=np.float64)
            # Forward transform voxel (z,y,x) → LPS (x,y,z) → flip to (z,y,x)
            centre_xyz = origin[::-1] + dirmat.T @ (voxel_centroid * spacing)
            centre_of_rotation_mm = centre_xyz[::-1]
        # else: falls back to image centre inside rigid_to_dvf

    # Build full rigid DVF centred on CTV centroid
    dvf = rigid_to_dvf(
        tx, ty, tz, rx, ry, rz,
        reference_shape, spacing_mm, origin_mm, direction,
        centre_of_rotation_mm,
    )

    # Compute CTV falloff weight via distance transform
    # distance_transform_edt returns distance in voxels → convert to mm
    ctv_dist_voxels = distance_transform_edt(~ctv_mask)  # (nz, ny, nx)
    # Use mean voxel spacing as representative mm/voxel
    mean_spacing = float(np.mean(spacing))
    ctv_dist_mm = ctv_dist_voxels * mean_spacing

    ctv_falloff = _smoothstep(ctv_dist_mm, 0.0, config.falloff_mm)  # (nz, ny, nx)

    # Combined weight: attenuate by bone AND distance from CTV
    combined_weight = bone_weight * ctv_falloff  # (nz, ny, nx)

    # Apply to DVF
    masked_vectors = dvf.vectors * combined_weight[np.newaxis, :, :, :]
    dvf = DeformationField(
        vectors=masked_vectors,
        spacing_mm=dvf.spacing_mm.copy(),
        origin_mm=dvf.origin_mm.copy(),
        direction=dvf.direction.copy(),
        source_description=(
            f"localized_tx{tx}_ty{ty}_tz{tz}_rx{rx}_ry{ry}_rz{rz}"
        ),
    )

    logger.debug(
        "localized_rigid_to_dvf: t=(%g,%g,%g)mm r=(%g,%g,%g)deg "
        "falloff=%.1fmm max_disp=%.3fmm",
        tx, ty, tz, rx, ry, rz,
        config.falloff_mm,
        float(np.max(np.linalg.norm(dvf.vectors, axis=0))),
    )
    return dvf

