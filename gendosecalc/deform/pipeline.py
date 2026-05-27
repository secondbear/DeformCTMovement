"""Ensemble CT deformation pipeline.

Orchestrates the full workflow:

1. Load reference CT (DICOM series)
2. Load RTSTRUCT and rasterise CTV mask
3. Ingest motion data (Synchrony XML or CSV)
4. Compute bone tissue weight map (once)
5. Select N representative states via k-medoids
6. For each state:
   a. Build CTV-localised DVF
   b. Save DVF as ``state_{i:03d}_dvf.mha``
   c. Apply deformation → deformed CT volume
   d. Export DICOM CT series as ``state_{i:03d}_ct/``
7. Write ``manifest.json``

GPU path:
    When ``config.use_gpu=True`` and ``cupy`` is available, the distance
    transform in ``localized_rigid_to_dvf`` is run on the GPU.  The
    per-state SimpleITK resampling remains on CPU (SimpleITK does not use
    CuPy); batching gains come from the vectorised distance transform.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from gendosecalc.deform.apply_deformation import deform_ct
from gendosecalc.deform.bone_mask import compute_bone_mask, compute_tissue_weight
from gendosecalc.deform.dicom_export import load_ct_series, save_ct_series
from gendosecalc.deform.dvf_generate import localized_rigid_to_dvf
from gendosecalc.deform.dvf_io import save_dvf
from gendosecalc.deform.models import (
    DeformationConfig,
    EnsembleManifestEntry,
    MotionSamples,
)
from gendosecalc.deform.motion_io import load_motion_csv, load_synchrony_xml
from gendosecalc.deform.motion_select import select_representative_states
from gendosecalc.deform.rtstruct_deform import deform_rtstruct
from gendosecalc.deform.structures import load_ctv_mask

logger = logging.getLogger(__name__)

_ENGINE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epoch_ms_to_iso(epoch_ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000.0, tz=datetime.timezone.utc)
    return dt.isoformat(timespec="milliseconds")


def _load_motion(motion_path: Path, tolerance_mode: str) -> MotionSamples:
    suffix = motion_path.suffix.lower()
    if suffix == ".xml":
        return load_synchrony_xml(motion_path, tolerance_mode=tolerance_mode)
    if suffix in (".csv", ".txt"):
        return load_motion_csv(motion_path)
    # Try XML first, fall back to CSV
    try:
        return load_synchrony_xml(motion_path, tolerance_mode=tolerance_mode)
    except Exception:
        return load_motion_csv(motion_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ensemble(
    ct_dir: str | Path,
    motion_path: str | Path,
    out_dir: str | Path,
    config: DeformationConfig | None = None,
    rtstruct_path: str | Path | None = None,
    deform_rtstruct_path: str | Path | None = None,
    n_states: int | None = None,
    rotation_weight_mm_per_deg: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full ensemble deformation pipeline.

    Parameters:
        ct_dir: Directory of the reference DICOM CT series.
        motion_path: Synchrony ``MotionData.xml`` or generic CSV.
        out_dir: Root output directory.  Sub-directories and files are created
            here.
        config: Deformation configuration.  Defaults to ``DeformationConfig()``.
        rtstruct_path: Path to RTSTRUCT DICOM.  If ``None``, the full rigid
            DVF (centred on image centre) is used without CTV localisation.
        deform_rtstruct_path: If provided, the RTSTRUCT at this path is
            deformed alongside each CT state and saved as
            ``state_{i:03d}_rs.dcm`` in ``out_dir``.  May be the same file
            as ``rtstruct_path``.
        n_states: Number of representative states.  Overrides ``config.n_states``
            if supplied.
        rotation_weight_mm_per_deg: Override ``config.rotation_weight_mm_per_deg``.
        seed: Random seed for k-medoids.

    Returns:
        The manifest as a dict (also written to ``manifest.json``).
    """
    if config is None:
        config = DeformationConfig()
    if n_states is not None:
        config.n_states = n_states
    if rotation_weight_mm_per_deg is not None:
        config.rotation_weight_mm_per_deg = rotation_weight_mm_per_deg

    ct_dir = Path(ct_dir)
    motion_path = Path(motion_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deform_rs = Path(deform_rtstruct_path) if deform_rtstruct_path else None

    # -------------------------------------------------------------------------
    # 1. Load reference CT
    # -------------------------------------------------------------------------
    logger.info("Loading CT from %s …", ct_dir)
    ct_array, spacing_mm, origin_mm, direction, source_datasets = load_ct_series(ct_dir)
    nz, ny, nx = ct_array.shape
    logger.info("CT shape: %s  spacing: %s mm", ct_array.shape, spacing_mm)

    source_series_uid = str(getattr(source_datasets[0], "SeriesInstanceUID", ""))

    # -------------------------------------------------------------------------
    # 2. CTV mask
    # -------------------------------------------------------------------------
    if rtstruct_path is not None:
        logger.info("Loading CTV mask from RTSTRUCT %s …", rtstruct_path)
        ctv_mask, ctv_centroid = load_ctv_mask(
            rtstruct_path,
            ct_shape=(nz, ny, nx),
            ct_spacing_mm=spacing_mm,
            ct_origin_mm=origin_mm,
            ct_direction=direction,
            roi_names=config.ctv_roi_names,
        )
        logger.info(
            "CTV mask: %d voxels, centroid LPS=(%s) mm",
            int(np.sum(ctv_mask)),
            ", ".join(f"{v:.1f}" for v in ctv_centroid),
        )
    else:
        # No RTSTRUCT: treat the whole soft-tissue volume as CTV
        ctv_mask = np.ones((nz, ny, nx), dtype=bool)
        ctv_centroid = None
        logger.warning(
            "No RTSTRUCT provided; CTV mask covers whole volume. "
            "Deformation will be bone-rigidity-only without CTV localisation."
        )

    # -------------------------------------------------------------------------
    # 3. Bone tissue weight (computed once)
    # -------------------------------------------------------------------------
    logger.info("Computing bone tissue weight map …")
    bone_mask = compute_bone_mask(ct_array, config)
    bone_weight = compute_tissue_weight(bone_mask, spacing_mm, config)
    logger.info(
        "Bone voxels: %d / %d  (%.1f %%)",
        int(np.sum(bone_mask)),
        bone_mask.size,
        100.0 * np.sum(bone_mask) / bone_mask.size,
    )

    # -------------------------------------------------------------------------
    # 4. Motion ingestion
    # -------------------------------------------------------------------------
    logger.info("Loading motion data from %s …", motion_path)
    samples = _load_motion(motion_path, config.motion_tolerance_mode)
    logger.info("Loaded %d motion samples", len(samples))

    if not samples.has_rotations:
        logger.warning(
            "Motion source has no rotation data; all rotations set to zero. "
            "Deformation will be translation-only per state."
        )

    # -------------------------------------------------------------------------
    # 5. k-medoids state selection
    # -------------------------------------------------------------------------
    n_states_eff = min(config.n_states, len(samples))
    logger.info(
        "Selecting %d representative states from %d samples …",
        n_states_eff, len(samples),
    )
    selection = select_representative_states(
        samples,
        n_states=n_states_eff,
        rotation_weight_mm_per_deg=config.rotation_weight_mm_per_deg,
        seed=seed,
    )

    # -------------------------------------------------------------------------
    # 6. Per-state processing
    # -------------------------------------------------------------------------
    manifest_entries: list[dict] = []

    for i, med_idx in enumerate(selection.medoid_indices):
        epoch_ms = int(samples.timestamps_ms[med_idx])
        dx, dy, dz = samples.offsets_mm[med_idx]
        rx, ry, rz = samples.rotations_deg[med_idx]
        cluster_weight = int(selection.cluster_weights[i])

        logger.info(
            "State %03d / %03d: ts=%d  t=(%+.2f,%+.2f,%+.2f)mm  "
            "r=(%+.2f,%+.2f,%+.2f)deg  weight=%d",
            i, n_states_eff - 1, epoch_ms,
            dx, dy, dz, rx, ry, rz, cluster_weight,
        )

        # DVF path
        dvf_filename = f"state_{i:03d}_dvf.mha"
        dvf_path = out_dir / dvf_filename
        ct_out_dir = out_dir / f"state_{i:03d}_ct"

        # --- Build DVF ---
        dvf = localized_rigid_to_dvf(
            tx=float(dx), ty=float(dy), tz=float(dz),
            rx=float(rx), ry=float(ry), rz=float(rz),
            reference_shape=(nz, ny, nx),
            spacing_mm=spacing_mm,
            origin_mm=origin_mm,
            direction=direction,
            ctv_mask=ctv_mask,
            bone_weight=bone_weight,
            config=config,
            centre_of_rotation_mm=ctv_centroid,
        )

        # --- Save DVF ---
        save_dvf(dvf, dvf_path)

        # --- Deform CT ---
        deformed_array, _, _, _ = deform_ct(
            ct_array, spacing_mm, origin_mm, direction, dvf, config,
        )

        # --- Export DICOM CT ---
        new_series_uid = save_ct_series(
            deformed_array=deformed_array,
            source_datasets=source_datasets,
            out_dir=ct_out_dir,
            state_index=i,
            epoch_ms=epoch_ms,
            dvf_filename=dvf_filename,
            tx=float(dx), ty=float(dy), tz=float(dz),
            rx=float(rx), ry=float(ry), rz=float(rz),
            config=config,
            cluster_weight=cluster_weight,
            source_series_uid=source_series_uid,
        )

        # --- Deform RTSTRUCT (optional) ---
        rs_rel_path = ""
        if deform_rs is not None:
            rs_filename = f"state_{i:03d}_rs.dcm"
            rs_out_path = out_dir / rs_filename
            try:
                deform_rtstruct(
                    rtstruct_path=deform_rs,
                    dvf=dvf,
                    out_path=rs_out_path,
                    state_index=i,
                    epoch_ms=epoch_ms,
                    source_series_uid=source_series_uid,
                    deformed_series_uid=new_series_uid,
                    config=config,
                )
                rs_rel_path = rs_filename
            except Exception as exc:
                logger.error(
                    "Failed to deform RTSTRUCT for state %03d: %s", i, exc, exc_info=True
                )

        entry = EnsembleManifestEntry(
            state_index=i,
            epoch_ms=epoch_ms,
            iso_timestamp=_epoch_ms_to_iso(epoch_ms),
            cluster_weight=cluster_weight,
            tx=float(dx),
            ty=float(dy),
            tz=float(dz),
            rx=float(rx),
            ry=float(ry),
            rz=float(rz),
            ct_dir=str(ct_out_dir.relative_to(out_dir)),
            dvf_path=dvf_filename,
            deformed_series_instance_uid=new_series_uid,
            source_ct_series_instance_uid=source_series_uid,
            rtstruct_path=rs_rel_path,
        )
        manifest_entries.append(entry.to_dict())

    # -------------------------------------------------------------------------
    # 7. Write manifest.json
    # -------------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "engine_version": _ENGINE_VERSION,
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "source_ct_dir": str(ct_dir),
        "source_ct_series_instance_uid": source_series_uid,
        "motion_path": str(motion_path),
        "rtstruct_path": str(rtstruct_path) if rtstruct_path else None,
        "deform_rtstruct_path": str(deform_rs) if deform_rs else None,
        "n_states": n_states_eff,
        "total_samples": len(samples),
        "k_medoids_cost": selection.total_cost,
        "config": {
            "bone_threshold_hu": config.bone_threshold_hu,
            "transition_width_mm": config.transition_width_mm,
            "falloff_mm": config.falloff_mm,
            "interpolation": config.interpolation,
            "rotation_weight_mm_per_deg": config.rotation_weight_mm_per_deg,
        },
        "states": manifest_entries,
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "Ensemble complete: %d states written to %s  (manifest: %s)",
        n_states_eff, out_dir, manifest_path,
    )
    return manifest
