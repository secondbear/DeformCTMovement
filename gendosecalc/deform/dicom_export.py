"""Export deformed CT volumes as DICOM CT series and manage RTDose tags.

Each deformed state is written as a new DICOM CT series with:
    - New ``SeriesInstanceUID`` (fresh UUID4-based UID)
    - Same ``FrameOfReferenceUID`` as the source series
    - ``ContentDate`` / ``ContentTime`` set to the motion-state timestamp
    - Private tag block (group ``0x6363``, creator ``"DEFORMCT"``) carrying
      full deformation provenance

Private tag layout (group 0x6363):
    0x6363,0x0010  — Creator = "DEFORMCT"
    0x6363,0x1000  — State index (int as string)
    0x6363,0x1001  — Epoch ms (int as string)
    0x6363,0x1002  — Source CT SeriesInstanceUID
    0x6363,0x1003  — DVF filename
    0x6363,0x1004  — tx,ty,tz (comma-separated mm)
    0x6363,0x1005  — rx,ry,rz (comma-separated degrees)
    0x6363,0x1006  — bone_threshold_hu
    0x6363,0x1007  — transition_width_mm
    0x6363,0x1008  — falloff_mm
    0x6363,0x1009  — cluster_weight
    0x6363,0x100A  — engine_version
"""

from __future__ import annotations

import copy
import datetime
import logging
import uuid
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from gendosecalc.deform.models import DeformationConfig, EnsembleManifestEntry

logger = logging.getLogger(__name__)

_PRIVATE_CREATOR = "DEFORMCT"
_PRIVATE_BLOCK = 0x6363
_ENGINE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# DICOM CT series I/O
# ---------------------------------------------------------------------------

def load_ct_series(series_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Dataset]]:
    """Load a DICOM CT series from a directory.

    Slices are sorted by ``ImagePositionPatient[2]`` (z coordinate).

    Parameters:
        series_dir: Directory containing ``.dcm`` files.

    Returns:
        ``(array, spacing_mm, origin_mm, direction, source_datasets)`` where:
        - array: int16 ``(nz, ny, nx)`` in HU
        - spacing_mm: ``(sz, sy, sx)`` in mm — row/col/slice spacing
        - origin_mm: ``(oz, oy, ox)`` of the first slice in LPS (z, y, x)
        - direction: ``(3, 3)`` direction cosine matrix (z, y, x convention)
        - source_datasets: list of pydicom Datasets for all slices (sorted)
    """
    series_dir = Path(series_dir)
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        # Some systems write files without extension
        dcm_files = sorted(series_dir.iterdir())

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "ImagePositionPatient"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No readable CT DICOM files found in {series_dir}")

    # Sort slices by z position
    datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    # Determine geometry from first slice
    first = datasets[0]
    row_spacing = float(first.PixelSpacing[0])
    col_spacing = float(first.PixelSpacing[1])

    if len(datasets) > 1:
        z0 = float(datasets[0].ImagePositionPatient[2])
        z1 = float(datasets[1].ImagePositionPatient[2])
        slice_spacing = abs(z1 - z0)
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    # Image orientation (row/col cosines → direction matrix)
    iop = np.array(first.ImageOrientationPatient, dtype=np.float64)
    row_cos = iop[:3]    # direction of increasing column (x direction in image)
    col_cos = iop[3:]    # direction of increasing row (y direction in image)
    slice_cos = np.cross(row_cos, col_cos)  # z direction

    # Our convention: direction rows = (z_cos, y_cos, x_cos) in (z,y,x) order
    # We define: dim0=z, dim1=y(row), dim2=x(col) following LPS convention
    direction = np.stack([slice_cos, col_cos, row_cos], axis=0)  # (3,3) (z,y,x)

    # Origin: position of voxel (0,0,0) in LPS
    ipp = np.array(first.ImagePositionPatient, dtype=np.float64)
    # DICOM gives origin in (x, y, z); convert to our (z, y, x)
    origin_mm = np.array([ipp[2], ipp[1], ipp[0]], dtype=np.float64)

    # Spacing in (z, y, x) order: slice_spacing, row_spacing, col_spacing
    spacing_mm = np.array([slice_spacing, row_spacing, col_spacing], dtype=np.float64)

    # Stack pixel data
    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.int16)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = (arr * slope + intercept).astype(np.int16)
        slices.append(hu)

    volume = np.stack(slices, axis=0)  # (nz, ny, nx)

    return volume, spacing_mm, origin_mm, direction, datasets


def _make_uid() -> str:
    """Generate a DICOM-safe UID using uuid4."""
    return generate_uid()


# ---------------------------------------------------------------------------
# Private tag helpers
# ---------------------------------------------------------------------------

def _set_private_block(
    ds: Dataset,
    state_index: int,
    epoch_ms: int,
    source_series_uid: str,
    dvf_filename: str,
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    config: DeformationConfig,
    cluster_weight: int,
) -> None:
    """Write the DEFORMCT private tag block onto ``ds`` in-place."""
    g = _PRIVATE_BLOCK

    # Creator — add only if not already present
    creator_tag = pydicom.tag.Tag(g, 0x0010)
    if creator_tag not in ds:
        ds.add_new(creator_tag, "LO", _PRIVATE_CREATOR)

    def _add(tag_offset: int, vr: str, value: str) -> None:
        tag = pydicom.tag.Tag(g, 0x1000 + tag_offset)
        ds.add_new(tag, vr, value)

    _add(0x00, "IS", str(state_index))
    _add(0x01, "LO", str(epoch_ms))          # epoch too large for DS; use LO
    _add(0x02, "UI", source_series_uid)
    _add(0x03, "LO", dvf_filename)
    _add(0x04, "LO", f"{tx:.6f},{ty:.6f},{tz:.6f}")   # compound — use LO
    _add(0x05, "LO", f"{rx:.6f},{ry:.6f},{rz:.6f}")   # compound — use LO
    _add(0x06, "DS", str(config.bone_threshold_hu))
    _add(0x07, "DS", str(config.transition_width_mm))
    _add(0x08, "DS", str(config.falloff_mm))
    _add(0x09, "IS", str(cluster_weight))
    _add(0x0A, "LO", _ENGINE_VERSION)


# ---------------------------------------------------------------------------
# RTDose tag helper (reusable by downstream engine)
# ---------------------------------------------------------------------------

def write_rtdose_state_tags(
    ds: Dataset,
    manifest_entry: EnsembleManifestEntry,
    config: DeformationConfig | None = None,
) -> None:
    """Write motion-state provenance tags onto an RTDose dataset.

    Sets ``ContentDate`` / ``ContentTime`` from the motion epoch, adds the
    DEFORMCT private block, and adds a ``ReferencedSeriesSequence`` pointing
    to the deformed CT.

    Parameters:
        ds: RTDose pydicom Dataset (modified in-place).
        manifest_entry: The state record from the ensemble manifest.
        config: Deformation config used for the state (for private tags).
    """
    if config is None:
        config = DeformationConfig()

    epoch_s = manifest_entry.epoch_ms / 1000.0
    dt = datetime.datetime.fromtimestamp(epoch_s, tz=datetime.timezone.utc)
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")[:13]

    _set_private_block(
        ds,
        state_index=manifest_entry.state_index,
        epoch_ms=manifest_entry.epoch_ms,
        source_series_uid=manifest_entry.source_ct_series_instance_uid,
        dvf_filename=Path(manifest_entry.dvf_path).name,
        tx=manifest_entry.tx,
        ty=manifest_entry.ty,
        tz=manifest_entry.tz,
        rx=manifest_entry.rx,
        ry=manifest_entry.ry,
        rz=manifest_entry.rz,
        config=config,
        cluster_weight=manifest_entry.cluster_weight,
    )

    # ReferencedSeriesSequence
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = manifest_entry.deformed_series_instance_uid
    ds.ReferencedSeriesSequence = Sequence([ref_series])


# ---------------------------------------------------------------------------
# CT series export
# ---------------------------------------------------------------------------

def save_ct_series(
    deformed_array: np.ndarray,
    source_datasets: list[Dataset],
    out_dir: str | Path,
    state_index: int,
    epoch_ms: int,
    dvf_filename: str,
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    config: DeformationConfig,
    cluster_weight: int,
    source_series_uid: str | None = None,
) -> str:
    """Write a deformed CT as a new DICOM series.

    Parameters:
        deformed_array: HU volume ``(nz, ny, nx)`` int16.
        source_datasets: Sorted list of source DICOM datasets (templates).
        out_dir: Directory to write the new series into (created if needed).
        state_index: Zero-based state index for naming / tags.
        epoch_ms: Motion-state epoch milliseconds.
        dvf_filename: Name of the paired DVF file (for private tags).
        tx, ty, tz: Applied translation in mm.
        rx, ry, rz: Applied rotation in degrees.
        config: Deformation config.
        cluster_weight: Number of source samples this state represents.
        source_series_uid: SeriesInstanceUID of the source CT; inferred from
            ``source_datasets[0]`` if not supplied.

    Returns:
        The new ``SeriesInstanceUID`` assigned to this series.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_series_uid is None:
        source_series_uid = str(getattr(source_datasets[0], "SeriesInstanceUID", ""))

    new_series_uid = _make_uid()

    # Epoch → DICOM date/time
    epoch_s = epoch_ms / 1000.0
    dt = datetime.datetime.fromtimestamp(epoch_s, tz=datetime.timezone.utc)
    content_date = dt.strftime("%Y%m%d")
    content_time = dt.strftime("%H%M%S.%f")[:13]
    series_desc = f"DeformCT state {state_index:03d} {dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    nz = deformed_array.shape[0]
    if len(source_datasets) != nz:
        logger.warning(
            "Source dataset count (%d) != deformed array slices (%d); "
            "padding/truncating source templates",
            len(source_datasets), nz,
        )

    for z in range(nz):
        src_idx = min(z, len(source_datasets) - 1)
        src_ds = copy.deepcopy(source_datasets[src_idx])

        # Update UIDs
        src_ds.SOPInstanceUID = _make_uid()
        src_ds.SeriesInstanceUID = new_series_uid

        # Timestamps
        src_ds.ContentDate = content_date
        src_ds.ContentTime = content_time

        # Series description
        src_ds.SeriesDescription = series_desc
        src_ds.SeriesNumber = 9000 + state_index

        # Write pixel data
        slice_arr = deformed_array[z]
        # Apply inverse rescale to go from HU back to stored integer
        slope = float(getattr(src_ds, "RescaleSlope", 1))
        intercept = float(getattr(src_ds, "RescaleIntercept", 0))
        stored = np.round((slice_arr.astype(np.float32) - intercept) / slope).astype(np.int16)
        src_ds.PixelData = stored.tobytes()
        src_ds.BitsAllocated = 16
        src_ds.BitsStored = 16
        src_ds.HighBit = 15
        src_ds.PixelRepresentation = 1  # signed

        # Private tags
        _set_private_block(
            src_ds,
            state_index=state_index,
            epoch_ms=epoch_ms,
            source_series_uid=source_series_uid,
            dvf_filename=dvf_filename,
            tx=tx, ty=ty, tz=tz,
            rx=rx, ry=ry, rz=rz,
            config=config,
            cluster_weight=cluster_weight,
        )

        out_path = out_dir / f"CT.{z:04d}.dcm"
        src_ds.save_as(str(out_path), write_like_original=False)

    logger.info(
        "Saved state %03d DICOM CT series (%d slices) → %s  [UID=%s]",
        state_index, nz, out_dir, new_series_uid,
    )
    return new_series_uid
