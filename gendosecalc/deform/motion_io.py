"""Parse motion data from Synchrony MotionData.xml or generic CSV files.

Synchrony XML (v1–v3):
    ``<DataPoint>`` elements under ``MotionData/DeliverySegment/*/ModelData``.
    Each point has ``<Timestamp>`` (epoch ms), ``<TargetOffset><X/Y/Z>``.
    Version 3 does not carry rotational data; rotations default to zero.

Generic CSV:
    Header row required. Recognised column names (case-insensitive):
        ``timestamp_ms``          — epoch milliseconds
        ``dx`` / ``x`` / ``lx``  — LR displacement in mm
        ``dy`` / ``y`` / ``ay``  — AP displacement in mm
        ``dz`` / ``z`` / ``sz``  — SI displacement in mm
        ``rx``                   — rotation x (degrees)  [optional]
        ``ry``                   — rotation y (degrees)  [optional]
        ``rz``                   — rotation z (degrees)  [optional]

Tolerance filtering (XML only):
    ``<PotentialDifferenceTolerance>`` and ``<MeasuredDifferenceTolerance>``
    are read per segment.  Points with ``PotentialDiff > tolerance`` or
    ``MeasuredDifference > tolerance`` (excluding NaN) are handled according
    to ``tolerance_mode``:
        ``"warn"``  — log a warning, keep the sample (default)
        ``"drop"``  — silently discard the sample
        ``"keep"``  — do not filter at all
"""

from __future__ import annotations

import csv
import logging
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from gendosecalc.deform.models import MotionSamples

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchrony XML parser
# ---------------------------------------------------------------------------

_XML_TRANSLATION_AXES = ("X", "Y", "Z")  # TargetOffset sub-elements → LR, AP, SI


def load_synchrony_xml(
    path: str | Path,
    tolerance_mode: str = "warn",
) -> MotionSamples:
    """Parse a Synchrony MotionData.xml file into a ``MotionSamples`` object.

    Parameters:
        path: Path to ``MotionData_*.xml``.
        tolerance_mode: How to handle out-of-tolerance data points.
            ``"warn"`` keeps them with a warning, ``"drop"`` removes them,
            ``"keep"`` performs no filtering.

    Returns:
        A ``MotionSamples`` with translations from ``<TargetOffset>``.
        Rotations are all zero (Synchrony v1–v3 does not carry them);
        ``has_rotations`` is set to ``False``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion XML not found: {path}")

    tree = ET.parse(str(path))
    root = tree.getroot()

    timestamps: list[int] = []
    offsets: list[tuple[float, float, float]] = []  # (x, y, z) = (LR, AP, SI)

    for segment in root.iter("DeliverySegment"):
        for results in segment.iter("RadiationResults"):
            for model_data in results.iter("ModelData"):
                # Read tolerances for this block
                try:
                    pot_tol = float(
                        model_data.findtext("PotentialDifferenceTolerance", default="1e9")
                    )
                except (TypeError, ValueError):
                    pot_tol = float("inf")
                try:
                    meas_tol = float(
                        model_data.findtext("MeasuredDifferenceTolerance", default="1e9")
                    )
                except (TypeError, ValueError):
                    meas_tol = float("inf")

                for dp in model_data.iter("DataPoint"):
                    ts_text = dp.findtext("Timestamp")
                    if ts_text is None:
                        continue
                    try:
                        ts = int(ts_text)
                    except ValueError:
                        continue

                    # Tolerance check
                    if tolerance_mode != "keep":
                        pot_diff_text = dp.findtext("PotentialDiff", default="0")
                        meas_diff_text = dp.findtext("MeasuredDifference", default="NaN")
                        try:
                            pot_diff = float(pot_diff_text)
                        except (TypeError, ValueError):
                            pot_diff = 0.0
                        try:
                            meas_diff = float(meas_diff_text)
                        except (TypeError, ValueError):
                            meas_diff = float("nan")

                        out_of_tol = pot_diff > pot_tol or (
                            not np.isnan(meas_diff) and meas_diff > meas_tol
                        )
                        if out_of_tol:
                            if tolerance_mode == "warn":
                                logger.warning(
                                    "DataPoint ts=%d out of tolerance "
                                    "(PotDiff=%.3f>%.3f or MeasDiff=%.3f>%.3f)",
                                    ts, pot_diff, pot_tol, meas_diff, meas_tol,
                                )
                            else:
                                continue  # drop

                    offset_el = dp.find("TargetOffset")
                    if offset_el is None:
                        continue
                    try:
                        ox = float(offset_el.findtext("X", default="0") or "0")
                        oy = float(offset_el.findtext("Y", default="0") or "0")
                        oz = float(offset_el.findtext("Z", default="0") or "0")
                    except (TypeError, ValueError):
                        continue

                    timestamps.append(ts)
                    offsets.append((ox, oy, oz))

    if not timestamps:
        raise ValueError(f"No valid DataPoints found in {path}")

    ts_arr = np.array(timestamps, dtype=np.int64)
    off_arr = np.array(offsets, dtype=np.float32)  # (N, 3): LR, AP, SI
    rot_arr = np.zeros((len(timestamps), 3), dtype=np.float32)

    logger.info("Loaded %d motion samples from %s", len(ts_arr), path.name)

    return MotionSamples(
        timestamps_ms=ts_arr,
        offsets_mm=off_arr,
        rotations_deg=rot_arr,
        has_rotations=False,
        source_path=str(path),
    )


# ---------------------------------------------------------------------------
# Generic CSV parser
# ---------------------------------------------------------------------------

# Map of normalised column names → internal field
_CSV_TIMESTAMP_COLS = {"timestamp_ms", "timestamp", "ts", "ts_ms", "time_ms"}
_CSV_DX_COLS = {"dx", "x", "lx", "d_lr", "lr"}
_CSV_DY_COLS = {"dy", "y", "ay", "d_ap", "ap"}
_CSV_DZ_COLS = {"dz", "z", "sz", "d_si", "si"}
_CSV_RX_COLS = {"rx", "rot_x", "roll"}
_CSV_RY_COLS = {"ry", "rot_y", "pitch"}
_CSV_RZ_COLS = {"rz", "rot_z", "yaw"}


def _map_header(headers: list[str]) -> dict[str, int]:
    """Return a mapping {field: column_index} from CSV headers."""
    h_lower = [h.strip().lower() for h in headers]
    mapping: dict[str, int] = {}

    for col_set, key in [
        (_CSV_TIMESTAMP_COLS, "ts"),
        (_CSV_DX_COLS, "dx"),
        (_CSV_DY_COLS, "dy"),
        (_CSV_DZ_COLS, "dz"),
        (_CSV_RX_COLS, "rx"),
        (_CSV_RY_COLS, "ry"),
        (_CSV_RZ_COLS, "rz"),
    ]:
        for idx, h in enumerate(h_lower):
            if h in col_set:
                mapping[key] = idx
                break

    return mapping


def load_motion_csv(path: str | Path) -> MotionSamples:
    """Parse a generic motion CSV file into a ``MotionSamples`` object.

    Required columns: timestamp (ms), dx, dy, dz.
    Optional columns: rx, ry, rz (degrees).  If absent, rotations are set to
    zero and a warning is emitted.

    Parameters:
        path: Path to the CSV file.

    Returns:
        A ``MotionSamples`` object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or no valid rows are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion CSV not found: {path}")

    timestamps: list[int] = []
    offsets: list[tuple[float, float, float]] = []
    rotations: list[tuple[float, float, float]] = []

    col_map: dict[str, int] = {}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header_row: list[str] | None = None

        for row_idx, row in enumerate(reader):
            if not any(cell.strip() for cell in row):
                continue  # skip blank lines

            if header_row is None:
                header_row = row
                col_map = _map_header(header_row)

                required = {"ts", "dx", "dy", "dz"}
                missing = required - col_map.keys()
                if missing:
                    raise ValueError(
                        f"CSV {path} missing required columns: {missing}. "
                        f"Available headers: {header_row}"
                    )

                has_rot = {"rx", "ry", "rz"}.issubset(col_map.keys())
                if not has_rot:
                    warnings.warn(
                        f"Rotation columns (rx/ry/rz) not found in {path}; "
                        "setting all rotations to zero.",
                        stacklevel=2,
                    )
                continue

            # Data row
            try:
                ts = int(float(row[col_map["ts"]]))
                dx = float(row[col_map["dx"]])
                dy = float(row[col_map["dy"]])
                dz = float(row[col_map["dz"]])
            except (ValueError, IndexError):
                logger.debug("Skipping malformed CSV row %d: %s", row_idx, row)
                continue

            rx = ry = rz = 0.0
            if has_rot:
                try:
                    rx = float(row[col_map["rx"]])
                    ry = float(row[col_map["ry"]])
                    rz = float(row[col_map["rz"]])
                except (ValueError, IndexError):
                    pass

            timestamps.append(ts)
            offsets.append((dx, dy, dz))
            rotations.append((rx, ry, rz))

    if not timestamps:
        raise ValueError(f"No valid data rows found in CSV {path}")

    ts_arr = np.array(timestamps, dtype=np.int64)
    off_arr = np.array(offsets, dtype=np.float32)
    rot_arr = np.array(rotations, dtype=np.float32)
    has_rotations = bool(has_rot) and np.any(rot_arr != 0)

    logger.info("Loaded %d motion samples from %s", len(ts_arr), path.name)

    return MotionSamples(
        timestamps_ms=ts_arr,
        offsets_mm=off_arr,
        rotations_deg=rot_arr,
        has_rotations=has_rotations,
        source_path=str(path),
    )
