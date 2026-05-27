"""Command-line interface for the DeformCTMovement ensemble pipeline.

Usage::

    python -m gendosecalc.deform.cli \\
        --ct exampledata/ct_063 \\
        --motion exampledata/motion_063/MotionData_fraction01.xml \\
        --out runs/063_fx01/ \\
        --n-states 20 \\
        [--rtstruct RS.dcm] \
        [--deform-rtstruct RS.dcm] \\
        [--ctv-roi CTV_Prostate] \\
        [--falloff 25] \\
        [--bone-threshold 300] \\
        [--interpolation linear] \\
        [--gpu] \\
        [--seed 0] \\
        [--log-level INFO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gendosecalc.deform.models import DeformationConfig
from gendosecalc.deform.pipeline import generate_ensemble


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gendosecalc.deform.cli",
        description=(
            "Generate an ensemble of N deformed CT volumes driven by "
            "a Synchrony motion log or CSV."
        ),
    )

    # Required
    p.add_argument(
        "--ct", required=True, metavar="DIR",
        help="Directory containing the reference DICOM CT series.",
    )
    p.add_argument(
        "--motion", required=True, metavar="FILE",
        help="Synchrony MotionData.xml or generic CSV (timestamp_ms,dx,dy,dz[,rx,ry,rz]).",
    )
    p.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output root directory (created if absent).",
    )

    # Optional
    p.add_argument(
        "--rtstruct", default=None, metavar="FILE",
        help="RTSTRUCT DICOM for CTV localisation.  If omitted, no CTV falloff.",
    )
    p.add_argument(
        "--deform-rtstruct", default=None, metavar="FILE",
        dest="deform_rtstruct",
        help=(
            "RTSTRUCT DICOM to deform alongside each CT state.  "
            "Produces state_{i:03d}_rs.dcm per state in --out.  "
            "May be the same file as --rtstruct."
        ),
    )
    p.add_argument(
        "--ctv-roi", default=None, metavar="NAME",
        help=(
            "ROI name in the RTSTRUCT to use as CTV (case-insensitive). "
            "Can be supplied multiple times.  Defaults: CTV, CTV_prostate, "
            "CTV_Prostate, ctv."
        ),
        action="append",
        dest="ctv_roi",
    )
    p.add_argument(
        "--n-states", type=int, default=20, metavar="N",
        help="Number of representative motion states to generate (default: 20).",
    )
    p.add_argument(
        "--falloff", type=float, default=25.0, metavar="MM",
        help="CTV-edge falloff distance in mm (smoothstep, default: 25).",
    )
    p.add_argument(
        "--bone-threshold", type=float, default=300.0, metavar="HU",
        help="HU threshold for bone segmentation (default: 300).",
    )
    p.add_argument(
        "--transition-width", type=float, default=3.0, metavar="MM",
        help="Gaussian sigma for bone–tissue boundary (mm, default: 3).",
    )
    p.add_argument(
        "--interpolation", choices=["linear", "bspline", "nearest"],
        default="linear",
        help="CT resampling interpolation (default: linear).",
    )
    p.add_argument(
        "--rotation-weight", type=float, default=10.0, metavar="MM_PER_DEG",
        help=(
            "Scale factor (mm/deg) to weight rotations in k-medoids 6D "
            "feature space (default: 10)."
        ),
    )
    p.add_argument(
        "--tolerance-mode",
        choices=["warn", "drop", "keep"],
        default="warn",
        help="How to handle Synchrony out-of-tolerance data points (default: warn).",
    )
    p.add_argument(
        "--gpu", action="store_true", default=False,
        help="Enable GPU acceleration via CuPy (falls back to CPU if unavailable).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for k-medoids initialisation (default: 0).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    ctv_roi_names = args.ctv_roi if args.ctv_roi else None

    config = DeformationConfig(
        bone_threshold_hu=args.bone_threshold,
        transition_width_mm=args.transition_width,
        interpolation=args.interpolation,
        falloff_mm=args.falloff,
        ctv_roi_names=(
            ctv_roi_names
            if ctv_roi_names
            else ["CTV", "CTV_prostate", "CTV_Prostate", "ctv"]
        ),
        rotation_weight_mm_per_deg=args.rotation_weight,
        n_states=args.n_states,
        motion_tolerance_mode=args.tolerance_mode,
        use_gpu=args.gpu,
    )

    if args.gpu:
        try:
            import cupy  # noqa: F401
            logger.info("CuPy available — GPU acceleration enabled.")
        except ImportError:
            logger.warning(
                "--gpu requested but CuPy is not installed.  Falling back to CPU. "
                "Install with: pip install cupy-cuda12x"
            )
            config.use_gpu = False

    try:
        manifest = generate_ensemble(
            ct_dir=args.ct,
            motion_path=args.motion,
            out_dir=args.out,
            config=config,
            rtstruct_path=args.rtstruct,
            deform_rtstruct_path=args.deform_rtstruct,
            n_states=args.n_states,
            rotation_weight_mm_per_deg=args.rotation_weight,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1

    n = len(manifest.get("states", []))
    rs_msg = f", {n} deformed RTSTRUCTs" if args.deform_rtstruct else ""
    print(
        f"Done: {n} deformed CT states written to {args.out}  "
        f"(manifest.json, {n} DVF .mha files, {n} DICOM CT series{rs_msg})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
