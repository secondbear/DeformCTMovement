"""Deformable CT generation for motion-resolved radiotherapy dose calculation."""

from gendosecalc.deform.models import (
    DeformationConfig,
    DeformationField,
    MotionSamples,
    StateSelection,
    EnsembleManifestEntry,
)
from gendosecalc.deform.dvf_io import load_dvf, save_dvf
from gendosecalc.deform.sitk_bridge import (
    ct_to_sitk,
    sitk_to_ct,
    dvf_to_sitk,
    sitk_to_dvf,
)
from gendosecalc.deform.bone_mask import compute_bone_mask, compute_tissue_weight
from gendosecalc.deform.apply_deformation import deform_ct
from gendosecalc.deform.dvf_generate import (
    rigid_to_dvf,
    motion_log_entry_to_dvf,
    localized_rigid_to_dvf,
)
from gendosecalc.deform.motion_io import load_synchrony_xml, load_motion_csv
from gendosecalc.deform.motion_select import select_representative_states
from gendosecalc.deform.structures import load_ctv_mask
from gendosecalc.deform.rtstruct_deform import deform_rtstruct
from gendosecalc.deform.pipeline import generate_ensemble

__all__ = [
    # Models
    "DeformationConfig",
    "DeformationField",
    "MotionSamples",
    "StateSelection",
    "EnsembleManifestEntry",
    # DVF I/O
    "load_dvf",
    "save_dvf",
    # SimpleITK bridge
    "ct_to_sitk",
    "sitk_to_ct",
    "dvf_to_sitk",
    "sitk_to_dvf",
    # Bone mask
    "compute_bone_mask",
    "compute_tissue_weight",
    # Deformation
    "deform_ct",
    # DVF generation
    "rigid_to_dvf",
    "motion_log_entry_to_dvf",
    "localized_rigid_to_dvf",
    # Motion ingestion
    "load_synchrony_xml",
    "load_motion_csv",
    # State selection
    "select_representative_states",
    # Structures
    "load_ctv_mask",
    # RTSTRUCT deformation
    "deform_rtstruct",
    # Pipeline
    "generate_ensemble",
]

