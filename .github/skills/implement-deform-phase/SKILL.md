---
name: implement-deform-phase
description: "Use when: implementing a DeformCTMovement phase (Phase 1 DVF infrastructure, Phase 2 bone masking, Phase 3 DVF generation, Phase 4 pipeline integration, Phase 5 validation). Provides the correct workflow for each phase including file targets, test strategy, and common pitfalls."
---

# Implement DeformCTMovement Phase

## Before You Start

1. Read `plan_deform.md` sections §5 (Data Model), §6 (Module Design), and the relevant phase in §7
2. Check `AGENTS.md` for the latest status of all phases
3. Run `pytest tests/deform/ -v` to confirm the baseline is clean

## Phase 1 — Core DVF Infrastructure

**Files**: `gendosecalc/deform/__init__.py`, `models.py`, `sitk_bridge.py`, `dvf_io.py`

Steps:
1. Add `SimpleITK >= 2.3` to `pyproject.toml`
2. Create `models.py` with `DeformationField` and `DeformationConfig` dataclasses (exact signatures from §5)
3. Implement `sitk_bridge.py` — use `sitk.GetImageFromArray` / `sitk.GetArrayFromImage`; always copy spacing, origin, direction
4. Implement `dvf_io.py` — use `sitk.ReadImage` / `sitk.WriteImage`; validate vector components == 3
5. Write round-trip tests in `tests/deform/test_sitk_bridge.py` and `test_dvf_io.py`

**Critical**: The DVF SimpleITK image must be a vector image with `sitk.sitkVectorFloat64`. Set direction from `DeformationField.direction` using `SetDirection(dvf.direction.flatten().tolist())`.

## Phase 2 — Bone Masking and DVF Application

**Files**: `bone_mask.py`, `apply_deformation.py`

Steps:
1. `compute_bone_mask`: `ct.array > config.bone_threshold_hu` → bool array `(nz, ny, nx)`
2. `compute_tissue_weight`: `gaussian_filter(mask.astype(float32), sigma=config.transition_width_mm / ct.spacing_mm)` → invert: `1.0 - smoothed`
3. `deform_ct`:
   - Multiply DVF vectors by weight map (broadcast over 3 components)
   - If DVF convention is forward, invert with `sitk.InvertDisplacementField()`
   - Apply via `sitk.DisplacementFieldTransform` → `sitk.Resample`
   - Clamp result to `[-1024, 3071]` and cast to `int16`

**Critical sigma calculation**: `sigma = transition_width_mm / spacing_mm` — this is per-axis. `gaussian_filter` accepts a sequence.

## Phase 3 — DVF Generation

**Files**: `dvf_generate.py`

Steps:
1. `rigid_to_dvf`: Build 4×4 homogeneous transform from (tx,ty,tz) + Euler rotations (rz·ry·rx convention); evaluate at each voxel centre in LPS; subtract original position → displacement
2. `motion_log_entry_to_dvf`: Extract keys `SI`, `LR`, `AP`, `Pitch`, `Roll`, `Yaw` from dict → map to (tx,ty,tz,rx,ry,rz) → delegate to `rigid_to_dvf`
3. Centre of rotation defaults to image centre in LPS mm

## Phase 4 — Pipeline Integration

**Files**: `scripts/run_pipeline.py`, `plan/motion_dose.py`

- Add argparse flag `--deform-ct` (boolean, default False)
- When enabled: for each motion state, call `rigid_to_dvf` → `deform_ct` → pass to `compute_static_dose`
- DICOM export: extend `dicom_ct.py` with `save_ct_series(ct, output_dir, series_uid)`

## Testing Rules

- All fixtures in `tests/conftest.py` — reuse before creating new ones
- Synthetic phantoms only — no real patient data
- For bone masking tests: create a 3D array with a bone-HU region (>300) and verify those voxels are unchanged after deformation
- For DVF generation tests: identity params must produce a DVF with `np.allclose(dvf.vectors, 0, atol=1e-5)`
