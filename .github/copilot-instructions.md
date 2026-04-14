# DeformCTMovement — Copilot Workspace Instructions

## Project Identity

This is **DeformCTMovement**, a Python module for generating anatomically plausible deformed CT volumes for motion-resolved radiotherapy dose calculation. It is part of the **GenDoseCalc** ecosystem.

**Domain**: Medical physics — prostate radiotherapy, SBRT, Synchrony motion log processing.

---

## Architecture

```
gendosecalc/deform/
├── models.py           — DeformationField, DeformationConfig dataclasses
├── sitk_bridge.py      — CT ↔ SimpleITK, DVF ↔ SimpleITK vector image
├── dvf_io.py           — Load/save DVF (MHA, NIfTI)
├── dvf_generate.py     — DVF from 6DOF params or Synchrony motion log
├── bone_mask.py        — HU thresholding + Gaussian tissue weight map
└── apply_deformation.py — Core deformation with bone rigidity masking
```

**Key dependency**: `SimpleITK >= 2.3`. `scipy.ndimage` for bone mask smoothing (already present).

---

## Coding Conventions

- **Language**: Python 3.11+, typed with `typing` / PEP 604 union syntax (`X | None`)
- **Array layout**: `(3, nz, ny, nx)` for DVF vectors, `(nz, ny, nx)` for scalar volumes
- **Coordinate system**: LPS (left-posterior-superior) throughout — match SimpleITK conventions
- **SimpleITK mapping**: always use **inverse mapping** (output → input space); forward DVFs must be inverted via `sitk.InvertDisplacementField()` before passing to `Resample`
- **HU handling**: preserve int16 HU range after resampling; clamp to `[-1024, 3071]`
- **Units**: displacements in **mm**, spacing in **mm**, angles in **degrees** (converted to radians internally)
- **Bone threshold**: default 300 HU — configurable via `DeformationConfig`
- **No patient data in tests**: all tests use synthetic phantoms only

---

## Testing

- All tests live in `tests/deform/`
- Fixtures are in `tests/conftest.py` — prefer reusing existing synthetic CT/DVF fixtures over creating new files
- `pytest` with no arguments should pass cleanly before any commit

---

## Common Pitfalls

1. **Direction cosine handedness** — SimpleITK uses a row-major direction matrix; verify with `image.GetDirection()` before and after conversion
2. **DVF convention** — forward vs. inverse mapping; always document which convention a function accepts/returns
3. **Bone mask sigma** — `transition_width_mm` is in mm; divide by voxel spacing before passing to `gaussian_filter`
4. **HU clamp** — Resample with B-spline can produce values slightly outside int16 range; always clamp before returning

---

## Current Implementation Status

See `plan_deform.md` for the complete specification and phase breakdown.

- Phase 1 (DVF infrastructure): **not started**
- Phase 2 (Bone masking + apply): **not started**
- Phase 3 (DVF generation): **not started**
- Phase 4 (Pipeline integration): **not started**
- Phase 5 (Validation): **not started**
