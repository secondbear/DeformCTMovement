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

## RTSTRUCT ↔ CT Coordinate System

### Coordinate spaces

| Space | Order | Source |
|-------|-------|--------|
| DICOM LPS | `(x, y, z)` | ContourData, ImagePositionPatient, DVF displacement output |
| Project voxel | `(iz, iy, ix)` = `(z, y, x)` | NumPy CT array `(nz, ny, nx)`, DVF array `(3, nz, ny, nx)` |

`load_ct_series()` returns `spacing_mm=(sz,sy,sx)`, `origin_mm=(oz,oy,ox)`, and a `direction` matrix where **each row is a direction-cosine unit vector expressed in DICOM LPS (x,y,z)**:

```python
direction = np.stack([slice_cos, col_cos, row_cos], axis=0)  # (3,3)
# row 0: LPS unit vector along increasing iz (slice direction)
# row 1: LPS unit vector along increasing iy (row direction)
# row 2: LPS unit vector along increasing ix (column direction)
```

For a standard axial HFS scan `ImageOrientationPatient=[1,0,0, 0,1,0]` this gives:
```
direction = [[0, 0, 1],   # slice → LPS +z
             [0, 1, 0],   # row   → LPS +y
             [1, 0, 0]]   # col   → LPS +x
```

### Forward transform (voxel → LPS mm)

```python
# physical_lps (x,y,z) = origin_lps + direction.T @ diag(spacing_zyx) @ [iz, iy, ix]
origin_lps = origin_mm[::-1]                           # (oz,oy,ox) → (ox,oy,oz)
displacement_lps = direction.T @ np.diag(spacing_mm) @ np.array([iz, iy, ix])
physical_lps = origin_lps + displacement_lps
```

### Inverse transform (LPS mm → voxel)

```python
# [iz, iy, ix] = diag(1/spacing) @ direction @ (physical_lps - origin_lps)
origin_lps = origin_mm[::-1]                           # (oz,oy,ox) → (ox,oy,oz)
M_inv = np.diag(1.0 / spacing_mm) @ direction         # (3,3)
rel   = pts_xyz - origin_lps                           # (N,3) in LPS (x,y,z)
vox   = (M_inv @ rel.T).T                              # (N,3) → (iz, iy, ix)
```

**Critical**: do NOT convert `pts_xyz` to `(z,y,x)` before applying `M_inv`. The direction rows are already in LPS `(x,y,z)`, so the dot product must operate in LPS space.

### Heatmap pixel conventions (`_get_slice` + Plotly Heatmap)

Plotly Heatmap renders `z[i][j]` at `y=i, x=j` with y increasing upward (row 0 at bottom). `_get_slice` pre-flips rows so superior anatomy appears at the top of the viewport:

| Orientation | `_get_slice` output | heatmap col | heatmap row |
|-------------|---------------------|-------------|-------------|
| Axial (0)   | `vol[iz, ::-1, :]`  | `ix`        | `ny-1-iy`   |
| Coronal (1) | `vol[::-1, iy, :]`  | `ix`        | `nz-1-iz`   |
| Sagittal (2)| `vol[::-1, :, ix]`  | `iy`        | `nz-1-iz`   |

Scatter overlay coordinates must match: use the same `col` / `row` formula when projecting ROI contour voxel indices onto the heatmap axes.

---

## Common Pitfalls

1. **Direction cosine handedness** — direction rows are in LPS `(x,y,z)`, not project `(z,y,x)`. The inverse transform is `diag(1/sp) @ direction @ rel_xyz`, **not** `inv(direction @ diag(sp)) @ rel_zyx` — the latter swaps iz and ix for standard axial CTs.
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
