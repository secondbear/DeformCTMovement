# Deformable CT Generation — Project Plan

## Goal

Add the ability to generate deformed CT volumes from a reference CT plus a movement/displacement specification. The deformation must be **anatomically plausible**: soft tissue around the prostate deforms elastically, while pelvic bone remains rigid. This enables motion-resolved dose recalculation on deformed anatomy rather than simple isocenter shifts.

---

## 1. Recommended Library

### Primary: **SimpleITK** (`pip install SimpleITK`)

| Criterion | Assessment |
|-----------|------------|
| Apply known DVF to CT | First-class: `DisplacementFieldTransform` + `Resample` |
| DICOM support | Excellent — reads/writes series, preserves metadata, LPS coords |
| Physical coordinate handling | Native — origin, spacing, direction cosines |
| Bone rigidity | Not built-in, but trivial to implement via DVF masking (see §4) |
| Install | `pip install SimpleITK` (pure wheel, no build tools needed) |
| License | Apache 2.0 |
| Maintenance | Very active — v2.5.3 (Nov 2025), 9,600+ commits |

**Why SimpleITK over alternatives:**

- **vs. ANTsPy** — ANTs excels at *computing* registrations (SyN) but has weak DICOM support and no built-in rigidity penalty. Our primary need is *applying* known deformations, not computing them from image pairs.
- **vs. ITKElastix** — Elastix's `TransformRigidityPenalty` is designed for the registration optimisation loop, not for post-hoc DVF modification. If we later need CT-to-CBCT registration, we add ITKElastix as a complement.
- **vs. VoxelMorph / TorchIO** — Deep-learning frameworks for augmentation/learning-based registration. Wrong paradigm; adds PyTorch dependency for no benefit.
- **vs. scipy.ndimage alone** — `map_coordinates` works but requires manual handling of all spatial metadata (origin, spacing, direction cosines, LPS↔voxel). SimpleITK wraps this correctly.

### Complement: **scipy.ndimage**

Already a project dependency. `gaussian_filter` for bone-mask smoothing, `map_coordinates` as a fallback for pure-numpy paths in tight engine loops.

### Future addition (if needed): **ITKElastix** (`pip install itk-elastix`)

For computing DVFs from image pairs (e.g., planning CT → daily CBCT) with bone rigidity penalty. Not needed for Phase 1.

---

## 2. Background: How This Fits GenDoseCalc

The current pipeline applies motion as a **rigid isocenter shift** per projection (from Synchrony motion logs). The CT itself is never modified. This is accurate for small translations but ignores:

- Soft-tissue deformation from bladder/rectal filling changes
- Rotational components of prostate motion
- Dose perturbation from changed radiological path through deformed anatomy

A deformed CT allows the existing pencil-beam engine to see the *actual* anatomy each beam traverses, improving dose accuracy for large-motion fractions.

---

## 3. Deformation Vector Field (DVF) — Concepts & Formats

A **DVF** is a 3D vector image where each voxel stores a displacement `(dx, dy, dz)` in mm. It maps every point in the reference CT to its deformed position:

```
deformed_position = original_position + displacement_vector
```

### Mapping convention

SimpleITK `Resample` uses **inverse mapping** (output → input). A forward DVF must be inverted before use. SimpleITK provides `InvertDisplacementField()` for this.

### Storage formats

| Format | Extension | Use case |
|--------|-----------|----------|
| MetaImage | `.mha` | SimpleITK/ITK native; recommended for project internal use |
| NIfTI | `.nii.gz` | Research interchange; ANTs, FSL, 3D Slicer |
| NRRD | `.nrrd` | 3D Slicer, Plastimatch |
| DICOM DSRO | `.dcm` | Clinical TPS import (Eclipse, RayStation) |

**Project default**: `.mha` for internal DVFs (lossless, single-file, includes full spatial metadata). NIfTI as optional export.

### Movement file input

The module should accept displacement specifications in multiple forms:

1. **Pre-computed DVF file** (`.mha`, `.nii.gz`) — loaded directly
2. **Rigid transform parameters** (tx, ty, tz, rx, ry, rz) — converted to a uniform DVF
3. **Synchrony motion log entry** — 6DOF extracted from existing `MotionData.xml` parser, converted to DVF centred on prostate region

---

## 4. Tissue-Dependent Deformation Strategy

### Bone rigidity via DVF masking

Rather than full FEM biomechanical modelling (which requires mesh generation and material property assignment), we use a pragmatic **post-hoc DVF masking** approach:

```
1. Segment bone from CT:       bone_mask = (ct_hu > 300)
2. Smooth mask boundary:        bone_smooth = gaussian_filter(bone_mask, sigma=transition_mm / spacing)
3. Compute attenuation:         weight = 1.0 − bone_smooth     # 0 in bone, 1 in soft tissue
4. Mask DVF:                    dvf *= weight                   # zero displacement in bone
5. Apply masked DVF to CT:      deformed_ct = resample(ct, masked_dvf)
```

**Parameters:**
- `bone_threshold_hu`: HU threshold for bone segmentation (default: 300 HU)
- `transition_width_mm`: Gaussian sigma for smooth bone→tissue boundary (default: 3.0 mm)

**Why this works for our case:**
- Pelvic bone does not move between fractions — it is the rigid reference frame
- The prostate displacement is a known input (from motion logs or external DVF)
- The smooth transition prevents discontinuities at bone–soft-tissue interfaces
- The approach is deterministic, fast, and transparent

**Limitations (documented, not blocking):**
- No biomechanical fidelity at the tissue interface (acceptable for dose perturbation studies)
- Assumes bone is correctly identified by HU thresholding (valid for pelvic CT without metal implants)

---

## 5. Data Model

### New dataclass: `DeformationField`

```python
@dataclass
class DeformationField:
    """3D displacement vector field in LPS coordinates."""
    vectors: np.ndarray          # (3, nz, ny, nx), float32, displacement in mm (dz, dy, dx)
    spacing_mm: np.ndarray       # (3,) — (dz, dy, dx)
    origin_mm: np.ndarray        # (3,) — LPS mm
    direction: np.ndarray        # (3, 3)
    source_description: str      # e.g. "rigid_shift_tx5_ty-2_tz1" or "dvf_file.mha"
```

### New dataclass: `DeformationConfig`

```python
@dataclass
class DeformationConfig:
    """Parameters controlling CT deformation behaviour."""
    bone_threshold_hu: float = 300.0
    transition_width_mm: float = 3.0
    interpolation: str = "linear"    # "linear", "bspline", "nearest"
    preserve_hounsfield: bool = True  # remap to int16 HU after resampling
```

---

## 6. Module Design

### New package: `gendosecalc/deform/`

```
gendosecalc/deform/
    __init__.py
    dvf_io.py           — Load/save DVF files (MHA, NIfTI)
    dvf_generate.py     — Generate DVF from rigid params or motion log entry
    bone_mask.py        — HU-based bone segmentation + smoothing
    apply_deformation.py — Core: mask DVF for bone rigidity, apply to CT
    sitk_bridge.py      — Convert between project CT/DVF dataclasses and SimpleITK images
```

### Key functions

```python
# dvf_io.py
def load_dvf(path: Path) -> DeformationField: ...
def save_dvf(dvf: DeformationField, path: Path) -> None: ...

# dvf_generate.py
def rigid_to_dvf(
    tx: float, ty: float, tz: float,
    rx: float, ry: float, rz: float,
    reference_ct: CT,
    centre_of_rotation_mm: np.ndarray | None = None,
) -> DeformationField: ...

def motion_log_entry_to_dvf(
    motion_entry: dict,          # single row from MotionData.xml
    reference_ct: CT,
) -> DeformationField: ...

# bone_mask.py
def compute_bone_mask(ct: CT, config: DeformationConfig) -> np.ndarray: ...
def compute_tissue_weight(bone_mask: np.ndarray, ct: CT, config: DeformationConfig) -> np.ndarray: ...

# apply_deformation.py
def deform_ct(
    ct: CT,
    dvf: DeformationField,
    config: DeformationConfig | None = None,
) -> CT: ...

# sitk_bridge.py
def ct_to_sitk(ct: CT) -> sitk.Image: ...
def sitk_to_ct(image: sitk.Image, source_ct: CT) -> CT: ...
def dvf_to_sitk(dvf: DeformationField) -> sitk.Image: ...
def sitk_to_dvf(image: sitk.Image) -> DeformationField: ...
```

### Integration point

The existing `compute_motion_dose()` in `plan/motion_dose.py` currently shifts the isocenter per projection. The deformed-CT path would instead:

1. For each motion state (or fraction), generate a deformed CT
2. Pass the deformed CT through the existing `compute_static_dose()` pipeline
3. Compare deformed-anatomy dose vs. planned dose

This keeps the pencil-beam engine unchanged — deformation is a preprocessing step.

---

## 7. Implementation Phases

### Phase 1 — Core DVF infrastructure (foundation)

**Files:** `deform/__init__.py`, `sitk_bridge.py`, `dvf_io.py`, `models.py`

- [ ] Add `SimpleITK >= 2.3` to `pyproject.toml` dependencies
- [ ] Add `DeformationField` and `DeformationConfig` to `models.py`
- [ ] Implement `sitk_bridge.py` — bidirectional conversion CT ↔ SimpleITK Image, DVF ↔ SimpleITK vector image
- [ ] Implement `dvf_io.py` — load/save DVF in MHA and NIfTI formats via SimpleITK
- [ ] Tests: round-trip CT conversion, DVF I/O with known test field

### Phase 2 — Bone masking and DVF application

**Files:** `bone_mask.py`, `apply_deformation.py`

- [ ] Implement `compute_bone_mask()` — HU thresholding with configurable threshold
- [ ] Implement `compute_tissue_weight()` — Gaussian-smoothed attenuation map
- [ ] Implement `deform_ct()` — mask DVF, apply via SimpleITK `DisplacementFieldTransform` + `Resample`, preserve HU int16
- [ ] Tests: deform water phantom (expect no bone masking effect), deform phantom with bone insert (verify bone voxels unchanged), verify HU preservation

### Phase 3 — DVF generation from motion parameters

**Files:** `dvf_generate.py`

- [ ] Implement `rigid_to_dvf()` — create a DVF from 6DOF rigid parameters (translation + Euler rotation about a configurable centre of rotation)
- [ ] Implement `motion_log_entry_to_dvf()` — extract 6DOF from existing Synchrony motion log data and delegate to `rigid_to_dvf()`
- [ ] Tests: identity transform produces zero DVF, pure translation DVF matches expected values, rotation about prostate centre produces expected displacement pattern

### Phase 4 — Pipeline integration

**Files:** `scripts/run_pipeline.py`, `plan/motion_dose.py` (optional)

- [ ] Add `--deform-ct` flag to `run_pipeline.py` that activates deformed-CT dose calculation
- [ ] Workflow: load CT → per motion state, generate DVF → deform CT → compute dose on deformed CT → compare with static dose
- [ ] Export deformed CTs as DICOM series (extend `dicom_ct.py` with a `save_ct_series()` function) for visual verification in TPS
- [ ] Integration test: full pipeline with example data

### Phase 5 — Validation and refinement

- [ ] Visual validation: overlay deformed CT on original in TPS, verify bone stability and soft-tissue plausibility
- [ ] Dose validation: compare deformed-CT dose vs. isocenter-shift dose for known motion cases
- [ ] Parameter sensitivity study: vary `bone_threshold_hu` and `transition_width_mm`, document effect on dose
- [ ] Document the module in `docs/deformation.md`

---

## 8. Testing Strategy

| Test | What it verifies |
|------|------------------|
| `test_sitk_bridge.py` | CT → SimpleITK → CT round-trip preserves HU, spacing, origin, direction |
| `test_dvf_io.py` | DVF save/load round-trip (MHA, NIfTI); loaded DVF matches saved vectors |
| `test_bone_mask.py` | Bone mask correct for known HU distribution; weight map smooth at boundary |
| `test_apply_deformation.py` | Zero DVF returns identical CT; uniform translation DVF shifts CT correctly; bone voxels remain unchanged after masked deformation |
| `test_dvf_generate.py` | Identity params → zero DVF; known translation → expected DVF; rotation produces expected displacement near/far from centre |
| `test_deform_pipeline.py` | End-to-end: CT + motion params → deformed CT → dose; smoke test for no crashes |

All tests use synthetic phantoms (existing `conftest.py` fixtures or new minimal ones). No real patient data in tests.

---

## 9. Dependencies Change

```toml
# pyproject.toml — add to [project.dependencies]
"SimpleITK >= 2.3",
```

No other new dependencies required. `scipy` (already present) provides `gaussian_filter`.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SimpleITK inverse-mapping convention causes DVF direction errors | Wrong deformation direction | Unit test with known translation; visual check in 3D Slicer |
| LPS/voxel coordinate mismatch between project CT and SimpleITK | Shifted/flipped deformed CT | Round-trip test in `test_sitk_bridge.py`; compare origin/spacing/direction |
| Bone mask too aggressive (clips soft tissue near bone) | Artificially rigid soft tissue near pelvis | Configurable threshold + smooth transition; sensitivity test in Phase 5 |
| HU interpolation artefacts at tissue boundaries after deformation | Incorrect density → dose error | Use linear interpolation (not nearest-neighbour); clamp HU to valid range |
| Large rotational DVFs cause volume boundary artefacts | Missing data at CT edges after deformation | Pad CT before deformation; document field-of-view requirements |

---

## 11. Future Extensions (out of scope for now)

- **ITKElastix-based DIR**: compute DVFs from CT-to-CBCT registration with bone rigidity penalty
- **4DCT phase deformation**: chain DVFs across respiratory phases
- **Structure propagation**: deform RTSTRUCT contours alongside CT (requires nearest-neighbour interpolation for label maps)
- **Biomechanical FEM**: replace DVF masking with finite-element tissue simulation (FEniCS or FEBio) for higher fidelity
- **Population motion models**: PCA-based statistical prostate motion model for Monte Carlo robustness studies
