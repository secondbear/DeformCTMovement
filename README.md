# DeformCTMovement

> Anatomically plausible deformable CT generation for motion-resolved radiotherapy dose calculation.

Part of the **GenDoseCalc** ecosystem. Applies a displacement vector field (DVF) to a reference planning CT, preserving pelvic bone rigidity while elastically deforming surrounding soft tissue — enabling per-fraction dose recalculation on realistic anatomy rather than simple isocenter shifts.

---

## Motivation

Current pencil-beam pipelines apply prostate motion as a rigid isocenter shift per projection (from Synchrony motion logs). This ignores:

- Bladder/rectal filling-driven soft-tissue deformation
- Rotational components of prostate motion  
- Dose perturbation from changed radiological path through deformed anatomy

A deformed CT lets the existing dose engine see the *actual* anatomy each beam traverses.

---

## Features

| Feature | Status |
|---------|--------|
| Load/save DVF (MHA, NIfTI) | Phase 1 |
| Convert CT ↔ SimpleITK Image (preserve HU, LPS coords) | Phase 1 |
| HU-based bone segmentation with Gaussian-smoothed boundary | Phase 2 |
| Apply DVF with bone rigidity masking | Phase 2 |
| Generate DVF from 6DOF rigid parameters | Phase 3 |
| Generate DVF from Synchrony motion log entry | Phase 3 |
| Pipeline flag `--deform-ct` in `run_pipeline.py` | Phase 4 |
| DICOM export of deformed CT series | Phase 4 |

---

## Architecture

```
gendosecalc/deform/
├── __init__.py
├── models.py           — DeformationField, DeformationConfig dataclasses
├── sitk_bridge.py      — CT ↔ SimpleITK, DVF ↔ SimpleITK vector image
├── dvf_io.py           — Load/save DVF files (MHA, NIfTI) via SimpleITK
├── dvf_generate.py     — DVF from 6DOF params or Synchrony motion log entry
├── bone_mask.py        — HU thresholding + Gaussian-smoothed tissue weight map
└── apply_deformation.py — Core: mask DVF for bone rigidity, resample CT
```

### Bone rigidity strategy

DVF masking via HU-threshold segmentation — fast, deterministic, no mesh required:

```
bone_mask   = ct_hu > 300
bone_smooth = gaussian_filter(bone_mask, σ = transition_mm / spacing)
weight      = 1.0 − bone_smooth        # 0 in bone, 1 in soft tissue
masked_dvf  = dvf * weight
deformed_ct = SimpleITK.Resample(ct, DisplacementFieldTransform(masked_dvf))
```

---

## Installation

```bash
pip install SimpleITK>=2.3
# scipy is already a project dependency
```

Or add to `pyproject.toml`:

```toml
[project.dependencies]
"SimpleITK >= 2.3",
```

---

## Quick Start

```python
from pathlib import Path
from gendosecalc.deform import deform_ct, load_dvf, rigid_to_dvf
from gendosecalc.deform.models import DeformationConfig

# Option A: apply a pre-computed DVF file
dvf = load_dvf(Path("motion_state_01.mha"))
config = DeformationConfig(bone_threshold_hu=300, transition_width_mm=3.0)
deformed = deform_ct(ct, dvf, config)

# Option B: generate DVF from known 6DOF shift
dvf = rigid_to_dvf(tx=5.0, ty=-2.0, tz=1.0, rx=0, ry=0, rz=0, reference_ct=ct)
deformed = deform_ct(ct, dvf)
```

---

## Data Models

### `DeformationField`

```python
@dataclass
class DeformationField:
    vectors: np.ndarray       # (3, nz, ny, nx) float32, displacement in mm (dz, dy, dx)
    spacing_mm: np.ndarray    # (3,) — (dz, dy, dx)
    origin_mm: np.ndarray     # (3,) — LPS mm
    direction: np.ndarray     # (3, 3)
    source_description: str   # e.g. "rigid_shift_tx5_ty-2_tz1"
```

### `DeformationConfig`

```python
@dataclass
class DeformationConfig:
    bone_threshold_hu: float = 300.0
    transition_width_mm: float = 3.0
    interpolation: str = "linear"   # "linear" | "bspline" | "nearest"
    preserve_hounsfield: bool = True
```

---

## DVF Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MetaImage | `.mha` | Project default — lossless, single file, full spatial metadata |
| NIfTI | `.nii.gz` | Research interchange; 3D Slicer, ANTs, FSL |
| NRRD | `.nrrd` | 3D Slicer, Plastimatch |

**Mapping convention**: SimpleITK `Resample` uses inverse mapping (output → input). Forward DVFs are inverted automatically via `InvertDisplacementField()`.

---

## Testing

All tests use synthetic phantoms — no patient data.

```bash
pytest tests/deform/ -v
```

| Test file | Covers |
|-----------|--------|
| `test_sitk_bridge.py` | CT → SimpleITK → CT round-trip (HU, spacing, origin, direction) |
| `test_dvf_io.py` | DVF save/load round-trip (MHA, NIfTI) |
| `test_bone_mask.py` | Bone mask correctness; weight map smooth at boundary |
| `test_apply_deformation.py` | Zero DVF → identical CT; translation DVF; bone voxels unchanged |
| `test_dvf_generate.py` | Identity params → zero DVF; known translation; rotation pattern |
| `test_deform_pipeline.py` | End-to-end smoke test |

---

## Implementation Phases

- **Phase 1** — Core DVF infrastructure (models, sitk_bridge, dvf_io)
- **Phase 2** — Bone masking and DVF application
- **Phase 3** — DVF generation from motion parameters
- **Phase 4** — Pipeline integration (`--deform-ct` flag, DICOM export)
- **Phase 5** — Validation and refinement

See [plan_deform.md](plan_deform.md) for full specification.

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| SimpleITK inverse-mapping direction errors | Unit test with known translation; visual check in 3D Slicer |
| LPS/voxel coordinate mismatch | Round-trip test in `test_sitk_bridge.py` |
| Bone mask too aggressive near pelvis | Configurable threshold + Gaussian smoothing |
| HU interpolation artefacts at boundaries | Linear interpolation; clamp HU to int16 range |

---

## License

Apache 2.0 — same as SimpleITK. See [LICENSE](LICENSE).
