# How to Use DeformCTMovement

End-to-end guide from motion data to deformed CT ensemble.

---

## Contents

1. [What the pipeline does](#1-what-the-pipeline-does)
2. [Input files](#2-input-files)
3. [Motion CSV format](#3-motion-csv-format)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Output structure](#5-output-structure)
6. [Viewing results](#6-viewing-results)
7. [Bone rigidity masking](#7-bone-rigidity-masking)
8. [State selection (k-medoids)](#8-state-selection-k-medoids)
9. [Python API](#9-python-api)

---

## 1. What the pipeline does

Given a reference CT and a list of rigid-body motion samples it:

1. Selects **N representative states** from all samples (k-medoids clustering).
2. For each state, builds a **Displacement Vector Field (DVF)** that translates/rotates soft tissue while keeping bone rigid.
3. Applies the DVF to produce a **deformed DICOM CT series** per state.
4. Optionally deforms the **RTSTRUCT contours** alongside the CT.
5. Saves everything (DVFs, CTs, RTSTRUCTs) and a **manifest JSON** in the output directory.

---

## 2. Input files

| Input | Flag | Required | Notes |
|-------|------|----------|-------|
| Reference CT directory | `--ct` | Yes | DICOM series directory (`.dcm` files) |
| Motion file | `--motion` | Yes | CSV or Synchrony XML — see §3 |
| Output directory | `--out` | Yes | Created if absent |
| CTV RTSTRUCT | `--rtstruct` | Recommended | Used to localise the deformation falloff around the CTV |
| ROI name(s) to use as CTV | `--ctv-roi` | If `--rtstruct` given | Default names tried: `CTV`, `CTV_prostate`, `CTV_Prostate`, `ctv` |
| RTSTRUCT to deform | `--deform-rtstruct` | No | Contours displaced per-state; may be same file as `--rtstruct` |
| Number of states | `--n-states` | No | Default 20; capped to total samples |

---

## 3. Motion CSV format

The simplest input is a plain CSV file:

```
timestamp_ms,dx,dy,dz
0,5.0,0.0,0.0
1000,0.0,-5.0,0.0
2000,0.0,0.0,5.0
```

### Columns

| Column | Alias names accepted | Units | LPS axis |
|--------|---------------------|-------|----------|
| `timestamp_ms` | `timestamp`, `ts`, `ts_ms`, `time_ms` | ms | — |
| `dx` | `x`, `lx`, `d_lr`, `lr` | mm | x = Left–Right (positive = Left) |
| `dy` | `y`, `ay`, `d_ap`, `ap` | mm | y = Anterior–Posterior (positive = Posterior) |
| `dz` | `z`, `sz`, `d_si`, `si` | mm | z = Superior–Inferior (positive = Superior) |
| `rx` | `rot_x`, `roll` | deg | Rotation about LPS x |
| `ry` | `rot_y`, `pitch` | deg | Rotation about LPS y |
| `rz` | `rot_z`, `yaw` | deg | Rotation about LPS z |

Rotation columns are optional — omitting them sets all rotations to zero. Timestamps are only used for cluster ordering; they don't need to be real wall-clock times.

### Common direction signs (LPS convention)

| Intended motion | Column | Sign |
|----------------|--------|------|
| Left | `dx` | +5 |
| Right | `dx` | −5 |
| Anterior | `dy` | −5 |
| Posterior | `dy` | +5 |
| Superior (cranial) | `dz` | +5 |
| Inferior (caudal) | `dz` | −5 |

### Synchrony XML

Pass the Synchrony `MotionData_fraction*.xml` file directly — the format is auto-detected:

```bash
--motion exampledata/motion_063/MotionData_fraction01.xml
```

---

## 4. Running the pipeline

### Minimal run (translations only, no RTSTRUCT)

```bash
.venv/bin/python -m gendosecalc.deform.cli \
  --ct  path/to/ct_series/ \
  --motion  motion.csv \
  --out  runs/my_run/
```

### Full run with RTSTRUCT deformation

```bash
.venv/bin/python -m gendosecalc.deform.cli \
  --ct  exampledata/ct_rs_063 \
  --motion  exampledata/mock_motion_3states.csv \
  --out  runs/mock_3states \
  --n-states 3 \
  --rtstruct  exampledata/ct_rs_063/RS.*.dcm \
  --deform-rtstruct  exampledata/ct_rs_063/RS.*.dcm \
  --ctv-roi  "CTVT_42.7"
```

**`--rtstruct` vs `--deform-rtstruct`:**

- `--rtstruct` — used internally to locate the CTV and compute the spatial falloff of the deformation (soft tissue near the CTV moves with the prostate; structures far away stay put). Not saved to output.
- `--deform-rtstruct` — this RTSTRUCT file's contour points are displaced per-state and saved as `state_NNN_rs.dcm`. It may be the same file as `--rtstruct`.

**`--ctv-roi`:**

Specifies which ROI name in the RTSTRUCT to use as the target structure (CTV). The default names tried are `CTV`, `CTV_prostate`, `CTV_Prostate`, `ctv`. If none of those exist, pass the actual name explicitly, e.g. `--ctv-roi "CTVT_42.7"`. Can be supplied multiple times to try several names.

If no `--rtstruct` is given at all, the deformation falloff is applied uniformly to the whole volume (no spatial localisation).

### All CLI options

```
--ct          DIR    Reference CT directory
--motion      FILE   CSV or Synchrony XML
--out         DIR    Output root
--rtstruct    FILE   RTSTRUCT for CTV localisation
--deform-rtstruct FILE  RTSTRUCT to deform per state
--ctv-roi     NAME   ROI name for CTV (repeatable)
--n-states    N      Number of representative states (default 20)
--falloff     MM     Deformation falloff distance in mm (default 25)
--bone-threshold HU  HU value separating bone from soft tissue (default 300)
--interpolation     linear|bspline|nearest (default linear)
--seed        INT    Random seed for k-medoids (default 0)
--log-level   LEVEL  DEBUG|INFO|WARNING (default INFO)
```

---

## 5. Output structure

```
runs/my_run/
├── manifest.json          — full provenance: config, states, UIDs
├── state_000_ct/          — deformed DICOM CT series (state 0)
│   ├── CT.*.dcm
│   └── ...
├── state_000_dvf.mha      — forward DVF for state 0 (3-component, float32, mm)
├── state_000_rs.dcm       — deformed RTSTRUCT for state 0 (if requested)
├── state_001_ct/
├── state_001_dvf.mha
├── state_001_rs.dcm
└── ...
```

### manifest.json

Records everything needed to reproduce or consume the ensemble:

```json
{
  "source_ct_dir": "...",
  "motion_path": "...",
  "n_states": 3,
  "config": { "bone_threshold_hu": 300, "falloff_mm": 25, ... },
  "states": [
    {
      "state_index": 0,
      "tx_mm": 5.0, "ty_mm": 0.0, "tz_mm": 0.0,
      "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0,
      "ct_dir": "state_000_ct",
      "dvf_path": "state_000_dvf.mha",
      "rtstruct_path": "state_000_rs.dcm",
      "deformed_series_instance_uid": "..."
    },
    ...
  ]
}
```

### DVF convention

DVFs are saved as **forward** fields: each voxel stores the displacement from its original to its deformed position. The `deform_ct()` function inverts them internally before applying (SimpleITK uses inverse mapping).

---

## 6. Viewing results

Pass the manifest to the viewer to compare original and all deformed states side-by-side:

```bash
.venv/bin/python -m viewer.app \
  --orig-ct  exampledata/ct_rs_063 \
  --orig-rtstruct  exampledata/ct_rs_063/RS.*.dcm \
  --ensemble  runs/mock_3states
```

`--ensemble` accepts either the run directory (recommended) or the full path to `manifest.json`.

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050).

| Control | Purpose |
|---------|---------|
| Deformed state dropdown | Select which motion state to show in the right panel |
| Orientation radio | Axial / Coronal / Sagittal |
| W/L / Gamma sliders | Window-level and display gamma |
| Diff range slider | Clamp ±HU range on the difference panel |
| Diff overlay on Deformed | Show ΔHU colour overlay on the deformed CT panel |
| Show contours | Toggle ROI contour overlay (requires `--orig-rtstruct`) |
| All ROIs dropdown | Filter which ROIs are drawn |

The **difference panel** shows `deformed_HU − original_HU`; positive (red) = tissue has moved into that voxel, negative (blue) = tissue has moved away. Contours shown on the difference panel are from the deformed RTSTRUCT for that state.

---

## 7. Bone rigidity masking

The DVF is not applied uniformly. Instead it is blended between "full motion" (soft tissue near the CTV) and "no motion" (bone):

```
bone_mask    = ct_hu > bone_threshold_hu        # e.g. 300 HU
bone_weight  = gaussian_smooth(bone_mask, σ = transition_width_mm / spacing)
tissue_weight = 1 − bone_weight                 # 0 inside bone, 1 in soft tissue

# Also apply spatial falloff from CTV centroid
falloff_weight = exp(−dist_from_CTV / falloff_mm)

final_dvf = raw_dvf × tissue_weight × falloff_weight
```

This means:
- **Bone voxels** are never displaced (pelvic bones stay in place).
- **Soft tissue near the CTV** moves as specified.
- **Soft tissue far from the CTV** transitions smoothly to zero displacement.
- The `--falloff` parameter controls how far the deformation extends (larger = more of the body moves).

---

## 8. State selection (k-medoids)

When a Synchrony motion log has hundreds of samples but you only want N states (for computational efficiency), the pipeline clusters the motion samples and picks the **medoid** (most representative sample) from each cluster.

The clustering distance uses both translation and rotation:

```
distance = sqrt(dx² + dy² + dz² + (rotation_weight × dθ)²)
```

`rotation_weight_mm_per_deg` converts degrees to mm-equivalent so both contribute meaningfully (default 10 mm/deg).

With only 3 CSV rows and `--n-states 3`, k-medoids just returns all 3 rows (trivial case). The `manifest.json` records `cluster_weight` per state — how many original samples each medoid represents.

---

## 9. Python API

```python
from pathlib import Path
from gendosecalc.deform.pipeline import generate_ensemble
from gendosecalc.deform.models import DeformationConfig

manifest = generate_ensemble(
    ct_dir="exampledata/ct_rs_063",
    motion_path="exampledata/mock_motion_3states.csv",
    out_dir="runs/my_run",
    config=DeformationConfig(
        bone_threshold_hu=300,
        transition_width_mm=3.0,
        falloff_mm=25.0,
        interpolation="linear",
    ),
    rtstruct_path="exampledata/ct_rs_063/RS.*.dcm",
    deform_rtstruct_path="exampledata/ct_rs_063/RS.*.dcm",
    n_states=3,
    seed=0,
)
# manifest is a dict matching the JSON structure above
print(manifest["states"][0]["ct_dir"])
```

### Lower-level: apply a DVF directly

```python
import numpy as np
from gendosecalc.deform.dvf_generate import rigid_to_dvf
from gendosecalc.deform.apply_deformation import deform_ct
from gendosecalc.deform.dicom_export import load_ct_series

ct, spacing, origin, direction, _ = load_ct_series("path/to/ct/")

dvf = rigid_to_dvf(
    tx=5.0, ty=0.0, tz=0.0,     # 5 mm Left, LPS convention
    rx=0.0, ry=0.0, rz=0.0,
    reference_shape=ct.shape,
    spacing_mm=spacing,
    origin_mm=origin,
    direction=direction,
)

deformed_ct, _, _, _ = deform_ct(ct, spacing, origin, direction, dvf)
```
