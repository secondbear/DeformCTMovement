# Agent Handoff — DeformCTMovement

**Last updated**: 2026-04-14  
**Author**: GitHub Copilot (initial scaffold)  
**Repo**: https://github.com/secondbear/DeformCTMovement

---

## What This Project Does

Generates anatomically plausible deformed CT volumes for motion-resolved radiotherapy dose calculation (prostate SBRT). Applies a displacement vector field (DVF) to a planning CT with bone-rigidity masking via HU thresholding.

Part of the **GenDoseCalc** ecosystem. The existing pipeline applies prostate motion as rigid isocenter shifts; this module enables per-state CT deformation so the pencil-beam engine sees realistic anatomy.

---

## Current State (as of handoff)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | DVF infrastructure (`models`, `sitk_bridge`, `dvf_io`) | **NOT STARTED** |
| Phase 2 | Bone masking + DVF application (`bone_mask`, `apply_deformation`) | **NOT STARTED** |
| Phase 3 | DVF generation from motion params (`dvf_generate`) | **NOT STARTED** |
| Phase 4 | Pipeline integration (`--deform-ct` flag, DICOM export) | **NOT STARTED** |
| Phase 5 | Validation and refinement | **NOT STARTED** |

No source files exist yet — only planning and scaffold documents.

---

## Files in This Repo

```
README.md                                     — User-facing overview
plan_deform.md                                — Full technical specification (read this first)
.gitignore                                    — Python + data file exclusions
.github/
  copilot-instructions.md                     — Always-on Copilot workspace instructions
  hooks/
    session.json                              — SessionStart hook: injects project context
  skills/
    implement-deform-phase/
      SKILL.md                                — On-demand skill: step-by-step phase implementation guide
AGENTS.md                                     — This file
```

---

## How to Start Implementation

### Recommended order

1. **Read `plan_deform.md`** in full — especially §5 (Data Model) and §6 (Module Design)
2. **Phase 1 first** — all other phases depend on `DeformationField`, `sitk_bridge`, and `dvf_io`
3. Use the `implement-deform-phase` skill (`.github/skills/implement-deform-phase/SKILL.md`) — it contains exact implementation steps, critical details, and test guidance for each phase

### Triggering the skill

In a Copilot chat session, the skill is auto-loaded when you ask to implement a phase. Alternatively, read it directly:

```
Read .github/skills/implement-deform-phase/SKILL.md and implement Phase 1.
```

---

## Key Technical Facts

| Fact | Detail |
|------|--------|
| Coordinate system | LPS throughout — match SimpleITK |
| DVF array shape | `(3, nz, ny, nx)` float32, displacement in mm |
| DVF mapping convention | **Inverse** (output → input); forward DVFs need `sitk.InvertDisplacementField()` |
| Bone threshold | 300 HU (default), configurable via `DeformationConfig` |
| HU range | Always clamp to `[-1024, 3071]` and cast to int16 after resampling |
| Sigma units | `transition_width_mm` is in mm — divide by voxel spacing before `gaussian_filter` |
| Test data | Synthetic phantoms only — **no patient data ever** |

---

## Critical Pitfalls to Avoid

1. **Direction cosines**: SimpleITK direction matrix is row-major. Always copy direction via `image.GetDirection()` / `image.SetDirection(direction.flatten().tolist())`.
2. **B-spline out of range**: B-spline interpolation can produce HU values outside int16. Always clamp.
3. **Forward vs inverse DVF**: document the convention in every function's docstring. Current convention: `deform_ct()` expects a **forward** DVF and inverts internally.
4. **Bone mask broadcasting**: tissue weight map is `(nz, ny, nx)` — must be unsqueezed to `(1, nz, ny, nx)` before multiplying the `(3, nz, ny, nx)` DVF array.

---

## Dependencies

```toml
# pyproject.toml (to add)
"SimpleITK >= 2.3",
# scipy already present (provides gaussian_filter)
```

---

## Testing

```bash
pytest tests/deform/ -v
```

Test structure (all files to be created):

```
tests/deform/
  conftest.py              — Synthetic CT and DVF fixtures
  test_sitk_bridge.py
  test_dvf_io.py
  test_bone_mask.py
  test_apply_deformation.py
  test_dvf_generate.py
  test_deform_pipeline.py
```

---

## What the Next Agent Should Do

1. Read `plan_deform.md` in full
2. Load the `implement-deform-phase` skill
3. Implement Phase 1 — create `gendosecalc/deform/__init__.py`, `models.py`, `sitk_bridge.py`, `dvf_io.py`, and their tests
4. Update this file: change Phase 1 status from **NOT STARTED** → **IN PROGRESS** → **COMPLETE**
5. Commit after each phase with message: `feat(deform): phase N — <short description>`

---

## Update Instructions

After completing each phase, update the status table above. Use the following status values:
- `NOT STARTED` — no work done
- `IN PROGRESS` — partially implemented  
- `COMPLETE` — implemented and all tests passing
- `BLOCKED` — waiting on external dependency (describe below table)
