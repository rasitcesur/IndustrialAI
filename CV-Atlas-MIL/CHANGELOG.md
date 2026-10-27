# CV Atlas MIL — Changelog

## 1.0.0 — 2026-10-26

- Unified five-expert inspection: pooled Extra Trees, healthy Atlas, original MIL, spatially co-located Regional Joint MIL, and high-resolution morphology.
- Added training-partition-only normalizers and reference-gallery protection for the Regional Joint MIL.
- Added CUDA/Cython/NumPy guarded feature dispatch, complete training/evaluation/prediction CLIs, and five-expert CSV inference scores.
- Added controlled spatial correspondence study and qualified prior out-of-fold performance in third-party documentation.
- Distributed under PolyForm Noncommercial License 1.0.0.

## Package consolidation

- All inference, training, descriptor and GPU-backend modules now reside in `cv_atlas_mil/`; the separate `casting_atlas/` compatibility package and root CUDA shim were removed.
- Public trained-model serialization uses `cv_atlas_mil.model.CVAtlasMIL` and its native submodules.
- The classifier architecture, feature definitions, weights and reported research evaluation are unchanged.
