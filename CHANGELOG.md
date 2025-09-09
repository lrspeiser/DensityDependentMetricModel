# Changelog

All notable changes to this repository will be documented in this file.

## 2025-09-09

### Added
- Optional lensing systematics (defaults off to preserve baseline results):
  - Two‑halo ΔΣ template for stacked profiles via `--twohalo-csv` (CSV: `R_kpc,DeltaSigma_2h`). Stack CSV/Source‑Data now include baseline and `_sys` series when provided.
  - Miscentering kernel for stacked Σ/ΔΣ via `--miscenter-f-off` and `--miscenter-sigma-kpc` (Rayleigh offsets), applied at stack time using per‑lens Σ_tot profiles.
  - $\kappa_\mathrm{ext}$ prior marginalization for $\theta_E$ metrics via `--kappa-ext-mean/--kappa-ext-sigma/--kappa-ext-samples`. Metrics JSON contains stat‑only and `_kappa` entries.

### Docs
- README.md and docs/lensing.md updated with systematics usage and flags.

## 2025-09-08

### Changed
- Orchestrator: Paper preset now enforces D_max=50 in the xi (nu) mapping and threads it across all xi calls (SPARC scans, lensing, Kz) for low-g stability.
- README.md: Box 1 updated to document min(…, D_max) and the paper preset plateau (D_max=50); removed conflicting “no hard cap” clarification.
- README.md: Switched all Milky Way Kz references to full‑3D figure; added PPN CSV note under Solar (paper preset), Extended Data section for Wide Binaries, and paper preset reproduction snippet.
- README.md: Added BTFR note about bootstrap CI shading and a model‑comparison (BIC) figure note.
- README.md: Replaced Milky Way Kz/Σ_1.1 figure and description to use the new full 3‑D phantom density computation:
  - Old: images/next_steps/enhanced_20250805_115400/mw_kz_sigma.png (scaled approximation)
  - New: images/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.png (full 3‑D via ρ_ph = (ξ−1)ρ_b − (4πG)^{-1}∇ξ·g_bar)
  - Added Source‑Data link: results/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.csv
- README.md Figures list: Updated “Fig. 2” path to the full 3‑D figure.
- README.md Methods/Code sections: Added notes on run_metadata.json and sparc_selection.json.
- README.md Figures: Added Δlog Z (BIC approximation) figure reference; added Reproduction (paper preset) snippet.

### Rationale
- Align the manuscript and documentation with the improved method requested by reviewers: replace the scaled DGG proxy with the full 3‑D phantom‑density computation consistent with the Poisson identity. Surface the paper preset and provenance locations.

### Notes
- The previous figure and CSV remain in the results directory for reproducibility (`mw_kz_sigma.png` and `mw_kz_sigma.csv`). The README now points to the full‑3D outputs.
- If needed, revert by restoring the prior README image paths to `mw_kz_sigma.png` and updating the text back to the scaled‑approximation description.
- The paper preset can be invoked via reproduce_paper.py; remove or adjust the section if using a different workflow.

