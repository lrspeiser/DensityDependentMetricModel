# Changelog

All notable changes to this repository will be documented in this file.

## 2025-09-08

### Changed
- Orchestrator: Paper preset now enforces D_max=50 in the xi (nu) mapping and threads it across all xi calls (SPARC scans, lensing, Kz) for low-g stability.
- README.md: Added PPN CSV note under Solar (paper preset) and a new Extended Data section for Wide Binaries; included reproduce snippet for paper preset.
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

