# Changelog

All notable changes to this repository will be documented in this file.

## 2025-09-10

### Added
- Milky Way Kz baryon‑prior propagation: new `--mw-kz-prior-band` path in `scripts/next_steps_from_run.py` draws baryonic parameter priors (disk/bulge M/L, gas, scale heights/lengths, bulge scale proxy) and propagates through the full‑3D phantom density to produce a Kz(z) band at R0. Outputs `mw_kz_prior_band.csv` and overlays a 16–84% shading on the Kz figure.
- CLI knobs for MW baryon priors: `--mw-prior-samples`, `--mw-prior-ml-sigma`, `--mw-prior-gas-frac-sigma`, `--mw-prior-height-frac-sigma`, `--mw-prior-Rd-frac-sigma`, `--mw-prior-bulge-a-frac-sigma`, and `--mw-kz-zlist`.
- README.md: expanded MW Kz section with assumptions (R0, z cuts, tracer‑kinematics) and description of the baryon‑prior band; lists Source‑Data CSVs.
- REPRODUCIBLE.md: added one‑command example to generate the MW Kz prior band; clarified overlay CSV usage.
- SLACS importer: `scripts/import_slacs_asu_tsv.py` to convert VizieR ASU‑TSV (Auger+ 2009; J/ApJ/705/1099) into curated/orchestrator CSVs.

### Changed
- MW Kz baseline plot label now explicitly indicates “baseline baryons” when no band is requested.
- Lensing docs aligned to SLACS (homogeneous) sample: updated README caption (N=70), metrics table, and per‑lens Extended Data examples (J0037‑0942; J1402+6321). Replaced CASTLES mentions for main figure to avoid mixed samples.
- README lensing metrics table now states definitions (Bias_abs, Bias_rel) and notes n_sersic=4 ETG baseline.

### Fixed
- SLACS Einstein radius units: RE provided in kpc were previously treated as arcsec in the curated table, inflating observed θE and producing large residuals. Now converted kpc→arcsec using flat ΛCDM (H0=70, Ωm=0.3) before writing theta_E_obs_arcsec. Metrics recalculated (RAR RMSE_abs ≈ 0.553″; GR ≈ 0.655″).

### Rationale
- Addresses reviewer request to use a homogeneous lens sample, clarify metric definitions, and ensure unit consistency.

### Notes
- No changes to formulas or the ξ(g) mapping. Changes are limited to data ingest correctness and documentation alignment.

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

