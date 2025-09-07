# Next-Step Analyses Index

Run: `rar_plateau_mw_full`

Artifacts:
- SPARC summary: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/sparc_a0_summary.csv`
- Solar table: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/solar_system_table.csv`, plot: `images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/solar_rar_plateau.png`
- Lensing baseline table: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/lensing_table.csv` (if present)
- Lensing RAR table: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/lensing_rar_table.csv` (if present)
- BTFR subset: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/btfr_summary.csv`
- Global a0: `results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p75_jaffe/global_a0.json` (if present)

Method Notes:
- RAR-plateau: D = 0.5 + sqrt(0.25 + a0_eff/g_bar); xi == D multiplies Vbar^2
- g_bar = (Vbar^2 / R) × 3.240779289e-14 in SI (m/s^2) for V in km/s and R in kpc
- a0_eff = a0 × (1 + zeta_env × s_rho × W(T)); see docs/cassini.md and docs/lensing.md
- Lensing baselines (GR point-mass, SIS) use Planck-like flat-ΛCDM distances; see docs/lensing.md

## Combined metrics summary
Generated: 2025-09-07 05:25 UTC

### Hernquist
- Alpha-only (zeta=0, env=constant): best alpha=1.5, RMS_rel=0.4318, N=3
- Tapered zeta @ alpha=2.0: best zeta=-0.5, RMS_rel=0.4295, N=3

### Jaffe
- Alpha-only (zeta=0, env=constant): best alpha=1.5, RMS_rel=0.4281, N=3
- Tapered zeta @ alpha=2.0: best zeta=-0.5, RMS_rel=0.4293, N=3

Artifacts:
- All runs: results\next_steps\btfr_fix_20250906_lastcross\combined_metrics\metrics_all_runs.csv
- Plots: images\next_steps\btfr_fix_20250906_lastcross\metrics
  - RMS vs alpha (per profile): rms_rel_vs_alpha_*.png
  - RMS vs zeta at alpha=2.0 (per profile): rms_rel_vs_zeta_alpha2_*.png

Notes:
- Lensing environment scaling uses clamp to keep (1 + zeta_env_lens * f(R)) >= 0, avoiding unphysical negative convergence for negative zeta.
- Einstein solver sometimes reported non-monotone <Sigma>(R); monotone envelope was applied during integration (expected for noisy grids).
