# Next-Step Analyses Index

Run: `btfr_fix_20250906`

Artifacts:
- SPARC summary: `results/next_steps/btfr_fix_20250906/lensing_alpha10/sparc_a0_summary.csv`
- Solar table: `results/next_steps/btfr_fix_20250906/lensing_alpha10/solar_system_table.csv`, plot: `images/next_steps/btfr_fix_20250906/lensing_alpha10/solar_rar_plateau.png`
- Lensing baseline table: `results/next_steps/btfr_fix_20250906/lensing_alpha10/lensing_table.csv` (if present)
- Lensing RAR table: `results/next_steps/btfr_fix_20250906/lensing_alpha10/lensing_rar_table.csv` (if present)
- BTFR subset: `results/next_steps/btfr_fix_20250906/lensing_alpha10/btfr_summary.csv`
- Global a0: `results/next_steps/btfr_fix_20250906/lensing_alpha10/global_a0.json` (if present)

Method Notes:
- RAR-plateau: D = 0.5 + sqrt(0.25 + a0_eff/g_bar); xi == D multiplies Vbar^2
- g_bar = (Vbar^2 / R) × 3.240779289e-14 in SI (m/s^2) for V in km/s and R in kpc
- a0_eff = a0 × (1 + zeta_env × s_rho × W(T)); see docs/cassini.md and docs/lensing.md
- Lensing baselines (GR point-mass, SIS) use Planck-like flat-ΛCDM distances; see docs/lensing.md