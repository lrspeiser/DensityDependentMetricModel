# Dmax plateau cap (RAR-plateau mapping)

Summary
- The RAR-plateau weak-field mapping uses D ≡ ξ = 0.5 + sqrt(0.25 + a0_eff/g_bar) as the multiplicative factor on V_bar^2.
- An optional finite plateau cap D_max can be applied in post-processing: D ← min(D, D_max), with D_max > 1.
- Purpose: a pragmatic guardrail to prevent extreme-field excursions in very deep regimes (ultra-low g_bar) that are outside the data domain for most figures, and to reflect a plausible saturation from microphysical screening/UV completion.
- Scope: the cap is applied in the NumPy mapping used by the analysis orchestrator (rotation curves, lensing, MW Kz) when provided via CLI. The CuPy engine returns the raw D without an internal cap; analysis code threads D_max explicitly downstream.

Where it is implemented
- NumPy path (cap is applied when D_max is provided):
  - scripts/next_steps_from_run.py → xi_rar_plateau_numpy(..., D_max=None)
    - Optional clamp: if D_max>1 → D = np.minimum(D, D_max)
- CuPy path (no internal clamp; returns raw D):
  - core/density_metric_cupy.py → xi_rar_plateau_cupy(...)
- CLI flag and preset behavior:
  - scripts/next_steps_from_run.py → --rar-dmax (float). If unset in the "paper" preset, defaults to 50. Custom preset leaves it unset (no cap).

Physical interpretation (why a cap might be reasonable)
- The RAR-plateau form approaches D ~ sqrt(a0_eff/g_bar) in the deep regime; as g_bar→0, D→∞ formally. Realistic microphysics may saturate the enhancement (e.g., strong-field screening, finite sound/signal speeds in the effective medium, or stability bounds), limiting D. We therefore test a finite plateau as a modeling assumption, quantify its impact, and disclose sensitivity in figures.
- This is not a first-principles derivation. It is a controlled hypothesis subjected to sweeps and model-comparison metrics.

How to run a Dmax sweep (reproducible commands)
Assumptions: You have a working run folder and SPARC rotmods (see WARP.md quickstart). Use the orchestrator with metric-only lensing.

1) No cap ("inf")

```
python scripts/next_steps_from_run.py \
  --preset custom \
  --run-dir runs/enhanced_20250805_115400 \
  --sparc-dir external_data/Rotmod_LTG \
  --posterior-samples 400 \
  --sample q2plus \
  --min-npts 12 --min-rmax-kpc 8 --max-quality 2 \
  --mw-kz \
  --lensing-sample-csv docs/lensing_targets.csv \
  --metric-lensing-only --density-profile sersic --write-ppn-table \
  --out-root results/dmax_sweep/inf
```

2) Finite caps (80, 50, 30) under the paper preset

```
# Dmax = 80
python scripts/next_steps_from_run.py \
  --preset paper \
  --rar-dmax 80 \
  --run-dir runs/enhanced_20250805_115400 \
  --sparc-dir external_data/Rotmod_LTG \
  --posterior-samples 400 \
  --mw-kz \
  --lensing-sample-csv docs/lensing_targets.csv \
  --metric-lensing-only --density-profile sersic \
  --out-root results/dmax_sweep/80

# Dmax = 50 (paper default)
python scripts/next_steps_from_run.py \
  --preset paper \
  --run-dir runs/enhanced_20250805_115400 \
  --sparc-dir external_data/Rotmod_LTG \
  --posterior-samples 400 \
  --mw-kz \
  --lensing-sample-csv docs/lensing_targets.csv \
  --metric-lensing-only --density-profile sersic \
  --out-root results/dmax_sweep/50

# Dmax = 30
python scripts/next_steps_from_run.py \
  --preset paper \
  --rar-dmax 30 \
  --run-dir runs/enhanced_20250805_115400 \
  --sparc-dir external_data/Rotmod_LTG \
  --posterior-samples 400 \
  --mw-kz \
  --lensing-sample-csv docs/lensing_targets.csv \
  --metric-lensing-only --density-profile sersic \
  --out-root results/dmax_sweep/30
```

Outputs to look for (per root)
- SPARC: sparc_a0_summary.csv (per-galaxy a0 best & chi2), model_comparison_bic.csv (ΔlogZ approx via BIC), sparc_selection.json
- MW Kz: mw_kz_sigma_full3d.csv and mw_kz_sigma_full3d.png
- Lensing (metric-only): lensing_metric_table.csv, lensing_thetaE_metrics.json, lensing_profile CSVs per lens, scatter and profile plots
- Solar System: solar_system_table.csv, ppn_table.csv, solar figure

One-line summary CSV
- Combine the sweep results into a single CSV (lens θE metrics + SPARC + MW Kz) using:

```
python tools/summarize_dmax_sweep.py \
  --roots results/dmax_sweep/30 results/dmax_sweep/50 results/dmax_sweep/80 results/dmax_sweep/inf \
  --out results/qa/dmax_summary.csv
```

Notes and caveats
- The cap is a post-processing clamp used in analysis products. The GPU engine’s raw dynamics can be inspected without the cap if desired; for manuscript figures, we disclose whether a cap is used and quantify sensitivity.
- Observationally, typical SPARC, θE and Kz domains do not probe the extreme tail where D would try to diverge; sensitivity is expected to be modest within [30, 80], but we verify with the sweep above.

File/code pointers
- scripts/next_steps_from_run.py: xi_rar_plateau_numpy (cap), orchestrator CLI, and outputs
- core/density_metric_cupy.py: xi_rar_plateau_cupy (no internal cap)
- tools/summarize_dmax_sweep.py: summary aggregator (extended to include SPARC and MW Kz)

