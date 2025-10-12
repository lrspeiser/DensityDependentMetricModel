# Wide-binary constraints (Gaia DR3) — quick start

This document explains how to generate the RAR-plateau prediction for the wide-binary (WB) statistic in the Solar neighborhood and, optionally, overlay a vetted observational summary from Gaia DR3.

Prerequisites
- No APIs or web access required. If you want to compare to an observed WB statistic, prepare a local CSV with columns: s_AU, v_ratio (binned or per-system as appropriate to your method). See referenced analyses in feedback.md for suitable methodologies and quality cuts.

Steps
1) Ensure you have a rar_plateau run directory (or use a placeholder name); the orchestrator will write a metadata snapshot used by this script. You may also proceed without it; defaults are applied.

2) Generate the theory prediction curve

```
python scripts/analyze_wide_binaries.py \
  --run-dir runs/rar_plateau_mw_full \
  --out-root results/next_steps/rar_plateau_mw_full \
  --images-root images/next_steps/rar_plateau_mw_full
```

This writes:
- results/next_steps/rar_plateau_mw_full/wide_binaries_pred.csv
- images/next_steps/rar_plateau_mw_full/wide_binaries_pred.png

3) Optional overlay with your local WB catalog summary

```
python scripts/analyze_wide_binaries.py \
  --run-dir runs/rar_plateau_mw_full \
  --out-root results/next_steps/rar_plateau_mw_full \
  --images-root images/next_steps/rar_plateau_mw_full \
  --wb-csv data/wide_binaries_dr3_summary.csv
```

Notes
- The theory curve shows sqrt(xi) − 1 vs projected separation s (AU) for a representative two-solar-mass binary. It is a proxy for the fractional velocity excess under the model relative to Newtonian gravity. Screening/gating suppresses the effect at small s; deviations grow only in the deep low-acceleration regime.
- If you include this in the manuscript, cite the Gaia DR3 WB analyses you compare against and describe the external-field modeling and quality cuts.

