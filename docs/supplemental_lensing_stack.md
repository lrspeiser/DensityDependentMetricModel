# Supplemental: Theoretical stacked ΔΣ (RAR vs GR)

Purpose
- This page documents the theory-only stacked excess surface density ΔΣ(R) built from the model’s metric mapping.
- It compares RAR (stars + phantom via ξ) to GR (stars only) using measured lens properties (z_l, z_s, log10 M_*, R_e) but does not overlay observed stacked lensing data. For accuracy claims, see the main paper’s θ_E comparison (predicted vs observed) and metrics.

What’s computed
- Per lens (scripts/next_steps_from_run.py):
  - Build a spherical baryon profile (Sérsic by default) from log10M_star and Re_kpc.
  - Compute enclosed baryonic mass M_b(<r). Under RAR, form M_eff(<r) = ξ(r)·M_b(<r) using xi_rar_plateau_numpy.
  - Project to Σ(R); compute \barΣ(<R) and ΔΣ(R)=max(\barΣ−Σ, 0) for RAR. Also store stars-only Σ_*, \barΣ_* to reconstruct GR ΔΣ.
  - Write per-lens profiles: results/.../lensing_metric_profiles/<lens>_profiles.csv with columns:
    R_kpc, Sigma_star, Sigma_tot, mean_star, mean_tot, DeltaSigma_tot, Sigma_cr.
- Stack across lenses on a common R grid (overlapping domain), report mean and the 16–84% spread.

Artifacts
- RAR-only stack (source for the original figure):
  - CSV (source-data): results/next_steps/enhanced_20250805_115400/lensing_metric_stack_source.csv
  - CSV (compact): results/next_steps/enhanced_20250805_115400/lensing_metric_stack.csv
  - Figure (cropped R≥0.1 kpc; log-safe shading): images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png
- GR vs RAR comparison (new):
  - CSV: results/next_steps/enhanced_20250805_115400/lensing_stack_compare.csv
  - Figure (cropped R≥0.1 kpc): images/next_steps/enhanced_20250805_115400/lensing_stack_compare.png

Code references
- Stack computation and per-lens profiles: scripts/next_steps_from_run.py (lensing stack and profiles steps).
- Replot the theoretical stack from source CSV with log-safe floor and R-min crop:
  - tools/replot_lensing_stack.py (default R_min = 0.1 kpc)
- Build GR vs RAR stacked overlay directly from per-lens profiles:
  - tools/stack_lensing_gr_vs_rar.py (default R_min = 0.1 kpc)

Interpretation (no observed stack overlaid)
- GR and RAR stacks are similar at small R (stellar-dominated), diverging to higher ΔΣ for RAR at larger R due to ξ’s phantom contribution. This demonstrates the expected qualitative behavior of the metric mapping.
- Without an observed stack overlay, the plot does not establish accuracy. For a quantitative, data-anchored comparison we use the θ_E (Einstein-radius) figures and metrics in the main paper, where RAR shows improved residuals vs GR on the SLACS sample (see lensing_thetaE_metrics.json in results/.../).

Reproducing the figures
- Generate the theoretical stack (already done by the orchestrator):
  - outputs written under results/next_steps/<run>/ and images/next_steps/<run>/
- Replot stack with cropping to avoid log-floor artifacts at very small radii:
  - python tools/replot_lensing_stack.py \
      --source results/next_steps/enhanced_20250805_115400/lensing_metric_stack_source.csv \
      --out images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png \
      --rmin-kpc 0.1
- Build GR vs RAR comparison overlay:
  - python tools/stack_lensing_gr_vs_rar.py \
      --profiles-dir results/next_steps/enhanced_20250805_115400/lensing_metric_profiles \
      --out-csv results/next_steps/enhanced_20250805_115400/lensing_stack_compare.csv \
      --out-png images/next_steps/enhanced_20250805_115400/lensing_stack_compare.png \
      --rmin-kpc 0.1

Notes
- Theoretical stacks are a useful sanity check for amplitude/shape and for comparing GR vs RAR behavior. However, statements about accuracy require observed stacked ΔΣ data with a well-documented shear/photo-z/boost/miscentering pipeline and error model. Those comparisons are intentionally omitted from the main paper at this stage and can be added later as a separate analysis.

