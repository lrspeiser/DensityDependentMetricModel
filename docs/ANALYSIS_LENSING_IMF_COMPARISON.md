# Lensing IMF Comparison Artifacts

This page tracks the manuscript-facing artifacts comparing Chabrier (δIMF=0.00) vs Salpeter-like (δIMF=+0.23) assumptions for SLACS ETGs, using circularized Re and propagated structural uncertainties.

Generated artifacts (via scripts/analysis/generate_imf_artifacts.py):

- Table: docs/tables/lensing_imf_comparison.md
- Histogram: docs/figures/lensing_imf_f_theta_hist.png
- ΔAIC: docs/metrics/lensing_imf_delta_aic.json
- q-axis summary: docs/stats/lensing_q_axis_ratio_summary.json
- Paper snippets:
  - Methods: docs/paper_snippets/methods_delta_imf_sentence.md
  - Paragraph: docs/paper_snippets/lensing_imf_paragraph.md

Notes
- These artifacts are generated from the latest results/next_steps/enhanced_* run. Re-run the script after new runs to refresh.
- For fairness, generate and include GR+baryons rows with the SAME IMF normalization if available; otherwise, keep the two-row GG table.
- Acceptance targets: Bias_rel ∈ [-0.1, +0.1], Coverage_68 ≳ 0.6, Coverage_95 ≳ 0.9.

