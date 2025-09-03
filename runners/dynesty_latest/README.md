# Dynesty pipeline (latest scripts)

This folder contains the canonical, current dynesty runner and supporting utilities.
Use these scripts for new runs.

Included:
- run_dynesty_stellar_fit_cupy.py (primary runner; CUDA/CuPy-enabled)
- resume_with_summary.py (resume helpers and summary extraction)
- enhanced_summary.py (post-run metrics and tables)
- run_full_analysis_parallel.py (orchestrates full analysis jobs)
- run_production_stellar_fits.py (production orchestration)
- run_production_stellar_fits_all.py (batch orchestration)
- generate_comparison_plots.py (post-run plot generator)

Typical usage:
  python runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py --help

Notes:
- If you had older runner scripts elsewhere in runners/, consider them legacy or experimental.
- If any downstream code imports these by path, update to the new locations, or add import shims.

