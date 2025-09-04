# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

1) Quickstart environment setup
- Python: use 3.11 (the CI uses 3.11). Create and activate a virtual environment.
  Windows (PowerShell):
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
  macOS/Linux:
    python3.11 -m venv .venv
    source .venv/bin/activate
- Install base dependencies (packaged under ./utils):
    pip install ./utils
- Install dynesty (used by runners and tools):
    pip install dynesty
- Install CuPy matching your CUDA version (GPU strongly recommended):
    # CUDA 12.x example
    pip install cupy-cuda12x
    # CUDA 11.x example
    pip install cupy-cuda11x
  Note: Many runners/tests import CuPy directly and will fail without a compatible GPU install.

2) Common commands (build, tests, runs)
- Build/install (editable) of utilities package for development:
    pip install -e ./utils
- Run full test suite (requires pytest; GPU recommended as many tests use CuPy paths):
    pip install pytest
    python -m pytest tests/ -v
- Run a single test file or filter by expression:
    python -m pytest tests/test_xi_simple.py -q
    python -m pytest -k "split_regions" -q
- Quick model smoke test (reduced sample, GPU path):
    python runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py --xi power --sample_max 1000 --maxcall 10000
- Compare models with test settings:
    python runners/run_all_stellar_fits.py --test
- Production orchestrator (CuPy runner underneath):
    python runners/run_production_fits.py --auto --priority
- GPU diagnostics/monitoring (optional):
    python runners/diagnose_gpu_utilization.py
    python runners/monitor_dashboard.py

3) High-level architecture (big-picture only)
- Core physics (core/)
  - density_metric_cupy.py: primary GPU implementation. Provides v_total_kms_cupy and xi_* functions; used by GPU runners and tools.
  - density_metric2.py: JAX-based reference/alternative engine.
  - data_io.py: data loading and preprocessing (Gaia, etc.).
  - xi_* modules and xi_registry.py: variants of the enhancement function ξ(ρ) and their registry.
- Runners (runners/)
  - dynesty_latest/: canonical, current dynesty-based CuPy runner suite (use these for new runs). Primary entry: dynesty_latest/run_dynesty_stellar_fit_cupy.py. Orchestrators for production/batch are here as well.
  - Other runners (run_dynesty.py, run_dynesty_cupy.py, run_dynesty_split_regions.py, run_production_fits.py): orchestration for sampling, diagnostics, and specialized workflows.
- Models (models/): NFW halo and ER/TFR environment models used for SPARC/galaxy comparisons.
- Data loaders (data_loaders/): survey-specific loaders (Gaia, SPARC, DES, BAO, etc.).
- Analysis (analysis/): post-run analyzers and plotting utilities used from periodic/final analysis steps.
- Scripts and tools (scripts/, tools/): analysis pipelines, plotting, SPARC evidence fitting helpers.
- Tests (tests/): mix of pytest suites and diagnostic scripts; many rely on CuPy/GPU paths.
- Outputs (results/, production_results/, runs/): intermediate and final artifacts (JSON/NPZ/PNG and run folders) collected in CI as artifacts.

Interdependencies (example flow):
- runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py → core/density_metric_cupy.py (physics) → core/data_io.py (data) → analysis/analyze_results.py (periodic/final analysis).

4) CI notes
- .github/workflows/ci.yml installs with pip install ./utils (no root pyproject or requirements.txt present), verifies Python 3.11, and uploads *.png/*.json artifacts. It does not run tests by default.
- Keep dependency pins inside utils/pyproject.toml current; add any new runtime deps there or document separate installs in this WARP.

5) Important docs to consult
- README.md: overview, installation examples, usage (quick tests, production runs, GPU optimization notes).
- runners/dynesty_latest/README.md: the canonical runner family and usage.
- runners/README_STELLAR_FIT.md: stellar-focused likelihood and flags.
- instructions/README_TESTS.md: how to run subsets of tests and GPU-oriented test notes.
- instructions/README_CORE.md and instructions/README_RUNNERS.md: module overviews and common parameters.

6) Data and GPU prerequisites (operational heads-up)
- Many runners/tests require a CUDA-capable GPU with a matching CuPy wheel (see CuPy install above).
- Gaia/SPARC workflows expect local data or will use cached/standardized paths as defined in core/data_io.py and tools/scripts; review instructions/ and docs/ before launching large runs.

Repository conventions and gaps captured from current state
- No top-level requirements.txt; base deps live in utils/pyproject.toml. Dynesty is required by runners and must be installed separately if not bundled.
- No formal lint configuration found (ruff/flake8/mypy/tox not present). Skip linting commands here.

