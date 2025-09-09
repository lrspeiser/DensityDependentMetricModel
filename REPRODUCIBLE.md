# REPRODUCIBLE.md

This note describes how to reproduce the main figures and tables in this repository with exact commands, package versions, and data artifacts. Update the DOI placeholders after minting Zenodo records.

## Single-command reproduction (recommended)

- From repo root (host):
  - Ensure SPARC rotmods exist under `external_data/Rotmod_LTG/` and a paper run NPZ exists under `runs/<run_name>/`.
  - Then run:

  ```bash
  RUN_DIR=runs/enhanced_20250805_115400 \
  SPARC_DIR=external_data/Rotmod_LTG \
  LENS_CSV=docs/lensing_targets.csv \
  ./reproduce_paper.sh
  ```

- Docker (CPU-first):
  ```bash
  docker build -t dgg-repro .
  docker run --rm -it \
    -e RUN_DIR=runs/enhanced_20250805_115400 \
    -e SPARC_DIR=external_data/Rotmod_LTG \
    -e LENS_CSV=docs/lensing_targets.csv \
    -v "$PWD/runs:/app/runs" \
    -v "$PWD/external_data/Rotmod_LTG:/app/external_data/Rotmod_LTG:ro" \
    -v "$PWD/results:/app/results" \
    -v "$PWD/images:/app/images" \
    dgg-repro
  ```

Notes
- The dynesty run regeneration (to create the NPZ) requires GPU/CuPy. If you need to regenerate the run, set `RUN_GENERATE=1` and ensure a working GPU/CuPy environment; otherwise, provide an existing run NPZ.

## Milky Way RAR‑plateau run & paper preset

If you need to run the Milky Way fit that produces the paper’s run NPZ, use the dynesty CuPy runner (GPU/CuPy recommended) and then invoke the paper preset orchestrator.

- Fit the Milky Way (RAR‑plateau):

  python runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py \
    --xi rar_plateau \
    --nlive 2000 --maxcall 1500000 --dlogz_target 0.01 \
    --seed 42 --num_threads 8 \
    --run_analysis \
    --out runs/rar_plateau_mw_full

  If runners/dynesty_latest/ is not present, fallback script: runners/run_dynesty_stellar_fit_cupy.py with the same flags.

- Generate paper figures/tables (paper preset):

  python scripts/reproduce_paper.py \
    --run-dir runs/rar_plateau_mw_full \
    --sparc-dir external_data/Rotmod_LTG \
    --lensing-csv docs/lensing_targets.csv \
    --sample gold --preset paper

- Helper: reproduce_paper.sh performs the orchestrator step and (optionally) the MW Kz overlay band plot. Set RUN_GENERATE=1 to let it attempt the dynesty run on a GPU/CuPy machine; otherwise provide an existing NPZ.
- Container: the provided Dockerfile runs reproduce_paper.sh in a CPU-first image; mount runs/ and external_data/Rotmod_LTG to reproduce outputs without GPU.

### Data requirements
- SPARC rotmod files under external_data/Rotmod_LTG/ (fetch via `git lfs pull`).
- Lensing targets table at docs/lensing_targets.csv with measured columns (lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec]).
- Milky Way Kz overlay CSV (optional): docs/mw_kz_overlay_two_bands.csv.
- Gaia annuli and posterior snapshots (if referenced) can be downloaded via the DOIs listed in §2 of this document once minted.

1) Environment and packages
- Python: 3.10+
- Core packages:
  - numpy
  - scipy
  - matplotlib
  - dynesty
  - cupy (for GPU runs; optional)
  - cupyx.scipy (for GPU Bessel in CuPy path)
- Freeze your environment (example):
  pip freeze > docs/requirements-freeze.txt

2) Data artifacts and DOIs (to be minted)
- Gaia annuli CSV and metadata.json for Milky Way analysis
  DOI: {{DOI_GAIA_ANNULI}}
- SPARC per-galaxy JSON sidecars (ER/TFR, GR, NFW evidences and fits)
  DOI: {{DOI_SPARC_JSON}}
- Runner commit snapshots (git commit hashes and tarball)
  DOI: {{DOI_RUNNERS_COMMITS}}
- Posterior NPZ snapshots (samples, weights, names)
  DOI: {{DOI_POSTERIORS_NPZ}}

3) One-command reproducible runs (as in docs/paper.md §10)
- SPARC single-galaxy matched triad (NGC 3198):
  python tools/fit_sparc_gr_evidence.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --sigma-floor 5.0 --mode evidence --nlive 1000 --maxcall 200000 --dlogz-target 0.01 --seed 42
  python tools/fit_sparc_nfw_evidence.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --sigma-floor 5.0 --mode evidence --nlive 1000 --maxcall 200000 --dlogz-target 0.01 --seed 42
  python tools/fit_sparc_er_env.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --mode fit --model er --sigma-floor 5.0 --gas-truncation RHI --T-proxy epicyclic --tidal-norm robust

- SPARC overlays and PPC:
  python scripts/plot_sparc_rotation_overlay.py --galaxy-id NGC3198 --sparc-dir external_data/Rotmod_LTG --fit-nfw-if-missing --out images/overlay_ngc3198.png
  python scripts/ppc_plots.py residual-envelope --json images/sparc_env_fit_ngc3198.json --out images/ppc_ngc3198_envelope.png
  python scripts/ppc_plots.py stacked-hist --json-glob "images/sparc_env_fit_*.json" --out images/ed_sparc_residual_hist.png

- SPARC batch evidence and ED table:
  python tools/batch_sparc_env_fit.py --sparc_dir external_data/Rotmod_LTG --mode evidence --sigma-floor 5.0 --nlive 1000 --maxcall 200000 --dlogz-target 0.01 --seed 42 --galaxies NGC3198 NGC2403 NGC6503 NGC5055 NGC7793 NGC2903 NGC7331 NGC3521 NGC2841 NGC5907 NGC4013 NGC5005 NGC5033 NGC598 NGC4157 NGC4010 NGC5985 UGC12506 UGC06917 UGC05918
  python tools/generate_sparc_ed_table.py

- Lensing pilot:
  python tools/lensing_predict.py --worked_example
  python tools/lensing_slacs_examples.py --A-env 0.3 --p-env 1.1 --r0-kpc 5.0 --a-env 1.0 --b-env 1.0 --mc 1000

4) Provenance and checksums
- Save SHA256 checksums for key inputs/outputs:
  - data/mw_annuli.csv, data/mw_annuli_metadata.json
  - images/sparc_* JSON sidecars
  - runs/*/*.npz (posterior snapshots)
- Place checksums under docs/checksums/.

5) Git commit references
- Record the exact commit for every major run; the post-analysis JSON sidecars already contain "commit" or "run_dir" in many tools. If missing, add a small helper to write git rev-parse HEAD into the JSON metadata.

6) Contact
- Maintainers: {{YOUR_NAME}} <{{YOUR_EMAIL}}>

