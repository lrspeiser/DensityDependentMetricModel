# REPRODUCIBLE.md

This note describes how to reproduce the main figures and tables in this repository with exact commands, package versions, and data artifacts. Update the DOI placeholders after minting Zenodo records.

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

