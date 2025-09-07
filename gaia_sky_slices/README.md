# Gaia DR3 regional slices — inputs directory

This folder is where processed Gaia DR3 regional slices are placed for local analyses.

Where files come from
- See docs/gaia_slices_readme.md for full instructions and ADQL examples.
- Typical workflow: query Gaia (UI or TAP), download CSV/FITS locally, then convert to Parquet using the helper.

Recommended conversion
- For LMC/SMC or custom regions, use the existing loader to convert CSV → Parquet:
  - python -m data_loaders.load_existing_gaia_lmc_smc --input "<download_path>/*.csv" --object LMC --out-dir gaia_sky_slices
  - Or with TAP (public, no credentials needed):
    - python -m data_loaders.load_existing_gaia_lmc_smc --api --object LMC --limit 100000 --out-dir gaia_sky_slices

Expected file pattern and columns
- Files should be written here as processed_*.parquet (e.g., processed_L000-030.parquet).
- Downstream loaders expect at minimum these columns:
  - R_kpc, v_obs, sigma_v
- Optional but supported if present:
  - z_kpc, source_id, quality_flag, v_R_kms, v_z_kms
- See data_loaders/load_existing_gaia.py and core/data_io.py for exact usage and validation.

Why this location?
- Several scripts read from this path by default (gaia_sky_slices/ at repo root),
  and .gitignore is configured to allow tracking Parquet/CSV in this folder.

Git LFS and versioning
- Parquet/CSV in this folder are tracked via Git LFS (.gitattributes rules).
- Do not place secrets or credentials here. Public Gaia queries require none.

Troubleshooting
- If a loader logs "No processed parquet files found", ensure your files match the
  processed_*.parquet naming and include the required columns listed above.
- To check coverage: python validation/check_data_coverage.py

