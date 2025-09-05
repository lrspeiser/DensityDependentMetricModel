# Gaia DR3 regional slices (LMC/SMC) — How to prepare inputs

This project does not call the Gaia Archive directly. To include external Gaia
stars for extended tests (e.g., LMC/SMC), follow these steps:

1) Open the Gaia Archive web UI and run the ADQL query for your target.
   - LMC (12°):
```
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
  POINT('ICRS', ra, dec),
  CIRCLE('ICRS', 80.894, -69.756, 12)
)
```
   - SMC (7°): change center to (13.186, -72.828), radius to 7.

2) Download the result as CSV (or FITS) to a local folder.

3) Convert to Parquet for faster local use with:
```
python -m data_loaders.load_existing_gaia_lmc_smc \
  --input "<download_path>/*.csv" \
  --object LMC \
  --out-dir data/gaia_slices
```

Notes
- This repository never embeds API keys or secrets in code or in commands.
- If a future workflow requires credentials, keep them outside source control
  and export them into your environment (never echo them). Document the steps
  here without placing any secrets in plain text.
