# Baseline and external datasets (tariff/data)

This folder documents and organizes the observational datasets used to establish GR baselines and evaluate the unified gate + energy-tariff add-on.

Data we use (minimum viable set)
- Supernova Hubble diagram (Pantheon+SH0ES)
  - Path in repo: external_data/pantheon/Pantheon+SH0ES.dat (already referenced by energy_tariff_model.py)
  - Columns parsed: zHD (3rd), MU_SH0ES (11th), MU_SH0ES_ERR_DIAG (12th)
- CMB spectral shape (COBE/FIRAS-like)
  - Provide a CSV with columns: frequency_GHz, intensity_Wsr_m2_Hz (or a normalized intensity column)
  - Place at: tariff/data/cmb_firas_like.csv (example path)
- BAO compilations (optional)
  - CSV with either D_M_over_rd, D_H_over_rd (with errors) or DV_over_rd; include a z column.
  - Place at: tariff/data/bao_compilation.csv
- Tolman surface-brightness dataset (optional)
  - CSV with columns: z, SB (surface brightness, consistent units), SB_err
  - Place at: tariff/data/tolman_sb.csv
- SN time-dilation dataset (optional)
  - CSV with columns: z and either timescale(+err) or stretch(+err)
  - Place at: tariff/data/sn_timedilation.csv

Where to get the data
- Pantheon+SH0ES: already referenced by path in this repo (ensure the file exists under external_data/pantheon/)
- FIRAS-like CMB spectrum: export from a public source or your own derived spectrum and save as CSV as described above
- BAO compilations: export a table from a standard BAO dataset and save as CSV with the specified column names
- Tolman and SN time-dilation datasets: export or collate into CSVs with the specified columns

Loaders and analysis
- The loaders live in tariff/data_ingest.py (functions: load_pantheon, load_cmb_spectrum_csv, load_bao_csv, load_tolman_csv, load_sntd_csv)
- Baseline checks live in tariff/analysis_baselines.py (linear μ(z), blackbody fits, Tolman and time-dilation exponents; simple FRW helper included)
- Unified gate comparisons live in tariff/analysis_unified_gate.py (Hubble diagram overlays from the unified gate + tariff)

Output plots
- All analysis scripts write plots under tariff/images/

Notes
- This add-on remains confined to tariff/; no changes to core galaxy/lensing code.
- If you add new datasets, update this README and loaders accordingly.
