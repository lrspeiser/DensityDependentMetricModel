# ΛCDM RAR band overlay — instructions

Purpose
- Provide an overlay for the RAR master panel that shows a band from ΛCDM hydrodynamical simulations (e.g., EAGLE/NIHAO) for context and contrast.

Data format
- Create a CSV with three columns (no header comments):
  - log10_gbar: log10 of the baryonic acceleration (m s^-2)
  - log10_gobs_lo: lower edge of the band at that g_bar (e.g., 16th percentile of log10 g_obs)
  - log10_gobs_hi: upper edge of the band at that g_bar (e.g., 84th percentile of log10 g_obs)
- File path recommendation: docs/lcdm_rar_band.csv
- A blank template is provided at docs/lcdm_rar_band_template.csv (just a header line). Copy to docs/lcdm_rar_band.csv and fill.

Curation guidance
- Digitize the band from the relevant published RAR figure (e.g., EAGLE or NIHAO) using a figure digitizer, or export directly from a public dataset if available.
- Ensure units match this repo’s RAR convention: accelerations in m s^-2, and values are log10-transformed before writing to CSV.
- Use a reasonably fine grid (e.g., 50–150 points) spanning log10 g_bar ~ [−13, −9] where relevant; sort rows by log10_gbar ascending.
- Prefer 16–84% bands when possible; if only min–max are available, note this in the manuscript caption.

How to generate the panel with overlay
```bash
python scripts/make_rar_master_panel.py \
  --sparc-dir external_data/Rotmod_LTG \
  --results-root results/next_steps/rar_plateau_mw_full \
  --lcdm-band docs/lcdm_rar_band.csv
```

Typical discriminants to annotate (in the paper)
- Low-acceleration slope: whether the ΛCDM band exhibits a different slope than the DGG posterior band at log10 g_bar ≲ −11.5.
- Mass/size dependence of scatter: note any trend with stellar mass bins, if available.
- Outer-disk falloffs: highlight specific high-SB systems where DGG predicts gentle declines vs ΛCDM “flatter” behavior at the same M⋆.

Reproducibility
- Commit the CSV to the repository (LFS not required; small text). Include the CSV path in the figure’s Source Data list.

