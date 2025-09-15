# Energy Tariff add-on (tariff/)

This folder contains an optional cosmology add-on that explores a gated, energy-proportional photon loss model layered on top of the Gravity Gates weak-field response ξ(g). It is confined to tariff/ and does not modify the core galaxy/lensing pipeline.

Key idea
- Energy loss: dE/dr = -k [ξ(r) - 1] E, giving 1+z = exp(k ∫ (ξ-1) dr).
- ξ(g) is the same RAR-plateau gate used in the paper (capped by D_max), so dense regions (high g) contribute negligibly and void-weighted paths dominate the integral.
- This is a phenomenological add-on; it does not replace FRW expansion. CMB/time-dilation/Tolman/BAO checks below are provided to assess viability.

Model box — one gate, two observables (Option A)
- Energy → gravity mapping used by tariff when energy coupling is enabled:
  - y = g_bar / a0; H(y; β) = 1 / (1 + y^β)
  - a0_eff = a0 [ 1 + ζ (ρ_γ / E0) H(y; β) ], with E0 ≈ 0.26 eV/cm^3
  - ξ(g_bar, ρ_γ) = min[ 1/2 + sqrt(1/4 + a0_eff/g_bar), D_max ]
  - Gravity: g_obs = ξ · g_bar
- Photon tariff (unchanged): d ln E / dℓ = − κ [ξ − 1] f_env(…)
- Defaults: a0 = 1.2e-10 m/s^2, β = 2, ζ = 1, D_max ∈ {30, 50}, ρ_γ ≈ 0.26 eV/cm^3.
- Solar-System safety: for y ≫ 1 and β ≳ 2, H → y^−β and ξ−1 ≈ 0.5 a0/g. With a0 ≈ 1.2×10^-10 m/s^2:
  - 1 AU: |ΔG/G| ≈ 1.0×10^-8
  - 10 AU: |ΔG/G| ≈ 1.0×10^-6
  - 30 AU: |ΔG/G| ≈ 9.1×10^-6
  All are below the Cassini bound ~2.3×10^-5.

Files
- energy_tariff_model.py
  - PhotonJourney simulator for z(r) and μ(z), with k calibrated from an H0 anchor by default.
  - Optional environment mixing f_env(r) or f_env(z), and an energy-coupled a0 scaffold via energy_coupled_gate.py.
  - Outputs plots under tariff/images/.
- energy_coupled_gate.py
  - Minimal “Sakharov-style” coupling scaffold that modulates a0_eff(g) for the RAR-plateau formula. Disabled by default.
- sweep_tariff_params.py
  - Grid-search sweep over D_max, g_bar_void, r0_void, gamma_void. Writes tariff/sweep_results.csv, prints top settings by reduced chi^2 vs Pantheon+ μ(z).
- tariff_major_tests.py
  - Batteries of checks with subcommands:
    - cmb: CMB spectral-shape fit (blackbody preservation check with Liouville vs energy-only intensity mapping)
    - tolman: Fit p in d_L = r (1+z)^p from μ(z)
    - sntd: Fit SN time-dilation exponent p_t from light-curve summaries
    - bao: H_eff(z), D_M(z), D_H(z) proxies and optional BAO CSV comparison
    - los: Correlate SN residuals with LOS density proxy
    - posteriors: Summarize sweep_results.csv as posterior-like histograms
- images/
  - Output directory for figures created by the above scripts (kept under version control with .gitkeep).
- researchpaper.md
  - Draft framing for a cosmology add-on section (introduction/methods/scope) referencing this scaffold.
- issues.md
  - Open issues and a short-term plan to lift the add-on to submission quality (SN time-dilation, FIRAS/CMB, Tolman, BAO, LOS correlations, etc.).

Data dependencies
- Pantheon+SH0ES Hubble diagram data is expected at external_data/pantheon/Pantheon+SH0ES.dat.
  - If the path differs, pass --data-file to energy_tariff_model.py or sweep_tariff_params.py.
- Optional BAO CSVs can be provided to tariff_major_tests.py bao.

Usage examples
1) Redshift curve and Hubble Diagram overlay

- Calibrate k from H0 and plot z(r); optionally overlay μ(z) with Pantheon+:
  python tariff/energy_tariff_model.py --distance-max 4000 --plot-hubble \
      --data-file external_data/pantheon/Pantheon+SH0ES.dat
  - Images saved to tariff/images/energy_tariff_redshift_model.png and hubble_diagram_with_data.png.

2) Parameter sweep

- Write tariff/sweep_results.csv, print top 10 by reduced chi^2:
  python tariff/sweep_tariff_params.py \
      --data-file external_data/pantheon/Pantheon+SH0ES.dat \
      --out-csv tariff/sweep_results.csv

3) Major tests

- CMB shape (Liouville vs energy-only mapping):
  python tariff/tariff_major_tests.py cmb --k 7.75e-6 --dmax 30 --distance-mpc 14000 --transport liouville
- Tolman exponent p from μ(z):
  python tariff/tariff_major_tests.py tolman --k 7.75e-6 --dmax 30 --data path/to/sn_mu.csv
- SN time-dilation exponent p_t from light-curve summaries:
  python tariff/tariff_major_tests.py sntd --data path/to/sn_timedilation.csv
- BAO proxies and optional CSV comparison:
  python tariff/tariff_major_tests.py bao --k 7.75e-6 --dmax 30 --bao path/to/bao.csv
- LOS correlation (residuals vs delta_LOS):
  python tariff/tariff_major_tests.py los --sn path/to/sn_mu.csv --los path/to/los_density.csv
- Posterior-like summaries of sweep:
  python tariff/tariff_major_tests.py posteriors --sweep tariff/sweep_results.csv

Notes and scope
- This add-on preserves the main paper’s scope: it does not alter galaxy rotation/lensing/Solar-System results in the core. It is an exploratory scaffold to test whether a small, gated, energy-proportional effect can follow SN Hubble-diagram shape trends while remaining consistent with CMB, time-dilation, Tolman, and BAO constraints.
- All outputs are restricted to tariff/ and images under tariff/images/.
