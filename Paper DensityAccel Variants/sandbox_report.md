# Density-gated vs Acceleration-gated Analyses (Sandbox Report)

This report summarizes sandbox runs comparing a density-gated (DG) / hybrid gate to the original acceleration-gated (AG, RAR-like) approach and GR (baryons only). All artifacts live under “Paper DensityAccel Variants/” and were generated without modifying the production code or README.

Scope of this sandbox
- Unified gate interface build_gate(kind): accel | density | hybrid
- Clusters (CLASH × ACCEPT): DG/hybrid diagnostics (ξ vs r, with ρ from ACCEPT gas + optional stars)
- Solar ephemeris: ΔG/G(r) with DG
- SPARC: AG overlays (CSV rotmods), ready for DG once ρ(R) provided
- Lensing: DG at θ_E from a Hernquist deprojection surrogate
- Milky Way Kz: baryons-only + gate metadata (DG phantom not included yet)

Artifacts
- Clusters (hybrid gate)
  - Diagnostics CSV: Paper DensityAccel Variants/results/cluster_hybrid/cluster_gate_diagnostics.csv
  - Summary JSON: Paper DensityAccel Variants/results/cluster_hybrid/summary_variants.json
  - Notes: ρ_bary(r) = μ_e m_p n_e + ρ_star(Hernquist) when stars CSV provided; ξ = build_gate('hybrid')(ḡ, ρ, r)

- Solar ephemeris (density gate)
  - Console sample: ΔG/G ≈ 4.9×10^{-2} (flat with r for a constant ρ_env), parameters: ρ_c=1e-24 g cm^{-3}, Dmax=50

- SPARC (accel gate; CSV rotmods converted from .dat)
  - Converted CSVs (first 20): Paper DensityAccel Variants/sparc_csv/*.csv
  - Models: Paper DensityAccel Variants/results/sparc_accel/CamB_gate_model.csv, D631-7_gate_model.csv, ...
  - Each CSV: R_kpc, Vbar_kms, ξ, V_model_kms
  - Next: add ρ(R) proxy to run DG/hybrid for SPARC (see “Planned extensions” below)

- Lensing (density gate)
  - JSON summary: Paper DensityAccel Variants/results/lensing_density.json
  - Fields: xi_at_R_E, R_E_kpc, Re_kpc, log10Mstar, gate params

- Milky Way Kz (baryons only; gate metadata)
  - CSV: Paper DensityAccel Variants/results/mw_kz/mw_kz_baryons_only.csv
  - Gate meta: Paper DensityAccel Variants/results/mw_kz/mw_kz_gate_meta.json

Key outcomes (qualitative)
- Clusters: The hybrid gate provides ξ(ḡ, ρ) using ρ from ACCEPT (plus optional stars), allowing the same parameter family to respond to missing inner baryons (BCG/ICL) and outer environment (ICM density). This mirrors the defensible configuration we used, now as a sandbox pipeline.
- Solar: A density screening produces roughly constant ΔG/G across AU for uniform ρ_env; the mapping is consistent with Solar safety when ρ suppresses the gate towards unity.
- SPARC: AG overlays generated; DG/hybrid ready once we estimate ρ(R) (midplane proxy from Σ/2h_z or a small deprojection helper). This will let us compare pure AG vs hybrid DG on galaxies with minimal extra nuisance.
- Lensing: DG changes the amplitude at θ_E via ξ(ρ_E). The sandbox writes xi_at_R_E so we can compare to the production lensing metrics.
- MW Kz: Left as baryons-only here (DG-phantom adds ∂ξ/∂ρ · ∇ρ terms); we recorded gate params to reproduce settings.

Planned extensions (to complete the paper-style comparison)
1) SPARC ρ(R) proxy
   - Build rho_proxy.csv from rotmods using midplane density: ρ ≈ Σ/(2 h_z) with h_z inferred from scaling relations.
   - Re-run sparc_variants.py with --gate density|hybrid to produce DG overlays.

2) Cluster comparison tables
   - Include a simple NFW/GR baseline in the sandbox diagnostics for a fair like-for-like table (RMS/coverage).

3) Lensing distances
   - Add a small distance helper (H0=70, Ωm=0.3) to compute Σ_cr from (z_l, z_s) and compare xi_at_R_E across gates and IMFs.

4) MW Kz DG-phantom (experimental)
   - Implement the extra phantom term proportional to ∂ξ/∂ρ to quantify differences vs AG-phantom in the vertical-force band.

5) End-to-end “sandbox_reproduce.py”
   - One command to run clusters, SPARC, lensing, Solar, and MW Kz variants, then compile a unified report.

Appendix: Commands used
- Clusters:
  python "Paper DensityAccel Variants/cluster_dg_ag_variants.py" \
    --accept external_data/accept_database.dat \
    --results "Paper DensityAccel Variants/results/cluster_hybrid" \
    --images "Paper DensityAccel Variants/images/cluster_hybrid" \
    --gate hybrid --a0 1.93e-7 --rho-c 1e-27 --rho-gamma 1.5 --zeta 1.0 --Dmax 50 \
    --stars-csv external_data/clash_stars.csv

- Ephemeris:
  python "Paper DensityAccel Variants/ephemeris_variants.py" \
    --gate density --rho-c 1e-24 --gamma 1.0 --Dmax 50 --rho-env 1e-21

- SPARC (CSV conversion + accel gate):
  python "Paper DensityAccel Variants/sparc_rotmod_to_csv.py" \
    --src-dir external_data/Rotmod_LTG \
    --out-dir "Paper DensityAccel Variants/sparc_csv" --limit 20

  python "Paper DensityAccel Variants/sparc_variants.py" \
    --sparc-rotmods "Paper DensityAccel Variants/sparc_csv" \
    --galaxies CamB D631-7 \
    --outdir "Paper DensityAccel Variants/results/sparc_accel" \
    --gate accel

- Lensing:
  python "Paper DensityAccel Variants/lensing_variants.py" \
    --gate density --rho-c 1e-27 --gamma 1.0 --Dmax 50 \
    --log10Mstar 11.6 --Re_kpc 8.0 --Sigma_crit_cgs 1.5e9 \
    --out "Paper DensityAccel Variants/results/lensing_density.json"

- MW Kz:
  python "Paper DensityAccel Variants/mw_kz_variants.py" \
    --gate density --rho-c 1e-27 --gamma 1.0 --Dmax 50 \
    --outdir "Paper DensityAccel Variants/results/mw_kz"
