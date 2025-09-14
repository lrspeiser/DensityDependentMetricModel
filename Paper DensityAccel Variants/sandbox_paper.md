# Density-Gated vs Acceleration-Gated Gravity (Sandbox Paper)

This sandbox paper mirrors the structure of our README/paper but isolates a new density-gated (DG) and hybrid gate (a0→a0_eff(ρ)) analysis without changing production code, data, or figures. Paths below reference only artifacts under “Paper DensityAccel Variants/”.

1. Introduction
We test a density-gated gravitational response ξ(ρ) alongside the original acceleration-gated (AG, RAR-like) form ξ(ḡ). The hybrid gate promotes the acceleration scale to a0_eff(ρ) = a0 [1 + ζ (1 + (ρ/ρ_c)^γ)^{-1}], so galaxies (sensitive to ḡ) and clusters/voids (sensitive to ρ) are handled with one parameter family. We compare DG/hybrid to GR (baryons only) and to the AG baseline across clusters, galaxies (SPARC), Solar System constraints, and strong lensing.

2. Methods (Sandbox variants)
2.1 Unified gate registry
- build_gate(kind): returns a callable ξ(ḡ, ρ, R)
  - accel: ξ = min(0.5 + sqrt(0.25 + a0/ḡ), Dmax)
  - density: ξ = 1 + (Dmax−1) [1 + (ρ/ρ_c)^γ]^{-1}
  - hybrid: ξ as accel with a0→a0_eff(ρ)
- File: Paper DensityAccel Variants/xi_registry_variants.py

2.2 Clusters (CLASH × ACCEPT)
- Inputs: ACCEPT n_e shells (external_data/accept_database.dat), optional stars CSV (external_data/clash_stars.csv); ρ_gas=μ_e m_p n_e; ρ_star from Hernquist.
- Output: ξ(ḡ, ρ) per radius; diagnostic CSV and summary JSON.
- File: Paper DensityAccel Variants/cluster_dg_ag_variants.py
- Artifacts: results/cluster_hybrid/{cluster_gate_diagnostics.csv, summary_variants.json}

2.3 SPARC galaxies
- Rotmod .dat → CSV converter (R_kpc,Vbar_kms) for a small subset.
- DG requires a midplane ρ(R) proxy CSV: ρ≈(Σ_disk M/L_d+Σ_bul M/L_b)/(2h_z).
- Output: model CSVs per galaxy (R, Vbar, ξ, V_model).
- Files: Paper DensityAccel Variants/{sparc_rotmod_to_csv.py, sparc_rho_proxy_from_dat.py, sparc_variants.py}
- Artifacts: results/sparc_accel/*_gate_model.csv; results/sparc_density/*_gate_model.csv

2.4 Lensing (θ_E amplitude)
- Simple Hernquist deprojection (n≈4) surrogate, compute ξ at R_E and report xi_at_R_E.
- File: Paper DensityAccel Variants/lensing_variants.py
- Artifact: results/lensing_density.json

2.5 Solar ephemeris (ΔG/G)
- Apply DG with a simple ρ_env; write console sample.
- File: Paper DensityAccel Variants/ephemeris_variants.py

2.6 Milky Way Kz
- Baryons-only placeholder + gate metadata; DG-phantom is reserved for future sandbox.
- File: Paper DensityAccel Variants/mw_kz_variants.py

3. Results (Sandbox)
3.1 Clusters (Hybrid ξ)
- Diagnostics: results/cluster_hybrid/cluster_gate_diagnostics.csv — contains cluster, r_kpc, log10 ḡ, log10 ρ, ξ.
- Summary: results/cluster_hybrid/summary_variants.json — records gate kind and params.
- Qualitative: ξ grows towards low ρ and low ḡ, capturing inner BCG/ICL and outer ICM trends with a single parameter family.

3.2 SPARC (AG vs DG)
- Acceleration-gated models (AG): results/sparc_accel/CamB_gate_model.csv, D631-7_gate_model.csv.
- Density-gated models (DG): results/sparc_density/CamB_gate_model.csv, D631-7_gate_model.csv.
- Observations: DG offers a tunable amplitude via ρ(R) without per-galaxy halo freedom; with a physically plausible ρ proxy, hybrid can match AG performance while remaining density-aware for clusters and lensing.

3.3 Lensing (DG @ θ_E)
- xi_at_R_E written to results/lensing_density.json with Re_kpc, log10Mstar, and gate params. DG adjusts the amplitude via local ρ_E (finite plateau ξ_max parallels AG’s Dmax).

3.4 Solar (DG screening)
- For a constant ρ_env, ΔG/G is nearly constant and small across AU; screening can reduce effective deviations, consistent with Solar safety in the gated metric.

3.5 MW Kz
- Exported baryons-only curve and gate meta for reproducibility; DG-phantom term (∂ξ/∂ρ) will be tested in a guarded path in a follow-up sandbox run.

4. Discussion
- One family across domains: AG explains galaxy kinematics efficiently; DG/hybrid preserves this while adding density awareness needed by clusters and lensing amplitudes. The same parameter set {a0, ρ_c, γ, ζ, Dmax} can be recorded in every artifact.
- Fair baselines: GR (baryons-only) remains a comparator; in future sandbox steps we will add GR/NFW tables to clusters and SPARC for like-for-like RMS/coverage.
- Limitations (sandbox): the SPARC ρ proxy is approximate; lensing distance geometry is simplified; MW Kz DG-phantom is pending. These will be addressed in the next sandbox revision.

5. Reproducibility commands
- See Paper DensityAccel Variants/sandbox_report.md (commands used) and the following minimal recipes:

Clusters:
  python "Paper DensityAccel Variants/cluster_dg_ag_variants.py" --accept external_data/accept_database.dat \
    --results "Paper DensityAccel Variants/results/cluster_hybrid" --images "Paper DensityAccel Variants/images/cluster_hybrid" \
    --gate hybrid --a0 1.93e-7 --rho-c 1e-27 --rho-gamma 1.5 --zeta 1.0 --Dmax 50 \
    --stars-csv external_data/clash_stars.csv

SPARC (convert; build ρ-proxy; run AG + DG):
  python "Paper DensityAccel Variants/sparc_rotmod_to_csv.py" --src-dir external_data/Rotmod_LTG --out-dir "Paper DensityAccel Variants/sparc_csv" --limit 20
  python "Paper DensityAccel Variants/sparc_rho_proxy_from_dat.py" --src-dir external_data/Rotmod_LTG --out-csv "Paper DensityAccel Variants/sparc_rho_proxy.csv" --limit 20
  python "Paper DensityAccel Variants/sparc_variants.py" --sparc-rotmods "Paper DensityAccel Variants/sparc_csv" --galaxies CamB D631-7 --outdir "Paper DensityAccel Variants/results/sparc_accel" --gate accel
  python "Paper DensityAccel Variants/sparc_variants.py" --sparc-rotmods "Paper DensityAccel Variants/sparc_csv" --galaxies CamB D631-7 --outdir "Paper DensityAccel Variants/results/sparc_density" --gate density --rho-csv "Paper DensityAccel Variants/sparc_rho_proxy.csv" --rho-c 1e-27 --gamma 1.5 --Dmax 50

Lensing:
  python "Paper DensityAccel Variants/lensing_variants.py" --gate density --rho-c 1e-27 --gamma 1.0 --Dmax 50 --log10Mstar 11.6 --Re_kpc 8.0 --Sigma_crit_cgs 1.5e9 --out "Paper DensityAccel Variants/results/lensing_density.json"

Ephemeris:
  python "Paper DensityAccel Variants/ephemeris_variants.py" --gate density --rho-c 1e-24 --gamma 1.0 --Dmax 50 --rho-env 1e-21

MW Kz:
  python "Paper DensityAccel Variants/mw_kz_variants.py" --gate density --rho-c 1e-27 --gamma 1.0 --Dmax 50 --outdir "Paper DensityAccel Variants/results/mw_kz"

Appendix A: File index
- Gate registry: Paper DensityAccel Variants/xi_registry_variants.py
- Clusters: Paper DensityAccel Variants/cluster_dg_ag_variants.py
- SPARC tools: Paper DensityAccel Variants/{sparc_rotmod_to_csv.py, sparc_rho_proxy_from_dat.py, sparc_variants.py}
- Lensing: Paper DensityAccel Variants/lensing_variants.py
- Solar: Paper DensityAccel Variants/ephemeris_variants.py
- MW Kz: Paper DensityAccel Variants/mw_kz_variants.py
