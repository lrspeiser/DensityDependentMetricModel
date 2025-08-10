# SPARC ER Replication Guide

Short answer
- Gas reconstruction is fixed and physically reasonable now (truncated exponential with correct units). ER is not competitively fit yet for NGC 3198, but we’re close. With a few concrete tweaks and sanity checks (sigma floor parity, slightly broader λ_max, lower w_min, robust tidal normalization, and gas truncation control), we expect χ²/dof to drop by 1–2 orders to an academically defensible range.

What’s included here
- Exact commands and settings to reproduce our current runs on NGC 3198
- How to set up data, gas reconstruction, and baselines
- A checklist of next changes with rationale to reach publication-grade results

Prereqs
- Python 3.10+
- Optional SciPy (for faster NFW fitting); falls back to a coarse grid otherwise
- This repository at the commit containing this document

1) Data setup
- Use the robust fetcher with Zenodo mirrors wired in (already committed):
  - scripts/fetch_sparc_hirad_sb_v2.py supports direct Zenodo URLs for:
    - MasterSheet_SPARC.mrt
    - MassModels_SPARC.mrt
    - Rotmod_LTG.zip
- Recommended destination: external_data/Rotmod_LTG
- After download, ensure external_data/Rotmod_LTG contains:
  - NGC3198_rotmod.dat (and others)
  - MasterSheet_SPARC.csv

2) Gas reconstruction (what’s active now)
- Default: truncated exponential (“Option A, truncated”) that enforces Σ(RHI)=1 Msun/pc^2 and total M_gas = 1.33 M_HI via the truncated integral:
  - Σ(R) = Σ0 exp(−R/Rd), 0 ≤ R ≤ Rmax
  - Σ0 = exp(RHI/Rd)
  - M_gas = 2π Σ0 Rd^2 [1 − e^{−x}(1+x)] × (kpc^2→pc^2 factor), x=Rmax/Rd
- Rmax mode:
  - Default is Rmax = RHI (metadata). For now, you can override with env vars:
    - SPARC_GAS_RMAX_MODE=KRD and SPARC_GAS_KRD=3.0 to use Rmax = 3 Rd
- Fallback: Option B shape-from-V_gas normalized to M_gas if metadata is missing.

3) Reproduce current NGC 3198 runs
A. ER fit (current implementation)
- Command (PowerShell):
  - python tools/fit_sparc_er_env.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --mode fit --model er
- Expected gas reconstruction log snippets:
  - [gas_profile] Truncated-A: Rd≈11.79 kpc, Σ0≈20.6 Msun/pc^2, Rmax≈35.66 kpc (with Rmax=RHI)
  - Max gas ρ_mid ~ 1e8 Msun/kpc^3
- Current outcome:
  - χ²/dof ≈ 5792.38/37 ≈ 156.55.
  - This is intentionally “pre-tweak” and serves as the baseline to improve.

B. GR/NFW baselines (with sigma floor parity)
- Command (PowerShell):
  - python tools/fit_sparc_baselines.py --galaxy NGC3198 --sparc-dir external_data/Rotmod_LTG --sigma-floor 5.0
- Expected results:
  - GR(baryons-only): χ²/dof ~ 44
  - NFW: χ²/dof ~ 1.8 with sensible V200, c

4) Make the comparison apples-to-apples (next changes to apply)
- Sigma floor parity: add --sigma-floor to the ER tool and include it in the likelihood (target 5 km/s to match baselines).
- Fit M/L like the baselines: allow Υ_disk, Υ_bulge to float with Gaussian priors (e.g., 0.5±0.1 and 0.7±0.1). Locking them can force ER to compensate elsewhere.

5) Two ER hyperprior adjustments
- Broaden λ_max prior to [0, 10] for extragalactic runs, so ER can actually “boost” where needed.
- Reduce w_min floor to [0, 0.05] (not 0.1). A high floor flattens W(T) and makes ER compensate via other parameters.

6) Gas truncation flags (to be added)
- Replace env vars with explicit flags in the ER CLI:
  - --gas-truncation RHI|KRD
  - --gas-krd 3.0
- After adding, test Rmax = RHI vs Rmax = 3 Rd and pick whichever yields lower χ²/dof without pushing Σ0 to unrealistic values.

Sanity targets (NGC 3198):
- Σ0 ~ 10^2–10^3 Msun/pc^2, ρ_mid,max ~ 10^8–10^11 Msun/kpc^3, parameters off bounds.

7) Tidal proxy and normalization (to be added)
- Normalize T robustly: median+MAD (or percentile scaling). Add --tidal-norm robust.
- Expose tidal proxies and try all: curvature | shear | epicyclic. Add --T-proxy <choice>.
- Narrow the starting width: allow σ_lnT ∈ [0.5, 2.0] but start tighter (e.g., 1.0) so the window is not overly broad.

8) Nuisance parity with baselines (to be added)
- Add Gaussian priors for distance and inclination from the MasterSheet and marginalize them in the ER likelihood.
- Ensure any masking (beam smearing, warps) matches whatever you use for NFW.

9) Quick bug traps
- Log min/max of ξ(R) over fitted radii; if ξ≈1 everywhere or spikes only at the center, ER can’t address the outer flat tail.
- Verify degrees of freedom and that σ-floor is applied identically to ER and baselines.
- Print a short hash of Σ_gas(R) at start so cache changes are guaranteed when flipping gas options.

10) A helpful initialization
- Start near: log10 ρ_c ≈ 15.0, γ_exp ≈ 3.0, λ_max ≈ 6.0, ln T0 ≈ 0.0, σ_lnT ≈ 1.0, w_min ≈ 0.01.
- This avoids bad plateaus and speeds convergence.

11) What “good enough” looks like before evidence runs
- Aim for ER χ²/dof ≤ 3–5 on NGC 3198 with σ-floor=5 km/s.
- Parameters not pegged to bounds, and ξ(R) shows a modest outer-disk boost (≈1.5–4), not a central spike.

Appendix: Commands (PowerShell)
- ER (current, Rmax=RHI):
  python tools/fit_sparc_er_env.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --mode fit --model er
- ER (override with Rmax = k Rd via env vars for now):
  $env:SPARC_GAS_RMAX_MODE='KRD'; $env:SPARC_GAS_KRD='3.0'; python tools/fit_sparc_er_env.py --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG --mode fit --model er
- Baselines with σ-floor:
  python tools/fit_sparc_baselines.py --galaxy NGC3198 --sparc-dir external_data/Rotmod_LTG --sigma-floor 5.0

Current best NGC 3198 (as of this commit)
- Command:
  python tools/fit_sparc_er_env.py \
    --galaxy_id NGC3198 --sparc_dir external_data/Rotmod_LTG \
    --mode fit --model er \
    --sigma-floor 5.0 \
    --gas-truncation KRD --gas-krd 3.0 \
    --T-proxy epicyclic --tidal-norm robust \
    --prior-lambda-max 10.0 --prior-wmin-max 0.05 \
    --fit-ml disk bulge
- Outcome:
  - chi2/dof ≈ 42.30/35 ≈ 1.21
  - Params: log10_rho_c≈14.0, gamma_exp≈1.0, lambda_max≈9.10, lnT0≈-0.233, sigma_lnT≈2.0, w_min=0.05, ups_disk≈0.10, ups_bul≈0.70
  - Gas (KRD=3) reconstruction log (example): Rd≈228 kpc, Σ0≈1.17 Msun/pc^2, Rmax≈683 kpc; Max gas ρ_mid≈5.8e6 Msun/kpc^3
- Notes:
  - Tidal proxy epicyclic performed best here; curvature was worse (~33 chi2/dof).
  - M/L moved to the prior/allowed boundary for disk (0.10). You may tighten the prior or revisit SB calibration if desired.

Contact
- Please open an issue or PR if any step is unclear or you want to contribute improvements to the ER CLI flags.

