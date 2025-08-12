# Preregistered modeling choices

This document captures the modeling choices fixed prior to cohort analysis to ensure parity and guard against cherry-picking.

- Tidal proxy: epicyclic (κ-based), robust normalization; frozen across cohort unless otherwise noted.
- Velocity floor (σ_floor): 5 km s^-1 baseline for GR, NFW, and TFR, added in quadrature to σ_obs. Deviations must be noted per run.
- Distance/inclination priors: SPARC MasterSheet values where available; Gaussian priors applied equivalently across GR/NFW/TFR.
- Mass-to-light priors (Υ_3.6): Gaussian 0.45 ± 0.10 M_⊙/L_⊙ (disk) and 0.70 ± 0.10 (bulge), with clamps [0.3–0.8]/[0.5–1.0] unless otherwise specified.
- Gas treatment: R_HI truncation with Σ(R_HI) ≈ 1 M_⊙ pc^-2; fallback to V_gas-shaped reconstruction only for plotting when required (never used for evidence).
- TFR hyperpriors: log10 ρ_c ∈ [14,17], γ_exp ∈ [1,5], λ_max ∈ [0,6], ln T_0 ∈ [−1,1], σ_lnT ∈ [0.3,2.0], w_min ∈ [0,0.1].
- NFW priors: broad on (V200, c) or (M200, c); optional Gaussian priors disabled by default unless explicitly recorded.
- Dynesty controls (paper-grade): nlive=1000 (per-galaxy triad) or higher as needed; maxcall=200000; dlogz_target=0.01; sample=rslice; bound=multi; seed=42 (unless varied for sensitivity).
- Masks: Identical masks across GR/NFW/TFR per galaxy.

Recording
- Every run emits a JSON sidecar with: model, priors, dynesty controls, σ_floor, tidal proxy, tidal normalization, D/i priors, and logZ ± err.
- Cohort aggregation reads these JSONs to build docs/ED-SPARC.md and compute ΔlogZ(TFR−GR), ΔlogZ(TFR−NFW) medians and IQRs, and the fraction with ΔlogZ>10.

Versioning
- This preregistration file is referenced in Methods and should be updated only with explicit change logs (append below).

Change log
- v1 (pre-submission): initial preregistration.
