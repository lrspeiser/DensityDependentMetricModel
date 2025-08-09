# SPARC Extragalactic Validation – Actionable TODOs

This checklist tracks the remaining work to move the manuscript from nearly-complete to submission-ready, focused on §8A (SPARC) and related sections. Items are ordered by impact on scientific completeness and reviewer perception.

Note: Check items as they are completed and link artifacts (JSON, CSV, PNG, figures) and commit SHAs.

---

## Tier 1 – Core science gaps (highest priority)

- [ ] 1) Full evidence-mode fits for NGC 3198
  - [ ] Implement evidence mode interfaces
    - [ ] Extend tools/fit_sparc_er_env.py with a --mode evidence that writes JSON with logZ and best-fit params
    - [ ] Provide matched GR (baryons-only), ER (density-aware), and NFW modes
  - [ ] Run NGC 3198 for all three models
    - [ ] GR: record logZ_GR, params
    - [ ] ER: record logZ_ER and best-fit (ρ_c, γ_exp, λ_max, T_0, σ_lnT, w_min, Υ_3.6), σ_floor, T-proxy name
    - [ ] NFW: record logZ_NFW and best-fit (M_200, c) or (V_200, c)
  - [ ] T proxies (ER): run curvature, shear, epicyclic
  - [ ] Deliverables
    - [ ] Fill §8A.4 placeholders for NGC 3198 (evidences, parameters)
    - [ ] Populate NGC 3198 row in ED-SPARC table
    - [ ] Update ED figure caption with actual χ²/dof and evidence values

- [ ] 2) Implement & run NFW baseline for SPARC
  - [ ] Minimal NFW in fitter with priors from abundance matching
  - [ ] Matched evidence run for NGC 3198
  - [ ] Report ΔlogZ(ER−NFW) in §8A.4 and ED table

- [ ] 3) Batch 5–10 galaxy run (SPARC)
  - [ ] Select clean set: NGC 3198, NGC 2403, M33, NGC 5055, NGC 2903, NGC 6946, NGC 2841, UGC 128
  - [ ] For each galaxy, run GR, ER, NFW in evidence mode and write JSON
  - [ ] Aggregate CSV (sparc_batch_summary.csv) with logZs, best-fit params, and T-proxy
  - [ ] Deliverables
    - [ ] Fill Extended Data Table ED-SPARC
    - [ ] Optional: Figure ED1 (stacked residuals GR vs ER)

- [ ] 4) Robustness tests for T proxy
  - [ ] For NGC 3198 and ≥1 other galaxy, run all 3 proxies
  - [ ] Compute ΔlogZ spread across proxies; compile Table ED-T
  - [ ] Text: “ΔlogZ varies by ≤ X across curvature/shear/epicyclic proxies.”

---

## Tier 2 – Completeness & Reviewer Confidence

- [ ] 5) σ_floor and M/L prior sensitivity (NGC 3198)
  - [ ] Re-run ER with σ_floor ∈ {0, 2, 3} km s^{-1}
  - [ ] Re-run with flat Υ_3.6 ∈ [0.3, 0.7] vs Gaussian prior
  - [ ] Deliverables: small Supplement figure/table; note in §8A.4 sanity checks

- [ ] 6) Solar-system tight-screening variant
  - [ ] Implement exponential S_ρ(ρ) = exp[−(ρ/ρ_c)^{γ_exp}]
  - [ ] Re-run MW and NGC 3198 ER fits; compare evidences and key params
  - [ ] Deliverables: Supplementary table; constraints “radar” plot

- [ ] 7) Milky Way ΛCDM baseline
  - [ ] Add NFW halo to MW run with matched sampler settings
  - [ ] Report logZ_NFW(MW) vs ER and GR in §7; optional MW panel for NFW

---

## Tier 3 – Presentation polish

- [ ] 8) Figure consistency
  - [x] Standardize axis labels and legend styling (SPARC NGC 3198)
  - [ ] Ensure figure numbering: Fig. 1 concept; Fig. 2 MW; ED figs for SPARC
  - [ ] Units typography: km s^{-1}, M_⊙ kpc^{-3}
  - [ ] Match colors/line styles across MW and SPARC plots

- [ ] 9) Extended Data & Supplement completion
  - [ ] Table S2: MW star counts per annulus
  - [ ] Table S3: Solar System |ξ − 1| with Cassini/LLR/Ephemeris comparison
  - [ ] Fig. S1: T proxy comparison curves and impact on ξ
  - [ ] Fig. S2: k-fold CV residuals and calibration curves

- [ ] 10) Environment split (optional pre-submission)
  - [ ] Void vs group rotation curve offsets; ER prediction check

---

## Artifacts and paths

- Plots: images/sparc_<galaxy>.png; ED figures under images/
- Per-galaxy JSON: images/sparc_<galaxy>.json (parameters, evidences)
- Batch CSV: sparc_batch_summary.csv (root or data/)
- Paper edits: docs/paper.md (sections §8A.4, ED tables, captions)

## Notes

- NGC 3198 current plot: images/sparc_ngc3198_fit.png (labels standardized)
- When implementing evidence mode and NFW, prefer non-interactive, reproducible CLI entry points and write all outputs with checksums.

