# Ranked galaxy-lens shortlist for forward predictions (baryons → lensing)

This list prioritizes systems where the baryonic surface density can be mapped well enough to make a forward prediction of the lensing signal (Einstein radius, arc morphology, ΔΣ), allowing a direct test of the density-gated gravity model without free lensing scalars.

A. Baryons-dominated inside the ring (cleanest)

1) ESO 325-G004 (E325) — z≈0.034 (nearby elliptical, strong lens)
- Why: Local, baryon-dominated within the Einstein radius; excellent HST imaging and VLT/MUSE kinematics.
- Data: HST ring imaging; spatially resolved stellar kinematics for stellar M/L (allow a radial gradient).
- Compare: Using a stellar mass map, project Σb(θ), predict θE and azimuthal shear; benchmark vs. published ring and PPN γ ∼ 1 (extragalactic GR test context).
- Action: Treat as the calibration case. Fill docs/lensing_targets.csv with measured log10M⋆ and Re.

2) SWELLS: SDSS J2141-0001 (spiral lens)
- Why: Tests disk+bulge (+gas) in a case with rich data.
- Data: HST/Keck imaging; gas and stellar kinematics; joint lensing–dynamics decompositions.
- Compare: θE, arc morphology, and shear profile using baryon map. Contrast with GR+baryons and with RAR-gate prediction.
- Action: Include a gas component in extended analyses; for the baseline CSV provide stellar M⋆, Re for spherical runs.

3) Q2237+0305 (Einstein Cross)
- Why: Very nearby spiral lens; rich rotation and stellar-kinematics literature.
- Caveat: Microlensing affects flux ratios. Emphasize image positions / ring geometry rather than magnifications.
- Action: Supply stellar M⋆ and Re in the CSV; use metric-only predictions for θE; defer full non-spherical modeling to a follow-up.

B. Recent JWST kinematics (time-delay lenses)

4) RX J1131−1231
- What’s new: JWST/NIRSpec IFU stellar kinematics tighten stellar M/L and dynamical modeling.
- Why useful: Better baryon maps enable forward predictions under a single theory for imaging + time delays + kinematics.
- Action: Fill CSV with updated stellar M⋆, Re; consider time-delay predictions in an extended pipeline.

C. Cluster-scale stress test (bonus)

Abell 2744 (Pandora)
- What exists: Ultra-deep JWST imaging; public strong+weak-lensing models; weak-lensing shear measurements.
- Note: Baryons are hot gas dominated; requires X-ray/SZ maps. This is a stretch test for density-gated models.
- Action: Not in the galaxy CSV. Use a separate cluster workflow (future work).

How to use in this repo

1) Populate docs/lensing_targets.csv with measured values for log10M_star and Re_kpc (and optionally n_sersic, profile).
2) Run the orchestrator in metric-only mode to produce forward predictions and ΔΣ stacks (no lensing scalars):

```bash
python scripts/next_steps_from_run.py \
  --run-dir runs/<your_run> \
  --sparc-dir external_data/Rotmod_LTG \
  --lensing-sample-csv docs/lensing_targets.csv \
  --metric-lensing-only --density-profile sersic --write-ppn-table
```

Outputs:
- Table: results/next_steps/<run>/lensing_metric_table.csv (includes observed θE if provided, GR θE, RAR θE)
- Per-lens profiles: results/next_steps/<run>/lensing_metric_profiles/<lens>_profiles.csv (R, Σ, ⟨Σ⟩, ΔΣ, Σcr)
- ΔΣ stack: results/next_steps/<run>/lensing_metric_stack.csv and images/next_steps/<run>/lensing_metric_stack.png

Notes on theory mapping (what’s being tested)
- In weak field, we take Φ=Ψ (no anisotropic stress) and cT=1, so lensing depends on Φ+Ψ=2Φ.
- The effective mass that sources Φ is baryons plus the “phantom” component implied by the same xi(...) used for dynamics.
- This ensures a single-theory prediction across dynamics and lensing without lensing-only scalars.
