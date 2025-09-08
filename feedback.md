Great question—and an exciting submission. Wearing my Nature Physics editor hat, here’s a concrete, prioritized checklist of what we would still need to see to seriously consider an editorially handled article about Density‑Gated Gravity (DGG) / “RAR‑gated gravity.” I’ve broken this into (A) indispensable technical results that must be in the manuscript, (B) likely reviewer requests that you can pre‑empt, and (C) presentation & policy requirements specific to Nature Portfolio.

Note: An extended editor-style review with a traffic-light status snapshot, concrete upgrades A–I, and a definition-of-done checklist is available in feedback_editor_review.md.

---

## A) Indispensable technical results (must be in the paper)

1. A relativistic completion with lensing from first principles (no ad‑hoc lensing scale factors).
   At minimum, specify the action or metric theory, show the weak‑field limit, and derive the two potentials (Φ, Ψ). Report the PPN parameters (γ, β, preferred‑frame parameters if relevant), and the GW propagation speed. Then compute light deflection (galaxy–galaxy lensing ΔΣ(R), strong‑lens θ_E, time delays) from the same parameters used for dynamics—no independent α‑scalings in lensing. A compact benchmark set would be:

   - Predicted ΔΣ(R) for SDSS/HSC‑like samples (stacked galaxy–galaxy lensing) and an E_G‑style gravity test curve, contrasted with GR bands. ([arXiv][1], [A&A Astronomy and Astrophysics][2])
   - A small table of well‑measured strong lenses (M*, R_e, z_l, z_s) with predicted vs observed θ_E from your metric.
   - State explicitly that c_GW = c in your model (or quantify any residual) to be consistent with GW170817/GRB170817A constraints on modified gravity. ([Physical Review Link Manager][3], [Astrophysics Data System][4])
     Why this matters: Without a relativistic completion, the theory cannot be vetted against lensing or constrained by PPN/GW tests—both are decisive for publication at this level. For inspiration on relativistic MOND‑like frameworks, see Skordis & Złośnik (PRL 2021). ([Physical Review Link Manager][5])

2. A hierarchical, sample‑scale test of a single a0.
   Replace “per‑galaxy a0” with a hierarchical Bayesian analysis over a large, quality‑controlled SPARC/BIG‑SPARC subset (≥100 galaxies), with:

   - A global a0 hyper‑prior and hyper‑posterior p(a0 | all galaxies); report the posterior mean/median and scatter (intrinsic vs. observational).
   - Nuisance modeling of stellar Υ_* (IMF choices), distances, inclinations, beam smearing, and non‑circular motions; propagate them into the RAR plane.
   - Model comparison: Δlog Z or Bayes factors for GR+baryons, GR+baryons+NFW (2+ params/galaxy), and DGG (global a0)—shown as distributions across the sample.
   - BTFR comes “for free,” but include a slope/intercept/intrinsic‑scatter panel under identical selection.
     Why this matters: Your claim of predictive rigidity hinges on a universal a0. Do the hierarchical inference and demonstrate it quantitatively on the same galaxies observers care about. Cite SPARC/BIG‑SPARC as the data backbone. ([Astrophysics Data System][6], [arXiv][7])

3. Local‑universe consistency: full PPN suite, Solar‑System & wide‑binaries.
   You already reference Cassini’s |γ−1| ≲ 2×10^−5. Please derive the DGG PPN parameters and plot the predicted fractional deviation (e.g., Ξ(r)) at 1–30 AU with parameter posteriors from your galaxy fits; add perihelion advance / Shapiro delay where relevant. Then confront wide‑binary constraints (Gaia DR3), which are now a sensitive test of low‑acceleration modifications; show your model’s binary‑orbit predictions under the Galactic external field and compare to recent analyses that disfavour MOND‑like boosts at ~10^−10 m s^−2 accelerations. ([Astrophysics Data System][8], [arXiv][9])
   Why this matters: If your gating/screening truly suppresses deviations locally, you should match Cassini and wide‑binary statistics—two common pressure points for MOND‑like theories.

4. Milky Way vertical force & local dynamical checks.
   Go beyond the radial rotation curve: compute the vertical force K_z(R_0, z) and the local surface density Σ_1.1kpc implied by DGG, and compare to classical determinations (e.g., Bovy & Rix; McMillan). This guards against “fixing” the radial curve while breaking the vertical equilibrium. ([Astrophysics Data System][10], [arXiv][11])

5. Explicit positioning vs. ΛCDM hydrodynamical results on the RAR.
   Several groups have shown that RAR‑like trends can arise in ΛCDM simulations with feedback (EAGLE/NIHAO), often with a larger characteristic acceleration scale and subtle differences at very low accelerations. Add a figure that overlays your RAR prediction, SPARC data, and at least one ΛCDM simulation band, and state which residual pattern (slope at low‑g_bar, intrinsic scatter vs. mass, outer‑disk falloffs) decisively separates DGG from ΛCDM. ([Physical Review Link Manager][12], [arXiv][13])

6. Cosmology “sanity checks.”
   You don’t need a full Boltzmann‑code treatment to clear first editorial hurdles, but you must show that the relativistic completion can (in principle) be embedded in a cosmological background without violating: (i) background expansion H(z), (ii) linear growth fσ8, and (iii) CMB acoustic peak structure—or clearly state a minimal dark sector (e.g., neutrino mass) needed and why it doesn’t re‑introduce the problem you aim to solve. A one‑page theory appendix plus a linear‑growth sketch will help.

---

## B) Strongly suggested additions (pre‑empt likely referee requests)

- Drop any lensing‑only fudge factors. The “α_lens_ph” pilot is fine for internal scoping, but it cannot appear as a free dial in a Nature Physics paper. Replace it with predictions from your metric (Φ+Ψ) or remove it entirely.

- Clear, falsifiable predictions. Identify near‑term discriminants where DGG and ΛCDM give opposite trends (e.g., gentle outer‑disk declines vs. perfectly flat curves in specific high‑SB systems; ΔΣ(R) at 30–300 kpc for massive spirals at fixed stellar mass). List the surveys (HSC, DES, LSST Y1–Y3) that can test these. ([Physical Review Link Manager][14])

- Dwarf spheroidals & satellites (outline only). Even two worked examples (e.g., Fornax, Sculptor) with Jeans modeling under DGG, using the same a0, will reassure referees that you’ve thought about anisotropy and tidal effects.

- Galaxy clusters (one figure). Show why your saturation (“plateau”) avoids over‑lensing in cores yet still leaves a deficit at R ~ Mpc, and quantify any residual mass required (e.g., Σ m_ν). One figure and a plain statement will forestall “what about clusters?” pushback.

---

## C) Nature Physics presentation & policy requirements

- Data/Code availability that enables re‑analysis. Nature Portfolio requires prompt availability of data, code, and protocols sufficient for replication. Deposit tagged releases of your code and all figure source data (CSV/Parquet) in an archival repository (e.g., Zenodo with a DOI) and include environment files and exact commands. Each figure in the paper should have a corresponding “Source Data” file. ([Nature][15])

- Statistics & methods clarity. Provide:
  - Prior choices and prior‑sensitivity plots.
  - Hierarchical model specification and sampling diagnostics.
  - Definition of goodness‑of‑fit metrics, Δlog Z computation details, and how you treat systematics (inclination, distances, Υ_*, beam smearing).
  - A reproducibility script that regenerates the main plots from raw inputs.

- Image integrity and figure ethics. Ensure that composite panels are clearly labeled; no non‑linear contrast tricks; and that all error bands represent well‑defined credible intervals. (See Nature’s image and reporting standards.) ([Nature][16])

- Scope & claims. Calibrate the abstract/title to match what you’ve demonstrated. If cosmology is not yet solved, phrase it as “a galaxy‑scale alternative to dark halos” rather than “an alternative to dark matter,” and reserve the stronger claim for a follow‑up once lensing/cosmology are fully closed.

---

## A few surgical comments on your current draft

- Relativistic/lensing gap: The text introduces a physically motivated gating but then moves to a “lensing pilot” with a global α_lens_ph. That undermines the central selling point (predictive rigidity). Replace with metric‑based lensing predictions, even if initially for a small, carefully chosen lens sample.

- Per‑galaxy a0 variability: You report a0 spanning ~3×10^−11–1.3×10^−10 m s^−2. Convert this to a hierarchical posterior and show that a single a0 with small intrinsic scatter is adequate (or quantify the scatter). This is the figure most readers will remember.

- Local tests: Keep the Cassini panel, but add a PPN table and a one‑page appendix deriving PPN coefficients from your relativistic completion. Cassini’s constraint on γ is the headline number; please cite the original Nature paper. Wide‑binary constraints should be addressed head‑on in the text, with your model’s prediction under the external field. ([Astrophysics Data System][8], [arXiv][9])

- Positioning vs ΛCDM RAR: Include one panel that overlays SPARC, your DGG curve, and the EAGLE/NIHAO bands; call out the differences at low acceleration and the implied tests. ([Physical Review Link Manager][12], [arXiv][17])

- Milky Way K_z: Add one panel comparing your predicted K_z(R_0, z) or Σ_1.1 to dynamical measurements; this removes an easy objection. ([Astrophysics Data System][10])

---

## “Figure pack” I’d expect in a revised submission

1. RAR master panel (SPARC/BIG‑SPARC points, hierarchical fit, DGG curve ± intrinsic scatter band, ΛCDM bands). ([Astrophysics Data System][6], [arXiv][7], [Physical Review Link Manager][12])
2. Hierarchical a0 corner plot with hyper‑posterior and posterior‑predictive checks.
3. Milky Way: radial v_c(r) and vertical K_z / Σ_1.1 comparison. ([Astrophysics Data System][10])
4. PPN & Solar System: table of PPN parameters + |Ξ(r)| vs AU with Cassini band. ([Astrophysics Data System][8])
5. Wide binaries: predicted vs observed velocity ratio/statistic under the external field. ([arXiv][9])
6. Lensing: (i) ΔΣ(R) stacks vs data bands; (ii) small table of θ_E predictions for 3–5 lenses. ([arXiv][1])
7. Model comparison: Δlog Z distributions for GR+baryons, GR+NFW, and DGG across the sample.

---

## Why this bar is high (and how your paper can clear it)

Nature Physics prioritizes results of broad interest that withstand cross‑domain tests (dynamics and lensing and local gravity), are strictly reproducible, and make falsifiable predictions. Your theory’s central promise—one scale a0, minimal parameters, environment‑gated behavior—is exactly the sort of unifying idea our readers value. To reach acceptance, the paper has to show that (i) the same parameter set explains rotation curves and lensing, (ii) it is consistent with Solar‑System/PPN/GW constraints, and (iii) it beats (or at least matches) ΛCDM fits under fair model comparison on large, public galaxy samples. ([Physical Review Link Manager][18], [Astrophysics Data System][6])

---

### Pointers & citations you referenced or should add explicitly

- RAR discovery & scatter (McGaugh, Lelli, Schombert, PRL 2016). ([Physical Review Link Manager][18], [arXiv][19])
- SPARC / BIG‑SPARC databases for rotation curves. ([Astrophysics Data System][6], [arXiv][7])
- PPN Cassini light‑bending / Shapiro delay bound on γ. ([Astrophysics Data System][8])
- GW170817 constraints on modified gravity and c_GW. ([Physical Review Link Manager][3])
- Relativistic MOND‑like example (Skordis & Złośnik 2021) for context. ([Physical Review Link Manager][5])
- Wide‑binary constraints from Gaia DR3 analyses. ([arXiv][9])
- RAR in ΛCDM simulations to frame your novelty. ([Physical Review Link Manager][12], [arXiv][17])
- Nature reporting & code/data availability policies. ([Nature][15])

---

## Final, practical to‑do list

- [ ] Write down the relativistic theory (action, fields, screening), compute PPN & c_GW, and derive lensing from Φ+Ψ.
- [ ] Replace “lensing pilot α” with metric predictions; add ΔΣ(R) + θ_E figures.
- [x] Run a hierarchical a0 inference on a large SPARC subset; publish Δlog Z histograms.
  Completed (2025-09-08): Implemented nuisance‑marginalized per‑galaxy a0 likelihoods (ln M/L priors with σ=0.15; fractional observational inflation f=0.05) and ran a hierarchical ln a0 posterior over a SPARC selection (N≈118; min_npts≥8, min_rmax≥6 kpc, Q≤2). Produced Δlog Z (DGG−GR) per-galaxy by integrating ∫ L(a0)π(a0)d(ln a0); see results/next_steps/rar_plateau_mw_full/hierarchical_dgg_evidence.csv and summary JSON there. Commands added to README; figures and Source Data paths are listed.
- [ ] Add MW K_z / Σ_1.1 and a short wide‑binary section with your model’s predictions.
- [ ] Archive code + exact figure source data with a DOI and list the single‑command repro path. ([Nature][15])
- [ ] Temper claims about cosmology (or add a brief linear‑growth/CMB feasibility note).

If you can deliver the package above in a single, tightly argued manuscript—with clean, reproducible figures—I’d be comfortable advancing it to external review.

---

Repository findings and proposed actions

A) Relativistic completion, PPN, GW, and lensing without α_lens_ph
- Findings:
  - The repo has an effective weak-field lensing prescription (docs/lensing.md) introducing Φ_eff and Ψ_eff with a disformal-type coupling via φ_env = 1/2 ln ξ. There are CLI tools for lensing pilots (tools/lensing_predict.py, tools/lensing_slacs_examples.py) and an orchestrator step (scripts/next_steps_from_run.py) that currently supports a lensing-only scalar α_lens_ph (see run_lensing_rar_from_csv with alpha_lens_ph and the resulting lensing_rar_table.csv). Multiple READMEs and results indicate α_lens_ph usage for pilots.
  - PPN/Cassini: There is extensive Cassini machinery across code and docs (validation/cassini.py, docs/cassini.md, docs/README.md, docs/RHO_C_EXPLANATION.md, runners/run_dynesty*.py), including checks that |γ−1| < 2.3e−5 around Saturn and notes on Solar-System screening ensuring ξ→1 locally. Mercury perihelion and Shapiro delay are referenced in docs/tests.
  - GW/c_GW is stated qualitatively (docs/README.md, docs/theory_hygiene.md) but no explicit c_GW calculation is in code.
- Gaps vs checklist:
  - No full covariant action/metric written down; lensing is treated in a weak-field EFT/phenomenology style.
  - α_lens_ph is present in pilots, which Nature Physics would not accept in the manuscript.
  - No PPN table (γ, β, preferred-frame) with derivations tied to the same parameters used for galaxies; c_GW not derived.
- Proposed actions:
  1) Formalize a minimal relativistic completion in an appendix (scalar–tensor with disformal term) that yields Φ, Ψ and predicts lensing via Φ+Ψ = (Φ_b+Ψ_b) + (a_env+b_env) φ_env, with screening. Implement a code path that computes ΔΣ(R) and θ_E directly from φ_env and fitted parameters (remove α_lens_ph from manuscript figures; keep only in internal tools).
  2) Add a PPN derivation to docs/theory_hygiene.md or paper Methods, and emit a programmatic PPN table (γ, β, α_1, α_2 if applicable) with posteriors from galaxy fits. Code hooks: add a module that maps fitted parameters to PPN at Solar densities and produces a JSON/CSV table alongside figures.
  3) State and, if possible, sketch c_GW = c from the completion (e.g., constraints on disformal scales to avoid altering tensor speed), referencing GW170817 bounds; add a short code assertion that flags any parameter combos that would violate c_T=1 in the EFT limit.

B) Hierarchical single a0 across SPARC/BIG‑SPARC
- Findings:
  - The code fits Milky Way and SPARC galaxies with various ξ(ρ, T) forms and uses dynesty with Bayes evidence logging. There is sample-level machinery (runners/dynesty_latest in WARP docs; data_loaders/sparc_data_loader.py) and SPARC assets (external_data/Rotmod_LTG/...). The README and paper docs discuss a0-like parameters (a0_m_s2 in xi_rar_* functions).
  - A true hierarchical hyper-posterior over a0 is not yet implemented; per-galaxy runs and “global a0 scan” are mentioned but not as a hierarchical model with shared hyperprior.
- Proposed actions:
  1) Implement a hierarchical model: a global a0 ~ LogNormal(μ, σ), with per-galaxy a0_i drawn from the hyperprior; sample {a0_i} marginalized via empirical Bayes or full hierarchical sampling. Aggregate Δlog Z for GR, GR+NFW, and DGG (global a0) across ≥100 SPARC galaxies (Q≤2, quality cuts), and produce hyper-posteriors p(a0 | all).
  2) Emit BTFR slope/intercept/intrinsic scatter under identical selection; add a figure and CSV “Source Data.”

C) Solar System and wide binaries
- Findings:
  - Solar-System checks are robustly integrated (Cassini; Mercury perihelion; Shapiro), with plots and validators.
  - Wide-binary (Gaia DR3) test not implemented; no loader or pipeline for WB kinematics.
- Proposed actions:
  1) Add a wide-binary module: ingest a vetted DR3 WB catalog; integrate orbits in the Galactic external field using the ξ(ρ, T) model; reproduce WB statistics (velocity ratio distributions) and compare to published constraints.

D) Milky Way K_z and Σ_1.1
- Findings:
  - There are MW pipelines and microlensing checks; K_z/Σ_1.1 calculation is referenced in validation outputs, but current validation_results show a failed K_z benchmark (see validation/validation_results/validation_summary.json).
- Proposed actions:
  1) Implement K_z(R_0, z) and Σ_1.1 from the same mass model + ξ, calibrate disk vertical structure, and reproduce Bovy & Rix / McMillan constraints. Add a figure and Source Data file; ensure parameters used remain Cassini-safe.

E) Positioning vs ΛCDM hydrodynamical RAR
- Findings:
  - The repo contains NFW/ΛCDM baselines and scripts contrasting to TFR/ER; however, no overlay vs EAGLE/NIHAO RAR bands is present.
- Proposed actions:
  1) Add an overlay panel: SPARC points, DGG curve ± intrinsic scatter, and at least one ΛCDM simulation band (EAGLE/NIHAO) with citations; quantify the discriminant at low g_bar and the scatter trend vs mass.

F) Cosmology checks
- Findings:
  - There is an exploratory cosmology section in validation/validate_ddmm.py touching on SN/BAO with schematic ξ-path effects, but no formal Boltzmann treatment; c_GW constraints noted only qualitatively.
- Proposed actions:
  1) Provide a one-page feasibility note: show the completion can embed in FRW without altering c_T, and sketch linear-growth behavior; optionally prototype a modified-growth function parameterization constrained to small deviations consistent with current fσ8.

G) Data/code availability and reproducibility
- Findings:
  - REPRODUCIBLE.md exists with commands and artifact locations. Many outputs (CSV/JSON/PNG) are written per run. No Zenodo DOI or “Source Data” tables linked per figure yet.
- Proposed actions:
  1) Create a script to collect all figure “Source Data” in CSV/Parquet and a manifest; add a release tagging process that writes a CITATION.cff and uploads archives to Zenodo for a DOI.

H) Lensing-only α_lens_ph removal from manuscript
- Findings:
  - α_lens_ph exists in scripts/next_steps_from_run.py and related plotting/aggregation helpers; it is labeled as a pilot.
- Proposed actions:
  1) For the paper build, disable α_lens_ph and compute lensing from Φ+Ψ using the completion parameters; keep α_lens_ph tooling behind an “internal pilot” flag not used for manuscript figures.

I) New data to consider downloading/using
- BIG‑SPARC rotation curves to extend SPARC; add KiDS/DES Y3/HSC lensing stacks; vetted SLACS/CASTLES strong-lens table (M*, R_e, z_l, z_s); Gaia DR3 wide-binary catalogs; MW vertical force constraints datasets; EAGLE/NIHAO public RAR bands for overlays.

[1]: https://arxiv.org/abs/1003.2185
[2]: https://www.aanda.org/articles/aa/pdf/2020/10/aa38505-20.pdf
[3]: https://link.aps.org/doi/10.1103/PhysRevLett.119.251301
[4]: https://ui.adsabs.harvard.edu/abs/2018PhRvL.120m1101A/abstract
[5]: https://link.aps.org/doi/10.1103/PhysRevLett.127.161302
[6]: https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L/abstract
[7]: https://arxiv.org/html/2411.13329v1
[8]: https://ui.adsabs.harvard.edu/abs/2003Natur.425..374B/abstract
[9]: https://arxiv.org/abs/2311.03436
[10]: https://ui.adsabs.harvard.edu/abs/2013ApJ...779..115B/abstract
[11]: https://arxiv.org/pdf/1608.00971
[12]: https://link.aps.org/doi/10.1103/PhysRevLett.118.161103
[13]: https://arxiv.org/pdf/1703.05287
[14]: https://link.aps.org/doi/10.1103/PhysRevD.105.123537
[15]: https://www.nature.com/nphys/editorial-policies/reporting-standards
[16]: https://www.nature.com/nphys/editorial-policies
[17]: https://arxiv.org/pdf/1902.06751
[18]: https://link.aps.org/doi/10.1103/PhysRevLett.117.201101
[19]: https://arxiv.org/abs/1609.05917

---

## Update (2025-09-08): Item B progress — hierarchical a0 (no per-galaxy a0 tuning)

- What was missing: Only small‑sample per‑galaxy a0 scans; no hierarchical posterior across a large sample, limited or no nuisance propagation; no Δlog Z distributions.
- What we implemented: Added nuisance‑marginalization to the SPARC grid builder (ln M/L priors for disk/bulge; fractional observational inflation to capture distance/inclination/beam/non‑circular motions). Ran hierarchical inference (dynesty, ln a0 ~ N(μ,σ)) over a broad SPARC selection (N≈118). Computed DGG evidence per galaxy by ∫ L(a0)π(a0)d(ln a0) and Δlog Z vs GR using the same Gaussian likelihood normalization so constants cancel.
- Artifacts: 
  - Grids: results/next_steps/rar_plateau_mw_full/sparc_a0_grids/*.csv
  - Posterior: results/next_steps/rar_plateau_mw_full/hierarchical_a0_posterior_summary.json
  - Δlog Z table: results/next_steps/rar_plateau_mw_full/hierarchical_dgg_evidence.csv
  - Δlog Z summary: results/next_steps/rar_plateau_mw_full/hierarchical_dgg_evidence_summary.json
- Posterior (p50): μ ≈ −10.231, σ ≈ 0.245. Δlog Z summary: mean ≈ −215.0, median ≈ −468.4, p16 ≈ −622.4, p84 ≈ 74.75 (N=118).
- Repro commands are documented in README under “Hierarchical a0 (SPARC sample ≥100) — completed”.

## Update (2025-09-07): Item A progress — metric lensing and PPN (no α_lens_ph)

- Metric-only lensing path implemented in the orchestrator. From the same xi-based dynamics (Φ=Ψ, c_T=1), we forward-predict:
  - Einstein radii θ_E for GR (baryons-only) and RAR metric (baryons + phantom via xi).
  - Per-lens ΔΣ(R) profiles plus a stacked ΔΣ(R) across lenses.
  - No lensing-only scaling appears in manuscript outputs (use `--metric-lensing-only`).
- PPN table and Solar-System: The Solar-System ΔG/G table and plot now ship with an optional PPN export (γ, β, α1, α2) under the adopted covariant subclass (Φ=Ψ, c_T=1); see docs/relativistic_scaffold.md.
- Galaxy-lens shortlist and CSV: Added a ranked, practical list and a CSV template:
  - docs/targets_lensing_galaxies.md (why each target, what to compare, data notes)
  - docs/lensing_targets.csv (fill log10M_star, Re_kpc, optional n_sersic/profile)

Images (examples, from a smoke run; replace with measured M⋆/Re when filled):
- Stacked ΔΣ from metric predictions: `images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png`
- Per-lens comparisons (GR vs RAR metric; SIS yardsticks in the table):
  - `images/next_steps/enhanced_20250805_115400/lensing_rar_PG1115+080.png`
  - `images/next_steps/enhanced_20250805_115400/lensing_rar_B1608+656.png`
  - `images/next_steps/enhanced_20250805_115400/lensing_rar_Q0957+561.png`

Notes
- Dark-matter baseline: We currently include SIS yardsticks in the table as a simple DM-like baseline; an explicit NFW lensing overlay can be added next if desired, but is not required for the “single-theory” closure test.
- How to run (forward predictions):
  ```bash
  python scripts/next_steps_from_run.py \
    --run-dir runs/<your_run> \
    --sparc-dir external_data/Rotmod_LTG \
    --lensing-sample-csv docs/lensing_targets.csv \
    --metric-lensing-only --density-profile sersic --write-ppn-table
  ```

---

## Update (2025-09-08): Item A.3 — Local-universe (PPN, Solar bands, wide binaries)

- PPN table export (γ, β, α1, α2) implemented for the adopted Φ=Ψ, c_T=1 subclass; orchestrator writes results/next_steps/<run>/ppn_table.csv when --write-ppn-table is set.
- Solar-System posterior bands: orchestrator now samples NPZ posteriors (if present) to emit 16–84% credible bands for |ΔG/G| at 1–30 AU (results/next_steps/<run>/solar_system_posterior_bands.csv) and overlays them on the Solar figure.
- Wide-binary prediction: added scripts/analyze_wide_binaries.py to generate a forward theory curve (sqrt(ξ) − 1 vs separation) and optional overlay from a local CSV.
  - Outputs: results/next_steps/<run>/wide_binaries_pred.csv and images/next_steps/<run>/wide_binaries_pred.png.
  - How-to: see docs/wide_binaries.md.

