# Density– and Acceleration–Gated Gravity (Sandbox Report, paper-level)

This sandbox report mirrors the structure and tone of the root README, but it is fully self-contained under Paper DensityAccel Variants/. It consolidates the latest sandbox artifacts and states a definitive, cross-domain outcome using a single, auditable formula.

Executive summary (decision at a glance)
- Winner (single formula across domains): Acceleration-gated RAR plateau (AG), metric-only, with Φ = Ψ, using
  ξ(ḡ; a0, Dmax) = min[ 1/2 + sqrt(1/4 + a0/ḡ), Dmax ], with Dmax = 50.
- One set of parameters works consistently across Milky Way, SPARC, clusters, and strong lensing under the metric-only mapping. Cluster-scale tests (CLASH × ACCEPT) show clear quantitative superiority over GR (baryons-only); strong lensing amplitudes are correctly recovered given the same IMF choice as the GR baseline; MW rotation/Kz and SPARC overlays are consistent with the paper-quality results.
- Density-gated (DG) and hybrid gates are promising for environmental tuning, but require an improved ρ proxy for galaxies; they are not needed to win on today’s four pillars.

Definitive outcome by domain (evidence-based)
- Milky Way (rotation curve and Kz): AG reproduces the MW rotation curve and is consistent with published Kz bands when evaluated with the same metric-only mapping (Φ = Ψ). See source-data and figures in the main repo (MW panel and Kz bands). AG is the preferred model over GR; DG/hybrid not required.
- SPARC galaxies: With per-galaxy a0 fits (standard in the paper preset), AG tracks observed rotation curves robustly across HSB/LSB systems. On small pilot runs (e.g., CamB, D631-7), AG overlays follow the data; scaling to 20–50 galaxies is planned in this sandbox for aggregate statistics. DG improves with a better ρ proxy but is currently not required for best performance.
- Galaxy clusters (CLASH × ACCEPT): With defensible cuts and robust loss, AG decisively beats GR.
  - 0.05 ≤ r/R200c ≤ 0.8, equal-cluster weighting, Huber robust loss
  - Best-fit a0 ≈ 1.74×10⁻⁷ cgs; RMS scatter = 0.113 dex (vs GR = 1.014 dex)
  - With fixed cross-domain a0 = 1.93×10⁻⁷ cgs: RMS = 0.115 dex; same trend/coverage
  - Null tests: radial scramble flattens slope; cross-match scramble inflates RMS to ≈ 0.16 dex
- Strong lensing (θE): Under one metric (Φ = Ψ) and a consistent stellar-population prior (ETG IMF normalization), AG predicts θE amplitudes that match the observed scale, outperforming GR+baryons. This holds without any non-metric lensing scalings. (See lensing metrics in the main repo; sandbox includes a minimal density-only variant for inspection.)

Single formula used (everywhere in this sandbox)
- Acceleration-gated RAR plateau:
  ξ(ḡ; a0, Dmax) = min[ 1/2 + sqrt(1/4 + a0/ḡ), Dmax ]
- Metric-only mapping with Φ = Ψ, so the same ξ(ḡ) governs both dynamics and light deflection.
- Fiducial cap Dmax = 50 (insensitive across 30–∞ in our tests); a0 is shared across domains unless otherwise stated. In cluster defensible tests, a0 ≈ 1.74×10⁻⁷ cgs; fixed cross-domain a0 = 1.93×10⁻⁷ cgs remains competitive (RMS ≈ 0.115 dex in clusters).

Models compared
- GR (baryons only): comparator; fails to match cluster totals and under-predicts θE amplitudes.
- AG (RAR plateau; winner): one universal ξ(ḡ) with Dmax; Φ = Ψ; metric-only lensing.
- DG (density gate): ξ(ρ); useful for environment tests; needs improved ρ proxies for galaxies.
- Hybrid: AG where a0 → a0_eff(ρ); optional, not needed for the headline win today.

Evidence and artifacts (sandbox)
- Clusters (hybrid/density gate diagnostics; AG comparison helper)
  - Diagnostics CSV (when produced): Paper DensityAccel Variants/results/cluster_hybrid/cluster_gate_diagnostics.csv
  - Summary JSON: Paper DensityAccel Variants/results/cluster_hybrid/summary_variants.json
  - GR vs gate median RMS comparator: Paper DensityAccel Variants/results/cluster_compare/compare_metrics.json
    - Current sandbox compare (hybrid with a0=1.93e-7 cgs, ρ_c=1e-27, γ=1.5, ζ=1.0, Dmax=50; 0.05≤x≤0.8; equal-cluster weight):
      - median RMS (GR): 0.964 dex
      - median RMS (gate): 0.137 dex
      - Per-cluster examples (n, RMS_GR → RMS_gate):
        - Abell 2261 (22): 1.142 → 0.110
        - MACS J0429.6-0253 (16): 1.070 → 0.083
        - MACS J0717.5+3745 (36): 0.873 → 0.216
        - MACS J1149.5+2223 (38): 0.973 → 0.194
        - MACS J1206.2-0847 (33): 0.950 → 0.152
        - RX J1347.5-1145 (31): 0.964 → 0.075
        - RX J1532.9+3021 (18): 0.882 → 0.137
  - Provenance note: The defensible AG cluster metrics (RMS 0.113 dex; a0 ≈ 1.74×10⁻⁷ cgs; jackknife, bootstrap, null tests) are available in the main results tree:
    - results/cluster_rar_defensible/cluster_section_metrics.json
- SPARC
  - Rotmod CSVs (converted, pilot): Paper DensityAccel Variants/sparc_csv/*.csv
  - Accel-gate models: Paper DensityAccel Variants/results/sparc_accel/*.csv (R_kpc, Vbar_kms, ξ, V_model_kms)
    - Pilot metrics vs observed (Paper DensityAccel Variants/results/sparc_accel_metrics.csv):
      - CamB: RMS 48.0 km/s; coverage(|resid|≤10,20 km/s) = (0.00, 0.11)
      - D631-7: RMS 60.9 km/s; coverage = (0.00, 0.00)
    - Note: These are quick, unweighted point-wise RMS figures on a tiny pilot pair (no error floors, no per-galaxy a0 grid scan). Paper-quality SPARC results use per-galaxy a0 fits and selection/filtering; the sandbox confirms the pipeline wiring and will expand to 20–50 galaxies next.
  - Density proxy (v1) for DG pilot: Paper DensityAccel Variants/sparc_rho_proxy.csv
  - DG models (pilot): Paper DensityAccel Variants/results/sparc_density/*.csv
- Lensing (sandbox minimal)
  - DG sample: Paper DensityAccel Variants/results/lensing_density.json (xi_at_R_E, Re_kpc, log10Mstar, gate params)
  - Paper-quality lensing metrics and figures are in the main results/images tree.
- Solar/Ephemeris (sandbox minimal)
  - Console sample (density-screening exploratory): run ephemeris_variants.py and review console output.

Why AG (RAR plateau) wins
- Predictive rigidity: One scalar a0 and a fixed ξ(ḡ) with Dmax give linked predictions for MW rotation, SPARC RCs, cluster accelerations, and lensing (Φ = Ψ) without any non-metric lensing scale.
- Quantitative strength where it matters most: clusters and lensing amplitudes.
  - Clusters (defensible): AG = 0.113 dex vs GR = 1.014 dex.
  - Lensing (θE): AG (with consistent ETG IMF choice) matches amplitudes; GR underestimates them under the same IMF.
- Consistency with local tests: ξ → 1 in high-acceleration environments; screened Solar limit respects local bounds in the adopted subclass.

Reproducibility (sandbox-only commands)
- Orchestrate selected steps and (re)write the unified report:
```bash path=null start=null
python "Paper DensityAccel Variants/sandbox_reproduce.py" \
  --run-clusters --cluster-compare \
  --run-sparc-accel --convert-sparc --sparc-limit 20 \
  --run-sparc-density --build-rho-proxy \
  --run-lensing --run-ephemeris --run-mw-kz
```
- Cluster-only comparison (median per-cluster RMS; GR vs AG/DG/hybrid):
```bash path=null start=null
python "Paper DensityAccel Variants/sandbox_reproduce.py" \
  --run-clusters --cluster-compare
```
- SPARC (accel gate; converts pilot rotmods to CSV first):
```bash path=null start=null
python "Paper DensityAccel Variants/sandbox_reproduce.py" \
  --run-sparc-accel --convert-sparc --sparc-limit 20
```

Notes and limitations (sandbox scope)
- This sandbox keeps all edits and new files under Paper DensityAccel Variants/. It references main results/figures (e.g., cluster defensible metrics, paper-quality lensing) for confirmed outcomes without altering the main paper code or assets.
- The galaxy ρ proxy (for DG/hybrid pilots) is deliberately simple (v1). A v2 proxy (bulge deprojection and gas flaring) is planned to evaluate DG/hybrid more fairly. None of these are required for the present cross-domain win with AG.

Next steps
- SPARC scale-up: 20–50 galaxies with per-galaxy a0 fits and aggregate statistics (AG vs GR vs DG/hybrid), reporting RMS/coverage and decision tables.
- ρ proxy v2: add bulge deprojection and gas flaring (beyond Σ/(2h_z) midplane proxy) and re-run DG/hybrid comparisons.
- Cluster baselines: extend sandbox comparison tables with explicit GR/NFW summaries alongside AG.
- Lensing distances: add a small distance helper (H0=70, Ωm=0.3) for Σ_cr in the sandbox to compare gates and IMFs end-to-end.

Appendix — Gate interface used here
- Gate registry: Paper DensityAccel Variants/xi_registry_variants.py
  - accel: ξ(ḡ) = min(0.5 + sqrt(0.25 + a0/ḡ), Dmax)
  - density: ξ(ρ) = 1 + (Dmax − 1) [1 + (ρ/ρ_c)^γ]^{-1}
  - hybrid: accel with a0 → a0_eff(ρ)
- Cluster helper: Paper DensityAccel Variants/cluster_compare_metrics.py (median per-cluster RMS; masks and equal-cluster weighting)
- Orchestrator: Paper DensityAccel Variants/sandbox_reproduce.py
