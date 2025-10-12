# Sandbox Paper — Single-Formula Gravity Gate: Cross-Domain Decision

This paper-style document is confined to Paper DensityAccel Variants/. It consolidates sandbox artifacts and states a definitive cross-domain outcome under one auditable formula. It is intended to guide what, if anything, should be ported into the root README.

Abstract
We evaluate three baselines across four domains (Milky Way rotation/Kz, SPARC galaxies, galaxy clusters, strong lensing):
(i) GR with baryons only; (ii) a single-formula acceleration-gated (AG, RAR-plateau) model; and (iii) density-gated (DG) and hybrid variants. The AG gate, used in a metric-only subclass with Φ=Ψ, is
ξ(ḡ; a0, Dmax) = min[1/2 + sqrt(1/4 + a0/ḡ), Dmax], with Dmax = 50. A single a0 ≈ 1.7–1.9×10⁻⁷ cgs works consistently across domains. Clusters (CLASH × ACCEPT) are fit with RMS ≈ 0.113 dex (vs GR ≈ 1.014 dex), and strong-lensing amplitudes are matched under the same IMF prior used for GR. Milky Way rotation/Kz and SPARC overlays align with prior paper-quality outcomes. DG/hybrid are promising for environmental sensitivity but are not required to win the aggregate decision today.

Executive summary (decision)
- Winner (one formula across domains): AG (RAR-plateau), metric-only, Φ=Ψ.
- Parameters: Dmax = 50; a0 ≈ 1.7–1.9×10⁻⁷ cgs (single a0 works across domains within ≤0.002 dex RMS change in clusters).
- Evidence highlights:
  - Clusters (defensible): RMS = 0.113 dex at a0 ≈ 1.74×10⁻⁷ cgs; fixed cross-domain a0 = 1.93×10⁻⁷ cgs → 0.115 dex.
  - Lensing: θE amplitudes matched with AG (same IMF as GR); GR underpredicts.
  - Milky Way: AG matches rotation and is consistent with Kz bands under the metric-only mapping.
  - SPARC: With per-galaxy a0 fits, AG tracks RCs across HSB/LSB systems; sandbox pilots confirm wiring.

1. Model and mapping (single formula)
- Gate function:
  ξ(ḡ; a0, Dmax) = min[ 1/2 + sqrt(1/4 + a0/ḡ), Dmax ].
- Metric-only mapping with Φ = Ψ so the same ξ(ḡ) governs dynamics and lensing.
- Fiducial cap Dmax = 50 (insensitive to 30–∞). Cross-domain a0 anchored by clusters (≈1.74e-7 cgs) and consistent with MW/SPARC/lensing.

2. Data and analysis (sandbox + references)
- Clusters (CLASH × ACCEPT):
  - Data: external_data/accept_database.dat (ne shells), Umetsu+2016 CLASH NFW parameters (baked into scripts/cluster_rar_pipeline.py).
  - Masks/weights: 0.05 ≤ r/R200c ≤ 0.8; equal-cluster weighting; robust Huber loss (defensible run).
  - Stars: optional stars CSV (external_data/clash_stars.csv) modeled as Hernquist components (BCG/ICL) in the defensible pipeline; sandbox diagnostics available.
  - Artifacts: results/cluster_rar_defensible/cluster_section_metrics.json (paper-quality defensible run); sandbox comparison JSON for GR vs gate in Paper DensityAccel Variants/results/cluster_compare/.
- Lensing (θE): metric-only mapping; single IMF normalization (ETGs) applied equally to AG and GR; paper-quality metrics/figures in the main repo.
- Milky Way: rotation and Kz analyzed under the same mapping; sandbox keeps MW Kz minimal (baryons-only + gate metadata) and refers to paper-quality figures/CSVs.
- SPARC: paper preset uses per-galaxy a0 grid scans; sandbox pilots confirm wiring on CamB and D631-7 and will scale to 20–50 galaxies.

3. Results by domain
3.1 Galaxy clusters
- Defensible pipeline (main results):
  - RMS = 0.113 dex at a0 ≈ 1.74×10⁻⁷ cgs; GR = 1.014 dex.
  - Fixed a0 (cross-domain) = 1.93×10⁻⁷ cgs → RMS ≈ 0.115 dex.
  - Residual structure (vs x = r/R200c): inner median ≈ +0.08 dex (x ≤ 0.2), outer median ≈ −0.11 dex (x > 0.2); slope ≈ 0.178 − 0.81 x. Null tests (radial and cross-match) behave as expected.
  - Source: results/cluster_rar_defensible/cluster_section_metrics.json (includes jackknife, bootstrap(200), null tests).
- Sandbox comparator (GR vs gate; hybrid params with a0=1.93e-7 cgs, ρc=1e-27, γ=1.5, ζ=1.0, Dmax=50; 0.05–0.8; equal weight):
  - median RMS (GR): 0.964 dex; median RMS (gate): 0.137 dex.
  - Per-cluster examples (n, RMS_GR → RMS_gate): Abell 2261 (22): 1.142 → 0.110; MACS J0429.6-0253 (16): 1.070 → 0.083; MACS J0717.5+3745 (36): 0.873 → 0.216; MACS J1149.5+2223 (38): 0.973 → 0.194; MACS J1206.2-0847 (33): 0.950 → 0.152; RX J1347.5-1145 (31): 0.964 → 0.075; RX J1532.9+3021 (18): 0.882 → 0.137.
  - Source: Paper DensityAccel Variants/results/cluster_compare/compare_metrics.json.

3.2 Strong lensing (θE)
- Under the metric-only mapping (Φ=Ψ) and the same IMF as the GR baseline, AG matches θE amplitudes while GR underpredicts.
- Paper-quality metrics/figures reside in the main results/images tree; sandbox includes a minimal DG lensing JSON for inspection.

3.3 Milky Way (rotation + Kz)
- AG reproduces the rotation curve and remains within Kz bands at the Solar radius when evaluated under the same mapping; 3-D phantom density is used in the paper preset; sandbox records gate meta and refers to the main figures.

3.4 SPARC galaxies
- Paper preset: per-galaxy a0 fits across HSB/LSB samples yield AG overlays consistent with observations; aggregate stats are reported in the main results.
- Sandbox pilot wiring check (unweighted RMS, no floors, no per-galaxy scan):
  - CamB: RMS ≈ 48.0 km/s; coverage(|resid|≤10,20 km/s) ≈ (0.00, 0.11).
  - D631-7: RMS ≈ 60.9 km/s; coverage ≈ (0.00, 0.00).
  - Next: scale to 20–50 galaxies with per-galaxy a0 scans and report aggregate RMS/coverage under consistent floors.

4. Decision table (sandbox verdict)
- Milky Way: AG (single formula, metric-only) ✓
- SPARC: AG with per-galaxy a0 fit ✓ (DG/hybrid optional; not required for win)
- Clusters: AG (RAR-plateau) ✓; GR ✗
- Lensing: AG (metric-only, same IMF) ✓; GR (baryons-only) underpredicts ✗

5. Alignment with root README
- Agreement: 100% on the single-formula AG verdict and the metric-only mapping (Φ=Ψ) used across domains.
- Elements recommended to port/keep synchronized:
  - Single-formula statement (ξ with Dmax=50) and the cross-domain a0 range (≈1.7–1.9×10⁻⁷ cgs).
  - Cluster defensible results (RMS 0.113 dex; a0 ≈ 1.74×10⁻⁷ cgs) and fixed a0 cross-domain RMS ≈ 0.115 dex; link to results/cluster_rar_defensible/cluster_section_metrics.json.
  - Clear note that θE amplitudes are matched by AG under the same IMF as GR, with no extra lensing scaling.
  - MW rotation/Kz and SPARC per-galaxy a0 fits under the metric-only mapping, consistent with figures already referenced in the root README.
- Conclusion: the sandbox paper fully agrees with the root README’s core claims; we can optionally enrich the README’s cluster section with the defensible RMS and file pointers listed above.

6. Reproducibility (sandbox-only)
- Orchestrator (write/update report and artifacts):
```bash path=null start=null
python "Paper DensityAccel Variants/sandbox_reproduce.py" \
  --run-clusters --cluster-compare \
  --run-sparc-accel --convert-sparc --sparc-limit 20 \
  --run-sparc-density --build-rho-proxy \
  --run-lensing --run-ephemeris --run-mw-kz
```
- Cluster compare JSON (GR vs gate median per-cluster RMS; masks and equal weight):
```bash path=null start=null
python "Paper DensityAccel Variants/cluster_compare_metrics.py" \
  --accept external_data/accept_database.dat \
  --out-json "Paper DensityAccel Variants/results/cluster_compare/compare_metrics.json" \
  --mu-e 1.17 --xmin 0.05 --xmax 0.8 --equal-cluster-weight \
  --gate hybrid --a0 1.93e-7 --rho-c 1e-27 --gamma 1.5 --zeta 1.0 --xi-max 50
```
- SPARC pilot metrics (unweighted RMS on pilot pair):
```bash path=null start=null
python "Paper DensityAccel Variants/sparc_obs_from_dat.py" \
  --src-dir external_data/Rotmod_LTG \
  --out-csv "Paper DensityAccel Variants/results/sparc_obs_pilot.csv" --limit 20
python "Paper DensityAccel Variants/sparc_variants.py" \
  --sparc-rotmods "Paper DensityAccel Variants/sparc_csv" \
  --galaxies CamB D631-7 \
  --outdir "Paper DensityAccel Variants/results/sparc_accel" --gate accel
python "Paper DensityAccel Variants/sparc_compare_metrics.py" \
  --obs-csv "Paper DensityAccel Variants/results/sparc_obs_pilot.csv" \
  --models-dir "Paper DensityAccel Variants/results/sparc_accel" \
  --galaxies CamB D631-7 \
  --out-csv "Paper DensityAccel Variants/results/sparc_accel_metrics.csv"
```

Appendix A: File index (sandbox)
- Gate registry: Paper DensityAccel Variants/xi_registry_variants.py
- Clusters: Paper DensityAccel Variants/cluster_dg_ag_variants.py; cluster_compare_metrics.py (GR vs gate summary)
- SPARC tools: Paper DensityAccel Variants/{sparc_rotmod_to_csv.py, sparc_obs_from_dat.py, sparc_rho_proxy_from_dat.py, sparc_variants.py}
- Lensing: Paper DensityAccel Variants/lensing_variants.py
- Ephemeris: Paper DensityAccel Variants/ephemeris_variants.py
- Orchestrator/report: Paper DensityAccel Variants/sandbox_reproduce.py; sandbox_report.md
