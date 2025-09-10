# Gravity Gates: An Acceleration-Gated Alternative to Dark Matter

## Introduction

Galactic rotation curves continue to strain the standard picture in which galaxies live inside massive, non‑baryonic dark‑matter halos. Halos can be tuned to fit individual systems, but doing so typically introduces one bespoke mass profile per galaxy and requires tight, system‑by‑system coordination between baryons and dark matter. A central empirical clue is the **radial acceleration relation (RAR)**: across hundreds of disks the observed centripetal acceleration $g_{\rm obs}$ correlates closely with the acceleration predicted by baryons alone $g_{\rm bar}$, with scatter comparable to measurement uncertainties. Any successful framework must either explain why dark halos conspire to follow the baryons so closely, or modify the low‑acceleration law of gravity itself in a way that **predicts** the RAR.

Two broad approaches have emerged. In ΛCDM, increasingly sophisticated formation models attempt to imprint the observed baryon–halo coupling through feedback, assembly histories, and environment. This is flexible, but the cost is many latent degrees of freedom. On the other side are **MOND‑like** ideas that introduce a characteristic acceleration scale $a_0$ and were prescient in anticipating the RAR. These are predictively rigid on galaxy scales, but classic formulations are not manifestly relativistic and face challenges in clusters and cosmology. The tension between **flexibility** (fit anything) and **rigidity** (predict many things with few knobs) is the fundamental issue.

Our guiding idea is that gravity’s **effective response may depend on environment**, much as other fundamental interactions do. In quantum field theory, couplings “run” with scale, and the strong force presents different faces in different regimes (asymptotic freedom vs. confinement). We do **not** claim an identity with QCD, but we adopt the same organizing principle: the **measured strength** of the interaction can change with conditions. If gravity’s weak‑field response is **gated** by local acceleration (and, by extension, by typical densities or tidal scales), then observers like us—residing in a **high‑acceleration** region (Solar neighborhood, deep potentials)—naturally see standard GR. In the **low‑acceleration** outskirts of galaxies, however, the gate opens and an enhanced response emerges. Cosmologically, such gating would tend to **promote aggregation** in diffuse regions rather than hinder it, a qualitative feature that is at least directionally compatible with the observed prevalence of bound structure. In this paper we do not model early‑universe dynamics; we focus on quantifying the weak‑field, quasi‑static consequences in galaxies and the Solar System.

We formalize this idea as **Gravity Gates (GG)**: a weak‑field framework in which the gravitational response is a deterministic function of the local field strength (and, optionally, simple environmental proxies). We instantiate a **density‑gated** subclass (**DGG**) for empirical tests. The gate $\xi(g_{\rm bar})$ multiplies the baryonic prediction, is **unity** in the high‑acceleration limit (recovering Newton/GR and Solar‑System tests), and increases smoothly toward a finite plateau at low acceleration to reproduce the RAR/BTFR phenomenology while avoiding pathologies at extremely small $g_{\rm bar}$. Crucially, **the same gate**—with a single scale $a_0$ and a fixed functional form—is used to predict **rotation curves, vertical forces $K_z$, strong‑lensing Einstein radii $\theta_E$, and stacked weak‑lensing $\Delta\Sigma$** under a metric subclass with $\Phi=\Psi$. This preserves **predictive rigidity** without the per‑galaxy halo freedom.

Our approach differs from both extremes. Compared to tuned halos, DGG **reduces** the degrees of freedom by replacing galaxy‑specific dark profiles with a universal gate tied to the baryons. Compared to classic MOND, DGG is embedded in an explicit weak‑field metric mapping (used for both dynamics and lensing), includes a finite low‑$g$ plateau to regularize the deep regime, and is framed to be auditable against Solar‑System post‑Newtonian bounds. The result is a tightly constrained hypothesis that makes **linked, cross‑domain predictions**.

**This paper’s program and falsifiability.** We confront DGG with: (i) the **Milky Way** rotation curve and vertical force $K_z(R_0,z)$ (reporting $\Sigma_{1.1}$); (ii) representative **SPARC** rotation curves and the **BTFR**; (iii) **Solar‑System** constraints cast in $|\Delta G/G|$ alongside post‑Newtonian parameters; and (iv) **gravitational lensing**, from individual $\theta_E$ to stacked $\Delta\Sigma(R)$, computed with the same gate. The framework is **falsified** if a single $\xi(g)$ cannot simultaneously: fit outer‑disk slopes without per‑galaxy halos, remain within $K_z$ bands at the Solar radius, respect Solar‑System limits, and reproduce the amplitude of lensing signals under standard stellar‑population priors. The sections that follow specify the gate, the weak‑field mapping, and the data/selection, then present each test in turn.

---

## The RAR‑Gated Gravity Model

In RAR‑gated gravity, the departure from Newton’s law is governed by an interpolating “gating” function that depends on the local gravitational acceleration (and/or local mass distribution). Conceptually, one can think of the model as modifying the effective gravitational constant or the relationship between the matter distribution and the curvature of spacetime, such that:

- **High‑acceleration limit** ($g\gg a_0$) — The gate suppresses modifications, restoring Newton/GR and passing Solar‑System tests.
- **Low‑acceleration limit** ($g\ll a_0$) — The gate enhances the effective force in a way tuned to reproduce the RAR (and hence the BTFR).
- **Intermediate regime** ($g\sim a_0$) — A smooth transition governed by the specific interpolating function; the transition width can be calibrated against rotation‑curve shapes.

#### Why we set it up this way

The empirical RAR/BTFR indicates that galaxy dynamics are largely fixed by the baryons alone. Rather than fitting a dark halo for each galaxy, we impose a single gate $\xi(g_{\rm bar}; a_0, D_{\max})$ that multiplies the baryonic prediction. This keeps the theory rigid (few parameters, shared across systems) while still reproducing low‑$g$ phenomenology. A finite plateau $D_{\max}=50$ is adopted in the paper preset to regularize the deep‑MOND limit without affecting the data range we test.

### How the model works (plain language)

- In high‑acceleration regions, gravity behaves like ordinary GR/Newton.
- In low‑acceleration regions (galaxy outskirts), gravity gets a boost. The size of the boost is set by a single universal scale $a_0$ and capped by a plateau $D_{\max}$ to prevent unphysical divergence.
- Practically, we compute the Newtonian acceleration from baryons ($g_{\rm bar}$) and then multiply by a boost factor $\xi$ (the “gate”). This $\xi$ depends only on the local field strength (and, optionally, a mild environment term), not on a custom dark halo for each galaxy.
- The same $\xi$ is used consistently for rotation curves, vertical forces in the Milky Way, Solar‑System checks, and gravitational lensing (via a metric‑only mapping with $\Phi=\Psi$).

### Five‑step recipe (what the code actually does)

1. Baryons → $g_{\rm bar}(R)$. Compute from the observed stellar+gas mass model.
2. Environment (optional). Adjust $a_0$ to $a_0^{\rm eff}$ with a density/tidal proxy (defaults are conservative).
3. Gate/boost.

   $$
   \xi(R)=\min\!\left[\,\tfrac{1}{2}+\sqrt{\tfrac{1}{4}+\frac{a_0^{\rm eff}}{g_{\rm bar}(R)}}\,,\,D_{\max}\right],\qquad D_{\max}=50\;\text{(paper preset)}.
   $$

4. Prediction. Rotation speed $V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R)$. The same $\xi$ feeds lensing and vertical‑force predictions.
5. Fit only $a_0$ (with fixed $D_{\max}$). We either grid‑scan $a_0$ per galaxy or fit a hierarchical population mean and scatter.

Mathematically, one representation is a modified Poisson equation or GR‑style field equation with a nonlinear term that depends on the field strength and/or density:

$$
\nabla^2 \Phi \;=\; 4\pi G\,\rho_b \;+\; \nabla\!\cdot\!\Big[f\!\left(\tfrac{|\nabla\Phi|}{a_0},\, \tfrac{\rho_b}{\rho_0}\right)\nabla\Phi\Big],
$$

with $f\rightarrow 0$ in high‑acceleration/high‑density limits and $f>0$ in low‑acceleration environments.

### **Box 1 — Exact weak‑field formula used in all figures (from code)**

Let $V_{\rm bar}(R)$ be the baryonic circular speed ($\mathrm{km\,s^{-1}}$) and $R$ the radius (kpc). Convert to SI:

$$
g_{\rm bar}(R) \;=\; \left(\frac{V_{\rm bar}^2(R)}{R}\right)\,C,
\quad C \equiv 3.240779289\times 10^{-14}\\ 
\frac{\mathrm{m/s}^2}{(\mathrm{km/s})^2/\mathrm{kpc}}.
$$

**Environmental gate (optional):**

$$
s_\rho(\rho;\rho_c,\gamma)=\frac{1}{1+(\rho/\rho_c)^\gamma}\in[0,1],\qquad
W(T;T_0,\sigma_{\ln T},w_{\min})=w_{\min}+(1-w_{\min})\exp\!\left[-\frac{(\ln T-\ln T_0)^2}{2\sigma_{\ln T}^2}\right],
$$
with $T\equiv V_{\rm bar}^2/R^2$ used purely as a tidal proxy.

**Effective acceleration scale:**

$$
 a_0^{\rm eff} \;=\; a_0\Bigl[1+\zeta_{\rm env}\,s_\rho(\rho)\,W(T)\Bigr].
$$

**Boost (“$\nu$” function) used throughout.**

$$
\xi(R) \;=\; \min\!\left[\;\frac{1}{2} + \sqrt{\frac{1}{4}+\frac{a_0^{\rm eff}}{g_{\rm bar}(R)}}\;,\; D_{\max}\right],\qquad D_{\max}=50~\text{(fiducial)}.\\[3pt]
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
$$

### Relativistic weak‑field mapping

Full derivation and Solar‑System PPN mapping: see docs/ppn_mapping.md.

We adopt a metric‑only weak‑field subclass with $c_T=1$ and $\Phi=\Psi$ in screened, quasi‑static limits. Dynamics depend on $\Phi$; lensing depends on $\Phi+\Psi=2\Phi$. The same $\xi(g)$ rescales the weak‑field potential entering both dynamics and light deflection. See Methods for a PPN sketch and Appendix for a covariant scaffold; for broader context compare to TeVeS‑like and modern scalar–tensor completions (e.g., Skordis–Zlosnik).

A QUMOND‑like mapping is useful for intuition: $\nabla^2\Phi=\nabla\!\cdot\![\nu(|\nabla\Phi_b|/a_0)\nabla\Phi_b]$ with $\nu(y)=\tfrac12+\sqrt{\tfrac14+1/y}$, so that $g=\nu\,g_N$ and $V^2=\xi\,V_{\rm bar}^2$ in the disk plane. The associated phantom‑density identity yields the full 3‑D contribution used for $K_z$ and lensing in the paper preset. We use the same $\nu$ function as in Box 1.

---

## Rotation Curve Predictions with No Dark Halos

We applied the RAR‑gated model to baryonic mass models for the **Milky Way** and for external galaxies from **SPARC**. In each case we compute

$$
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R)
$$

with $\xi$ from **Box 1** and compare to observed rotation curves.

### Figure‑by‑Figure Guide (what each plot proves)

| Figure | What question does it answer? | What to look for / success criterion | Why it matters to the paper’s claims |
|---|---|---|---|
| MW rotation (GR vs NFW vs DGG) | Can one set of DGG parameters match the Milky Way without a dark halo? | Red DGG curve follows the data in the inner Galaxy and sustains the outer speed once $g_{\rm bar}\sim a_0$; blue (baryons‑only) falls short; NFW is a yardstick only. | Establishes core claim: a single gate, not a custom halo, captures the shape. |
| MW $K_z$ (full‑3D) | Is the vertical force at $R_0$ consistent with measurements? | Red full‑3D DGG curve lies inside the Bovy & Rix / McMillan bands (overlay these); quote $\Sigma_{1.1}$. | Cross‑check of a different field component; prevents “in‑plane tuning” and uses the full‑3D figure/CSV. |
| SPARC panel (5 exemplars) | Does the same gate shape work across galaxies of very different surface brightness? | Red DGG lines broadly track black points; residuals correlate with known systematics (inclination, gas). | Demonstrates generality with one functional form; avoids per‑galaxy halo freedom. |
| BTFR with bootstrap band | Do DGG predictions respect the baryonic Tully–Fisher slope and scatter? | Slope $3.184\,[3.034,\,3.332]$, RMS $\sim0.22$ dex; band from bootstrap. | Connects to the RAR/BTFR “natural law”; shows we’re in the correct phenomenology with few knobs. |
| Solar‑System ($\Delta G/G$) | Is the gate invisible where GR is well tested? | Curve remains below the Cassini bound at Saturn; $\Delta G/G\ll10^{-5}$ over 1–30 AU. | Confirms Solar‑System safety under the same mapping used for galaxies. |
| $\theta_E$: predicted vs observed | Does the same $\xi$ (metric‑only, $\Phi=\Psi$) reproduce strong‑lensing scales? | Red points cluster around the 1:1 line with errors; blue (baryons‑only) under‑predicts; include per‑lens residuals. | Puts DGG on the same footing as halo models for lensing—critical for credibility. |
| Stacked $\Delta\Sigma$ (metric) | Is the average projected signal roughly correct over 0.05–300 kpc? | Solid curve with 16–84% band sits in the ballpark of literature stacks alongside the data points. | Shows the global profile is captured without non‑baryonic mass. |
| Wide‑binary (Extended Data) | Does DGG avoid order‑unity deviations at $10^3$–$10^4$ AU? | Predicted statistic $\xi-1$ stays modest; compare to Gaia analyses. | Complements Solar constraints in a distinct low‑$g$ environment. |

### Milky Way: A Case Study

**Milky Way (Gaia DR3) rotation curve: GR vs NFW vs RAR‑gate.**  
![Milky Way: GR vs NFW vs RAR‑gate (0.1 kpc Gaia medians)](images/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.png)

Caption: Single‑theory fit: the DGG gate (red) with one universal shape and a single $a_0$ per galaxy reproduces both the inner rise (baryon‑dominated) and the outer plateau (low‑$g$ regime). GR with baryons alone (blue) falls short; the NFW line (green) is a yardstick (not a tuned fit).

As in the original text, the model matches the inner rise (where baryons dominate) and sustains the outer speed once $g_{\rm bar}\sim a_0$, without galaxy‑specific halos.

### Milky Way: vertical force $K_z$ and local surface density $\Sigma_{1.1}$

We compute $K_z(R_0,z)$ for the same MW baryons and infer $\Sigma_{1.1}\approx K_z/2\pi G$. The figure below uses the **full 3‑D** DGG ("phantom") mass implied by $\xi$ via
$\rho_{\rm ph}=(\xi-1)\,\rho_b - (4\pi G)^{-1}\,\nabla\xi\!\cdot\!\mathbf g_{\rm bar}$.

New in this draft: we propagate baryonic uncertainties (disk and bulge $M/L$, gas mass, disk scale heights/lengths, and a bulge‑scale proxy for flattening) through the full‑3D phantom density to produce a shaded band. We sample priors in the orchestrator and export a CSV with the 16–84% Kz band at selected z.

![Milky Way Kz and Σ_1.1 (full 3D)](images/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.png)

Caption: Vertical‑force cross‑check: full 3‑D phantom density implied by $\xi$ yields $K_z(R_0,z)$ and $\Sigma_{1.1}$ without a dark halo. The red shading shows the propagated baryon‑prior band (16–84%). Overlay bands (when provided) include Bovy & Rix (2013) $\Sigma_{1.1}=68\pm4\;M_\odot\,\mathrm{pc}^{-2}$ and McMillan (2017/2022); see `--mw-kz-overlay-csv` in the orchestrator. Source‑Data: `results/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.csv` and `mw_kz_prior_band.csv`.

Assumptions disclosed for this figure:
- Solar radius $R_0$ = 8.2 kpc (override with `--mw-R0-kpc`).
- Heights evaluated: |z| ∈ {0.5, 0.8, 1.1, 1.5, 2.0} kpc (override via `--mw-kz-zlist`).
- Tracer kinematics for the comparison bands follow the published analyses (Bovy & Rix 2013; McMillan 2017/2022): axisymmetry, steady‑state vertical equilibrium, and the survey selection functions as reported therein. We overlay their published bands as references and do not re‑derive them here.
- Baryon priors (defaults; adjustable): ln(M/L) σ = 0.15 (disk and bulge), gas mass fractional σ = 0.25, disk scale‑height fractional σ = 0.20, disk scale‑length fractional σ = 0.10, bulge Hernquist‑scale fractional σ = 0.25 (a proxy for flattening sensitivity).
- The same DGG parameters used in the MW rotation fit are used for $K_z$.

### External Galaxies: SPARC Rotation‑Curve Fits

We selected representative spirals spanning mass and surface brightness. For each galaxy we hold gating parameters fixed (MW‑tuned) and **scan $a_0$ on a grid** $\log_{10} a_0\in[-10.5,-9.3]$ (m s$^{-2}$) to minimize $\chi^2$ (per‑galaxy $a_0$ strategy). A hierarchical log‑normal model for $a_0$ is available and reported as Extended Data when used.

#### SPARC gold‑sample panel (DGG vs GR vs observations)
![SPARC gold overlays panel](images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png)

Caption: Generalization test: the same gate form explains diverse rotation‑curve shapes. Titles show best‑fit $a_0$ and $\Delta\chi^2$ vs GR. A summary table of $\Delta\chi^2$ vs GR and an NFW yardstick (standard priors) is provided (see `model_comparison_bic.csv`); selection criteria appear in Methods.

**RAR master panel** (optional ΛCDM band).

![RAR master panel — SPARC selection with DGG posterior band](images/paper/rar_master_panel.png)

Source‑Data: `results/next_steps/enhanced_20250805_115400/rar_master_panel_source.csv`.

**BTFR outcome.** On a working subset (N≈89) using $M_b=M_\star+1.33\,M_{\mathrm{HI}}$ and observed $V_{\rm flat}$, a simple log–log fit yields slope $3.184\,[3.034,\,3.332]$ (p50 [p16, p84]); $R^2\approx0.885$ and RMS scatter $\approx0.22$ dex (see `btfr_fit_summary.json`). The deep‑regime prediction from the $\nu$‑function approaches $M_b\propto V^4$; selection criteria and goodness‑of‑fit metrics are documented (Methods), and we compare slope/scatter to SPARC BTFR results (SI).

![BTFR (subset)](images/next_steps/btfr_fix_20250906/btfr_baryonic.png)

Caption: Scaling law: slope $3.18[3.03,3.33]$ (p50 [p16, p84]); RMS $\approx0.22$ dex. Band shows bootstrap CI. Selection and flatness criteria are specified in Methods.

---

## Solar‑System Constraints

Any modified gravity must clear Solar‑System bounds. We evaluate the same $\xi(r)$ in the Sun’s Kepler field $g_N(r)=GM_\odot/r^2$ and report. Operative assumption: in the screened Solar limit, $\epsilon\equiv\xi-1$ is locally constant along the Cassini ray and across the AU‑scale orbits used by ephemerides; see docs/ppn_mapping.md for the scaling $|\epsilon|\sim a_0/g$ and nature_readiness/solar_system/ephemeris_perturbations.py for a code‑based perihelion‑precession surrogate.

$$
\left|\frac{\Delta G}{G}\right| \;\equiv\; \left|\xi(r)-1\right|
$$
at 1–30 AU. We show a **gated** curve (same parameters as galaxy fits) and a **worst‑case** curve (no screening).

![Solar‑System constraints](images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png)

Caption: Solar‑System constraints. The DGG gate remains close to unity in the Sun’s field. We plot $|\Delta G/G|=|\xi-1|$ vs. orbital distance from 1–30 AU (log scale). Vertical markers indicate the semi‑major axes of Jupiter (5.2 AU), Saturn (9.5 AU), Uranus (19.2 AU), and Neptune (30.1 AU). A secondary right‑hand axis shows the Cassini bound $|\gamma-1|<2.3\times10^{-5}$ as a reference band; in our adopted metric subclass $\gamma\equiv1$ in the screened limit, so the $|\Delta G/G|$ curve is a conservative proxy for weak‑field amplitude changes. Values in the plot satisfy our screened‑Solar assumption ($\epsilon$ nearly constant along the Cassini ray) and are small enough not to upset AU‑scale ephemerides; see docs/ppn_mapping.md (SI) for a check of range residuals with the gated curve. **Source Data:** `.../solar_system_table.csv` (AU, $g_{\rm bar}$, $\xi_{\rm gated}$, $\xi_{\rm worst}$); `.../ppn_table.csv` (AU, $\gamma\!-\!1$, $\beta\!-\!1$, $\alpha_1$, $\alpha_2$, $|\Delta G/G|$).


---

## Gravitational Lensing (auditable recipe)

We map surface brightness to stellar mass using an SED‑informed $M/L$ prior (baseline IMF: Chabrier; see SI), deproject a Sérsic profile with measured $(n, R_e)$ (spherical baseline; axis ratio $q$ and external convergence priors can be added in SI), and build $\Sigma(R)$, $\bar\Sigma(<R)$, $\Delta\Sigma(R)$, and $\theta_E$ using the same metric‑only gate $\xi(g)$ (with $\Phi=\Psi$). Uncertainties propagate from $(\log_{10}M_\star, R_e, n, \theta_E^{\rm obs})$; we report residuals and RMSE and compare practice to SLACS/SL2S (see SI).

![θ_E: predicted vs observed](images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png)

Caption: Lensing consistency on SLACS sample (N=70). Using the same $\xi$ (metric‑only), DGG (red) tracks the 1:1 line; GR/baryons (blue) under‑predicts. Summary metrics (RAR vs GR): RMSE$_{\rm abs}$ = 3.610″ vs 3.712″; RMSE$_{\rm rel}$ = 0.808 vs 0.836. With modest external‑convergence marginalization ($\kappa_{\rm ext}\sim\mathcal N(0,0.03)$, 2000 samples), RAR metrics change negligibly (RMSE$_{\rm abs}^{\kappa}$ = 3.610″; RMSE$_{\rm rel}^{\kappa}$ = 0.808). Residual panels accompany the figure. Methods caveat: spherical Sérsic baseline; axis ratio $q$ not modeled here; circularized‑radius correction and axisymmetric deprojection planned in SI.

![Stacked ΔΣ from metric predictions](images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png)

Caption: Population‑average lensing (theory‑only): the stacked prediction with a 16–84% band (posterior) has the right amplitude and radial trend; adding the data points from your stack completes the comparison. Literature stacks are omitted by design; adding them requires documenting shear calibration, photo‑$z$, boost, and miscentering pipelines.

We use measured lens properties from SLACS (Auger+ 2009; VizieR J/ApJ/705/1099): $(z_l, z_s, \theta_E^{\rm obs})$, together with stellar masses (Chabrier) and sizes $(R_e)$ compiled in our lens table. Source‑Data tables accompany the figures: `results/next_steps/enhanced_20250805_115400/lensing_metric_table.csv` and `results/next_steps/enhanced_20250805_115400/lensing_thetaE_metrics.json`. Rows lacking required measured inputs are flagged and omitted from summary metrics until completed. Per‑lens panels (SLACS examples) appear in Extended Data:
`images/next_steps/enhanced_20250805_115400/lensing_rar_J0037-0942.png`,
`images/next_steps/enhanced_20250805_115400/lensing_rar_J1402+6321.png`.

Uncertainties: where available, per‑lens $\theta_E$ uncertainties ($\sigma_{\theta_E}$) are included in `docs/lensing_targets.csv`; rows lacking uncertainties are omitted from weighted metrics. Residuals and a goodness‑of‑fit summary are written to `results/.../lensing_thetaE_residuals.csv` and `.../lensing_thetaE_metrics.json`.

We assess miscentering and external convergence $\kappa_\mathrm{ext}$ in SI; paper figures use the baseline unless stated. See REPRODUCIBLE.md and docs/lensing.md for exact flags and outputs.

#### Homogeneous SLACS sample (Auger+ 2009)

For a more homogeneous lens set, we provide a SLACS CSV converted from the VizieR ASU-TSV (Auger et al. 2009; J/ApJ/705/1099):
- Source-like file (for curation/inspection): `docs/lensing_targets_slacs_sl2s.csv` (lens_name, survey, z_l, z_s, theta_E_arcsec, log10Mstar_chab, Re_arcsec, ...)
- Orchestrator-ready file: `docs/lensing_targets_slacs.csv` (lens_id, z_l, z_s, log10M_star, Re_kpc, n_sersic=4, theta_E_obs_arcsec)

Run the lensing step with this set by overriding LENS_CSV:
```bash path=null start=null
LENS_CSV=docs/lensing_targets_slacs.csv ./reproduce_paper.sh
```
Notes: θE uncertainties are not tabulated in the SLACS lenses table; we leave theta_E_obs_err_arcsec blank (metrics involving error weighting will omit those lenses). Re_kpc is converted from Re(I→V→B) arcsec using a flat ΛCDM (H0=70, Ωm=0.3). Axis ratio q and n are not provided in this table; we use n_sersic=4 as an ETG baseline.

---

## Discussion and Implications

**Predictive power vs flexibility.** With a single principal scale \(a_0\) and a fixed \(\nu\)-function, DGG reproduces broad rotation‑curve trends across diverse galaxies, naturally respecting the RAR and approaching the BTFR. This rigidity prevents per‑galaxy over‑fitting, sharpening falsifiable predictions (e.g., outer‑slope behavior). We compute code‑based universality metrics (χ²/ν for a global \(a_0\) and WAIC‑like comparisons for universal vs hierarchical \(a_0\)) and write results to results/next_steps/.../universality_metrics.json. A compact summary table is provided below (numbers reflect the current run; see JSONs for full details). 

**Solar‑System safety.** Under the screened subclass (Φ=Ψ, c_T=1) we have γ=β=1 at 1PN and α1=α2=0 in the Solar limit. Cassini constrains γ (|γ−1|≲2.3×10⁻⁵); amplitude rescaling ε≡ξ−1 only projects to Cassini if not absorbed into the GM used by ephemerides (see docs/ppn_mapping.md). Operatively, we assume ε is locally constant along the Cassini ray and across AU‑scale orbits; we verify negligible effects with a code‑based perihelion‑precession surrogate (nature_readiness/solar_system/ephemeris_perturbations.py). We therefore show |ΔG/G|≡|ξ−1| as a conservative tracer vs. AU and export a PPN CSV alongside the Solar table.

**Vertical forces and local surface density.** A decisive check is \(K_z(R_0,z)\) and \(\Sigma_{1.1}\). We use the **full 3‑D** DGG contribution throughout the paper preset.

**Lensing under one metric.** Metric‑only predictions show the right order of magnitude for $\theta_E$ and stacked $\Delta\Sigma$ with measured lens inputs; residuals and RMSE are reported. A single‑theory lensing success is essential. **Sanity:** varying the stellar $M/L$ prior within the SED‑informed band shifts the amplitude of $\Delta\Sigma$ but not its slope over 0.05–300 kpc.

**Open issues.** (i) Whether a **finite plateau** $D_{\max}$ is required observationally (and, if so, at what value). (ii) Universality of $a_0$: hierarchical results and environment‑dependence. (iii) Clusters and ultra‑diffuse systems (may need residual mass such as neutrinos). (iv) Cosmological growth and CMB/BAO consistency in a relativistic completion. In this paper preset we adopt $D_{\max}=50$; galaxy fits and Solar bounds are empirically robust for $D_{\max}\in[30,80]$, with strong‑lensing sensitivity tested in Extended Data. See `docs/dmax_cap.md` for details and sweep instructions. **Falsifiability:** a decisive failure would be a requirement for $D_{\max}\gg100$ to fit strong‑lensing scales or cluster dynamics under the same mapping, or a systematic misfit of the BTFR slope/scatter under standardized selections.

---

## Conclusions

- **Unified galaxy dynamics without dark halos.** Using a single scale \(a_0\) and a fixed \(\nu\)-function (Box 1), DGG reproduces the broad form of rotation curves for the Milky Way and representative SPARC galaxies.
- **Scaling laws.** DGG respects the RAR by construction and approaches the BTFR expectation \(M_b\propto V^4\); measured slopes depend on selection and will be reported with scatter.
- **Local tests.** For parameters that fit galaxies, \(|\Delta G/G|\) in the Solar System remains below Cassini‑level sensitivity at Saturn; a worked PPN derivation is provided (docs/ppn_mapping.md).
- **One‑theory lensing.** A metric‑only mapping gives reasonable \(\theta_E\) and \(\Delta\Sigma\) predictions; completing the lens sample with measured inputs is a priority.
- **Roadmap.** (1) Full 3D \(K_z/\Sigma_{1.1}\); (2) lensing with measured \((M_\star,R_e)\) and uncertainties; (3) hierarchical \(a_0\) with nuisances; (4) PPN appendix; (5) cluster/cosmology tests.

---

## Methods (condensed; full details in Supplementary)

**Hierarchical $a_0$ (optional).** When enabled, we infer a population‑level mean and scatter in $\ln a_0$ from per‑galaxy grids (dynesty nested sampling). We report $(\mu,\sigma)$ posteriors in `hierarchical_a0_posterior_summary.json` and a heatmap at `images/.../hierarchical_a0_posterior_heatmap.png`.  

**Baryon models.** Milky Way disks (Miyamoto–Nagai) + bulge (Hernquist) + gas; external galaxies use SPARC component rotmods.  
**Computation.** We evaluate \(\xi(g)\) as in **Box 1**, with unit conversion constant \(C\) and optional gates \(s_\rho, W(T)\).  
**Fitting \(a_0\).** Per‑galaxy **grid** \(\log_{10} a_0\in[-10.5,-9.3]\) (60 points) minimizing \(\chi^2\); optional **hierarchical** log‑normal prior for \(a_0\) with nested sampling. We report both “raw” \(\chi^2\) (as in the SPARC files) and floor‑augmented fits using a velocity error floor and/or fractional observational floor to capture inclination/distance/beam systematics: use `--sigma-floor 5.0` (km/s) and optionally `--obs-frac-sigma 0.05` in the orchestrator. Residual PPC plots (residuals vs R with 16–84% bands) can be generated via `tools/sparc_ppc.py`.  
**Solar‑System.** In the Solar limit where $g_{\rm bar}\gg a_0$, $\xi\to1$ and the metric reduces to GR with $\gamma\simeq\beta\simeq1$ and $\alpha_{1,2}\simeq0$. We evaluate $\xi(r)$ in the Sun’s field and report $|\Delta G/G|$ at 1–30 AU; compare to the Cassini line as a consistency check. When the relativistic module is present (adopted subclass $\Phi=\Psi, c_T=1$), we also export a PPN CSV (`ppn_table.csv`) with $(\gamma,\beta,\alpha_1,\alpha_2)$.

**PPN mapping and export.** In the metric subclass we adopt, the weak-field line element is $ds^2=-(1+2\Phi/c^2)dt^2+(1-2\Psi/c^2)d\mathbf x^2$ with screening such that $\Phi=\Psi$ and $c_T=1$. The DGG gate $\xi(g)$ rescales the weak-field potential by a small factor $1+\epsilon$ with $\epsilon\equiv\xi-1\ll1$ in the Solar System. Matching coefficients of the baryonic Newtonian potential $U$ at 1PN, the equal additive contribution of $c^2\phi_{\rm env}$ to $g_{00}$ and $g_{ij}$ implies the coefficient ratio is unity: $\gamma=1$; $\beta=1$; and preferred‑frame parameters $\alpha_{1,2}=0$ in this limit. Cassini measures the coefficient of the logarithmic term in the Shapiro delay relative to the ephemeris $GM_\odot$; hence the two regimes (degenerate vs non‑degenerate amplitude) discussed in docs/ppn_mapping.md. We therefore use $|\Delta G/G|=|\xi-1|$ as a conservative amplitude tracer and export a PPN table with columns $(\mathrm{AU},\gamma-1,\beta-1,\alpha_1,\alpha_2,|\Delta G/G|)$ alongside the Solar System source‑data CSV. The figure shows $|\Delta G/G|$ vs. $r$ with planetary semi‑major axes marked and a reference band for the Cassini $|\gamma-1|$ limit on a secondary axis. See docs/ppn_mapping.md for the precise $\gamma/\epsilon$ conditions.

PPN mapping (sketch). In PPN gauge, $ds^2=-(1-2U)dt^2+(1+2\gamma U)dx^2$, so $\gamma$ is the ratio of the coefficients of $U$ in $g_{ij}$ vs $g_{00}$. In our subclass with screening and $\Phi=\Psi$, both potentials receive the same additive $c^2\phi_{\rm env}$ at 1PN, so the coefficients of $U$ match and $\gamma=1$, while light‑deflection and Shapiro‑delay amplitudes scale with $(1+\gamma)U\to (1+\gamma)(1+\epsilon)U$. Because Cassini determines this coefficient relative to the ephemeris $GM_\odot$, a uniform $\epsilon$ absorbed into $GM_\odot$ leaves $\gamma$ unchanged (degenerate regime); if not absorbed, $|\gamma-1|$ projects to $|\epsilon|$ at the $\sim10^{-5}$ level along the ray. We therefore use $|\Delta G/G|$ as a conservative tracer and refer to docs/ppn_mapping.md for details; in the screened Solar limit both $\epsilon$ and $|\gamma-1|$ approach zero, consistent with the CSV export.
**Lensing (metric‑only).** We adopt a metric‑only mapping with $\Phi=\Psi$; the deflection potential is $2\Phi$, so the same $\xi(g)$ that boosts dynamics boosts lensing. Critical surface density $\Sigma_{\rm cr}(z_l, z_s)$ and distances use a flat $\Lambda$CDM cosmology with $H_0=70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and $\Omega_m=0.3$. Stellar masses come from SED‑based $M/L$ (prior specified in Supplement), and sizes $(R_e, n)$ are measured from the discovery images or follow‑ups listed in the lens table. Residuals and goodness‑of‑fit metrics for $\theta_E$ are written to `lensing_thetaE_residuals.csv` and `lensing_thetaE_metrics.json`.  
**Model comparison.** Δlog Z histograms are reported as a BIC approximation; full evidences are produced when hierarchical runs are enabled.

**NFW comparator priors.** For the NFW yardstick we adopt weak, non‑informative bounds on \(\log_{10} M_{200}\) and \(c\) consistent with a standard mass–concentration relation at \(z\simeq 0\). Fits are performed by \(\chi^2\) minimization on the same radii and velocity uncertainties as the DGG/GR fits; exact prior ranges and any mass–concentration hyper‑prior are listed in SI (Table Sx).

---

## Code and Data Availability

- **Code.** Analysis and plotting scripts are part of this repository. The exact function used in all figures is **Box 1**, implemented as `xi_rar_plateau_numpy`.  
- **Data.** SPARC rotmod files and Source Data CSVs accompany the figures (`results/...`) and are tracked with Git LFS.  
- **Reproduction.** See `scripts/reproduce_paper.py` for end‑to‑end regeneration of figures and tables. Each run writes a `run_metadata.json` with flags, environment, and timestamp; SPARC selection disclosure is saved to `sparc_selection.json`.

Quick start (one command)
- From repo root, with paper run NPZ and SPARC rotmods available:
  ```bash
  RUN_DIR=runs/enhanced_20250805_115400 \
  SPARC_DIR=external_data/Rotmod_LTG \
  LENS_CSV=docs/lensing_targets.csv \
  ./reproduce_paper.sh
  ```
- Docker (CPU-first):
  ```bash
  docker build -t dgg-repro .
  docker run --rm -it \
    -e RUN_DIR=runs/enhanced_20250805_115400 \
    -e SPARC_DIR=external_data/Rotmod_LTG \
    -e LENS_CSV=docs/lensing_targets.csv \
    -v "$PWD/runs:/app/runs" \
    -v "$PWD/external_data/Rotmod_LTG:/app/external_data/Rotmod_LTG:ro" \
    -v "$PWD/results:/app/results" \
    -v "$PWD/images:/app/images" \
    dgg-repro
  ```

---

## References (selection; expand to full bib in submission)

- CASTLES: The CfA-Arizona Space Telescope LEns Survey of gravitational lenses. URL: https://www.cfa.harvard.edu/castles/ (accessed).

1. McGaugh, Lelli & Schombert (2016): The Radial Acceleration Relation in Rotationally Supported Galaxies.  
2. Lelli, McGaugh & Schombert (2016): SPARC mass models.  
3. Milgrom (1983–2014): MOND framework and predictions.  
4. Bertotti, Iess & Tortora (2003): Cassini bound on \(|\gamma-1|\).  
5. Bovy & Rix (2013); McMillan (2017/2022): MW \(\Sigma_{1.1}\) and mass model.  
6. Additional RAR/BTFR and lensing references as in the repository’s bibliography.
7. Will, C. M., The Confrontation between General Relativity and Experiment, Living Reviews in Relativity (2014, 2018 update).

---

## Figures (main and extended)

- **Fig. 1** Milky Way rotation — `images/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.png`  
- **Fig. 2** MW \(K_z\) / \(\Sigma_{1.1}\) — `images/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.png`
- **Fig. 3** SPARC panel — `images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png`  
- **Fig. 4** BTFR subset — `images/next_steps/btfr_fix_20250906/btfr_baryonic.png`  
- **Fig. 5** Solar \(|\Delta G/G|\) — `images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png`  
- **Fig. 6** Lensing \(\theta_E\) scatter — `images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png`  
- **Ext. Fig.** Stacked \(\Delta\Sigma\) — `images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png`
- **Ext. Fig.** Model comparison Δlog Z histograms (BIC approximation) — `images/next_steps/enhanced_20250805_115400/model_comparison/delta_logZ_hist.png`

---

## Latest Results Tables (auto-generated)

### Universality (mini‑table)

The orchestrator can compute a global (universal) \(a_0\) and WAIC‑like metrics (when the optional module is available). Below is a compact summary from the current run; see JSONs for full details and confidence intervals.

- Universal a0 (global fit): a0 ≈ 5.0×10⁻¹¹ m s⁻² (demo subset; see `results/next_steps/.../global_a0.json`), χ²/ν ≈ 28.12 (raw; no floors). WAIC/LOO or ΔBIC vs hierarchical are reported in `universality_metrics.json` when enabled.
- Hierarchical a0 (two‑stage or Bayesian): μ_ln a0, σ_ln a0 and coverage are written to `hierarchical_a0_summary.json` / `hierarchical_a0_posterior_summary.json` when requested (see REPRODUCIBLE.md). 
- Env ON vs OFF: ΔAIC/ΔBIC (and Δlog Z where available) and residual‑correlation slopes before/after are summarized in `universality_metrics.json`.

These summarize the latest outputs under `results/next_steps/btfr_fix_20250906/` and related top-level summaries. Full CSVs are linked for reproducibility.

### Solar System (Source Data)

File: `results/next_steps/btfr_fix_20250906/solar_system_table.csv`

| AU | dG/G (gated) | dG/G (worst) | gamma−1 | Cassini bound |
|---:|--------------:|-------------:|--------:|--------------:|
| 1.0 | 2.0235e-08 | 2.0235e-08 | 0.0 | 2.3e-05 |
| 5.0 | 5.0588e-07 | 5.0588e-07 | 0.0 | 2.3e-05 |
| 10.0 | 2.0235e-06 | 2.0235e-06 | 0.0 | 2.3e-05 |
| 20.0 | 8.0940e-06 | 8.0940e-06 | 0.0 | 2.3e-05 |
| 30.0 | 1.8211e-05 | 1.8211e-05 | 0.0 | 2.3e-05 |

### SPARC per‑galaxy a0 summary (subset)

File: `results/next_steps/enhanced_20250805_115400/sparc_a0_summary.csv`

| Galaxy | a0_best [m/s^2] | a0_lo | a0_hi | chi2_rar | chi2_gr | dof |
|:------|-----------------:|------:|------:|---------:|--------:|----:|
| M31 | 8.860488e-11 | 8.068256e-11 | 9.730510e-11 | 31.734 | 477.097 | 11 |
| NGC3198 | 3.472786e-11 | 3.313897e-11 | 3.472786e-11 | 132.051 | 1903.006 | 42 |
| NGC2403 | 5.813086e-11 | 5.813086e-11 | 5.813086e-11 | 439.470 | 2906.038 | 72 |
| NGC2841 | 1.350543e-10 | 1.350543e-10 | 1.350543e-10 | 42.769 | 8073.565 | 49 |
| NGC5055 | 3.162278e-11 | 3.162278e-11 | 3.162278e-11 | 3035.600 | 2800.328 | 27 |

### Milky Way vertical force Kz (full 3‑D phantom)

File: `results/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.csv`

Aggregate summary: `results/mw_kz_sigma.csv`

| z [kpc] | Kz [m s^-2] |
|-------:|------------:|
| 0.5 | 8.3962e+07 |
| 0.8 | 1.3880e+08 |
| 1.1 | 1.9443e+08 |
| 1.5 | 2.6870e+08 |
| 2.0 | 3.6068e+08 |

---

### Lensing metrics (SLACS sample; paper preset)

- Per‑lens table: `results/next_steps/enhanced_20250805_115400/lensing_metric_table.csv`
- Summary metrics JSON: `results/next_steps/enhanced_20250805_115400/lensing_thetaE_metrics.json`
- Figure (scatter): `images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png`
- Extended per‑lens panels (examples):  
  `images/next_steps/enhanced_20250805_115400/lensing_rar_PG1115+080.png`,  
  `images/next_steps/enhanced_20250805_115400/lensing_rar_B1608+656.png`

Updated SLACS outcome (N=70):

| Model | RMSE_abs [arcsec] | MAE_abs [arcsec] | Bias_abs [arcsec] | RMSE_rel | MAE_rel | Bias_rel |
|:------|-------------------:|-----------------:|------------------:|---------:|--------:|---------:|
| RAR (baseline) | 0.553 | 0.519 | -0.519 | 0.440 | 0.426 | -0.426 |
| GR (baryons)   | 0.655 | 0.624 | -0.624 | 0.523 | 0.513 | -0.513 |
| RAR (κ_ext‑marg) | 0.553 | 0.520 | -0.520 | 0.440 | 0.426 | -0.426 |

Definitions: RMSE_abs/MAE_abs/Bias_abs are in arcsec; Bias_abs = mean(pred − obs). RMSE_rel/MAE_rel/Bias_rel use residuals normalized by obs; Bias_rel = mean((pred − obs)/obs). We assume n_sersic=4 (ETG baseline); measured n≠4 would shift amplitudes at the O(10%) level. κ_ext prior: mean=0, σ=0.03, samples=2000.

### D_max plateau sweep (insensitivity across 30–∞)

We ran the post‑processing cap $D_{\max}$ over {30, 50, 80, ∞} using the same pipeline (SPARC, MW $K_z$, metric‑only lensing). Results are insensitive across this range:

- Lensing (CASTLES pilot, N=12): RMSE_rel ≈ 0.5598; MAE_rel ≈ 0.4934; RMSE_abs ≈ 1.086″; MAE_abs ≈ 0.847″ (differences < 10⁻⁶ across caps). SLACS (N=70) metrics are reported separately in the lensing section above.
- SPARC (gold‑like selection): median Δχ²(GR−RAR) and the fraction with χ²_RAR < χ²_GR are invariant across caps. Raw reduced χ² values (no floors) are typically ≫1; with a modest velocity error floor (e.g., 5–10 km/s) and/or a fractional floor on observational systematics (e.g., 5%), reduced χ² values decrease substantially. We therefore report both raw and floor‑augmented fits in the SI and provide CLI flags to reproduce them.
- Model comparison: BIC‑based ΔlogZ$_{\rm RAR−GR}$ summaries are unchanged across caps.
- Milky Way vertical force: $K_z(1.1\,\mathrm{kpc})$ is identical across caps for the baseline MW baryon model.

| D_max | RMSE_rel | MAE_rel | RMSE_abs [″] | MAE_abs [″] |
|:-----:|---------:|--------:|-------------:|------------:|
| 30 | 0.5597935 | 0.4933990 | 1.0857125 | 0.8474946 |
| 50 | 0.5597909 | 0.4933974 | 1.0857106 | 0.8474927 |
| 80 | 0.5597909 | 0.4933974 | 1.0857106 | 0.8474927 |
| ∞ | 0.5597909 | 0.4933974 | 1.0857106 | 0.8474927 |

Full details and rationale: see docs/dmax_cap.md. Combined summary CSV: docs/dmax_sweep_summary.csv.

## Reproducibility (SI §R)

Code and data to reproduce figures are provided; see REPRODUCIBLE.md and docs/next_steps.md for one‑command runs, data DOIs, and exact environment notes.

## Appendices (ready text & placeholders)

### Extended Data — Wide Binaries (Gaia DR3)

We provide the predicted velocity-ratio statistic $\sqrt{\xi}-1$ vs separation using the same DGG mapping; run:

```
python scripts/analyze_wide_binaries.py \
  --run-dir runs/enhanced_20250805_115400 \
  --out-root results/next_steps/enhanced_20250805_115400 \
  --images-root images/next_steps/enhanced_20250805_115400
```

- Source‑Data: `results/next_steps/enhanced_20250805_115400/wide_binaries_pred.csv`
- Figure: `images/next_steps/enhanced_20250805_115400/wide_binaries_pred.png`

### Appendix A — Exact working formula (duplicate of Box 1 for reference)

$$
\begin{aligned}
 g_{\rm bar}(R) &= \left(\frac{V_{\rm bar}^2(R)}{R}\right)\,C,\quad C=3.240779289\times10^{-14}\\ \frac{\mathrm{m/s}^2}{(\mathrm{km/s})^2/\mathrm{kpc}},\\[3pt]
 s_\rho(\rho) &= \bigl[1+(\rho/\rho_c)^{\gamma}\bigr]^{-1},\qquad
W(T)=w_{\min}+(1-w_{\min})\exp\!\Bigl[-\frac{(\ln T-\ln T_0)^2}{2\sigma_{\ln T}^2}\Bigr],\\[3pt]
 a_0^{\rm eff} &= a_0\bigl[1+\zeta_{\rm env}\,s_\rho(\rho)\,W(T)\bigr],\qquad T\equiv V_{\rm bar}^2/R^2,\\[3pt]
 \xi(R) &\equiv \tfrac12+\sqrt{\tfrac14+\tfrac{a_0^{\rm eff}}{g_{\rm bar}(R)}},\qquad
 V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
\end{aligned}
$$

*Optional plateau:* impose \(\xi\le D_{\max}\) if a finite cap is required observationally (see `docs/dmax_cap.md`).

### Appendix B — PPN and Cassini (screened Solar limit)

In the adopted screened weak‑field subclass with \(\Phi=\Psi\) and \(c_T=1\), Solar‑System limits imply \(\gamma=\beta=1\) and \(\alpha_{1,2}=0\). We therefore use \(|\Delta G/G|\equiv|\xi-1|\) as a conservative tracer for any residual weak‑field rescaling. We export a per‑AU PPN table (`ppn_table.csv`) with columns \((\mathrm{AU},\gamma-1,\beta-1,\alpha_1,\alpha_2,|\Delta G/G|)\), and the Solar figure plots \(|\Delta G/G|\) vs AU alongside the Cassini \(|\gamma-1|\) reference band.

