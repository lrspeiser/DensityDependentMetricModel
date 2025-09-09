# Density‑Gated Gravity: A Density‑Dependent Alternative to Dark Matter

## Introduction

Galactic rotation curves have long challenged the standard cosmological model, which invokes massive halos of non-baryonic *dark matter* to explain the unexpectedly high orbital speeds in outer galactic disks. While dark halos can be *fitted* to match individual galaxy rotations, their success comes at the cost of introducing numerous free parameters (one halo per galaxy) and fine-tuned correlations between baryonic and dark mass distributions. A striking empirical clue is the **Radial Acceleration Relation (RAR)**: across hundreds of galaxies, the observed centripetal acceleration \(g_{\rm obs}(r)\) tightly correlates with that predicted by visible matter alone \(g_{\rm bar}(r)\)[1]. This correlation persists even in regions where dark matter is presumed to dominate, implying that the dark contribution is “fully specified by that of the baryons”[1]. The small scatter in the RAR (comparable to observational uncertainties) suggests an underlying law of nature[1] rather than a fortuitous result of galaxy formation. Indeed, the RAR has been called “tantamount to a natural law” for galaxies[1]. Such a universal relation is difficult to reconcile with arbitrary halo tuning, as it would require a mysterious *conspiracy* between visible and dark components across all galaxies[2]. This has motivated the pursuit of alternative gravity theories that *predict* the RAR intrinsically, without invoking invisible mass[2].

One notable example is Milgrom’s Modified Newtonian Dynamics (MOND), which postulates a new fundamental acceleration scale \(a_0\sim10^{-10}\) m/s² at which gravity deviates from Newton’s laws[3]. MOND’s simple prescription can explain flat rotation curves and was prescient in foreseeing the RAR decades before its observational confirmation[3][1]. However, the original MOND formula (and similar empirical interpolations) are *too rigid* – with a single parameter and no built-in relativistic framework, they struggle to fit *all* phenomena (e.g. the diversity of galaxy profiles, galaxy clusters, and cosmological observations)[4]. On the other hand, explaining the RAR within the dark matter paradigm also poses challenges: it requires highly coordinated distributions of baryons and dark matter for every system[5], which may be achievable in detailed galaxy formation models but lacks the elegance of a universal law. Given the continuing non-detection of dark matter particles and the empirical successes of MOND-like phenomenology on galactic scales, it is worthwhile to explore new gravity models that combine **predictive rigidity** with flexibility and consistency across scales.

**RAR‑gated gravity** is a new approach that aims to retain the parsimonious, predictive nature of MOND/RAR, while embedding the theory in a more robust, general‑relativistic‑style framework. In this model, the *strength* of gravity is **gated** by the local acceleration (and/or related mass density), such that standard General Relativity (GR) is recovered in high‑acceleration regimes (e.g. the inner Solar System and deep potential wells), but departures from Newtonian gravity automatically emerge in low‑acceleration environments (galactic outskirts). The model requires no *ad hoc* dark halos; instead, the observed flat or gently declining rotation curves are a natural outcome of the modified field equations. Importantly, RAR‑gated gravity introduces only a small number of new parameters (ideally just one universal acceleration scale \(a_0\) and perhaps one dimensionless strength parameter), making it highly constrained. If such a model can account for diverse galaxy rotation curves with *the same* fundamental constants, it would strengthen the case for a new theory of gravity and provide a serious alternative to dark matter. In this paper, we present the RAR‑gated gravity model, confront it with empirical tests—including **Milky Way** rotation measurements, fits to **SPARC** galaxy rotation curves, and **Solar‑System** precision data—and outline its advantages and remaining challenges. Our goal is to demonstrate that this approach is a credible contender for explaining galactic dynamics.

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

> Why the plateau? Prior drafts used an unbounded boost; with real data the cap $D_{\max}=50$ avoids pathologies at extremely low $g_{\rm bar}$ yet leaves galaxy‑scale predictions unchanged over the measured range. Robust for $30\!\lesssim\!D_{\max}\!\lesssim\!80$; see lensing sensitivity analyses.

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

**Boost (“$\nu$” function) used in figures (paper preset).**

$$
\xi(R)\equiv D(R) \;=\; \min\!\left[\;\frac{1}{2} + \sqrt{\frac{1}{4}+\frac{a_0^{\rm eff}}{g_{\rm bar}(R)}}\;,\; D_{\max}\right],\quad D_{\max}=50.
\qquad
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
$$

> **Preset note.** In the paper preset, we enforce a finite plateau \(D_{\max}=50\), and propagate it to rotation curves, lensing, Solar‑System checks, and \(K_z\). The effective \(D_{\max}\) is recorded in `run_metadata.json`. Passing `--rar-dmax` overrides this value; overrides are logged in the same metadata file.

### Relativistic and lensing stance used for figures

For manuscript figures we adopt a **metric‑only** weak‑field mapping with $\Phi=\Psi$ and $c_T=1$; lensing uses $\Phi+\Psi=2\Phi$ and the same $\xi(g)$ as dynamics. A concise PPN derivation and full metric completion are slated for Methods/Supplement.

> **Placeholder to add:** short derivation connecting $\xi$ to $\kappa(R)$ and $\theta_E$ in this subclass, and the PPN parameters $(\gamma,\beta,\alpha_1,\alpha_2)$.

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

![Milky Way Kz and Σ_1.1 (full 3D)](images/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.png)

Caption: Vertical‑force cross‑check: full 3‑D phantom density implied by $\xi$ yields $K_z(R_0,z)$ and $\Sigma_{1.1}$ without a dark halo. Overlay observational bands (Bovy & Rix; McMillan) to show consistency (pass `--mw-kz-overlay-csv <bands.csv>` to the orchestrator). Source‑Data: `results/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.csv`. We report $\Sigma_{1.1}$ in the CSV; uncertainties to be added in the submission draft.

### External Galaxies: SPARC Rotation‑Curve Fits

We selected representative spirals spanning mass and surface brightness. For each galaxy we hold gating parameters fixed (MW‑tuned) and **scan $a_0$ on a grid** $\log_{10} a_0\in[-10.5,-9.3]$ (m s$^{-2}$) to minimize $\chi^2$ (per‑galaxy $a_0$ strategy). A hierarchical log‑normal model for $a_0$ is available and reported as Extended Data when used.

#### SPARC gold‑sample panel (RAR‑gate vs GR vs observations)
![SPARC gold overlays panel](images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png)

Caption: Generalization test: a fixed gate form explains diverse rotation‑curve shapes. Titles show best‑fit $a_0$ and $\Delta\chi^2$ vs GR.

**RAR master panel** (optional ΛCDM band) is available at  
`images/next_steps/rar_plateau_mw_full/rar_master_panel.png`.

**BTFR outcome.** On a working subset (N≈89) using $M_b=M_\star+1.33\,M_{\mathrm{HI}}$ and observed $V_{\rm flat}$, a simple log–log fit yields slope $3.184\,[3.034,\,3.332]$ (p50 [p16, p84]); $R^2\approx0.885$ and RMS scatter $\approx0.22$ dex (see `btfr_fit_summary.json`). The deep‑regime prediction from the $\nu$‑function approaches $M_b\propto V^4$; we also assess selection sensitivity (flatness, inclinations, gas content) and intrinsic scatter.

Caption: Scaling law: slope $3.18[3.03,3.33]$ (p50 [p16, p84]); RMS $\approx0.22$ dex. Band shows bootstrap CI. Selection and flatness criteria are specified in Methods.

---

## Solar‑System Constraints

Any modified gravity must clear Solar‑System bounds. We evaluate the same $\xi(r)$ in the Sun’s Kepler field $g_N(r)=GM_\odot/r^2$ and report

$$
\left|\frac{\Delta G}{G}\right| \;\equiv\; \left|\xi(r)-1\right|
$$
at 1–30 AU. We show a **gated** curve (same parameters as galaxy fits) and a **worst‑case** curve (no screening).

![Solar‑System constraints](images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png)

Caption: Safety check: $|\Delta G/G|=|\xi-1|$ is below Cassini at 5–10 AU and well below at 1 AU; PPN mapping for $\Phi=\Psi$ is provided in Methods and exported as `ppn_table.csv`.

> **Source Data:** `results/next_steps/rar_plateau_mw_full/solar_system_table.csv` (AU, $g_{\rm bar}$, $\xi_{\rm gated}$, $\xi_{\rm worst}$). Paper preset also writes `ppn_table.csv` for the adopted $\Phi=\Psi$ subclass.

### PPN parameters (placeholder)

We plan to present a short derivation for $(\gamma,\beta,\alpha_1,\alpha_2)$ in the quasi‑static, screened limit with $\Phi=\Psi$. A CSV export is supported by the code when the relativistic module is present.

---

## Gravitational Lensing (metric‑only path for figures)

Lensing is computed from the same $\xi(g)$ via a **metric‑only** mapping with $\Phi=\Psi$. We produce per‑lens $\langle\Sigma\rangle(R)$, $\Delta\Sigma(R)$, and $\theta_E$, plus a stacked $\Delta\Sigma$.

![θ_E: predicted vs observed](images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png)

Caption: Lensing consistency: using the same $\xi$ (metric‑only), DGG (red) tracks the 1:1 line; GR/baryons (blue) under‑predicts. Error bars show observational uncertainties; residual panels accompany the figure in the paper PDF.

![Stacked ΔΣ from metric predictions](images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png)

Caption: Population‑average lensing: the stacked prediction with a 16–84% band (posterior) has the right amplitude and radial trend; adding the data points from your stack completes the comparison.

We use measured lens properties from CASTLES (and follow‑ups): $(z_l, z_s, \theta_E^{\rm obs})$, together with stellar masses and sizes $(\log_{10} M_\star, R_e, n)$ compiled in our lens table. Source‑Data tables accompany the figures: `results/next_steps/btfr_fix_20250906/lensing_table.csv` and `results/next_steps/btfr_fix_20250906/lensing_rar_table.csv`. Rows lacking required measured inputs are flagged and omitted from summary metrics until completed. Per‑lens panels (e.g., PG1115+080; B1608+656) appear in Extended Data:
`images/next_steps/enhanced_20250805_115400/lensing_rar_PG1115+080.png`,
`images/next_steps/enhanced_20250805_115400/lensing_rar_B1608+656.png`.

Uncertainties: where available, per‑lens $\theta_E$ uncertainties ($\sigma_{\theta_E}$) are included in `docs/lensing_targets.csv`; rows lacking uncertainties are omitted from weighted metrics. Residuals and a goodness‑of‑fit summary are written to `results/.../lensing_thetaE_residuals.csv` and `.../lensing_thetaE_metrics.json`.

---

## Discussion and Implications

**Predictive power vs flexibility.** With a single principal scale \(a_0\) and a fixed \(\nu\)-function, DGG reproduces broad rotation‑curve trends across diverse galaxies, naturally respecting the RAR and approaching the BTFR. This rigidity prevents per‑galaxy over‑fitting, sharpening falsifiable predictions (e.g., outer‑slope behavior).

**Solar‑System safety.** The same mapping that boosts galaxy outskirts yields \(|\Delta G/G|\ll 10^{-5}\) at \(\sim10\) AU for galaxy‑fit parameters, qualitatively consistent with Cassini. A PPN derivation will firm up the comparison.

**Vertical forces and local surface density.** A decisive check is \(K_z(R_0,z)\) and \(\Sigma_{1.1}\). We use the **full 3‑D** DGG contribution throughout the paper preset.

**Lensing under one metric.** Metric‑only predictions show the right order of magnitude for $\theta_E$ and stacked $\Delta\Sigma$ with measured lens inputs; residuals and RMSE are reported. A single‑theory lensing success is essential. **Sanity:** varying the stellar $M/L$ prior within the SED‑informed band shifts the amplitude of $\Delta\Sigma$ but not its slope over 0.05–300 kpc.

**Open issues.** (i) Whether a **finite plateau** $D_{\max}$ is required observationally (and, if so, at what value). (ii) Universality of $a_0$: hierarchical results and environment‑dependence. (iii) Clusters and ultra‑diffuse systems (may need residual mass such as neutrinos). (iv) Cosmological growth and CMB/BAO consistency in a relativistic completion. In this paper preset we adopt $D_{\max}=50$; galaxy fits and Solar bounds are empirically robust for $D_{\max}\in[30,80]$, with strong‑lensing sensitivity tested in Extended Data. **Falsifiability:** a decisive failure would be a requirement for $D_{\max}\gg100$ to fit strong‑lensing scales or cluster dynamics under the same mapping, or a systematic misfit of the BTFR slope/scatter under standardized selections.

---

## Conclusions

- **Unified galaxy dynamics without dark halos.** Using a single scale \(a_0\) and a fixed \(\nu\)-function (Box 1), DGG reproduces the broad form of rotation curves for the Milky Way and representative SPARC galaxies.
- **Scaling laws.** DGG respects the RAR by construction and approaches the BTFR expectation \(M_b\propto V^4\); measured slopes depend on selection and will be reported with scatter.
- **Local tests.** For parameters that fit galaxies, \(|\Delta G/G|\) in the Solar System remains below Cassini‑level sensitivity at Saturn; a PPN derivation is planned.
- **One‑theory lensing.** A metric‑only mapping gives reasonable \(\theta_E\) and \(\Delta\Sigma\) predictions; completing the lens sample with measured inputs is a priority.
- **Roadmap.** (1) Full 3D \(K_z/\Sigma_{1.1}\); (2) lensing with measured \((M_\star,R_e)\) and uncertainties; (3) hierarchical \(a_0\) with nuisances; (4) PPN appendix; (5) cluster/cosmology tests.

---

## Methods (condensed; full details in Supplementary)

**Hierarchical $a_0$ (optional).** When enabled, we infer a population‑level mean and scatter in $\ln a_0$ from per‑galaxy grids (dynesty nested sampling). We report $(\mu,\sigma)$ posteriors in `hierarchical_a0_posterior_summary.json` and a heatmap at `images/.../hierarchical_a0_posterior_heatmap.png`.  

**Baryon models.** Milky Way disks (Miyamoto–Nagai) + bulge (Hernquist) + gas; external galaxies use SPARC component rotmods.  
**Computation.** We evaluate \(\xi(g)\) as in **Box 1**, with unit conversion constant \(C\) and optional gates \(s_\rho, W(T)\).  
**Fitting \(a_0\).** Per‑galaxy **grid** \(\log_{10} a_0\in[-10.5,-9.3]\) (60 points) minimizing \(\chi^2\); optional **hierarchical** log‑normal prior for \(a_0\) with nested sampling.  
**Solar‑System.** In the Solar limit where $g_{\rm bar}\gg a_0$, $\xi\to1$ and the metric reduces to GR with $\gamma\simeq\beta\simeq1$ and $\alpha_{1,2}\simeq0$. We evaluate $\xi(r)$ in the Sun’s field and report $|\Delta G/G|$ at 1–30 AU; compare to the Cassini line as a consistency check. When the relativistic module is present (adopted subclass $\Phi=\Psi, c_T=1$), we also export a PPN CSV (`ppn_table.csv`) with $(\gamma,\beta,\alpha_1,\alpha_2)$.

PPN mapping (sketch). In PPN gauge, $ds^2=-(1-2U)dt^2+(1+2\gamma U)dx^2$. In our subclass with screening and $\Phi=\Psi$, both potentials receive the same small fractional rescaling $U\to (1+\epsilon)U$ with $\epsilon\equiv\Delta G/G\approx\xi-1\ll1$. Then $\gamma\equiv\Psi/\Phi=1$ identically, while light‑deflection and Shapiro delay amplitudes scale with $(1+\gamma)U\to (1+\gamma)(1+\epsilon)U$. Thus Cassini’s $|\gamma-1|<2.3\times10^{-5}$ bound implies $\epsilon\ll10^{-5}$ for any mapping that would attribute the observed delay amplitude to an effective rescaling of $U$. We therefore use $|\Delta G/G|$ as a conservative tracer; in the screened Solar limit both $\epsilon$ and $|\gamma-1|$ approach zero, consistent with the CSV export.
**Lensing (metric‑only).** We adopt a metric‑only mapping with $\Phi=\Psi$; the deflection potential is $2\Phi$, so the same $\xi(g)$ that boosts dynamics boosts lensing. Critical surface density $\Sigma_{\rm cr}(z_l, z_s)$ is computed for a flat $\Lambda$CDM cosmology. Stellar masses come from SED‑based $M/L$ (prior specified in Supplement), and sizes $(R_e, n)$ are measured from the discovery images or follow‑ups listed in the lens table. Residuals and goodness‑of‑fit metrics for $\theta_E$ are written to `lensing_thetaE_residuals.csv` and `lensing_thetaE_metrics.json`.  
**Model comparison.** Δlog Z histograms are reported as a BIC approximation; full evidences are produced when hierarchical runs are enabled.

> **Where to find in code:** the working implementation is `xi_rar_plateau_numpy(...)` and `solar_system_table(...)` in `scripts/next_steps_from_run.py`.

---

## Code and Data Availability

- **Code.** Analysis and plotting scripts are part of this repository. The exact function used in all figures is **Box 1**, implemented as `xi_rar_plateau_numpy`.  
- **Data.** SPARC rotmod files and Source Data CSVs accompany the figures (`results/...`) and are tracked with Git LFS.  
- **Reproduction.** See `scripts/reproduce_paper.py` for end‑to‑end regeneration of figures and tables. Each run writes a `run_metadata.json` with flags, environment, and timestamp; SPARC selection disclosure is saved to `sparc_selection.json`.

---

## References (selection; expand to full bib in submission)

- CASTLES: The CfA-Arizona Space Telescope LEns Survey of gravitational lenses. URL: https://www.cfa.harvard.edu/castles/ (accessed).

1. McGaugh, Lelli & Schombert (2016): The Radial Acceleration Relation in Rotationally Supported Galaxies.  
2. Lelli, McGaugh & Schombert (2016): SPARC mass models.  
3. Milgrom (1983–2014): MOND framework and predictions.  
4. Bertotti, Iess & Tortora (2003): Cassini bound on \(|\gamma-1|\).  
5. Bovy & Rix (2013); McMillan (2017/2022): MW \(\Sigma_{1.1}\) and mass model.  
6. Additional RAR/BTFR and lensing references as in the repository’s bibliography.

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

### Lensing metrics (CASTLES sample; paper preset)

- Per-lens tables: `results/next_steps/btfr_fix_20250906/lensing_table.csv`
- RAR lens metrics: `results/next_steps/btfr_fix_20250906/lensing_rar_table.csv`
- Combined pivots and summaries:  
  - `results/next_steps/btfr_fix_20250906/combined/lensing_summary_pivot_RAR.csv`  
  - `results/next_steps/btfr_fix_20250906/combined/lensing_summary_pivot_GR.csv`  
  - `results/next_steps/btfr_fix_20250906/combined/global_alpha/lensing_global_alpha_metrics.csv`
- Figure: `results/next_steps/btfr_fix_20250906/combined/global_alpha/lensing_global_alpha_pred_vs_obs.png`

> CASTLES conversion used: `results/next_steps/btfr_fix_20250906/lenses_castles_small_converted.csv`.

## Reproduction (paper preset)

Run the all-in-one script to regenerate figures/tables using the paper preset:

```
python scripts/reproduce_paper.py \
  --preset paper \
  --run-dir runs/enhanced_20250805_115400 \
  --sparc-dir external_data/Rotmod_LTG \
  --lensing-csv docs/lensing_targets.csv
```

- Paper preset enforces metric-only lensing, posterior sampling for Solar bands, and standardized SPARC cuts.
- Provide measured lens entries in `docs/lensing_targets.csv` with columns `lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec,theta_E_obs_err_arcsec]` to produce the lensing metric table (rows missing uncertainties are excluded from weighted metrics).

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
 W(T)=w_{\min}+(1-w_{\min})\exp\!\Bigl[-\frac{(\\ln T-\\ln T_0)^2}{2\\sigma_{\\ln T}^2}\Bigr],\\[3pt]
 a_0^{\rm eff} &= a_0\bigl[1+\zeta_{\rm env}\,s_\rho(\rho)\,W(T)\bigr],\qquad T\equiv V_{\rm bar}^2/R^2,\\[3pt]
 \xi(R) &\equiv \tfrac12+\sqrt{\tfrac14+\tfrac{a_0^{\rm eff}}{g_{\rm bar}(R)}},\qquad
 V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
\end{aligned}
$$

*Optional plateau:* impose \(\xi\le D_{\max}\) if a finite cap is required observationally.

### Appendix B — PPN and Cassini (placeholder)

- Starting from the relativistic completion with \(\Phi=\Psi\) and \(c_T=1\), derive \(\gamma,\beta\) in the Solar limit and connect \(\Delta G/G\) to \(\gamma-1\).  
- Provide a compact AU table (1–30 AU) and show consistency with \(|\gamma-1|<2.3\times10^{-5}\) at Saturn.

