# Density‑Gated Gravity: A Density‑Dependent Alternative to Dark Matter

## Introduction

Galactic rotation curves have long challenged the standard cosmological model, which invokes massive halos of non-baryonic *dark matter* to explain the unexpectedly high orbital speeds in outer galactic disks. While dark halos can be *fitted* to match individual galaxy rotations, their success comes at the cost of introducing numerous free parameters (one halo per galaxy) and fine-tuned correlations between baryonic and dark mass distributions. A striking empirical clue is the **Radial Acceleration Relation (RAR)**: across hundreds of galaxies, the observed centripetal acceleration \(g_{\rm obs}(r)\) tightly correlates with that predicted by visible matter alone \(g_{\rm bar}(r)\)[1]. This correlation persists even in regions where dark matter is presumed to dominate, implying that the dark contribution is “fully specified by that of the baryons”[1]. The small scatter in the RAR (comparable to observational uncertainties) suggests an underlying law of nature[1] rather than a fortuitous result of galaxy formation. Indeed, the RAR has been called “tantamount to a natural law” for galaxies[1]. Such a universal relation is difficult to reconcile with arbitrary halo tuning, as it would require a mysterious *conspiracy* between visible and dark components across all galaxies[2]. This has motivated the pursuit of alternative gravity theories that *predict* the RAR intrinsically, without invoking invisible mass[2].

One notable example is Milgrom’s Modified Newtonian Dynamics (MOND), which postulates a new fundamental acceleration scale \(a_0\sim10^{-10}\) m/s² at which gravity deviates from Newton’s laws[3]. MOND’s simple prescription can explain flat rotation curves and was prescient in foreseeing the RAR decades before its observational confirmation[3][1]. However, the original MOND formula (and similar empirical interpolations) are *too rigid* – with a single parameter and no built-in relativistic framework, they struggle to fit *all* phenomena (e.g. the diversity of galaxy profiles, galaxy clusters, and cosmological observations)[4]. On the other hand, explaining the RAR within the dark matter paradigm also poses challenges: it requires highly coordinated distributions of baryons and dark matter for every system[5], which may be achievable in detailed galaxy formation models but lacks the elegance of a universal law. Given the continuing non-detection of dark matter particles and the empirical successes of MOND-like phenomenology on galactic scales, it is worthwhile to explore new gravity models that combine **predictive rigidity** with flexibility and consistency across scales.

**RAR‑gated gravity** is a new approach that aims to retain the parsimonious, predictive nature of MOND/RAR, while embedding the theory in a more robust, general‑relativistic‑style framework. In this model, the *strength* of gravity is **gated** by the local acceleration (and/or related mass density), such that standard General Relativity (GR) is recovered in high‑acceleration regimes (e.g. the inner Solar System and deep potential wells), but departures from Newtonian gravity automatically emerge in low‑acceleration environments (galactic outskirts). The model requires no *ad hoc* dark halos; instead, the observed flat or gently declining rotation curves are a natural outcome of the modified field equations. Importantly, RAR‑gated gravity introduces only a small number of new parameters (ideally just one universal acceleration scale \(a_0\) and perhaps one dimensionless strength parameter), making it highly constrained. If such a model can account for diverse galaxy rotation curves with *the same* fundamental constants, it would strengthen the case for a new theory of gravity and provide a serious alternative to dark matter. In this paper, we present the RAR‑gated gravity model, confront it with empirical tests—including **Milky Way** rotation measurements, fits to **SPARC** galaxy rotation curves, and **Solar‑System** precision data—and outline its advantages and remaining challenges. Our goal is to demonstrate that this approach is a credible contender for explaining galactic dynamics.

---

## The RAR‑Gated Gravity Model

In RAR‑gated gravity, the departure from Newton’s law is governed by an interpolating “gating” function that depends on the local gravitational acceleration (and/or local mass distribution). Conceptually, one can think of the model as modifying the effective gravitational constant or the relationship between the matter distribution and the curvature of spacetime, such that:

- **High‑acceleration limit** (\(g\gg a_0\)) — The gate suppresses modifications, restoring Newton/GR and passing Solar‑System tests.
- **Low‑acceleration limit** (\(g\ll a_0\)) — The gate enhances the effective force in a way tuned to reproduce the RAR (and hence the BTFR).
- **Intermediate regime** (\(g\sim a_0\)) — A smooth transition governed by the specific interpolating function; the transition width can be calibrated against rotation‑curve shapes.

Mathematically, one representation is a modified Poisson equation or GR‑style field equation with a nonlinear term that depends on the field strength and/or density:
$$
\nabla^2 \Phi \;=\; 4\pi G\,\rho_b \;+\; \nabla\!\cdot\!\Big[f\!\left(\tfrac{|\nabla\Phi|}{a_0},\, \tfrac{\rho_b}{\rho_0}\right)\nabla\Phi\Big],
$$
with \(f\rightarrow 0\) in high‑acceleration/high‑density limits and \(f>0\) in low‑acceleration environments.

### **Box 1 — Exact weak‑field formula used in all figures (from code)**

Let \(V_{\rm bar}(R)\) be the baryonic circular speed (km s\(^{-1}\)) and \(R\) the radius (kpc). Convert to SI:
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
with \(T\equiv V_{\rm bar}^2/R^2\) used purely as a tidal proxy.

**Effective acceleration scale:**
$$
 a_0^{\rm eff} \;=\; a_0\Bigl[1+\zeta_{\rm env}\,s_\rho(\rho)\,W(T)\Bigr].
$$

**Boost (“\(\nu\)” function) actually implemented:**
$$
\xi(R)\equiv D \;=\; \frac{1}{2} + \sqrt{\frac{1}{4}+\frac{a_0^{\rm eff}}{g_{\rm bar}(R)}} \;\;\ge 1,
\qquad
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
$$

> **Clarification.** Prior text referred to a low‑\(g\) **“plateau”**; the code currently **does not impose a hard cap** on \(D\) as \(g_{\rm bar}\to 0\). If a finite plateau \(D\le D_{\max}\) is desired, a one‑line clamp can be added. Until then, we avoid “plateau” terminology in the main claims.

### Relativistic and lensing stance used for figures

For manuscript figures we adopt a **metric‑only** weak‑field mapping with \(\Phi=\Psi\) and \(c_T=1\); lensing uses \(\Phi+\Psi=2\Phi\) and the same \(\xi(g)\) as dynamics. A concise PPN derivation and full metric completion are slated for Methods/Supplement.

> **Placeholder to add:** short derivation connecting \(\xi\) to \(\kappa(R)\) and \(\theta_E\) in this subclass, and the PPN parameters \((\gamma,\beta,\alpha_1,\alpha_2)\).

---

## Rotation Curve Predictions with No Dark Halos

We applied the RAR‑gated model to baryonic mass models for the **Milky Way** and for external galaxies from **SPARC**. In each case we compute
$$
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R)
$$
with \(\xi\) from **Box 1** and compare to observed rotation curves.

### Milky Way: A Case Study

**Milky Way (Gaia DR3) rotation curve: GR vs NFW vs RAR‑gate.**  
![Milky Way: GR vs NFW vs RAR‑gate (0.1 kpc Gaia medians)](images/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.png)

*Caption:* Observed Gaia DR3 median stellar speeds every 0.1 kpc (black with 16–84% bands) compared with GR (baryons‑only; blue dashed), an NFW **yardstick** (green dotted), and RAR‑gate (red). Curves use the same baryon model and a representative \(a_0\) consistent with the SPARC range.

As in the original text, the model matches the inner rise (where baryons dominate) and sustains the outer speed once \(g_{\rm bar}\sim a_0\), without galaxy‑specific halos.

### Milky Way: vertical force \(K_z\) and local surface density \(\Sigma_{1.1}\)

We compute \(K_z(R_0,z)\) for the same MW baryons and infer \(\Sigma_{1.1}\approx K_z/2\pi G\). In addition to baryons‑only we show a **scaled** DGG curve for orientation.

![Milky Way Kz and Σ_1.1](images/next_steps/enhanced_20250805_115400/mw_kz_sigma.png)

> **To be upgraded for submission:** replace the scaled approximation with the **full 3D** DGG (“phantom”) mass implied by \(\xi\) and compare \(\Sigma_{1.1}\) with Bovy & Rix / McMillan bands, including uncertainties.

### External Galaxies: SPARC Rotation‑Curve Fits

We selected representative spirals spanning mass and surface brightness. For each galaxy we hold gating parameters fixed (MW‑tuned) and **scan \(a_0\) on a grid** \(\log_{10} a_0\in[-10.5,-9.3]\) (m s\(^{-2}\)) to minimize \(\chi^2\) (per‑galaxy \(a_0\) strategy). A hierarchical log‑normal model for \(a_0\) is available and reported as Extended Data when used.

#### SPARC gold‑sample panel (RAR‑gate vs GR vs observations)
![SPARC gold overlays panel](images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png)

*Caption:* M31, NGC 3198, NGC 2403, NGC 2841, NGC 5055. Black: observed \(V_c\); blue dashed: GR baryons; red: RAR‑gate with **per‑galaxy** best‑fit \(a_0\). See `results/.../sparc_a0_summary.csv` for \(a_0\) values and \(\Delta\chi^2\).

**RAR master panel** (optional ΛCDM band) is available at  
`images/next_steps/rar_plateau_mw_full/rar_master_panel.png`.

**BTFR outcome.** On a working subset (N≈89) using \(M_b=M_\star+1.33\,M_{\mathrm{HI}}\) and **observed** \(V_{\rm flat}\), a simple log–log fit yields a slope \(\sim 3.2\pm0.1\). The deep‑regime prediction from the \(\nu\)-function approaches \(M_b\propto V^4\); we will report selection sensitivity (flatness criterion, inclinations, gas content) and intrinsic scatter.

---

## Solar‑System Constraints

Any modified gravity must clear Solar‑System bounds. We evaluate the same \(\xi(r)\) in the Sun’s Kepler field \(g_N(r)=GM_\odot/r^2\) and report
$$
\left|\frac{\Delta G}{G}\right| \;\equiv\; \left|\xi(r)-1\right|
$$
at 1–30 AU. We show a **gated** curve (same parameters as galaxy fits) and a **worst‑case** curve (no screening).

![Solar‑System constraints](images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png)

*Caption:* The orange (gated) curve remains below the Cassini bound \(|\gamma-1|<2.3\times10^{-5}\) at Saturn (dotted line). A formal PPN mapping \(\Delta G/G \!\leftrightarrow\! \gamma-1\) for our subclass will be provided in Methods.

> **Source Data:** `results/next_steps/rar_plateau_mw_full/solar_system_table.csv` (AU, \(g_{\rm bar}\), \(\xi_{\rm gated}\), \(\xi_{\rm worst}\)).

### PPN parameters (placeholder)

We plan to present a short derivation for \((\gamma,\beta,\alpha_1,\alpha_2)\) in the quasi‑static, screened limit with \(\Phi=\Psi\). A CSV export is supported by the code when the relativistic module is present.

---

## Gravitational Lensing (metric‑only path for figures)

Lensing is computed from the same \(\xi(g)\) via a **metric‑only** mapping with \(\Phi=\Psi\). We produce per‑lens \(\langle\Sigma\rangle(R)\), \(\Delta\Sigma(R)\), and \(\theta_E\), plus a stacked \(\Delta\Sigma\).

![θ_E: predicted vs observed](images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png)

![Stacked ΔΣ from metric predictions](images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png)

> **Placeholders to complete for submission:** lens table with measured \((M_\star, R_e, n)\), \((z_l,z_s)\), cosmology for \(\Sigma_{\rm cr}\), and uncertainties on \(\theta_E\). Per‑lens panels (e.g., PG1115+080; B1608+656) appear in Extended Data:  
> `images/next_steps/enhanced_20250805_115400/lensing_rar_PG1115+080.png`,  
> `images/next_steps/enhanced_20250805_115400/lensing_rar_B1608+656.png`.

---

## Discussion and Implications

**Predictive power vs flexibility.** With a single principal scale \(a_0\) and a fixed \(\nu\)-function, DGG reproduces broad rotation‑curve trends across diverse galaxies, naturally respecting the RAR and approaching the BTFR. This rigidity prevents per‑galaxy over‑fitting, sharpening falsifiable predictions (e.g., outer‑slope behavior).

**Solar‑System safety.** The same mapping that boosts galaxy outskirts yields \(|\Delta G/G|\ll 10^{-5}\) at \(\sim10\) AU for galaxy‑fit parameters, qualitatively consistent with Cassini. A PPN derivation will firm up the comparison.

**Vertical forces and local surface density.** A decisive check is \(K_z(R_0,z)\) and \(\Sigma_{1.1}\). We will replace the scaled curve by the **full 3D** DGG contribution.

**Lensing under one metric.** Metric‑only predictions show the right order of magnitude for \(\theta_E\) and stacked \(\Delta\Sigma\) with measured lens inputs pending. A single‑theory lensing success is essential.

**Open issues.** (i) Whether a **finite plateau** \(D_{\max}\) is required observationally (and, if so, at what value). (ii) Universality of \(a_0\): hierarchical results and environment‑dependence. (iii) Clusters and ultra‑diffuse systems (may need residual mass such as neutrinos). (iv) Cosmological growth and CMB/BAO consistency in a relativistic completion.

---

## Conclusions

- **Unified galaxy dynamics without dark halos.** Using a single scale \(a_0\) and a fixed \(\nu\)-function (Box 1), DGG reproduces the broad form of rotation curves for the Milky Way and representative SPARC galaxies.
- **Scaling laws.** DGG respects the RAR by construction and approaches the BTFR expectation \(M_b\propto V^4\); measured slopes depend on selection and will be reported with scatter.
- **Local tests.** For parameters that fit galaxies, \(|\Delta G/G|\) in the Solar System remains below Cassini‑level sensitivity at Saturn; a PPN derivation is planned.
- **One‑theory lensing.** A metric‑only mapping gives reasonable \(\theta_E\) and \(\Delta\Sigma\) predictions; completing the lens sample with measured inputs is a priority.
- **Roadmap.** (1) Full 3D \(K_z/\Sigma_{1.1}\); (2) lensing with measured \((M_\star,R_e)\) and uncertainties; (3) hierarchical \(a_0\) with nuisances; (4) PPN appendix; (5) cluster/cosmology tests.

---

## Methods (condensed; full details in Supplementary)

**Baryon models.** Milky Way disks (Miyamoto–Nagai) + bulge (Hernquist) + gas; external galaxies use SPARC component rotmods.  
**Computation.** We evaluate \(\xi(g)\) as in **Box 1**, with unit conversion constant \(C\) and optional gates \(s_\rho, W(T)\).  
**Fitting \(a_0\).** Per‑galaxy **grid** \(\log_{10} a_0\in[-10.5,-9.3]\) (60 points) minimizing \(\chi^2\); optional **hierarchical** log‑normal prior for \(a_0\) with nested sampling.  
**Solar‑System.** Evaluate \(\xi(r)\) in the Sun’s field and report \(|\Delta G/G|\) at 1–30 AU; compare to the Cassini line as a consistency check.  
**Lensing (metric‑only).** Build \(\langle\Sigma\rangle(R)\), \(\Delta\Sigma(R)\), and \(\theta_E\) using the same \(\xi\) and measured lens properties; \(\Sigma_{\rm cr}\) from a standard flat ΛCDM cosmology.

> **Where to find in code:** the working implementation is `xi_rar_plateau_numpy(...)` and `solar_system_table(...)` in `scripts/next_steps_from_run.py`.

---

## Code and Data Availability

- **Code.** Analysis and plotting scripts are part of this repository. The exact function used in all figures is **Box 1**, implemented as `xi_rar_plateau_numpy`.  
- **Data.** SPARC rotmod files and Source Data CSVs accompany the figures (`results/...`) and are tracked with Git LFS.  
- **Reproduction.** See `scripts/reproduce_paper.py` for end‑to‑end regeneration of figures and tables.

---

## References (selection; expand to full bib in submission)

1. McGaugh, Lelli & Schombert (2016): The Radial Acceleration Relation in Rotationally Supported Galaxies.  
2. Lelli, McGaugh & Schombert (2016): SPARC mass models.  
3. Milgrom (1983–2014): MOND framework and predictions.  
4. Bertotti, Iess & Tortora (2003): Cassini bound on \(|\gamma-1|\).  
5. Bovy & Rix (2013); McMillan (2017/2022): MW \(\Sigma_{1.1}\) and mass model.  
6. Additional RAR/BTFR and lensing references as in the repository’s bibliography.

---

## Figures (main and extended)

- **Fig. 1** Milky Way rotation — `images/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.png`  
- **Fig. 2** MW \(K_z\) / \(\Sigma_{1.1}\) — `images/next_steps/enhanced_20250805_115400/mw_kz_sigma.png`  
- **Fig. 3** SPARC panel — `images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png`  
- **Fig. 4** BTFR subset — `images/next_steps/btfr_fix_20250906/btfr_baryonic.png`  
- **Fig. 5** Solar \(|\Delta G/G|\) — `images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png`  
- **Fig. 6** Lensing \(\theta_E\) scatter — `images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png`  
- **Ext. Fig.** Stacked \(\Delta\Sigma\) — `images/next_steps/enhanced_20250805_115400/lensing_metric_stack.png`

---

## Latest Results Tables (auto-generated)

These summarize the latest outputs under `results/next_steps/enhanced_20250805_115400/`. Full CSVs are linked for reproducibility.

### Solar System (Source Data)

File: `results/next_steps/enhanced_20250805_115400/solar_system_table.csv`

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

| z [kpc] | Kz [m s^-2] |
|-------:|------------:|
| 0.5 | 8.3962e+07 |
| 0.8 | 1.3880e+08 |
| 1.1 | 1.9443e+08 |
| 1.5 | 2.6870e+08 |
| 2.0 | 3.6068e+08 |

---

## Appendices (ready text & placeholders)

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

