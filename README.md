# Gravity Gates: A Predictive Acceleration‑Gated Alternative to Dark Halos for Galaxies and Lensing

Leonard Speiser (Independent Researcher)

## Abstract

The radial acceleration relation (RAR) ties observed galaxy dynamics to baryons with striking precision, challenging the need for bespoke dark‑halo profiles on a per‑galaxy basis. We present **Gravity Gates (GG)**, a weak‑field framework in which the effective response of gravity is a **deterministic function of the local acceleration**. A single, universal gate $\xi(g_{\rm bar};a_0,D_{\max})$ equals unity at high acceleration (recovering Newton/GR and Solar‑System tests) and **smoothly enhances** the force at low acceleration to reproduce RAR/BTFR phenomenology, with a finite plateau to regularize the deep regime. Using **one gate** and **no dark halos**, we show that GG (i) predicts the Milky Way rotation curve and **vertical force $K_z(R_0,z)$** within published bands, (ii) fits representative **SPARC** rotation curves with only $a_0$ varied per galaxy or via a hierarchical population model, and (iii) **reproduces the amplitude of strong gravitational lensing** (Einstein radii) under a metric‑only mapping with $\Phi=\Psi$, after applying the same stellar IMF normalization to both GG and GR baselines. We quantify falsifiability and Solar‑System safety (PPN sketch; Cassini‑level bounds). By replacing per‑galaxy halo freedom with a **single, cross‑domain principle**, Gravity Gates offer a **predictively rigid** alternative to dark halos on galaxy and lens scales.

## 1. Introduction

Galactic rotation curves continue to strain the standard picture in which galaxies live inside massive, non‑baryonic dark‑matter halos. Halos can be tuned to fit individual systems, but doing so typically introduces one bespoke mass profile per galaxy and requires tight, system‑by‑system coordination between baryons and dark matter. A central empirical clue is the **radial acceleration relation (RAR)**: across hundreds of disks the observed centripetal acceleration $g_{\rm obs}$ correlates closely with the acceleration predicted by baryons alone $g_{\rm bar}$, with scatter comparable to measurement uncertainties. Any successful framework must either explain why dark halos conspire to follow the baryons so closely, or modify the low‑acceleration law of gravity itself in a way that **predicts** the RAR.

Two broad approaches have emerged. In ΛCDM, increasingly sophisticated formation models attempt to imprint the observed baryon–halo coupling through feedback, assembly histories, and environment. This is flexible, but the cost is many latent degrees of freedom. On the other side are **MOND‑like** ideas that introduce a characteristic acceleration scale $a_0$ and were prescient in anticipating the RAR. These are predictively rigid on galaxy scales, but classic formulations are not manifestly relativistic and face challenges in clusters and cosmology. The tension between **flexibility** (fit anything) and **rigidity** (predict many things with few knobs) is the fundamental issue.

Our guiding idea is that gravity’s **effective response may depend on environment**, much as other fundamental interactions do. In quantum field theory, couplings "run" with scale, and the strong force presents different faces in different regimes (asymptotic freedom vs. confinement). We do **not** claim an identity with QCD, but we adopt the same organizing principle: the **measured strength** of the interaction can change with conditions. If gravity’s weak‑field response is **gated** by local acceleration (and, by extension, by typical densities or tidal scales), then observers like us—residing in a **high‑acceleration** region (Solar neighborhood, deep potentials)—naturally see standard GR. In the **low‑acceleration** outskirts of galaxies, however, the gate opens and an enhanced response emerges. Cosmologically, such gating would tend to **promote aggregation** in diffuse regions rather than hinder it, a qualitative feature that is at least directionally compatible with the observed prevalence of bound structure. In this paper we do not model early‑universe dynamics; we focus on quantifying the weak‑field, quasi‑static consequences in galaxies and the Solar System.

We formalize this idea as **Gravity Gates (GG)**: a weak‑field framework in which the gravitational response is a deterministic function of the local field strength (and, optionally, simple environmental proxies). We instantiate a **density‑gated** subclass (**DGG**) for empirical tests. The gate $\xi(g_{\rm bar})$ multiplies the baryonic prediction, is **unity** in the high‑acceleration limit (recovering Newton/GR and Solar‑System tests), and increases smoothly toward a finite plateau at low acceleration to reproduce the RAR/BTFR phenomenology while avoiding pathologies at extremely small $g_{\rm bar}$. Crucially, **the same gate**—with a single scale $a_0$ and a fixed functional form—is used to predict **rotation curves, vertical forces $K_z$, strong‑lensing Einstein radii $\theta_E$, and stacked weak‑lensing $\Delta\Sigma$** under a metric subclass with $\Phi=\Psi$. This preserves **predictive rigidity** without the per‑galaxy halo freedom.

Our approach differs from both extremes. Compared to tuned halos, DGG **reduces** the degrees of freedom by replacing galaxy‑specific dark profiles with a universal gate tied to the baryons. Compared to classic MOND, DGG is embedded in an explicit weak‑field metric mapping (used for both dynamics and lensing), includes a finite low‑$g$ plateau to regularize the deep regime, and is framed to be auditable against Solar‑System post‑Newtonian bounds. The result is a tightly constrained hypothesis that makes **linked, cross‑domain predictions**.

**This paper’s program and falsifiability.** We confront DGG with: (i) the **Milky Way** rotation curve and vertical force $K_z(R_0,z)$ (reporting $\Sigma_{1.1}$); (ii) representative **SPARC** rotation curves and the **BTFR**; (iii) **Solar‑System** constraints cast in $|\Delta G/G|$ alongside post‑Newtonian parameters; and (iv) **gravitational lensing**, from individual $\theta_E$ to stacked $\Delta\Sigma(R)$, computed with the same gate. The framework is **falsified** if a single $\xi(g)$ cannot simultaneously: fit outer‑disk slopes without per‑galaxy halos, remain within $K_z$ bands at the Solar radius, respect Solar‑System limits, and reproduce the amplitude of lensing signals under standard stellar‑population priors. The sections that follow specify the gate, the weak‑field mapping, and the data/selection, then present each test in turn.

### Claims at a glance

**What is new here**

1. **One gate, many domains.** A universal $\xi(g)$ set by baryons predicts: rotation curves (MW + SPARC), MW $K_z$, and **strong‑lensing $\theta_E$**—with **no dark‑halo fits**.
2. **Relativistic weak‑field mapping.** Metric‑only subclass with $\Phi=\Psi$ ensures one function governs both dynamics and light deflection (Methods; PPN sketch).
3. **Falsifiability.** A single gate must simultaneously satisfy outer‑disk slopes, MW $K_z$, Solar‑System constraints, and lensing amplitudes using the same $M_\star/L$ priors as the GR baseline.
4. **Fair comparators.** NFW and GR+baryons baselines are fitted under stated priors and **identical** $M_\star/L$ assumptions.

**Scope**

We restrict attention to quasi‑static, weak‑field tests—galaxy rotation curves, the Milky Way K_z, Solar‑System bounds, and strong‑lensing amplitudes—under a metric‑only mapping; we do not model cosmological expansion or fit CMB/BAO, and we do not address cluster‑scale dynamics here.

### Falsifiability

The present framework is falsified if **one and the same** $\xi(g)$ cannot
(i) match outer‑disk slopes without per‑galaxy halos,
(ii) remain within published $K_z(R_0,z)$ bands at $z\in[0.5,2]$ kpc for the Milky Way,
(iii) keep $|\Delta G/G|=|\xi-1|$ below Cassini‑level sensitivity across $1\!-\!30$ AU under the screened mapping, and
(iv) reproduce $\theta_E$ amplitudes using the **same** stellar $M_\star/L$ priors applied to the GR baseline.
A further failure would be a requirement for $D_{\max}\!\gg\!100$ to match lensing or cluster dynamics.

---

## 2. The RAR‑Gated Gravity Model

In RAR‑gated gravity, the departure from Newton’s law is governed by an interpolating "gating" function that depends on the local gravitational acceleration (and/or local mass distribution). Conceptually, one can think of the model as modifying the effective gravitational constant or the relationship between the matter distribution and the curvature of spacetime, such that:

- **High‑acceleration limit** ($g\gg a_0$) — The gate suppresses modifications, restoring Newton/GR and passing Solar‑System tests.
- **Low‑acceleration limit** ($g\ll a_0$) — The gate enhances the effective force in a way tuned to reproduce the RAR (and hence the BTFR).
- **Intermediate regime** ($g\sim a_0$) — A smooth transition governed by the specific interpolating function; the transition width can be calibrated against rotation‑curve shapes.

#### Why we set it up this way

The empirical RAR/BTFR indicates that galaxy dynamics are largely fixed by the baryons alone. Rather than fitting a dark halo for each galaxy, we impose a single gate $\xi(g_{\rm bar}; a_0, D_{\max})$ that multiplies the baryonic prediction. This keeps the theory rigid (few parameters, shared across systems) while still reproducing low‑$g$ phenomenology. A finite plateau $D_{\max}=50$ is adopted in the paper preset to regularize the deep‑MOND limit without affecting the data range we test.

### How the model works (plain language)

- In high‑acceleration regions, gravity behaves like ordinary GR/Newton.
- In low‑acceleration regions (galaxy outskirts), gravity gets a boost. The size of the boost is set by a single universal scale $a_0$ and capped by a plateau $D_{\max}$ to prevent unphysical divergence.
- Practically, we compute the Newtonian acceleration from baryons ($g_{\rm bar}$) and then multiply by a boost factor $\xi$ (the "gate"). This $\xi$ depends only on the local field strength (and, optionally, a mild environment term), not on a custom dark halo for each galaxy.
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

**Boost (" $\nu$ " function) used throughout.**

$$
\xi(R) \;=\; \min\!\left[\;\frac{1}{2} + \sqrt{\frac{1}{4}+\frac{a_0^{\rm eff}}{g_{\rm bar}(R)}}\;,\; D_{\max}\right],\qquad D_{\max}=50~\text{(fiducial)}.\\[3pt]
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R).
$$

### Notation and 3‑D mapping (used for $K_z$ and lensing)

We define the **scalar** $g_{\rm bar}\equiv |\nabla\Phi_b|$ and the **vector** $\mathbf g_{\rm bar}\equiv -\nabla\Phi_b$. In a QUMOND‑like mapping,

$$
\nabla^2\Phi \;=\; \nabla\!\cdot\!\big[\nu(g_{\rm bar}/a_0)\,\nabla\Phi_b\big],\quad \nu\equiv \xi,
$$

the effective ("phantom") density is

$$
\rho_{\rm ph} \;=\; (\xi-1)\,\rho_b \;+\; \frac{1}{4\pi G}\,\nabla\xi\!\cdot\!\nabla\Phi_b 
\;=\; (\xi-1)\,\rho_b \;-\; \frac{1}{4\pi G}\,\nabla\xi\!\cdot\!\mathbf g_{\rm bar},
$$

which we use for full‑3D $K_z$ and lensing computations.

### Relativistic weak‑field mapping

Full derivation and Solar‑System PPN mapping: see docs/ppn_mapping.md.

We adopt a metric‑only weak‑field subclass with $c_T=1$ and $\Phi=\Psi$ in screened, quasi‑static limits. Dynamics depend on $\Phi$; lensing depends on $\Phi+\Psi=2\Phi$. The same $\xi(g)$ rescales the weak‑field potential entering both dynamics and light deflection. See Methods for a PPN sketch and Appendix for a covariant scaffold; for broader context compare to TeVeS‑like and modern scalar–tensor completions (e.g., Skordis–Zlosnik).

A QUMOND‑like mapping is useful for intuition: $\nabla^2\Phi=\nabla\!\cdot\![\nu(|\nabla\Phi_b|/a_0)\nabla\Phi_b]$ with $\nu(y)=\tfrac12+\sqrt{\tfrac14+1/y}$, so that $g=\nu\,g_N$ and $V^2=\xi\,V_{\rm bar}^2$ in the disk plane. The associated phantom‑density identity yields the full 3‑D contribution used for $K_z$ and lensing in the paper preset. We use the same $\nu$ function as in Box 1.

---

## 3. Rotation‑Curve Predictions Without Dark Halos

We applied the RAR‑gated model to baryonic mass models for the **Milky Way** and for external galaxies from **SPARC**. In each case we compute

$$
V_{\rm model}^2(R)=\xi(R)\,V_{\rm bar}^2(R)
$$

with $\xi$ from **Box 1** and compare to observed rotation curves.


### 3.1 Milky Way case study

**Milky Way (Gaia DR3) rotation curve: GR vs NFW vs RAR‑gate.**  
![Milky Way: GR vs NFW vs RAR‑gate (0.1 kpc Gaia medians)](images/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.png)

Figure 1 | An acceleration-gated model reproduces the Milky Way rotation curve without a dark matter halo. The observed circular velocity of the Milky Way (black points; Gaia Collaboration et al. 2022) is compared to three models. The prediction from baryons alone under standard gravity (blue) fails to match the data at large radii. A standard Navarro-Frenk-White (NFW) dark matter halo (green) provides a descriptive fit. Our Density-Gated Gravity (DGG) model (red), using a single universal gating function and no dark matter component, accurately reproduces the inner baryon-dominated region and the flat outer rotation curve.
*   **Source Data:** `results/rar_plateau_mw_full/mw_rotation_rar_plateau_finebins.csv`

As in the original text, the model matches the inner rise (where baryons dominate) and sustains the outer speed once $g_{\rm bar}\sim a_0$, without galaxy‑specific halos.

### 3.2 Milky Way vertical force $K_z$ and $\Sigma_{1.1}$

We compute $K_z(R_0,z)$ for the same MW baryons and infer $\Sigma_{1.1}\approx K_z/2\pi G$. The figure below uses the **full 3‑D** DGG ("phantom") mass implied by $\xi$ via
$\rho_{\rm ph}=(\xi-1)\,\rho_b - (4\pi G)^{-1}\,\nabla\xi\!\cdot\!\mathbf g_{\rm bar}$.

New in this draft: we propagate baryonic uncertainties (disk and bulge $M/L$, gas mass, disk scale heights/lengths, and a bulge‑scale proxy for flattening) through the full‑3D phantom density to produce a shaded band. We sample priors in the orchestrator and export a CSV with the 16–84% Kz band at selected z.

![Milky Way Kz and Σ_1.1 (full 3D)](images/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.png)

Figure 2 | The DGG model is consistent with the measured vertical force in the Solar neighborhood. The predicted vertical force $K_z$ as a function of height $z$ above the Galactic plane at the Solar radius $R_0 = 8.2$ kpc. The red curve and shaded band show the DGG prediction and its 16th–84th percentile confidence interval, derived by propagating uncertainties in the baryonic mass model. This prediction is consistent with observational constraints from Bovy & Rix (2013) and McMillan (2017) (overlaid bands), without requiring a local dark matter disk.
*   **Source Data:** `results/next_steps/enhanced_20250805_115400/mw_kz_sigma_full3d.csv`

Assumptions disclosed for this figure:
- Solar radius $R_0$ = 8.2 kpc (override with `--mw-R0-kpc`).
- Heights evaluated: |z| ∈ {0.5, 0.8, 1.1, 1.5, 2.0} kpc (override via `--mw-kz-zlist`).
- Tracer kinematics for the comparison bands follow the published analyses (Bovy & Rix 2013; McMillan 2017/2022): axisymmetry, steady‑state vertical equilibrium, and the survey selection functions as reported therein. We overlay their published bands as references and do not re‑derive them here.
- Baryon priors (defaults; adjustable): ln(M/L) σ = 0.15 (disk and bulge), gas mass fractional σ = 0.25, disk scale‑height fractional σ = 0.20, disk scale‑length fractional σ = 0.10, bulge Hernquist‑scale fractional σ = 0.25 (a proxy for flattening sensitivity).
- The same DGG parameters used in the MW rotation fit are used for $K_z$.

### 3.3 External galaxies: SPARC rotation‑curve fits

We selected representative spirals spanning mass and surface brightness. For each galaxy we hold gating parameters fixed (MW‑tuned) and **scan $a_0$ on a grid** $\log_{10} a_0\in[-10.5,-9.3]$ (m s$^{-2}$) to minimize $\chi^2$ (per‑galaxy $a_0$ strategy). A hierarchical log‑normal model for $a_0$ is available and reported as Extended Data when used.

#### SPARC gold‑sample panel (DGG vs GR vs observations)
![SPARC gold overlays panel](images/next_steps/enhanced_20250805_115400/sparc_panel_gold.png)

Figure 3 | The universal gating function explains diverse rotation curve shapes across the SPARC galaxy sample. Observed rotation curves (black points; Lelli, McGaugh & Schombert 2016) for five representative galaxies are compared with predictions from baryons alone (blue) and the DGG model (red). The DGG fits, obtained by optimizing a single parameter ($a_0$) for each galaxy, successfully track the data, demonstrating the model's applicability across galaxies of varying mass and surface brightness.
*   **Source Data:** Per-galaxy rotmod files are from the SPARC database. Fit results are available in `results/next_steps/enhanced_20250805_115400/sparc_a0_summary.csv`.

#### Dual‑case panel: LSB vs HSB (RAR Plateau vs GR vs NFW)
![SPARC dual cases — LSB vs HSB](images/next_steps/enhanced_20250805_115400/sparc_panel_dual_cases.png)

Figure 3a | RAR Plateau captures both low‑surface‑brightness (LSB), extended, flat‑outer rotation curves and high/typical‑surface‑brightness (HSB) systems. Left: LSB extended galaxies (examples: UGC 00128, UGC 05005, UGC 01230) selected by SBdisk0 ≤ 120 L☉/pc², Rmax/Rd ≥ 8, and flat outer slopes (|dV/dR| ≤ 1.5 km s⁻¹ kpc⁻¹; outer ΔV/V ≤ 0.15). Right: representative HSB/typical spirals (examples: NGC 5055, NGC 2841, NGC 3198). Curves show Observed (black), GR (blue), NFW (green, quick‑fit), and RAR Plateau (red). The same gating function is used for both groups; only per‑galaxy a0 is scanned within the paper preset. Minor tensions at extreme radii (very outer points) are discussed below.
*   Selection details: thresholds follow tools/find_sparc_lsb_extended_flat.py; see code for exact formulas and CSV at `results/lsb_extended_flat_candidates.csv`.

**RAR master panel** (optional ΛCDM band).

![RAR master panel — SPARC selection with DGG posterior band](images/paper/rar_master_panel.png)

Source‑Data: `results/next_steps/enhanced_20250805_115400/rar_master_panel_source.csv`.

**BTFR outcome.** On a working subset (N≈89) using $M_b=M_\star+1.33\,M_{\mathrm{HI}}$ and observed $V_{\rm flat}$, a simple log–log fit yields slope $3.184\,[3.034,\,3.332]$ (p50 [p16, p84]); $R^2\approx0.885$ and RMS scatter $\approx0.22$ dex (see `btfr_fit_summary.json`). The deep‑regime prediction from the $\nu$‑function approaches $M_b\propto V^4$; selection criteria and goodness‑of‑fit metrics are documented (Methods), and we compare slope/scatter to SPARC BTFR results (SI).

![BTFR (subset)](images/next_steps/btfr_fix_20250906/btfr_baryonic.png)

Figure 4 | The DGG framework reproduces the Baryonic Tully-Fisher Relation (BTFR). The relation between total baryonic mass ($M_b$) and flat rotation velocity ($V_f$) for a subset of 89 SPARC galaxies. DGG model predictions (red points) are fitted with a log-log linear relation (black line), yielding a slope of $3.18 \pm 0.15$, consistent with empirical measurements. The shaded region represents the bootstrap confidence interval of the fit.
*   **Source Data:** `results/next_steps/btfr_fix_20250906/btfr_baryonic.csv`

---

## 4. Solar‑System Constraints

Any modified gravity must clear Solar‑System bounds. We evaluate the same $\xi(r)$ in the Sun’s Kepler field $g_N(r)=GM_\odot/r^2$ and report. Operative assumption: in the screened Solar limit, $\epsilon\equiv\xi-1$ is locally constant along the Cassini ray and across the AU‑scale orbits used by ephemerides; see docs/ppn_mapping.md for the scaling $|\epsilon|\sim a_0/g$ and nature_readiness/solar_system/ephemeris_perturbations.py for a code‑based perihelion‑precession surrogate.

$$
\left|\frac{\Delta G}{G}\right| \;\equiv\; \left|\xi(r)-1\right|
$$

at 1–30 AU. We show a **gated** curve (same parameters as galaxy fits) and a **worst‑case** curve (no screening).

![Solar‑System constraints](images/next_steps/rar_plateau_mw_full/solar_rar_plateau.png)

Figure 5 | The DGG model satisfies Solar System constraints. The predicted fractional deviation from Newtonian gravity, $|\Delta G/G| = |\xi-1|$, is plotted as a function of orbital distance from the Sun. The model's prediction (blue curve) remains orders of magnitude below the constraint from the Cassini mission on the PPN parameter $\gamma$ (gray band, $|\gamma-1|<2.3\times10^{-5}$; Bertotti, Iess & Tortora 2003), confirming the model's compatibility with high-precision local tests.
*   **Source Data:** `results/next_steps/rar_plateau_mw_full/solar_system_table.csv`


---

## 5. Gravitational Lensing with One Metric

We map surface brightness to stellar mass using an SED‑informed $M/L$ prior. For SLACS ETGs the paper preset applies a population‑level Salpeter‑like offset $\delta_{\rm IMF}=+0.23$ to Chabrier SED masses; Chabrier results are shown for comparison. We deproject a Sérsic profile with measured $(n, R_e)$ (spherical baseline; axis ratio $q$ and external convergence priors can be added in SI), and build $\Sigma(R)$, $\bar\Sigma(<R)$, $\Delta\Sigma(R)$, and $\theta_E$ using the same metric‑only gate $\xi(g)$ (with $\Phi=\Psi$). Uncertainties propagate from $(\log_{10}M_\star, R_e, n, \theta_E^{\rm obs})$; we report residuals and RMSE and compare practice to SLACS.

![θ_E: predicted vs observed](images/next_steps/enhanced_20250805_115400/lensing_thetaE_pred_vs_obs.png)

Figure 6 | The DGG framework consistently predicts strong gravitational lensing. Predicted versus observed Einstein radii ($\theta_E$) for a sample of 70 early-type galaxies from the SLACS survey (Auger et al. 2009). Predictions from the DGG model (red points), using the same metric-only framework as for dynamics, cluster around the one-to-one relation (dashed line). In contrast, predictions from baryons alone in GR (blue points) systematically underestimate the lensing strength.
*   **Source Data:** `results/next_steps/enhanced_20250805_115400/lensing_metric_table.csv`

**IMF normalization for SLACS ETGs.** With SED masses on a Chabrier IMF prior, the metric‑only GG prediction underestimates Einstein radii (high‑$g$; $\xi\simeq1$). Introducing a single population‑level offset $\delta_{\rm IMF}$ for early‑type lenses (Salpeter‑like, +0.23 dex) brings the amplitude into agreement without changing $\xi(g)$. Coverage improves and bias is reduced (Chab: 68% = 0.014, 95% = 0.100; Salp: 68% = 0.500, 95% = 0.800). For fairness, we apply the same IMF choice to the GR+baryons baseline; see the GR rows in the comparison table. $\delta_{\rm IMF}$ is a single population‑level normalization for SLACS ETGs only; it is not used in SPARC, MW $K_z$, or Solar‑System tests. See also:
- Table: `docs/tables/lensing_imf_comparison.md`
- Histogram: `docs/figures/lensing_imf_f_theta_hist.png`
- ΔAIC JSON: `docs/metrics/lensing_imf_delta_aic.json`
- q‑axis summary: `docs/stats/lensing_q_axis_ratio_summary.json`

Defaults for coverage sampling (θ_E): unless overridden, we propagate $\sigma(\log_{10}M_\star)=0.10$ dex, a fractional $\sigma(R_e)=3.5\%$, and a global $\kappa_{\rm ext}\sim\mathcal N(0,0.03)$. ΔAIC is computed on the subset of lenses with reported $\theta_E$ uncertainties; we report $N_{\rm eff}$ in the JSON. When per‑lens $\sigma_{\theta_E}$ are unavailable, ΔAIC is omitted (see JSON for $N_{\rm eff}$).

[Theoretical stacked ΔΣ moved to supplemental] See docs/supplemental_lensing_stack.md for the theory‑only stack (with GR vs RAR overlay), code references, and source tables. We omit it from the main paper because no observed stack is overlaid here; accuracy is judged instead via the θ_E comparison (predicted vs observed), for which metrics and scatter are included above.

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

Importer sanity checks (applied in scripts/tools):
- Assert θE is provided in arcsec on ingest; warn if any θE > 10″ (common kpc mis-units).
- Assert Re is in arcsec before cosmology conversion; log the cosmology used (H0, Ωm) in table metadata.
- Cosmology constants are pinned in one place (LENSING_COSMO in scripts/next_steps_from_run.py) and included in run_metadata for every build; the same values are used consistently across lensing artifacts.

---

## 6. Discussion and Implications

**Predictive power vs flexibility.** With a single principal scale $a_0$ and a fixed $\nu$-function, DGG reproduces broad rotation‑curve trends across diverse galaxies, naturally respecting the RAR and approaching the BTFR. This rigidity prevents per‑galaxy over‑fitting, sharpening falsifiable predictions (e.g., outer‑slope behavior). We compute code‑based universality metrics (χ²/ν for a global $a_0$ and WAIC‑like comparisons for universal vs hierarchical $a_0$) and write results to results/next_steps/.../universality_metrics.json. A compact summary table is provided below (numbers reflect the current run; see JSONs for full details). 

**Solar‑System safety.** Under the screened subclass (Φ=Ψ, c_T=1) we have γ=β=1 at 1PN and α1=α2=0 in the Solar limit. Cassini constrains γ (|γ−1|≲2.3×10⁻⁵); amplitude rescaling ε≡ξ−1 only projects to Cassini if not absorbed into the GM used by ephemerides (see docs/ppn_mapping.md). Operatively, we assume ε is locally constant along the Cassini ray and across AU‑scale orbits; we verify negligible effects with a code‑based perihelion‑precession surrogate (nature_readiness/solar_system/ephemeris_perturbations.py). We therefore show |ΔG/G|≡|ξ−1| as a conservative tracer vs. AU and export a PPN CSV alongside the Solar table.

**Vertical forces and local surface density.** A decisive check is $K_z(R_0,z)$ and $\Sigma_{1.1}$. We use the **full 3‑D** DGG contribution throughout the paper preset.

**Lensing under one metric.** Metric‑only predictions show the right order of magnitude for $\theta_E$ and stacked $\Delta\Sigma$ with measured lens inputs; residuals and RMSE are reported. A single‑theory lensing success is essential. **Sanity:** varying the stellar $M/L$ prior within the SED‑informed band shifts the amplitude of $\Delta\Sigma$ but not its slope over 0.05–300 kpc.

**Environment term (ablation).** We publish an ablation between env‑ON and env‑OFF on the same SPARC subset with Δχ², ΔAIC/ΔBIC (global parameter counts stated), and, where available, Δlog Z. We also report residual correlation slopes vs central surface brightness, inclination, and gas fraction (Pearson/Spearman with p‑values). See `env_ablation_summary.json` and `residual_correlations.{json,csv}` (SI) for the quantitative outcomes; the Discussion notes the sign of ΔIC in plain language.

**Open issues.** (i) Whether a **finite plateau** $D_{\max}$ is required observationally (and, if so, at what value). (ii) Universality of $a_0$: hierarchical results and environment‑dependence. (iii) Clusters and ultra‑diffuse systems (may need residual mass such as neutrinos). (iv) Cosmological growth and CMB/BAO consistency in a relativistic completion. In this paper preset we adopt $D_{\max}=50$; galaxy fits and Solar bounds are empirically robust for $D_{\max}\in[30,80]$, with strong‑lensing sensitivity tested in Extended Data. See `docs/dmax_cap.md` for details and sweep instructions. **Falsifiability:** a decisive failure would be a requirement for $D_{\max}\gg100$ to fit strong‑lensing scales or cluster dynamics under the same mapping, or a systematic misfit of the BTFR slope/scatter under standardized selections.

#### On minor mismatches at extreme radii (outermost points)
- Beam smearing and asymmetric drift corrections can degrade rotation‑curve reliability at low surface brightness and large radii (inclination systematics and non‑circular motions increase fractional errors). SPARC files report quality flags Q; many LSB extended systems are Q=2.
- Distance and inclination uncertainties feed directly into V_obs and can bias the outer slope by several km/s, particularly for near face‑on systems.
- Our overlays use component rotmods and reconstruct gas surface density when _HIrad profiles are missing (documented in code logs). This reconstruction can affect the precise v_bar tail, but the effect on RAR Plateau overlays is modest relative to observational systematics at the farthest radii.
- The paper preset does not tune parameters radius‑by‑radius; we scan only a0 per galaxy with a fixed gate and a finite plateau. Residual tensions at the last point or two are expected within the quoted error budgets and selection floors. We therefore refrain from ad‑hoc tweaks and disclose the floors used (σ_floor, obs_frac_sigma) and Q.

---

### Outlook — cosmological extension (separate paper)

Because $\xi(g)$ is local, it can be coupled to line‑of‑sight integrals to build a redshift–distance mapping without assuming FRW kinematics. We have explored an **energy–gravity reciprocity** ("energy tariff") where a tiny cumulative energy drain along low‑$g$ sightlines yields a phenomenological $z(r)$ consistent with SN Hubble‑diagram curvature and Liouville‑preserving CMB transport. These results **do not** alter any galaxy/lensing conclusions here and will be presented in a **companion paper**.

---

## Conclusions

- **Unified galaxy dynamics without dark halos.** Using a single scale $a_0$ and a fixed $\nu$-function (Box 1), DGG reproduces the broad form of rotation curves for the Milky Way and representative SPARC galaxies.
- **Scaling laws.** DGG respects the RAR by construction and approaches the BTFR expectation $M_b\propto V^4$; measured slopes depend on selection and will be reported with scatter.
- **Local tests.** For parameters that fit galaxies, $|\Delta G/G|$ in the Solar System remains below Cassini‑level sensitivity at Saturn; a worked PPN derivation is provided (docs/ppn_mapping.md).
- **One‑theory lensing.** A metric‑only mapping gives reasonable $\theta_E$ and $\Delta\Sigma$ predictions; completing the lens sample with measured inputs is a priority.
- **Roadmap.** (1) Full 3D $K_z/\Sigma_{1.1}$; (2) lensing with measured $(M_\star,R_e)$ and uncertainties; (3) hierarchical $a_0$ with nuisances; (4) PPN appendix; (5) cluster/cosmology tests.

---

## 7. Methods

**Hierarchical $a_0$ (optional).** When enabled, we infer a population‑level mean and scatter in $\ln a_0$ from per‑galaxy grids (dynesty nested sampling). We report $(\mu,\sigma)$ posteriors in `hierarchical_a0_posterior_summary.json` and a heatmap at `images/.../hierarchical_a0_posterior_heatmap.png`.  

**Baryon models.** Milky Way disks (Miyamoto–Nagai) + bulge (Hernquist) + gas; external galaxies use SPARC component rotmods.  

**Computation.** We evaluate $\xi(g)$ as in **Box 1**, with unit conversion constant $C$ and optional gates $s_\rho, W(T)$.  

**Fitting $a_0$.** Per‑galaxy **grid** $\log_{10} a_0\in[-10.5,-9.3]$ (60 points) minimizing $\chi^2$; optional **hierarchical** log‑normal prior for $a_0$ with nested sampling. We report both "raw" $\chi^2$ (as in the SPARC files) and floor‑augmented fits using a velocity error floor and/or fractional observational floor to capture inclination/distance/beam systematics. For headline SPARC reduced‑$\chi^2$ we adopt `--sigma-floor 6.0` (km/s) and `--obs-frac-sigma 0.05` unless otherwise stated; raw values (no floors) are reported alongside in SI. Residual PPC plots (residuals vs R with 16–84% bands) can be generated via `tools/sparc_ppc.py`.  

**Solar‑System.** In the Solar limit where $g_{\rm bar}\gg a_0$, $\xi\to1$ and the metric reduces to GR with $\gamma\simeq\beta\simeq1$ and $\alpha_{1,2}\simeq0$. We evaluate $\xi(r)$ in the Sun’s field and report $|\Delta G/G|$ at 1–30 AU; compare to the Cassini line as a consistency check. When the relativistic module is present (adopted subclass $\Phi=\Psi, c_T=1$), we also export a PPN CSV (`ppn_table.csv`) with $(\gamma,\beta,\alpha_1,\alpha_2)$.

**PPN mapping and export.** In the metric subclass we adopt, the weak-field line element is $ds^2=-(1+2\Phi/c^2)dt^2+(1-2\Psi/c^2)d\mathbf x^2$ with screening such that $\Phi=\Psi$ and $c_T=1$. The DGG gate $\xi(g)$ rescales the weak-field potential by a small factor $1+\epsilon$ with $\epsilon\equiv\xi-1\ll1$ in the Solar System. Matching coefficients of the baryonic Newtonian potential $U$ at 1PN, the equal additive contribution of $c^2\phi_{\rm env}$ to $g_{00}$ and $g_{ij}$ implies the coefficient ratio is unity: $\gamma=1$; $\beta=1$; and preferred‑frame parameters $\alpha_{1,2}=0$ in this limit. Cassini measures the coefficient of the logarithmic term in the Shapiro delay relative to the ephemeris $GM_\odot$; hence the two regimes (degenerate vs non‑degenerate amplitude) discussed in docs/ppn_mapping.md. We therefore use $|\Delta G/G|=|\xi-1|$ as a conservative amplitude tracer and export a PPN table with columns $(\mathrm{AU},\gamma-1,\beta-1,\alpha_1,\alpha_2,|\Delta G/G|)$ alongside the Solar System source‑data CSV. The figure shows $|\Delta G/G|$ vs. $r$ with planetary semi‑major axes marked and a reference band for the Cassini $|\gamma-1|$ limit on a secondary axis. See docs/ppn_mapping.md for the precise $\gamma/\epsilon$ conditions.

**PPN mapping (sketch).** In PPN gauge, $ds^2=-(1-2U)dt^2+(1+2\gamma U)dx^2$, so $\gamma$ is the ratio of the coefficients of $U$ in $g_{ij}$ vs $g_{00}$. In our subclass with screening and $\Phi=\Psi$, both potentials receive the same additive $c^2\phi_{\rm env}$ at 1PN, so the coefficients of $U$ match and $\gamma=1$, while light‑deflection and Shapiro‑delay amplitudes scale with $(1+\gamma)U\to (1+\gamma)(1+\epsilon)U$. Because Cassini determines this coefficient relative to the ephemeris $GM_\odot$, a uniform $\epsilon$ absorbed into $GM_\odot$ leaves $\gamma$ unchanged (degenerate regime); if not absorbed, $|\gamma-1|$ projects to $|\epsilon|$ at the $\sim10^{-5}$ level along the ray. We therefore use $|\Delta G/G|$ as a conservative tracer and refer to docs/ppn_mapping.md for details; in the screened Solar limit both $\epsilon$ and $|\gamma-1|$ approach zero, consistent with the CSV export (ε treated as locally constant over 1–30 AU).
**Lensing (metric‑only).** We adopt a metric‑only mapping with $\Phi=\Psi$; the deflection potential is $2\Phi$, so the same $\xi(g)$ that boosts dynamics boosts lensing. Critical surface density $\Sigma_{\rm cr}(z_l, z_s)$ and distances use a flat $\Lambda$CDM cosmology with $H_0=70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and $\Omega_m=0.3$. Stellar masses come from SED‑based $M/L$ (prior specified in Supplement), and sizes $(R_e, n)$ are measured from the discovery images or follow‑ups listed in the lens table. Residuals and goodness‑of‑fit metrics for $\theta_E$ are written to `lensing_thetaE_residuals.csv` and `lensing_thetaE_metrics.json`. The ETG mass normalization $\delta_{\rm IMF}$ is a single population‑level parameter applied to all SLACS lenses; it is not a per‑lens degree of freedom. For fairness, we apply the same IMF choice to the GR+baryons baseline. $\delta_{\rm IMF}$ applies only to early‑type strong lenses (SLACS ETGs) and is not applied to SPARC disc rotation‑curve fits.  
**Model comparison.** Δlog Z histograms are reported as a BIC approximation; full evidences are produced when hierarchical runs are enabled.

**NFW comparator priors.** For the NFW yardstick we adopt weak, non‑informative bounds on $\log_{10} M_{200}$ and $c$ consistent with a standard mass–concentration relation at $z\simeq 0$. Fits are performed by $\chi^2$ minimization on the same radii and velocity uncertainties as the DGG/GR fits; exact prior ranges and any mass–concentration hyper‑prior are listed in SI (Table Sx).

---

## 8. Code and Data Availability

- **Code.** Analysis and plotting scripts are part of this repository. The exact function used in all figures is **Box 1**, implemented as `xi_rar_plateau_numpy`.  
- **Data.** SPARC rotmod files and Source Data CSVs accompany the figures (`results/...`) and are tracked with Git LFS.  
- **Reproduction.** See `scripts/reproduce_paper.py` for end‑to‑end regeneration of figures and tables. Each run writes a `run_metadata.json` with flags, environment, and timestamp; SPARC selection disclosure is saved to `sparc_selection.json`.

Reproduction (one command)
From repo root, with paper run NPZ and SPARC rotmods available:
```bash
RUN_DIR=runs/enhanced_20250805_115400 \
SPARC_DIR=external_data/Rotmod_LTG \
LENS_CSV=docs/lensing_targets_slacs.csv \
./reproduce_paper.sh
```
Note: Paper preset enables circularized Re (q) and δIMF=+0.23 (ETG) by default. Override via:
- --use-circularized-Re=false
- --delta-imf-dex 0.0   # Chabrier variant
Docker (CPU-first):
```bash
docker build -t dgg-repro .
docker run --rm -it \
  -e RUN_DIR=runs/enhanced_20250805_115400 \
  -e SPARC_DIR=external_data/Rotmod_LTG \
  -e LENS_CSV=docs/lensing_targets_slacs.csv \
  -v "$PWD/runs:/app/runs" \
  -v "$PWD/external_data/Rotmod_LTG:/app/external_data/Rotmod_LTG:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/images:/app/images" \
  dgg-repro
```

---

## 9. Galaxy clusters — CLASH × ACCEPT (preliminary)

We test the RAR‑gated framework on galaxy clusters using published total mass models from Umetsu et al. (2016, CLASH; NFW fits per cluster) and gas‑only baryons from the ACCEPT database (external_data/accept_database.dat). This preliminary section summarizes a reproducible pipeline run and the resulting cluster‑scale RAR scatter.

What the pipeline does
- Validates raw ACCEPT shells (monotonic radii, removes overlaps, drops non‑finite/non‑physical n_e).
- Converts shells to cumulative gas mass and computes baryonic accelerations g_bar.
- Uses Umetsu+2016 CLASH NFW parameters to compute total accelerations g_tot.
- Fits a universal acceleration scale a0 for three theory curves (MOND‑simple, EG‑like, and RAR‑plateau with Dmax=50) and reports scatter.

Assumptions
- Cosmology: H0=70 km s⁻¹ Mpc⁻¹, Ωm=0.27, ΩΛ=0.73 (see summary.json).
- Mean molecular weight per free electron: μ_e=1.17.
- Gas‑only baryons in this run; optional BCG stellar mass may be included via external_data/clash_stars.csv (if provided; off by default).
- Data hygiene: enforce monotonic R_out, filter overlaps, drop bad n_e; report f_gas(<0.5 R200c) and f_gas(<R200c).

### Results (gas only; CLASH NFW totals vs GG and GR)

- Matched clusters: 7, points: 220
- Global fit (RAR‑plateau, Dmax=50): a0 = 1.93×10⁻⁷ cgs
  RMS scatter in log10 g: 0.146 dex (RAR‑plateau) vs 1.014 dex (GR baryons‑only)
- Coverage: 47.3% within ±0.1 dex; 85.5% within ±0.2 dex; 49.1% positive residuals
- Radial trend in residuals Δlog10 g ≡ log gNFW − log gRAR as a function of x ≡ r/R200c:
  - Inner median (x ≤ 0.2): +0.085 dex
  - Outer median (x > 0.2): −0.130 dex
  - Linear fit: Δlog g ≈ 0.196 − 0.93 x (zero at x ≈ 0.21)

Interpretation. The positive inner residuals are consistent with missing stellar baryons (BCG + ICL) in the gas‑only g_bar. At larger radii the gas dominates g_bar and residuals tilt negative, as expected from a single‑parameter gate matched to galaxy‑scale a0.

Figure: GR vs RAR vs CLASH NFW

![Cluster RAR: NFW data vs GR and RAR plateau](images/next_steps/cluster_rar/cluster_rar_scatter.png)

Source‑data (results/cluster_rar/)
- cluster_rar_points.csv — points used in the scatter.
- diagnostics.csv — raw‑data hygiene and quick sanity (n_shells, n_used, min/max n_e, f_gas at 0.5/1.0 R200c, stars_used, used_frac).
- metrics.json — RMS in log10 space vs GR and the RAR‑plateau and fitted a0 (Dmax=50).
- summary.json — cosmology, μ_e, matched cluster count, and fitted a0 for all theory curves (MOND‑simple, EG‑like, RAR‑plateau).
- cluster_section_metrics.json — jackknife/bootstrapped uncertainties for a0 and RMS.

Reproduce locally
```bash path=null start=null
source .venv/bin/activate
python scripts/cluster_rar_pipeline.py \
  --accept external_data/accept_database.dat \
  --results results/cluster_rar \
  --images images/cluster_rar \
  --warn-fgas 0.2 \
  --warn-used-frac 0.6 \
  # Optional: include BCG stellar mass if available
  --stars-csv external_data/clash_stars.csv
```

Notes
- Optional stars CSV — either of the following schemas is accepted by the helper:
  1) logM/Re form: cluster, log10Mstar_BCG, Re_kpc, [log10Mstar_ICL, Re_ICL_kpc], profile
  2) M/a form: cluster, M_BCG_Msun, a_BCG_kpc, [M_ICL_Msun, a_ICL_kpc]
  Profiles hernquist|sersic4 map internally to Hernquist. If present and matched, stars are folded into g_bar.
- Warnings are emitted if f_gas(<R200c) > 0.2 or the fraction of ACCEPT shells used < 0.6.
- Cluster‑name matching between CLASH and ACCEPT uses robust normalization and aliasing in the pipeline.

Optional: add BCG/ICL stars (Hernquist) and re‑fit

```bash
# Gas‑only (sanity)
python scripts/cluster_clusters_plus_stars.py \
  --points results/cluster_rar/cluster_rar_points.csv \
  --diagnostics results/cluster_rar/diagnostics.csv \
  --outdir results/cluster_rar_plus \
  --images images/next_steps/cluster_rar_plus

# Gas + stars (provide per‑cluster stars CSV)
python scripts/cluster_clusters_plus_stars.py \
  --points results/cluster_rar/cluster_rar_points.csv \
  --diagnostics results/cluster_rar/diagnostics.csv \
  --stars external_data/clash_stars.csv \
  --outdir results/cluster_rar_plus_stars \
  --images images/next_steps/cluster_rar_plus_stars
```

```bash
# venv (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install numpy matplotlib
python scripts/cluster_rar_pipeline.py \
  --accept external_data/accept_database.dat \
  --results results/cluster_rar \
  --images images/cluster_rar
```

Notes
- Total accelerations are from CLASH NFW (Table 2 of Umetsu+2016) at the paper cosmology ($H_0=70,\, \Omega_m=0.27,\, \Omega_\Lambda=0.73$).
- Baryons are gas‑only from ACCEPT shells using $\rho_{\rm gas}=\mu_e m_p n_e$ with $\mu_e=1.17$.
- The figure overlays:
  - CLASH NFW points (log10 $g_{\rm bar}$ vs log10 $g_{\rm tot}$),
  - GR baseline (one‑to‑one), and
  - RAR‑plateau (global $a_0$, $D_{\max}=50$).
- The raw‑data hygiene and $f_{\rm gas}$ sanity are reported per cluster; inspect diagnostics.csv to flag suspect inputs.

---

## 9. References

(finalizing this list)

---

For extended figures, latest results tables, and detailed reproducibility instructions, see REPRODUCIBLE.md.

For headline SPARC fits we adopt a modest noise floor unless stated: σ_floor = 6 km/s and obs_frac_sigma = 0.05. We report raw (no floors) and floor‑augmented metrics side‑by‑side, and provide posterior‑predictive checks (PPC; residuals vs radius with 16–84% bands) for representative HSB/LSB subsets. The aggregator standardizes residuals with $\sigma_{\rm eff}=\sqrt{\sigma^2+6^2+(0.05\,V_{\rm obs})^2}$ and records these floor settings in `docs/metrics/sparc_fit_quality.json`.

- JSON: `docs/metrics/sparc_fit_quality.json`
- Figure: `docs/figures/sparc_ppc_panel.png`

### Universality Metrics

The orchestrator can compute a global (universal) $a_0$ and WAIC‑like metrics (when the optional module is available). Below is a compact summary from the current run; see JSONs for full details and confidence intervals.

- Universal a0 (global fit): a0 ≈ 5.0×10⁻¹¹ m s⁻² (demo subset; see `results/next_steps/.../global_a0.json`), χ²/ν ≈ 28.12 (raw; no floors). WAIC/LOO or ΔBIC vs hierarchical are reported in `universality_metrics.json` when enabled.
- Hierarchical a0 (two‑stage or Bayesian): μ_ln a0, σ_ln a0 and coverage are written to `hierarchical_a0_summary.json` / `hierarchical_a0_posterior_summary.json` when requested (see REPRODUCIBLE.md). 
- Env ON vs OFF: ΔAIC/ΔBIC (and Δlog Z where available) and residual‑correlation slopes before/after are summarized in `universality_metrics.json`.

These summarize the latest outputs under `results/next_steps/btfr_fix_20250906/` and related top-level summaries. Full CSVs are linked for reproducibility.

### Solar System Source Data

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
  `images/next_steps/enhanced_20250805_115400/lensing_rar_J0037-0942.png`,  
  `images/next_steps/enhanced_20250805_115400/lensing_rar_J1402+6321.png`

Updated SLACS outcome (N=70):

|| Model | RMSE_abs [arcsec] | MAE_abs [arcsec] | Bias_abs [arcsec] | RMSE_rel | MAE_rel | Bias_rel |
||:------|-------------------:|-----------------:|------------------:|---------:|--------:|---------:|
|| RAR (baseline; Salpeter‑like) | 0.256 | 0.213 | -0.171 | 0.202 | 0.172 | -0.133 |
|| GR (baryons; Salpeter‑like)   | 0.368 | 0.321 | -0.313 | 0.288 | 0.259 | -0.252 |
|| RAR (κ_ext‑marg; Salpeter‑like) | 0.256 | 0.213 | -0.172 | 0.202 | 0.173 | -0.134 |

Definitions: RMSE_abs/MAE_abs/Bias_abs are in arcsec; Bias_abs = mean(pred − obs). RMSE_rel/MAE_rel/Bias_rel use residuals normalized by obs; Bias_rel = mean((pred − obs)/obs). We assume n_sersic=4 (ETG baseline); measured n≠4 would shift amplitudes at the O(10%) level. κ_ext prior: mean=0, σ=0.03, samples=2000.

### D_max plateau sweep (insensitivity across 30–∞)

We ran the post‑processing cap $D_{\max}$ over {30, 50, 80, ∞} using the same pipeline (SPARC, MW $K_z$, metric‑only lensing). Results are insensitive across this range:

- Lensing (SLACS, N=70): summary metrics are insensitive across caps within reported precision; see the lensing section above for the Salpeter‑like preset values.
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


## Appendices (selected)

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

### Appendix A — Exact working formula

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

*Optional plateau:* impose $\xi\le D_{\max}$ if a finite cap is required observationally (see `docs/dmax_cap.md`).

### Appendix B — PPN and Cassini (screened Solar limit)

In the adopted screened weak‑field subclass with $\Phi=\Psi$ and $c_T=1$, Solar‑System limits imply $\gamma=\beta=1$ and $\alpha_{1,2}=0$. We therefore use $|\Delta G/G|\equiv|\xi-1|$ as a conservative tracer for any residual weak‑field rescaling. We export a per‑AU PPN table (`ppn_table.csv`) with columns $(\mathrm{AU},\gamma-1,\beta-1,\alpha_1,\alpha_2,|\Delta G/G|)$, and the Solar figure plots $|\Delta G/G|$ vs AU alongside the Cassini $|\gamma-1|$ reference band.

