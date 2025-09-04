# RAR-Gated Gravity: Reproducing Flat Galactic Rotation Curves *without* Dark Matter

> **Repository:** https://github.com/lrspeiser/DensityDependentMetricModel  
> **Figure used below:**  
> https://github.com/lrspeiser/DensityDependentMetricModel/blob/main/images/rar_vs_gr_nfw_gaia.png

---

## Abstract

Spiral-galaxy rotation curves remain nearly flat far beyond the bright disk, in tension with the Keplerian fall-off expected from visible (baryonic) matter under General Relativity (GR) [1, 2]. The prevailing fix—postulating massive non-baryonic dark-matter halos—has no laboratory confirmation after decades of direct and indirect searches. We present a **RAR-gated** gravity model: a **bounded, acceleration-anchored** rescaling of the baryonic gravitational field that (i) recovers GR at high acceleration and high density, (ii) **enhances gravity** as the baryonic acceleration drops toward an empirical pivot \(a_0\), and (iii) **saturates** to a finite plateau at extremely low acceleration so it never diverges. Conceptually, spacetime’s “fabric” stretches more easily as the baryonic environment thins—akin to a **running coupling** (cf. QCD color) that strengthens at large separation but remains bounded.

Applied to **Gaia DR3** Milky Way kinematics (~144k stars; analysis window 6–14 kpc), the RAR-gated model reproduces the flat outer trend **without a dark halo**, attaining **NFW-level residuals** in the main disk (RMSE ≈ 22.7 km s\(^{-1}\) vs. ≈ 22.6 km s\(^{-1}\) for NFW; GR-only ≈ 64.6 km s\(^{-1}\)). Under identical likelihoods/sampler settings, **Bayesian evidence decisively favors RAR over GR** and is competitive with a quick NFW fit to the same dataset. We outline predictions (outer-disk shape, environmental response, lensing), a matched SPARC/THINGS test plan, and CLI steps to reproduce the analysis.

---

## 1. Introduction

Observed rotation curves remain roughly **flat** at large radii [1, 2], whereas a baryons-only GR model predicts \(V(R)\!\propto\!R^{-1/2}\). ΛCDM resolves this with **dark-matter halos** that keep \(V(R)\) elevated, but **no non-gravitational signal** of DM has been confirmed. A powerful empirical constraint is the **Radial Acceleration Relation (RAR)**: across diverse disks, the observed centripetal acceleration \(g_{\rm obs}(R)\) correlates tightly with the Newtonian baryonic acceleration \(g_{\rm bar}(R)\) [3]. At \(g_{\rm bar}\!\gg\!a_0\) one finds \(g_{\rm obs}\!\simeq\!g_{\rm bar}\) (Newtonian), while for \(g_{\rm bar}\!\lesssim\!a_0\) one finds \(g_{\rm obs}\!\approx\!\sqrt{a_0\,g_{\rm bar}}\)—the scaling that keeps \(V(R)\) flat (anticipated in MOND [4]).

We ask whether a **single, minimal** modification—tied to \(g_{\rm bar}\) and bounded—can fit the **Milky Way** at halo-level quality **without** invoking a dark halo.

---

## 2. RAR-gated gravity (bounded, acceleration-anchored)

We retain the baryonic mass model \(V_{\rm bar}(R)\) (thin+thick stellar disks, bulge, gas), and define the **effective** circular speed
\[
V^2(R)=\xi\!\big(g_{\rm bar}(R)\big)\,V_{\rm bar}^2(R),\qquad 
g_{\rm bar}(R)=\frac{V_{\rm bar}^2(R)}{R}.
\]

A simple, bounded **RAR gate** is
\[
\xi(g_{\rm bar}) \;=\; 1 \;+\; \frac{\lambda_{\max}}{\,1 + (g_{\rm bar}/a_0)^{\gamma}\,},\quad 
\lambda_{\max}>0,\;\gamma>0,
\]
so that:
- **High-acceleration / high-density** (\(g_{\rm bar}\!\gg\!a_0\)): \(\xi\to1\) (pure GR).
- **Low-acceleration** (\(g_{\rm bar}\!\ll\!a_0\)): \(\xi\to 1+\lambda_{\max}\) (**flat plateau**, no divergence).

> *Intuition.* As the baryonic environment thins, the metric response **strengthens modestly** and then **saturates**—a “running-and-saturation” picture reminiscent of QCD color (stronger at long range, yet bounded).

**Optional localization/screening (for surveys; not required in the MW figure below).**  
A **tidal window** \(W(T)\in[0,1]\) can localize the effect to tidally structured outer disks, and a **density screen** \(S_\rho(\rho)\) can exponentially suppress any deviation in high-density regions (Solar-System safe). The full coupling is
\[
\xi(g_{\rm bar},T,\rho)=1+\big(\xi(g_{\rm bar})-1\big)\,W(T)\,S_\rho(\rho).
\]

---

## 3. Data & method (Milky Way, Gaia DR3)

**Dataset.** ~144,000 **Gaia DR3** stars processed to Galactocentric coordinates with quality cuts; rotation medians in 0.5 kpc annuli. Analysis window: **6–14 kpc** (robust), with checks at 8–14 and 12–16 kpc [5, 6].

**Baselines under identical controls.**
- **GR (baryons-only)**: \(\xi\equiv1\).
- **ΛCDM/NFW** [7]: two halo parameters + same baryons.
- **RAR-gate:** \(\xi(g_{\rm bar})\) above, \(W\equiv1\) for MW; density-screen implicit via \(\xi\to1\) at high \(g_{\rm bar}\).

**Likelihood & sampler.** Gaussian in \(V(R)\) with a shared \(\sigma_{\rm floor}\); **dynamic nested sampling** (same \(n_{\rm live}\), `maxcall`, `dlogz_target` across all three) [8].

---

## 4. Results (Milky Way)

- **Fit quality (6–14 kpc):**  
  GR(baryons-only) **RMSE ≈ 64.6 km s\(^{-1}\)**;  
  NFW **RMSE ≈ 22.6 km s\(^{-1}\)**;  
  **RAR-gate RMSE ≈ 22.7 km s\(^{-1}\)** (no halo).
- **Evidence:** RAR vs GR: **Δ\(\log Z\) ≫ 10** (decisive). In our parity run, RAR was also favored over a quick NFW fit to the same Gaia set.
- **Asymptotic speed:** \(V_{\infty}\approx 205\) km s\(^{-1}\) (12–16 kpc), i.e. a **flat** outer curve with no halo.

![Milky Way rotation curve: Gaia vs GR, NFW, and RAR-gate](https://github.com/lrspeiser/DensityDependentMetricModel/blob/main/images/rar_vs_gr_nfw_gaia.png)

*Figure 1.* **Milky Way rotation curve** from Gaia DR3 medians (black dots; 16–84% shaded) versus **GR** *(blue dashed)*, **NFW** *(green dash-dot)*, and **RAR-gate** *(red)*. GR declines; NFW and RAR-gate remain flat. RAR-gate achieves NFW-level residuals **without** a dark halo.

---

## 5. Baryonic model (summary)

We model the Milky Way with two exponential stellar disks (thin/thick), a Hernquist bulge, and an exponential gas disk (with mild flaring), fit under literature-informed priors (masses, scale lengths/heights). This is consistent with standard Galactic mass models [9].

---

## 6. Reproducibility (CLI)

> **Repo:** https://github.com/lrspeiser/DensityDependentMetricModel

```bash
# 1) Clone & setup
git clone https://github.com/lrspeiser/DensityDependentMetricModel.git
cd DensityDependentMetricModel

# 2) Run matched baselines on Gaia DR3 (same processing/likelihood)
# GR (baryons-only)
python runners/run_dynesty_stellar_fit_cupy.py --xi gr --nlive 2000 --maxcall 1500000 \
  --dlogz_target 0.01 --sample_method rslice --bound_method multi \
  --periodic_analysis --analysis_interval_min 30 --summary_interval 60

# ΛCDM/NFW (baryons + halo)
python runners/run_dynesty_stellar_fit_cupy.py --xi nfw --include_halo --nlive 2000 \
  --maxcall 1500000 --dlogz_target 0.01 --sample_method rslice --bound_method multi \
  --periodic_analysis --analysis_interval_min 30 --summary_interval 60

# 3) RAR-gated gravity (no halo)
python runners/run_dynesty_stellar_fit_cupy.py --xi rar_gate --allow_experimental \
  --nlive 2000 --maxcall 1500000 --dlogz_target 0.01 \
  --sample_method rslice --bound_method multi \
  --periodic_analysis --analysis_interval_min 30 --summary_interval 60

# 4) Make the overlay plot used in Fig. 1
python scripts/generate_comparison_plots.py  # emits images/rar_vs_gr_nfw_gaia.png
All fits use identical sampler knobs and the same radial window so RMSE and Δ
log
⁡
𝑍
logZ are apples-to-apples. Posterior/evidence JSON and NPZ snapshots are saved under runs/.

7. Potential critiques & next steps
Universality across galaxies.
Critique: MW success may not generalize.
Plan: Matched triads (GR/NFW/RAR) on SPARC/THINGS with identical 
𝜎
f
l
o
o
r
σ 
floor
​
 , D/i priors, and tidal-proxy parity; cohort Δ
log
⁡
𝑍
logZ histograms and posterior-predictives.

Acceleration scale 
𝑎
0
a 
0
​
 .
Critique: Why 
𝑎
0
a 
0
​
  and is it universal?
Plan: Informative priors around canonical 
1.2
×
10
−
10
1.2×10 
−10
  m s
−
2
−2
 ; check consistency across galaxies; explore theoretical origins (cosmological links, EFT embedding).

Lensing.
Critique: Modified dynamics must reproduce galaxy-scale lensing.
Plan: Compute deflection in the density-modulated metric (bounded 
𝜉
ξ); compare to galaxy–galaxy weak lensing and Einstein ring masses.

Screening & Solar System.
Critique: Cassini/LLR constraints must be satisfied.
Plan: Keep explicit density screening; tabulate 
∣
𝜉
−
1
∣
∣ξ−1∣ at planetary densities; verify PPN 
𝛾
γ bounds [10].

Dwarfs & extremes.
Critique: Some LSB dwarfs exhibit 
𝑔
o
b
s
/
𝑔
b
a
r
≫
1
+
𝜆
max
⁡
g 
obs
​
 /g 
bar
​
 ≫1+λ 
max
​
 .
Plan: Test whether a single 
𝜆
max
⁡
λ 
max
​
  suffices; if not, quantify residuals and assess environment (
𝑊
(
𝑇
)
W(T)) or a modest 
𝜆
λ–scaling with surface brightness.

Cosmology.
Critique: ΛCDM matches CMB/BAO; any alternative must address structure growth.
Plan: Explore linear-growth predictions of a bounded RAR-gate in a weak-field relativistic extension; compare matter power spectrum against data.

References
V. C. Rubin & W. K. Ford Jr., “Rotation of the Andromeda Nebula from a spectroscopic survey of emission regions,” ApJ 159, 379–403 (1970). doi:10.1086/150317

A. Bosma, “21-cm line studies of spiral galaxies. II. The distribution and kinematics of neutral hydrogen,” AJ 86, 1825–1846 (1981). doi:10.1086/113062

S. S. McGaugh, F. Lelli & J. M. Schombert, “The Radial Acceleration Relation in Rotationally Supported Galaxies,” Phys. Rev. Lett. 117, 201101 (2016). doi:10.1103/PhysRevLett.117.201101

M. Milgrom, “A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis,” ApJ 270, 365–370 (1983). doi:10.1086/161130

Gaia Collaboration, “Gaia DR3: summary of the content and survey properties,” A&A 674, A1 (2023). doi:10.1051/0004-6361/202243940

D. Katz et al., “Gaia DR3: spectroscopic content,” A&A 674, A5 (2023). doi:10.1051/0004-6361/202243888

J. F. Navarro, C. S. Frenk & S. D. M. White, “A Universal Density Profile from Hierarchical Clustering,” ApJ 490, 493 (1997). doi:10.1086/304888

J. S. Speagle, “dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences,” MNRAS 493, 3132–3158 (2020). doi:10.1093/mnras/staa278

J. Binney & S. Tremaine, Galactic Dynamics (2nd ed.), Princeton Univ. Press (2008).

B. Bertotti, L. Iess & P. Tortora, “A test of general relativity using radio links with the Cassini spacecraft,” Nature 425, 374–376 (2003). doi:10.1038/nature01997
