Below is a **publication‑ready, self‑contained section** you can drop into the paper. It introduces the high‑level hypothesis (why “gates” might exist), shows one concrete cosmological consequence (redshift from an energy→gravity transfer), and then folds in the quantitative model and figures you already produced. It is explicitly **optional** and does not alter the validity of the rest of the paper.

---

## Optional Cosmological Hypothesis (independent of core results)

### Why a Gravity “Gate” Might Exist: Energy–Gravity Reciprocity

**Hypothesis.**  Gravity and energy are reciprocally linked at low acceleration. When the local gravitational field falls below a characteristic scale $a_0$, the vacuum/field degrees of freedom that *induce* the elastic response of spacetime become “under‑supplied.” The system responds by **drawing energy from traversing radiation and fields** to maintain the induced gravitational stiffness. In regions with $g_{\rm bar}\gg a_0$ (e.g., inside galaxies and the Solar System), the induced response is saturated and no draw is needed.

This motivates the **gate** we employ throughout the paper,

$$
\xi(g_{\rm bar})=\min\!\left[\tfrac12+\sqrt{\tfrac14+\frac{a_0}{g_{\rm bar}}},\,D_{\max}\right],
$$

which enhances the effective gravitational response only when $g_{\rm bar}\lesssim a_0$ and then saturates at a finite plateau $D_{\max}$. In this cosmological add‑on, the same gate also regulates a tiny, cumulative **energy→gravity transfer** in the intergalactic medium.

### A Single Observable Consequence: Redshift from Energy Drain

If the reciprocity above operates during photon propagation through low‑$g$ regions, a fraction of a photon’s energy is continuously converted into the induced gravitational response—an **energy tariff** proportional to how far the local response deviates from GR:

$$
\frac{d\ln E}{dr} \;=\; -\alpha(r),\qquad
\alpha(r)=k\,\big[\xi(r)-1\big]\,f_{\rm void}(r).
\tag{1}
$$

Here $r$ is path length (Mpc), $k$ is a coupling, and $f_{\rm void}(r)\in[0,1]$ is an *environmental mix* that down‑weights parts of the line of sight that are not deep void (e.g., filaments/halos). Integrating Eq. (1) gives the accumulated redshift:

$$
1+z(r)=\exp\!\left(k\int_{0}^{r}\!\big[\xi(l)-1\big]\,f_{\rm void}(l)\,dl\right).
\tag{2}
$$

Two limits reconcile the model with local tests:
(i) **Inside galaxies** $\xi\!\approx\!1$ or $f_{\rm void}\!\approx\!0\Rightarrow$ no tariff;
(ii) **Deep voids** $f_{\rm void}\!\to\!1$, $\xi\!\to\!D_{\max}\Rightarrow$ the tariff integrates to a measurable redshift.

We adopt a minimal, data‑driven shape for the environmental mix,

$$
f_{\rm void}(r)=\frac{1}{1+(r/r_0)^{\gamma}},
\tag{3}
$$

which captures the empirical fact that typical long sightlines intersect structure and therefore do not behave as perfectly empty voids at all distances.

---

## Worked Example: Hubble Diagram and Energy Balance (unchanged gate)

We keep the same gate $\xi(g_{\rm bar})$ used for rotation curves, vertical forces, and lensing in the main text (metric subclass with $\Phi=\Psi$). Calibrating $k$ on a small‑$z$ window to preserve the local Hubble slope and sweeping $(D_{\max},\,g_{\rm bar,void},\,r_0,\,\gamma)$, we find families of solutions that reproduce the **Pantheon+** Hubble diagram with excellent fidelity.

A representative fit is

$$
D_{\max}=30,\quad g_{\rm bar,void}=10^{-15}\ {\rm m\,s^{-2}},\quad
r_0=2000\ {\rm Mpc},\ \gamma=1.5,\quad
k\simeq 7.8\times10^{-6}\ {\rm Mpc}^{-1},
$$

yielding a **reduced $\chi^2\approx0.78$** for $\mu(z)$ under our static mapping convention, and an **energy‑balance RMSE $\approx 0.015$** for $E_{\rm obs}=1/(1+z)$ vs distance. Nearby parameter choices perform comparably, indicating that the **shape parameters** $(r_0,\gamma)$ lift the far‑tail without disturbing the local slope, whereas $k$ and $D_{\max}$ are largely degenerate once the small‑$z$ calibration is fixed.

**Figures (this section):**

* **Fig. X** — *Hubble Diagram (Pantheon+ vs Energy‑Tariff)*: `hubble_diagram_with_data.png`.
* **Fig. Y** — *Per‑Photon Energy Balance*: `energy_balance_plot.png`.
* **Fig. Z** — *Predicted $z(r)$ vs linear Hubble lines*: `energy_tariff_redshift_model.png`.

*(Captions: Fig. X shows agreement with the SN locus; Fig. Y shows modelled $E_{\rm obs}(r)$ tracking empirical $1/(1+z)$; Fig. Z illustrates how the environmental mix flattens the far‑tail.)*

---

## Why This Extends—But Does Not Alter—Our Main Results

* **Same mechanism, two regimes.** The gate $\xi$ explains galaxy‑scale “extra gravity”; Eqs. (1)–(3) say the *same deviation* also governs a tiny, cumulative energy drain for photons in low‑$g$ environments.
* **Screened where GR is tested.** In high‑$g$ regions the gate closes and the tariff vanishes; Solar‑System, binary‑pulsar, and inner‑galaxy constraints used in the main paper are unaffected.
* **Falsifiable environment dependence.** The model predicts a weak correlation between SN Hubble residuals and **line‑of‑sight large‑scale structure** (through $f_{\rm void}$), providing a direct test that does not rely on expansion kinematics.

---

## Methods Summary (for reproducibility)

$$
\begin{aligned}
&\xi(g_{\rm bar})=\min\!\left[\tfrac12+\sqrt{\tfrac14+\frac{a_0}{g_{\rm bar}}},\,D_{\max}\right],\\
&\frac{d\ln E}{dr}=-k\,[\xi(r)-1]\,f_{\rm void}(r),\qquad
f_{\rm void}(r)=\frac{1}{1+(r/r_0)^{\gamma}},\\
&1+z(r)=\exp\Big(k\int_0^r [\xi(l)-1]f_{\rm void}(l)\,dl\Big).
\end{aligned}
$$

We fit $k$ on a small‑$z$ window and then sweep $(D_{\max}, g_{\rm bar,void}, r_0, \gamma)$, ranking by reduced $\chi^2$ on $\mu(z)$ (Pantheon+). The galaxy‑scale fits and lensing predictions elsewhere in the paper **use the same $\xi$** and are numerically unchanged by this section.

---

## Editorial assessment (Nature Physics level) and recommended next steps

**Strengths for a top‑tier submission**

* **Single mechanism, cross‑scale reach.** A unified gate explains both galactic “extra gravity” and the SN Hubble diagram without adding dark components.
* **Tight link to data.** The redshift construction is minimal (Eqs. 1–3) and already gives a competitive $\chi^2$ on Pantheon+; the energy‑balance figure is a clear, independent check.
* **Falsifiability.** Predicts line‑of‑sight–dependent residuals and fixes relationships among $\{k,D_{\max}\}$ once the local slope and outer‑disk phenomenology are set.

**Major issues to address pre‑submission**

1. **SN time‑dilation.** Type Ia light curves exhibit $(1+z)$ time dilation. A static tariff must **either** reproduce the same dilation (via an independent mechanism) **or** show that light‑curve standardization pipelines do not bias the test. *Action:* reprocess a subset of SN light curves with explicit time‑domain fits under the tariff hypothesis.

2. **CMB blackbody and spectral distortions.** Any energy‑loss mechanism must preserve the CMB’s near‑perfect blackbody spectrum. *Action:* compute the transformation of a Planck distribution under Eq. (1). If necessary, constrain $k$–$\xi$ evolution with redshift and include photon‑number effects so the spectrum remains Planckian to FIRAS limits.

3. **Tolman surface‑brightness and $d_L$–$d_A$ duality.** Expansion predicts $S\!\propto\!(1+z)^{-4}$. In a static framework you must specify the luminosity‑distance mapping $d_L=r(1+z)^p$ and test $p$ jointly with deep‑imaging surface‑brightness data. *Action:* add $p$ to the inference and confront the latest Tolman tests.

4. **BAO and cosmic chronometers.** BAO set a standard ruler; chronometers measure $H(z)$. *Action:* derive the predicted $z(r)$ and inferred “effective” $H(z)$ under the tariff and test BAO peak positions and chronometer data.

5. **Large‑scale‑structure consistency.** The same gate modifies the Poisson equation in voids. *Action:* ray‑trace through N‑body mocks to compute a *first‑principles* $f_{\rm void}(r)$ and the predicted correlation between SN residuals and LOS density contrast.

6. **Lensing time delays and strong‑lens cosmography.** Determine whether the tariff changes photon flight times in a way that biases time‑delay distances. *Action:* compute Fermat‑potential integrals with/without tariff and test against well‑measured systems.

7. **Parameter identifiability and degeneracies.** Quantify the $k$–$D_{\max}$ degeneracy (given the local slope) and report marginalized posteriors for $(r_0,\gamma)$ from joint fits to SNe + lensing + RAR.

**Concrete research plan (short‑term)**

* **A. Joint inference.** Fit $(k,D_{\max},r_0,\gamma,p)$ to Pantheon+ with explicit treatment of time‑dilation and K‑corrections; report Bayesian evidence vs $\Lambda$CDM.
* **B. Structure‑correlation test.** Cross‑correlate SN residuals with tomographic LOS densities (DES/KiDS/SDSS), predicting the sign and amplitude from ray‑traced $f_{\rm void}$.
* **C. CMB spectrum check.** Compute CMB spectral distortions from Eq. (1) along a realistic thermal history and compare to COBE/FIRAS bounds; if needed, constrain allowed $k(z)$.
* **D. Lensing & dynamics coherence.** With the same $\xi$, re‑validate galaxy–galaxy lensing amplitudes and outer‑disk $K_z$ while scanning the SN‑permitted band of $(r_0,\gamma)$.
* **E. Public release.** Package the tariff integrator and the parameter sweep (`sweep_results.csv`) with a reproducible notebook and figure scripts.

**Go/no‑go criterion for submission**

* Green‑light once (i) SN time‑dilation is matched without ad‑hoc fixes, (ii) CMB blackbody is preserved within FIRAS limits, and (iii) at least one environment‑dependence prediction (LOS correlation of SN residuals) is borne out at $>2\sigma$.

