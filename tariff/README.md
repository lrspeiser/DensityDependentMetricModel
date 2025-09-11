# Optional Cosmological Hypothesis (independent of core results)

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

A representative fit (this run)

$$
D_{\max}=30,\quad g_{\rm bar,void}=10^{-15}\ {\rm m\,s^{-2}},\quad
r_0=2000\ {\rm Mpc},\ \gamma=1.5,\quad
k\simeq 7.75\times10^{-6}\ {\rm Mpc}^{-1},
$$

produced the following metrics directly from the scripts in this folder:

- Hubble Diagram (Pantheon+ overlay): **reduced $\chi^2 = 0.777$** using `external_data/pantheon/Pantheon+SH0ES.dat`.
- Energy balance (comparing $E_{\rm obs}^{\rm model}=1/(1+z_{\rm model}(r_\mu))$ to data $E_{\rm obs}^{\rm data}=1/(1+z)$): **RMSE = 0.0146**.
- CMB spectral-shape check at 14 Gpc: expansion-like mapping is essentially Planckian (rms $\sim 2\times10^{-16}$), while an energy-only mapping yields large distortions (rms $\sim 1.8\times10^{-1}$), echoing FIRAS constraints.
- BAO proxy curves `H_\mathrm{eff}(z), D_M(z), D_H(z)` generated for reference.

**Figures (this section):**

* **Fig. X** — *Hubble Diagram (Pantheon+ vs Energy‑Tariff)*: `hubble_diagram_with_data.png`.
* **Fig. Y** — *Per‑Photon Energy Balance*: `energy_balance_plot.png`.
* **Fig. Z** — *Predicted $z(r)$ vs linear Hubble lines*: `energy_tariff_redshift_model.png`.
* **Fig. W** — *BAO/chronometer proxies*: `bao_proxies.png`.

*(Captions: Fig. X shows the SN locus and model overlay; Fig. Y compares energy tracks vs distance; Fig. Z illustrates environmental‑mix effects on the far tail; Fig. W shows derived proxy curves.)*

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

## Open Issues and Next Steps

See `issues.md` in this folder for a full list of outstanding theoretical and observational checks (time‑dilation, CMB spectrum, Tolman test, BAO/chronometers, LSS consistency, lensing time delays, and parameter identifiability), along with a concrete work plan and submission criteria. The present README focuses on the hypothesis, construction, and results summary.

---

## Reproducibility (commands)

All figures and metrics above were generated with a fresh Python 3.11 venv using the following commands from the repository root:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy matplotlib pandas scipy dynesty

# Hubble diagram, energy balance, and z(r) plot (saves three PNGs)
./.venv/bin/python tariff/energy_tariff_model.py --distance-max 4000 --steps 200 --preset best --plot-hubble --plot-energy-balance --data-file external_data/pantheon/Pantheon+SH0ES.dat

# CMB spectral-shape test (saves cmb_distortion_test.png)
./.venv/bin/python tariff/tariff_major_tests.py cmb --distance-mpc 14000 --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000

# BAO/chronometer proxy curves (saves bao_proxies.png)
./.venv/bin/python tariff/tariff_major_tests.py bao --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000 --rmax-mpc 6000 --zmax 2.5
```

Notes:
- Pantheon+ data path: `external_data/pantheon/Pantheon+SH0ES.dat` is included in-repo.
- PNGs are stored at repo root by default from these scripts and are tracked under Git LFS per `.gitattributes`.

