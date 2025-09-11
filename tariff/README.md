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

## Major Observational Checks — Issues and Results

### 1) Hubble Diagram consistency (Pantheon+SH0ES)
- Data: `external_data/pantheon/Pantheon+SH0ES.dat` (comments starting with `#` and the header row starting with `CID` are ignored). The loader uses the zHD, MU_SH0ES, and MU_SH0ES_ERR_DIAG columns.
- Method: predict μ(z) from the model by inverting z(r) (monotonic lookup) rather than using a static Euclidean approximation; overlay against Pantheon+SH0ES with uncertainties and compute χ².
- Parameters (preset “best”): D_max=30, g_bar_void=1e-15 m/s², r0_void=2000 Mpc, gamma_void=1.5; energy-coupled enabled (ζ=1.0, β=2.0). Coupling calibrated to k≈7.75×10⁻⁶ 1/Mpc.
- Result: reduced χ² ≈ 0.777 for μ(z) across the usable SN sample; figure saved as `hubble_diagram_with_data.png`.
- Note on pitfalls: when we temporarily compared μ(r) via a static Euclidean law at large z and/or deviated from the tuned preset, the fit degraded substantially (reduced χ² ≈ 54.6). Using μ(z) from the inverse z(r) mapping and the tuned preset resolves this.

### 2) Per‑photon energy balance versus μ‑derived distances
- Method: convert μ to distance r (Euclidean), set E_emit=1, compare E_obs^data=1/(1+z) to E_obs^model(r) with the same r-sampling.
- Result: RMSE(E_obs_model vs E_obs_data) ≈ 1.46×10⁻² with preset "best"; earlier exploratory tuning yielded ≈ 7.76×10⁻².
- Figure: `energy_balance_plot.png`.

### 3) CMB spectral shape (FIRAS‑like constraint)
- Setup: r=14,000 Mpc, same parameters as above. We compare two mappings of an emitted Planck spectrum I_ν: (A) expansion‑like I/(1+z)³; (B) energy‑only I/(1+z).
- Result: (A) is essentially Planckian with rms fractional residual ≈ 2×10⁻¹⁶; (B) shows large distortions with rms ≈ 1.8×10⁻¹—well above FIRAS tolerances.
- Figure: `cmb_distortion_test.png`.

### 4) Tolman surface‑brightness scaling
- Model comparison: fit p in d_L = r (1+z)^p from μ(z).
- Result from our run: p ≈ 0.401 with χ²/dof ≈ 1.318.
- Repro command shown below (tolman subcommand).

### 5) BAO and chronometer proxies
- We compute effective H(z) ≡ c d/dz ln(1+z(r)), plus D_M(z) and D_H(z), and save `bao_proxies.png`.
- Optional: if a BAO CSV is provided, we fit the sound horizon r_d and report χ²/dof (not included in the baseline run here).

### 6) Strong‑lens time delays (qualitative check)
- Under an energy‑only tariff with group speed = c and unchanged Fermat potential, differential time delays are unchanged. The demo prints the SIS toy Δφ; no figure is produced.

### 7) SN time‑dilation scaling (status)
- Harness exists to fit p_t in t_obs ∝ (1+z)^{p_t} from light‑curve summaries. We have not executed it yet in this baseline; a CSV is required.

### Data handling notes (Pantheon+)
- The loader reads zHD, MU_SH0ES, MU_SH0ES_ERR_DIAG from the `.dat` file, skipping `#` comment lines and the header row beginning with `CID`. If the upstream file format changes, update `load_pantheon_data()` in `tariff/energy_tariff_model.py` accordingly.

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

# Tolman surface-brightness exponent p (prints best-fit p and χ²/dof)
./.venv/bin/python tariff/tariff_major_tests.py tolman --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000

# Strong-lens time-delay toy check (prints Δφ; delays unchanged under energy-only tariff)
./.venv/bin/python tariff/tariff_major_tests.py timedelay

# SN time-dilation (requires your CSV with z and timescale/stretch columns)
# ./.venv/bin/python tariff/tariff_major_tests.py sntd --data path/to/sntd_summary.csv
```

Notes:
- Pantheon+ data path: `external_data/pantheon/Pantheon+SH0ES.dat` is included in-repo.
- PNGs are stored at repo root by default from these scripts and are tracked under Git LFS per `.gitattributes`.

