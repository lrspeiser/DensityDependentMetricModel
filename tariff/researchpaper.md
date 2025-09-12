# A Unified Gate Linking Weak-Field Dynamics and Photon Energy Loss

Leonard Speiser, Independent Researcher

## Abstract

We propose a **unified gate** $G(y,\rho_\gamma)$ that links weak-field dynamical enhancements (RAR-like behavior) to a strictly **energy-proportional** photon energy loss (“tariff”) that accumulates along **void-weighted** lines of sight. The gate depends on the baryon-only acceleration ratio $y\equiv g_{\rm bar}/a_0$ and the ambient photon energy density $\rho_\gamma$ (CMB+EBL). By construction, $G\!\to\!1$ in high-acceleration regions (Solar-System safety) and the tariff is $\propto E$ (CMB spectral purity). In the weak field, we take

$$
 g_{\rm obs} \;=\; G\, g_{\rm bar}, \qquad \frac{d\ln E}{d\ell} \;=\; -\kappa\,[G-1].
$$

We outline a tariff-only analysis track that (i) establishes GR **baselines** from data (small-$z$ Hubble slope, CMB blackbody fit, Tolman exponent, SN time-dilation), and (ii) tests the unified gate by overlaying $\mu(z)$ on Pantheon+ with $\chi^2$, and by deriving $H_{\rm eff}(z)$ and BAO **shape-only** overlays with a fitted $r_d$. A minimal scalar–tensor wrapper preserves FRW dynamics, $\Phi=\Psi$, and $c_T=1$ in screened regimes. We enumerate **falsifiable** predictions and provide clear locations where numbers must be run.

**Keywords:** modified gravity; RAR; CMB; supernovae; BAO; weak field; conformal drift; energy–gravity reciprocity

---

## Unified Gate Law and Relativistic Scaffolding (addendum)

This section fuses our gated RAR gravity with a photon energy→gravity tariff using a single control function. It provides both a drop-in model and a relativistic wrapper so the mechanism is testable and spectrum-safe.

### A) Unified Gate Law (drop-in model)

Let g_bar(x) be the baryon-only Newtonian field, y ≡ g_bar/a0 (RAR handle), and ρ_γ(x) the local background photon energy density (CMB+EBL). Define one gate G that strengthens gravity when acceleration and ambient photon energy are both low:

G(y, ρ_γ) = 1 + η · (1 + y^p)^(-1) · (1 + (ρ_γ/ρ_⋆)^q)^(-1),  with p,q ≳ 1, η > 0.

Tie dynamics and tariff to the same gate:
- Gravity (RAR form): g_obs = ν g_bar, with ν ≡ G(y, ρ_γ). This reduces to standard RAR interpolation when ρ_γ is uniform (q=0), and automatically boosts gravity in low-ρ_γ voids.
- Photon tariff (blackbody-safe): along a photon path with affine length ℓ,
  d ln E / dℓ = −κ [G − 1]  ⇒  E(ℓ) = E0 · exp[−τ(ℓ)],  with  τ(ℓ) = κ ∫ (G − 1) dℓ.
  Calibrate κ so that τ ≃ ln(T_LSS/T0) ≈ ln(3000/2.725) ≈ 7 (CMB temperature drop). In high-g or high-ρ_γ regions, G→1 and the tariff vanishes.
- Optional back-reaction (gentle self-reinforcement): accumulate a gate potential ψ via dψ/dℓ = γ [G − 1] and let G → G · f(ψ) with f(ψ)=exp(σψ) or f(ψ)=1+σψ (σ small), keeping galaxy fits intact while allowing slight cosmology-level drift.

Why this works: one gate G controls both weak-field dynamics and a uniform (E-proportional) cooling along light paths, preserving blackbody shape and passing Solar-System constraints (G→1 as y→∞).

### B) Relativistic and quantum-compatible scaffolding (EEP-safe framing)

To keep conservation laws and the equivalence principle intact, we use a conservative conformal-drift framing. All species share one physical metric g_{μν} (EEP-safe); the tariff acts as an effective conformal drift of the photon distribution function with a slowly varying factor A(φ,χ) such that d ln A/dℓ = 1/2 κ [G(χ)-1] along null rays. This preserves c_T=1 in screened regimes and avoids species-dependent violations.

Essentials in the weak field:
- Modified Poisson: ∇·[ G(χ) ∇Φ ] = 4πG ρ_b, with G(χ) = 1 + η (1+y^p)^(−1) (1+(ρ_γ/ρ_⋆)^q)^(−1) f(φ).
- Photon energy and period drift (time-dilation safety): d ln E / dℓ = − d ln A / dℓ ≡ −κ [G − 1] and d ln(Δt) / dℓ = + d ln A / dℓ ≡ +κ [G − 1]. Energy and arrival periods rescale inversely with the same A, preserving SN time dilation and Tolman when expressed in terms of the observed 1+z.
- Total energy conservation (phenomenological): ∇_μ T^{μν}_{(γ)} = −Q^ν, ∇_μ T^{μν}_{(φ)} = +Q^ν with a null-aligned exchange Q^ν ∝ κ [G − 1] T^{νλ}_{(γ)} k_λ (k^μ: photon wavevector). We avoid over-specifying frame-dependent couplings.
- Mode-level statement: d ln ω̂_k / dℓ = −κ [G(χ) − 1] — a dilaton-like conformal drift (no scattering), hence no spectral distortion beyond a uniform rescaling (FIRAS-safe for smooth κ[G−1]).

### C) How to calculate this in our pipeline (practical plan)

We separate (i) baseline expectations under GR from current data (quantum/CMB/expansion tests), and (ii) unified gate tests.

Baselines (what GR must match):
- Hubble diagram (small-z linear slope, μ_lin): compare Pantheon+ vs μ_lin(z; H0=67.4,73.0); produce baseline_hubble.png.
- CMB spectral purity: fit Planck T′ to a FIRAS-like spectrum; report T′ and rms fractional residuals; baseline_cmb_spectrum.png.
- Tolman surface brightness: fit S ∝ (1+z)−p; report p and baseline_tolman.png.
- SN time dilation: fit t ∝ (1+z)pt; report pt and baseline_sntd.png.

Unified gate + tariff tests:
- μ(z) overlay and χ²: build z(r), invert to r(z), compute μ(z), overlay vs Pantheon+, report χ²/dof and write unified_gate_hubble_overlay.png.
- H_eff(z) and BAO shape-only overlays (FRW + small overlay recommended): map 1+z_obs = (1+z_FRW) e^{τ(z_FRW)}, r(z_obs) = ∫ c/H_FRW dz_FRW, then H_eff(z_obs) = H_FRW(z_FRW) e^{τ} [1 + (1+z_FRW) τ'(z_FRW)]. Report D_M(z)=r(z), D_H(z)=c/H_eff(z); fit r_d to BAO compilations; write unified_gate_bao_proxies.png and metrics to JSON.

1) Gate evaluation
- Compute y(R) = g_bar/a0 from the baryon-only Newtonian field already available in our runners (v_baryon^2/R → g_bar). We will reuse T ≡ v_baryon^2/R^2 if convenient and convert to g_bar as needed.
- Set ρ_γ to a spatially uniform baseline (ρ_CMB today), then allow simple perturbations for EBL or void-modulation if desired. Start with ρ_γ = ρ_⋆ = 0.26 eV/cm^3 (today’s CMB) so the energy factor is unity, turning on q>0 later to probe sensitivity.
- Evaluate G(y, ρ_γ) = 1 + η (1+y^p)^(−1) (1+(ρ_γ/ρ_⋆)^q)^(−1) [× f(ψ) if enabled). For the main analysis set q=0 (drop ρ_γ handle) and control environment via y and an explicit f_env(z, LOS).

2) Dynamics (galaxy fits)
- Replace the enhancement factor in our acceleration-space RAR bridge with ν ≡ G(y, ρ_γ) when testing gate variants; keep published xi for reproducibility unless explicitly flagged as experimental.
- Diagnostics: re-check BTFR slope and Solar-System screening (G→1 for y≫1), and ensure lensing mapping Φ+Ψ uses φ_env = 1/2 ln ξ consistently when interpreting G as an effective ξ.

3) Tariff integration (cosmology add-on)
- Along a path parameterized by comoving distance r (Mpc), integrate τ(r) = κ ∫_0^r [G(y(l), ρ_γ(l)) − 1] dl using the same LOS logic as energy_tariff_model.py. In FRW + overlay, map 1+z_obs=(1+z_FRW) e^{τ(z_FRW)} and r(z_obs)=∫ c/H_FRW dz_FRW before forming H_eff and distances.
- Calibration: anchor the small‑z slope (H0 or M_B) first; treat κ as a small overlay parameter such that |τ(z≲2)| ≪ 1. The “τ≈ln(1100)” route is an appendix scenario (speculative), not the mainline hypothesis.
- Build a monotone z(r) table: 1+z = exp[τ(r)], invert via PCHIP to obtain r(z) and μ(z) for Pantheon+ comparisons.

4) Optional back-reaction ψ
- Integrate dψ/dℓ = γ [G−1] with small σ in f(ψ) to keep galaxy-scale effects negligible while allowing late-time void enhancement. Log and bound ψ to avoid runaways; turn off by default.

5) Tests (pass/fail dials)
- Blackbody purity: tariff ∝ E ensures no frequency-dependent distortions; check residuals vs Planck fit (FIRAS tolerances ~few×10^−5).
- Time-dilation and Tolman: verify stretch ∝ (1+z) and surface brightness S ∝ (1+z)^−4 remain intact in the add-on framing.
- BAO/chronometer proxies: in FRW + overlay, use H_eff(z_obs) = H_FRW(z_FRW) e^{τ} [1+(1+z_FRW) τ'(z_FRW)] and compare D_M(z)=r(z), D_H(z)=c/H_eff(z) shapes to BAO; fit only an overall r_d if desired. For tariff-only comparisons, also show ratios to FRW to highlight shape impact.
- Local tests: ensure G→1 in high-g settings (Solar System, lab) and that our PPN table remains GR under screening.

### D) Implementation mapping (tariff-only code scaffold)

- New module tariff/unified_gate_scaffold.py provides:
  - GateParams(η, p, q, ρ_⋆, κ, σ, enable_backreaction): parameter dataclass.
  - gate_G(y, rho_gamma, params, psi=None) → G.
  - tariff_dlnE_dell(y, rho_gamma, params, psi=None) = −κ [G−1].
  - integrate_tau(path_sampler, params) → τ and (optionally) ψ; path_sampler supplies (y, ρ_γ, dl).
  - calibrate_kappa_to_cmb(f_void, D_LSS, G_cap) → κ to reach τ≈ln(1100).

- Baselines and unified-gate analysis (plots + metrics):
  - tariff/analysis_baselines.py: GR baselines — Hubble μ_lin(z), CMB Planck fit (FIRAS-like), Tolman p, SN p_t; writes images/baseline_*.png
  - tariff/analysis_unified_gate.py: Unified gate μ(z) overlay vs Pantheon+ with χ², H_eff(z) and BAO proxy overlays with r_d fit; writes
    images/unified_gate_hubble_overlay.png, images/unified_gate_bao_proxies.png and results/unified_gate_metrics.json.

All of this remains confined to tariff/ and is wired for our existing LOS machinery.

### E) Methodological checks (code-verified invariants and metrics)

- H_eff identity (BAO proxy): with z = z(r) or the FRW+overlay mapping, we enforce H_eff(z) ≡ c·dz/dr and its equivalence to c·(1+z)·d ln(1+z)/dr in the tariff-only limit. We record the RMS relative difference as heff_identity_rms in results/unified_gate_metrics.json; numerical targets are ≲ 1e−6 after tabulation smoothing.
- Distance duality and luminosity distance: in the flat case we use D_L(z) = (1+z)·D_M(z) with D_M(z) = r(z). We compute μ(z) from D_L and also recompute via the duality relation on the Pantheon grid; we store distance_duality_ok and distance_duality_mu_rms.
- BAO shape-only fit: after constructing H_eff(z) = c·dz/dr, we take D_M(z) = r(z) and D_H(z) = c/H_eff(z) and fit a single sound-horizon r_d to BAO compilations; metrics include rd_best_Mpc, bao_chi2, bao_red_chi2, and bao_dof.
- Small‑z slope anchor (optional): verify dμ/dz|_{z→0} = (5/ln 10)·c/H0 when the local H0 anchor is applied.
- Blackbody purity (optional): the tariff ∝ E preserves a blackbody; we fit a Planck curve to a FIRAS-like spectrum and record rms_frac_resid.
- Data dependencies: optional baselines require CSVs under tariff/data (see README_DATA.md).
- Reproducibility: plots are written under tariff/images/ and metrics to tariff/results/unified_gate_metrics.json.

---

# Figures and Tables (auto-generated from tariff/)

- Hubble baseline (GR): ![](images/baseline_hubble.png)
- CMB spectral shape baseline:
  - If present: ![](images/baseline_cmb_spectrum.png)
  - If missing: data file tariff/data/cmb_firas_like.csv not found (figure not generated)
- Tolman test baseline:
  - If present: ![](images/baseline_tolman.png)
  - If missing: data file tariff/data/tolman_sb.csv not found (figure not generated)
- SN time-dilation baseline:
  - If present: ![](images/baseline_sntd.png)
  - If missing: data file tariff/data/sn_timedilation.csv not found (figure not generated)
- Unified gate μ(z) overlay: ![](images/unified_gate_hubble_overlay.png)
- Unified gate H_eff(z) and BAO proxies: ![](images/unified_gate_bao_proxies.png)

Metrics (JSON): results/unified_gate_metrics.json (χ², reduced χ², fitted r_d if BAO is provided)

---

## 1. Introduction

The **radial acceleration relation (RAR)** ties the observed centripetal acceleration $g_{\rm obs}$ to the baryon-predicted $g_{\rm bar}$ with remarkably small scatter, challenging halo-tuning in ΛCDM and motivating weak-field modifications. Separately, the **CMB** is a near-perfect blackbody at $T_0\simeq2.725$ K, interpreted as relic radiation stretched by expansion from $T_{\rm LSS}\!\sim\!3000$ K.

We explore whether a **single gate** $G$ can (i) reproduce weak-field dynamical phenomenology already captured by RAR-like laws and (ii) supply a tiny, **spectrum-safe** photon tariff that accumulates primarily in **voids**. We **retain** the expanding FRW background; the tariff is a **phenomenological overlay** controlled by the same gate that governs dynamics. This paper defines the model, its relativistic scaffolding, the observational program, and explicit pass/fail dials.

> **Scope**: We do **not** claim a replacement for FRW/CMB/BAO nor cluster-scale success without extra mass; we present a tightly-constrained reciprocity that is independently falsifiable.

---

# 2. Background: “Energy Converts to Gravity” (what we mean—and don’t)

**Energy gravitates.** In GR, the source of gravity is the stress‑energy tensor; energy density and pressure both curve spacetime. In that sense, “energy → gravity” is embedded in the field equations, and redshift in FRW follows from geometry.

**Historic photon‑loss proposals.** Zwicky’s 1929 **tired‑light** idea posited that photons *lose* energy en route, producing a redshift–distance relation in a static spacetime. Such mechanisms are **ruled out** by multiple observations (image blurring from scattering, failure to reproduce SN time‑dilation and Tolman dimming, and CMB blackbody/anisotropy constraints). Our construction explicitly avoids those pathologies by (i) assuming expansion and (ii) imposing a **strictly proportional** loss law that preserves blackbody shape and is **gated** to operate only in **low‑acceleration** regions, leaving dense regions and the Solar System essentially untouched. ([arXiv][4])

**Our framing.** Within **Gravity Gates**, the same gate $\xi(g)$ that enhances weak‑field gravitational response (Paper I) also controls a tiny, *environment‑weighted* photon energy loss rate. This *reciprocity* provides a single control variable for both dynamics and a phenomenological redshift contribution. We emphasize: the **galaxy, $K_z$**, Solar‑System, and **lensing** results from Gravity Gates stand **independently** of this cosmology add‑on.

---

# 3. Gated energy‑to‑gravity reciprocity: model summary

We adopt the Gravity Gates gate $\xi(g_{\rm bar};a_0,D_{\max})$ from Paper I and use it to **modulate** a line‑of‑sight energy‑loss coefficient $\alpha(l)$. The loss is **proportional to photon energy** (no frequency dependence), so an initial energy $E_0$ evolves as

$$
\frac{dE}{dl}=-\alpha(l)\,E \quad\Rightarrow\quad E(l)=E_0\,e^{-\tau(l)},\quad \tau(l)=\int_0^l \alpha(l')\,dl'.
$$

Because $E\propto \nu$ for photons, this implies a **pure redshift** contribution without spectral distortion:

$$
1+z(l)\;=\;e^{\tau(l)}.
$$

We **gate** $\alpha$ by the low‑acceleration response and a simple environment weight $f_{\rm env}$:

$$
\boxed{\ \n\alpha(l)=k\,[\xi(l)-1]\;f_{\rm env}(l)\ },
\qquad 
1+z(r)=\exp\!\left(k\int_0^r [\xi(l)-1]f_{\rm env}(l)\,dl\right).
$$

Here $k$ is a small coupling to be fitted (or anchored locally), and $f_{\rm env}\in[0,1]$ is a void weight that increases when the photon path threads underdense, low‑$g$ regions. Two useful parameterizations are:
**distance‑based:** $f_{\rm env}(r)=\bigl[1+(r/r_0)^\gamma\bigr]^{-1}$;
**redshift‑based:** $f_{\rm env}(z)=\bigl[1+(z_\star/z)^\eta\bigr]^{-1}$ (rises toward unity at high $z$).
Implementation details, data hooks, and prior runs are given in your add‑on note.

> **Link to Gravity Gates.** $\xi(g)$ is **large** (gate “open”) only when $g\ll a_0$, i.e., in **void‑like** environments; it is $\simeq 1$ (gate “closed”) in dense regions and in the Solar System. This naturally predicts stronger line‑of‑sight tariff in void‑weighted sightlines, and negligible effects in high‑$g$ locales—exactly the phenomenology you test.

---

## Box A — Preserving a blackbody (why the loss must be $\propto E$)

To avoid FIRAS‑level CMB spectral distortions, the loss law must be **linear in $E$** (no extra frequency dependence). Then a blackbody emitted at temperature $T_{\rm em}$ remains a blackbody with

$$
T_{\rm obs}=\frac{T_{\rm em}}{1+z} = T_{\rm em}\,e^{-\tau},
\quad \text{with } \tau=\!\int \alpha\,dl,
$$

and phase‑space density $I_\nu/\nu^3$ (Liouville) remains invariant under a pure rescaling of $\nu$. This is the minimal, CMB‑safe loss law you sketched; we adopt it throughout. (FIRAS: $T_0=2.7255$ K with rms deviations $\lesssim 10^{-4}$.) ([arXiv][2])

---

## Box B — Calibrating the CMB temperature drop ($\sim\!3000$ K $\to$ 2.725 K)

Taking $T_{\rm LSS}\!\approx\!3000$ K at last scattering and $T_0=2.725$ K today,

$$
\tau_{\rm CMB}\;=\;\ln\!\frac{T_{\rm LSS}}{T_0}\;\approx\;\ln(1100)\;\approx\;7.0.
$$

If the tariff acts only along a fraction $f_{\rm void}$ of the comoving line‑of‑sight distance to last scattering $D_{\rm LSS}\simeq 14~\mathrm{Gpc}$, then

$$
\alpha_{\rm eff} \;\equiv\; \frac{\tau_{\rm CMB}}{f_{\rm void}\,D_{\rm LSS}}
\;\approx\; \frac{7}{0.8\times 14\ \mathrm{Gpc}}
\;\approx\;0.63~\mathrm{Gpc^{-1}} \;\;(0.19~\mathrm{Gly^{-1}}).
$$

In our gated model, $\alpha(l)=k[\xi(l)-1]f_{\rm env}(l)$. Thus **one parameter** $k$ is set by requiring $\int\alpha\,dl=\tau_{\rm CMB}$ (or, in practice, by matching the local Hubble‑diagram slope and then verifying the CMB constraint). This is the clean calibration route you proposed.

---

# 4. Observational program and falsifiable signatures

**(A) Supernova Hubble diagram (shape test).**
With a monotone inversion $r(z)$ from $1+z=\exp\!\int \alpha dl$, compute $\mu(z)$ and compare to Pantheon+; fit only $k$ (and, if used, $f_{\rm env}$ hyper‑parameters) after anchoring the small‑$z$ slope. Your implementation already reports reduced $\chi^2$ and provides a PCHIP inversion path.

**(B) Line‑of‑sight environment correlation (distinctive prediction).**
Define the **gated path integral**

$$
S \;\equiv\; \int_0^{r(z)} [\xi(l)-1]f_{\rm env}(l)\,dl.
$$

Then test whether SN residuals $\Delta\mu$ correlate with $S$ (or with a void‑fraction proxy along the line of sight). A **positive slope** would indicate extra tariff where the gate opens (void‑weighted paths), whereas **no correlation** falsifies the reciprocity at the reported sensitivity. (Your harness has LOS correlation hooks.)

**(C) CMB blackbody/anisotropy safety (hard constraint).**
Because the loss is $\propto E$ and nearly isotropic on large scales, the **shape** of the blackbody is preserved (FIRAS), and small‑scale anisotropies are not blurred by scattering. You already enforce Liouville‑respecting transport in code and flag that a naive intensity‑only mapping fails by orders of magnitude.   ([LAMBDA][5])

**(D) Time‑dilation and Tolman checks.**
Your pipeline includes tests for **SN stretch $\propto 1+z$** and **Tolman $(1+z)^{-4}$** behavior; those must match expansion expectations (the tariff alone cannot produce the correct scalings). ([arXiv][3])

**(E) BAO/chronometer proxies (shape only).**
Use the FRW+overlay mapping: $1+z_{\rm obs}=(1+z_{\rm FRW})\,e^{\tau(z_{\rm FRW})}$, $r(z_{\rm obs})=\int c/H_{\rm FRW}\,dz_{\rm FRW}}$, and $H_{\rm eff}(z_{\rm obs}) = H_{\rm FRW}(z_{\rm FRW}) e^{\tau}\,[1+(1+z_{\rm FRW})\,\tau'(z_{\rm FRW})]$. Then compare $D_M(z)=r(z)$ and $D_H(z)=c/H_{\rm eff}(z)$ shapes to BAO; fit only an overall $r_d$ if desired. ([arXiv][6])

---

# 5. Relation to Gravity Gates (Paper I)

* **Single control function.** The same $\xi(g)$ that fits **galaxy rotation curves**, **MW $K_z$**, **Solar‑System safety**, and **strong‑lensing** amplitudes (metric‑only mapping with $\Phi=\Psi$) also gates the cosmological tariff. No per‑object tweaking is introduced here.
* **Screening.** In high‑$g$ regimes (Solar System, galaxy interiors), $\xi\!\to\!1$ and $\alpha\!\to\!0$, automatically keeping local tests intact.
* **Scope.** This section **does not** claim a replacement for FRW/CMB/BAO; it asks whether a *small, gated* reciprocity can (i) follow the SN‑Hubble shape and (ii) remain CMB‑, Tolman‑, and time‑dilation‑safe, thereby offering a *phenomenological* link between cosmic web environment and residual redshift trends. (Full cosmology is deferred to a companion study.)

---

## Methods snapshot (for the cosmology add‑on)

1. **Gate & environment.** $\xi(g)=\tfrac12+\sqrt{\tfrac14+a_0/g}$ (capped by $D_{\max}$) from Paper I; environment weights $f_{\rm env}(r)$ or $f_{\rm env}(z)$.
2. **Loss law.** $\alpha(l)=k[\xi(l)-1]f_{\rm env}(l)$; integrate to get $1+z=\exp\!\int\alpha\,dl$.
3. **Calibration.** Fit $k$ to preserve the **local Hubble slope** (small $z$) and verify $\int\alpha\,dl\approx \ln(1100)$ to satisfy the **CMB temperature drop** if treating the tariff as cosmologically relevant over the full path.
4. **Inversion.** Build a monotone $z(r)$ grid; invert for $r(z)$ via PCHIP; compute $\mu(z)$.
5. **Tests.** SN Hubble diagram (shape), LOS‑environment correlation, CMB blackbody check (Liouville), Tolman $p$, SN time‑dilation $p_t$, and BAO/chronometer proxies—exactly as your codebase implements.

---

## Related Work: Skordis & Złośnik (AeST) and alternatives

Our proposal differs from relativistic MOND frameworks such as the Aether–Scalar–Tensor (AeST) theory of Skordis & Złośnik. AeST augments GR with a unit‑timelike vector and a scalar while keeping a single metric and Φ=Ψ; it was designed to satisfy c_T=c and to match Planck‑era observables on linear scales, with the scalar behaving effectively as dust so that CMB and matter power spectra can be fitted without particle CDM. [SZ‑PRL][SZ-PRL], [SZ‑PRD][SZ-PRD], [SZ‑arXiv][SZ-ARXIV]

In contrast, our unified gate uses the same weak‑field control function that boosts low‑g dynamics to modulate a tiny, energy‑proportional photon‑energy loss that accumulates primarily in voids; dense, high‑g regions are screened (G→1), preserving Solar‑System tests. This mechanism adds path‑dependent redshift on top of FRW expansion while preserving the CMB blackbody (no frequency dependence). A distinctive prediction is a correlation between SN residuals and a void‑weighted line‑of‑sight gate integral, which AeST (lacking any tariff) does not predict.

Broader landscape (one‑sentence context):
- Superfluid dark matter (Berezhiani & Khoury): a DM component whose phonons mediate MOND‑like forces in galaxies; cosmology remains essentially ΛCDM‑like. [SFDM][SFDM-PRD]
- Dipolar dark matter (Blanchet & Le Tiec): a polarizable DM fluid reproducing MOND phenomenology while keeping standard cosmology. [DDM][DDM-PRD]
- MOG/STVG (Moffat): modified‑gravity field content for galaxy/cluster dynamics with proposed CMB treatments; distinct predictions from both AeST and our reciprocity. [MOG][MOG-MDPI]

---

### Notes on positioning and prior art

* Our gated reciprocity is **not** the Zwicky‑style tired‑light scattering: it is expansion‑compatible, **frequency‑proportional**, near‑isotropic, and **screened** in high‑$g$ regions; it therefore avoids blurring and preserves the blackbody form by construction. ([arXiv][4])
* The novelty is the **link to a weak‑field gravity gate** already audited on galaxies/lensing, yielding *one function* $\xi(g)$ that controls both dynamics and a small, testable, environment‑dependent redshift contribution.

---

## (Optional) One‑paragraph “Scope and limitations” for the end of the Intro

**Scope and limitations.** We retain the expanding FRW background and test a **small, gated** energy‑to‑gravity reciprocity layered on top. We do **not** attempt full‑fidelity fits to CMB/BAO or a global replacement of FRW; cluster‑scale dynamics are also out of scope. The tariff is constrained to be $\propto E$ (blackbody‑safe), screened in high‑$g$ regimes, and weak enough to satisfy SN time‑dilation and Tolman tests; violations would falsify the mechanism.

---

### Where to paste

* If you’re preparing **Paper II**, keep Sections 1–5 as your Introduction & Model.
* If you’re appending to **Paper I**, insert **Section “Outlook / Cosmology Add‑On”** with a shortened version of Sections 1–3, and point readers to a companion manuscript for the full program. (The galaxy/lens results remain unchanged.)

[1]: https://www.pnas.org/doi/10.1073/pnas.15.3.168 "A relation between distance and radial velocity among ..."
[2]: https://arxiv.org/abs/0911.1955 "The Temperature of the Cosmic Microwave Background"
[3]: https://arxiv.org/abs/astro-ph/9602124 "[astro-ph/9602124] Observation of Cosmological Time Dilation using ..."
[4]: https://arxiv.org/abs/astro-ph/0106566 "The Tolman Surface Brightness Test for the Reality of the Expansion. IV. A Measurement of the Tolman Signal and the Luminosity Evolution of Early-Type Galaxies"
[5]: https://lambda.gsfc.nasa.gov/product/cobe/about_firas.html "The COBE Far Infrared Absolute Spectrophotometer (FIRAS)"
[6]: https://arxiv.org/abs/astro-ph/0501171 "[astro-ph/0501171] Detection of the Baryon Acoustic Peak ..."

[SZ-PRL]: https://link.aps.org/doi/10.1103/PhysRevLett.127.161302
[SZ-PRD]: https://link.aps.org/doi/10.1103/PhysRevD.100.104013
[SZ-ARXIV]: https://arxiv.org/abs/2007.00082
[SFDM-PRD]: https://link.aps.org/doi/10.1103/PhysRevD.92.103510
[DDM-PRD]: https://link.aps.org/doi/10.1103/PhysRevD.80.023524
[MOG-MDPI]: https://www.mdpi.com/2075-4434/1/1/65

