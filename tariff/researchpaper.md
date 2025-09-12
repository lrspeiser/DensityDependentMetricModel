Awesome—here’s a clean, drop‑in **Introduction + Background/Method** package for your cosmology add‑on, written to the tone of Paper II (or a clearly labeled Section in the SI/Appendix of Paper I). It assumes standard expansion, briefly motivates it (redshift–distance, CMB), gives a short history of “energy → gravity” ideas (including Zwicky’s tired light and why that fails), and then explains **how you apply the concept within Gravity Gates** to test whether **void‑weighted light paths** correlate with **extra redshift** via your gated loss law. I’ve kept your notation ($\xi, a_0, D_{\max}$, $f_{\rm env}$, $k$) and integrated your blackbody‑preserving derivation and the $T_{\rm LSS}\!\to\!T_0$ path‑length calibration. Citations to your two documents are included. 

I also cite a few standard cosmology references (Hubble–Lemaître law, FIRAS CMB temperature, SN time‑dilation, Tolman test, BAO) for context. ([PNAS][1])

---

# 1. Introduction (Cosmology Context)

We assume the universe is expanding. Empirically, the **redshift–distance relation**—first assembled by Hubble (building on Lemaître’s interpretation)—shows that more distant galaxies exhibit systematically larger redshifts, establishing a nearly linear relation at low redshift $v\simeq H_0 d$ and motivating an expanding background. ([PNAS][1])  Independent evidence comes from the **cosmic microwave background (CMB)**: a near‑perfect blackbody today at $T_0 = 2.7255\pm0.0006$ K (COBE/FIRAS recalibrated), with deviations $<\!10^{-4}$ across 0.5–5 mm—the textbook fossil of an early hot phase redshifting as the universe expands. ([arXiv][2])  Additional, independent expansion tests include **SN Ia time‑dilation** of light curves (stretch $\propto 1+z$), the **Tolman surface‑brightness** dimming ($\propto (1+z)^{-4}$, modulated by evolution), and the **BAO** standard‑ruler feature in galaxy clustering. ([arXiv][3])

Motivated by our **Gravity Gates** framework—where the weak‑field response depends on local acceleration—we ask a narrower question: *if* low‑acceleration environments also mediate a tiny, cumulative **energy→gravity reciprocity** for radiation, can a strictly frequency‑proportional “tariff” along **void‑weighted** sightlines (i) reproduce the **shape** of the SN Hubble diagram and (ii) remain consistent with the CMB’s blackbody spectrum, *without* altering any of our galaxy/lens results? We keep standard expansion as background and use the reciprocity as a controlled phenomenology layered on top. (The gated gravity model itself—used for galaxies, $K_z$, Solar‑System safety, and lensing—is summarized in Paper I. )

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
From the inferred $H_{\rm eff}(z)\equiv c\,d\!\ln(1+z)/dz$, compute $D_M(z)$ and $D_H(z)$ and compare **shape** to BAO compilations; fit only an overall $r_d$ if desired. ([arXiv][6])

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

[1]: https://www.pnas.org/doi/10.1073/pnas.15.3.168?utm_source=chatgpt.com "A relation between distance and radial velocity among ..."
[2]: https://arxiv.org/abs/0911.1955?utm_source=chatgpt.com "The Temperature of the Cosmic Microwave Background"
[3]: https://arxiv.org/abs/astro-ph/9602124?utm_source=chatgpt.com "[astro-ph/9602124] Observation of Cosmological Time Dilation using ..."
[4]: https://arxiv.org/abs/astro-ph/0106566?utm_source=chatgpt.com "The Tolman Surface Brightness Test for the Reality of the Expansion. IV. A Measurement of the Tolman Signal and the Luminosity Evolution of Early-Type Galaxies"
[5]: https://lambda.gsfc.nasa.gov/product/cobe/about_firas.html?utm_source=chatgpt.com "The COBE Far Infrared Absolute Spectrophotometer (FIRAS)"
[6]: https://arxiv.org/abs/astro-ph/0501171?utm_source=chatgpt.com "[astro-ph/0501171] Detection of the Baryon Acoustic Peak ..."

