# Energy–Gravity Reciprocity and the Cosmological “Energy Tariff”

A Worked Add‑On to Gravity Gates (Optional; does not alter galaxy/lensing results)

Authors: (Project team)

Abstract
We propose and test an optional cosmological mechanism—an energy→gravity reciprocity—that operates only in low‑acceleration environments. The same gate ξ(ḡ) used in the core Gravity Gates framework regulates a tiny, cumulative energy tariff along photon paths in void‑like regions. Integrating this tariff yields a redshift–distance relation z(r) without invoking expansion kinematics. Using a minimal environmental mix and calibrating the coupling k to preserve the local slope, we demonstrate: (i) an excellent Hubble Diagram overlay against Pantheon+ (reduced χ² ≈ 0.777); (ii) a close per‑photon energy balance against 1/(1+z) (RMSE ≈ 0.0146); (iii) CMB spectral preservation under Liouville‑respecting transport (rms residuals ≲ 2×10⁻¹⁶), while a naive energy‑only mapping fails FIRAS by orders of magnitude; and (iv) reasonable BAO/chronometer proxy trends after adopting a redshift‑dependent environment mix and a monotone z↔r inversion. All galaxy‑scale dynamical and lensing results in the main repository are unchanged by this optional section.

Keywords: gravity gates; RAR/BTFR; low‑acceleration regime; energy–gravity reciprocity; supernova Hubble diagram; CMB spectral invariance; BAO proxies; Tolman test

1. Introduction

Gravity Gates posit a weak‑field response that depends on local acceleration. In high‑g regions (Solar System, inner galaxies) ξ→1 and GR is recovered; in low‑g regions ξ rises smoothly and saturates at a finite plateau Dmax. The cosmological add‑on asserts a reciprocity: when the induced gravitational stiffness is under‑supplied, a tiny energy drain from traversing radiation sustains it. This “energy tariff” integrates to a measurable redshift only along long, predominantly void sightlines, and is negligible where GR is well tested.

2. Theory

2.1 Gate used throughout

ξ(ḡ) = min[ 1/2 + sqrt(1/4 + a0/ḡ), Dmax ].

2.2 Energy tariff and accumulated redshift

(1)   d lnE / dr = − α(r),     α(r) = k [ξ(r) − 1] fenv(r)
(2)   1 + z(r) = exp( k ∫₀^r [ξ(l) − 1] fenv(l) dl )

2.3 Environmental mix (two domains)

Legacy (distance‑based):   fenv(r) = 1 / [1 + (r/r0)^γ].
Redshift‑based (recommended for BAO):   fenv(z) = 1 / [1 + (z*/z)^η], increasing with z and tending to 1 at high z.

2.4 CMB spectral transport

To respect Liouville’s theorem, Iν/ν³ is invariant. A blackbody emitted at T transforms to an observed Planck spectrum with T′ = T/(1+z) when Iobs = Iem/(1+z)³. A naive energy‑only mapping I/(1+z) spoils the spectrum and is rejected by FIRAS; we keep it only as a diagnostic toggle.

3. Data and Methods

3.1 Dataset and loader

Pantheon+SH0ES: external_data/pantheon/Pantheon+SH0ES.dat (ASCII; comment lines “#”; header row “CID …”). Columns used: zHD (index 2), MU_SH0ES (index 10), MU_SH0ES_ERR_DIAG (index 11). The loader in tariff/energy_tariff_model.py parses these by position and filters NaNs and nonpositive uncertainties.

3.2 Coupling calibration and integration

We calibrate k to preserve the small‑z slope implied by the chosen anchor H0 (unless k is provided directly). Redshift is obtained by numerically integrating (ξ−1)fenv along the line of sight with fine uniform steps. To enable fenv(z), we maintain a running redshift during integration.

3.3 Inversion and distances

We tabulate z(r) on a dense grid and invert monotonically using a piecewise cubic Hermite (PCHIP) interpolator when SciPy is available; otherwise we fall back to linear interpolation after enforcing strict monotonicity with an ε‑jitter. Distance modulus μ(z) is computed from r(z) via Euclidean conversion to parsecs (paper‑neutral proxy for comparing curves).

3.4 BAO/chronometer proxies

We define an effective H(z) ≡ c d/dz ln(1+z(r)) by differentiating the monotone inverse r(z). From H(z) we construct DM(z) and DH(z) for reference curves and, when a BAO CSV is provided, fit the sound horizon rd in a single‑parameter regression with χ²/dof reported.

3.5 Ancillary tests

Tolman test: fit p in dL = r (1+z)^p from μ(z).   SN time dilation: fit pt in t ∝ (1+z)^pt from light‑curve timescales.   Strong‑lens time delays: invariance under energy‑only tariff (group speed c; Fermat potential unchanged).

Implementation references
- tariff/energy_tariff_model.py — simulator and μ(z) prediction; Pantheon loader
- tariff/tariff_major_tests.py — batteries for CMB, Tolman, SN time dilation, BAO proxies, LOS correlation, time delays

4. Results (this run)

Preset (“best”): Dmax = 30, ḡvoid = 1×10⁻¹⁵ m s⁻², r0 = 2000 Mpc, γ = 1.5; energy coupling enabled (ζ=1.0, β=2.0); k ≈ 7.75×10⁻⁶ Mpc⁻¹.

4.1 Supernova Hubble diagram

Reduced χ² ≈ 0.777 for μ(z) versus Pantheon+ (usable sample; see loader notes). Figure: hubble_diagram_with_data.png.

4.2 Per‑photon energy balance

RMSE(Eobs,model vs 1/(1+z)) ≈ 1.46×10⁻² using μ‑derived distances r(μ). Figure: energy_balance_plot.png.

4.3 CMB spectral shape

With Liouville‑preserving transport, the observed spectrum remains Planckian with rms fractional residuals ≲ 2×10⁻¹⁶ at 14 Gpc (diagnostic: energy‑only mapping yields ≈ 1.8×10⁻¹). Figure: cmb_distortion_test.png.

4.4 Tolman surface‑brightness

Best‑fit p ≈ 0.401 with χ²/dof ≈ 1.318.

4.5 BAO/chronometer proxies

Heff(z), DM(z), DH(z) are produced from the monotone inversion; optional rd fit is available given a BAO CSV. Figure: bao_proxies.png.

5. Discussion

- SNe and energy balance: the gate‑driven z(r) reproduces the curvature of μ(z) and the per‑photon trend without altering galaxy/lensing results.
- CMB: Liouville preservation is non‑negotiable; the tariff affects z(r) but not the blackbody form, consistent with FIRAS when transport is handled correctly.
- BAO/chronometers: adopting fenv(z) and a monotone inversion improves trends versus naive finite‑difference mappings; testing against public BAO compilations is the next step.
- Distinct predictions: weak correlations of SN residuals with line‑of‑sight structure are a targeted falsification test.

6. Limitations and Caveats

- The μ(z) comparison is a static‑geometry proxy for shape tests, not a claim of full cosmological distance modelling.   
- Results depend on the environmental mix; fenv(z) captures an increasing void fraction with redshift but is an effective description.
- Time‑dilation and LOS correlation tests require external CSVs; hooks are provided.

7. Reproducibility

All figures and metrics were generated on Python 3.11 with numpy, matplotlib, pandas, scipy, dynesty. Commands below reproduce the headline plots and numbers.

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy matplotlib pandas scipy dynesty

# Hubble diagram, energy balance, and z(r) plot (saves three PNGs)
./.venv/bin/python tariff/energy_tariff_model.py --distance-max 4000 --steps 200 --preset best --plot-hubble --plot-energy-balance --data-file external_data/pantheon/Pantheon+SH0ES.dat

# CMB spectral-shape test (Liouville by default)
./.venv/bin/python tariff/tariff_major_tests.py cmb --transport liouville --distance-mpc 14000 --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000

# BAO/chronometer proxies with redshift-based f_env(z) and PCHIP inversion
./.venv/bin/python tariff/tariff_major_tests.py bao --void-mix-mode redshift --zstar 0.5 --eta 1.5 --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000 --rmax-mpc 6000 --zmax 2.5

# Tolman p
./.venv/bin/python tariff/tariff_major_tests.py tolman --k 7.75e-6 --dmax 30 --gbar-void 1e-15 --r0-void 2000 --gamma-void 1.5 --steps 4000

# Time delays (SIS toy)
./.venv/bin/python tariff/tariff_major_tests.py timedelay

# SN time dilation (requires your CSV)
# ./.venv/bin/python tariff/tariff_major_tests.py sntd --data path/to/sntd_summary.csv
```

Notes
- Pantheon+ data path: external_data/pantheon/Pantheon+SH0ES.dat (in‑repo).   
- PNGs are saved at repo root by default from these scripts and tracked by Git LFS per .gitattributes.

