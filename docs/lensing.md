# Lensing in density-aware TFR: handling the conformal issue

Goal
- Provide a consistent, testable weak-field lensing prescription that does not duck the conformal invariance of null geodesics.
- Connect to your environment-dependent response ξ(ρ, T) used for dynamics, while passing Solar-System tests via screening.

Problem statement
- If we take a pure conformal rescaling g~_{\mu\nu} = ξ g_{\mu\nu} and assert that photons “see” the same ξ as matter, null geodesics are unchanged (up to affine reparametrization). The bending by a fixed baryon distribution stays at the GR prediction, while your dynamics are boosted: an immediate mass–anisotropy/lensing inconsistency.

Choice and rule (we choose B)
- We posit that photons do couple to the environment scalar, but not purely conformally. Instead, photons follow null geodesics of an effectively disformal weak-field metric whose two Newtonian potentials receive an additional environmental scalar contribution φ_env(ρ, T):

  g_{tt} = -(1 + 2 Φ_b + 2 a_env φ_env)\,,
  g_{ij} =  (1 - 2 Ψ_b - 2 b_env φ_env) δ_{ij}\,.

  where Φ_b ≈ Ψ_b are the baryonic GR potentials (no dark halo), and a_env, b_env are dimensionless response weights. In GR a_env = b_env = 0. In standard scalar–tensor with universal conformal coupling, a_env = b_env and null geodesics are conformally invariant. We explicitly allow a_env ≠ b_env (a disformal-type coupling) so that the lensing combination (Φ+Ψ) picks up φ_env while remaining screened in high-density (Solar-System) environments.

- Identification with ξ: In your dynamics, the circular velocity boost can be modeled as an effective addition to the time-time potential felt by massive particles. Define

  φ_env(r) = 1/2 ln ξ(r)\,.

  Then the matter-side dynamical boost corresponds to adding a_env φ_env to Φ_b in v_c^2 ≈ r ∂_r (Φ_b + a_env φ_env). For photons, the bending angle depends on the Weyl potential:

  Φ_W ≡ (Φ + Ψ)/2 = (Φ_b + Ψ_b)/2 + (a_env + b_env) φ_env / 2.

  Thus lensing receives an extra term proportional to (a_env + b_env) φ_env so long as a_env + b_env ≠ 0. Purely conformal corresponds to a_env = b_env, which by itself would not change light paths; here the disformal structure enters because φ_env is an inhomogeneous scalar and we work to leading order in gradients (see below), effectively sourcing additional convergence via ∇^2_⊥ φ_env.

Weak-field lensing formula
- In the thin-lens, weak-field limit (|Φ|,|Ψ| ≪ 1), the deflection angle for impact parameter R is

  α(R) = 2 ∇_⊥ ∫_{-∞}^{∞} [Φ_W(z, R)] dz.

  Under the above rule, we split

  α(R) = α_b(R) + α_env(R),
  α_b(R) = 2 ∇_⊥ ∫ (Φ_b + Ψ_b)/2 dz = 4 G M_b(<R) / (c^2 R),
  α_env(R) = (a_env + b_env) ∇_⊥ ∫ φ_env dz.

  The baryonic term is the GR deflection by the baryons only. The environmental term is new and is driven by transverse gradients of φ_env. For spherical systems, α_env can be computed efficiently by line-of-sight integration of φ_env(r), with φ_env(r) ≡ 1/2 ln ξ(ρ(r), T(r)).

Screening and Solar-System safety
- Your ξ(ρ, T) is screened at high ρ and high T. Because φ_env = 1/2 ln ξ, φ_env → 0 and ∇φ_env → 0 in Solar-System conditions, implying negligible Shapiro delay and deflection contributions. This preserves Cassini and planetary ranging constraints. In galaxies and group-scale lenses, lower ambient density and cooler environments allow φ_env gradients to be nonzero, increasing lensing in tandem with the dynamical boost.

Minimal parameterization for first-pass comparisons
- a_env controls how much of φ_env enters dynamics (time-time potential). Your SPARC fits effectively constrain this combination via v_c(r). We set a_env = 1 as a natural normalization (absorbed into ξ definition).
- b_env controls how much of φ_env enters spatial curvature (space-space potential). Lensing depends on (a_env + b_env). For a first pass we set b_env = 1 ("photon sees full φ_env"). Tighter bounds can be derived by SLACS-like lenses comparing θ_E.

Worked example structure (implemented in tools/lensing_predict.py)
- Lens mass model: spherical Hernquist with M_* and scale length a.
- φ_env model: φ_env(r) = (1/2) ln[1 + A_env (r / r0)^{-p}], a flexible, screened power-law that mimics your ξ(ρ, T) profiles outside central high-density regions. A_env, p > 0, r0 ~ few kpc. Alternative: exponential-screened variant.
- Compute α_b(R) analytically from M_b(<R). Compute α_env(R) numerically via line-of-sight integral of φ_env.
- Einstein radius θ_E solves α(D_l θ_E) = θ_E D_s / D_ls.
- Provide uncertainties by sampling (M_*, a, A_env, p) with Gaussian errors.

SLACS-style comparison plan
- For 2–3 lenses with published (z_l, z_s, θ_E, M_*±σ, R_e), choose Hernquist a ≈ R_e/1.8153.
- Fit/choose (A_env, p) using your galaxy-scale posteriors as priors (e.g., from SPARC), or set conservative ranges (e.g., log10 A_env ∈ [−2, 0], p ∈ [0.5, 2.5]).
- Report predicted θ_E under GR (baryons only) and under TFR-lensing (with α_env), compare to observed θ_E.

Notes and caveats
- This is a first-order, testable prescription. It is effectively a phenomenological disformal scalar coupling in the weak field. A full covariant action can be written to reproduce these linearized metric potentials while respecting screening; that development is beyond scope here but consistent with standard scalar–tensor effective field theory with environment-dependent functions.
- Cluster and CMB lensing would further constrain (a_env + b_env) at larger scales and temperatures; screening should suppress φ_env in hot intracluster media.

How to run (manuscript path)
- Use scripts/next_steps_from_run.py with --metric-lensing-only (no α_lens_ph). Provide docs/lensing_targets.csv with measured log10M_star and Re_kpc. See docs/paper_appendix_relativistic.md and theory/relativistic.py for the mapping used.

Internal pilot (not for manuscript figures)
- See tools/lensing_predict.py for a CLI that computes θ_E and α(R) for chosen parameters, including a worked example with uncertainties. You can plug in posterior-informed priors for A_env, p from your ER/TFR fits in SPARC.

