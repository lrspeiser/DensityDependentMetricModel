# Modified Poisson equation and ξ mapping (QUMOND derivation)

Purpose
- Provide a compact, action-level non-relativistic derivation for the kinematic boost ξ(g_N) used in this work, connect it to a well-posed modified Poisson equation (QUMOND form), and justify the ξ→lensing mapping by constructing the effective ("phantom") density seen by Φ+Ψ.
- Clarify Solar-System behavior and PPN scope under our adopted covariant subclass.

Non-relativistic field equation (QUMOND form)
- In QUMOND (Milgrom 2010), the non-relativistic action yields a modified Poisson equation that depends on the Newtonian (baryonic) potential Φ_b:

  ∇²Φ = ∇·[ ν(|∇Φ_b|/a0) ∇Φ_b ] ,   with  ∇²Φ_b = 4πG ρ_b.

- In spherical symmetry, this reduces to

  g(r) ≡ |∇Φ| = ν(y) g_N(r) ,  with  y ≡ g_N/a0 ,

  so that the circular speed satisfies V² = ξ V_bar² with ξ ≡ ν.

- For the "simple" MOND family μ(x) = x/(1+x), the corresponding ν-function is

  ν(y) = 1/2 + sqrt(1/4 + 1/y) ,

  yielding the boost used in the manuscript:

  ξ(g_N) = 1/2 + sqrt(1/4 + a0_eff/g_N).

- Environmental modulation enters our analysis via a0_eff = a0 [1 + ζ_env s_ρ W(T)], where s_ρ gates by density (ratio to a threshold ρ_c with exponent γ) and W(T) is a tidal/kinematic window. These appear as phenomenological gates multiplying a0 in the weak-field, and reduce to the standard MOND limit when ζ_env → 0.

Phantom density and ξ→lensing mapping
- The QUMOND equation can be re-written as standard Poisson with an effective density ρ_tot = ρ_b + ρ_ph that sources the potential Φ:

  ∇²Φ = 4πG (ρ_b + ρ_ph) ,  with  ρ_ph = (1/4πG) ∇·[(ν−1) ∇Φ_b].

- In spherical symmetry, this reproduces V² = ξ V_bar² for dynamics, and when projected along the line of sight the total surface density Σ_tot yields lensing observables (θ_E, ΔΣ) that are consistent with the same ν (hence ξ). This provides a constructive mapping from the kinematic boost to lensing without introducing lensing-only scalars.

Identity used in code (axisymmetric grids)
- On a 2D axisymmetric (R, Z) grid, write ξ ≡ ν(|∇Φ_b|/a0_eff).
  Expanding ∇·[(ν−1) ∇Φ_b] gives the implementable identity

  ρ_ph = (ξ−1) ρ_b − (1/4πG) (∇ξ · ∇Φ_b) ,

  which we use numerically with finite-difference gradients:
  ∇ξ · ∇Φ_b ≈ (∂ξ/∂R) g_R^b + (∂ξ/∂Z) g_Z^b , with g_R^b = −∂Φ_b/∂R, g_Z^b = −∂Φ_b/∂Z.

- This identity is equivalent to the QUMOND phantom-density expression provided ξ depends on |∇Φ_b|/a0_eff. It is what the orchestrator computes (see scripts/next_steps_from_run.py: phantom_density_from_xi), then it projects ρ_tot = ρ_b + ρ_ph to obtain Σ_tot, solves ⟨Σ⟩(R_E) = Σ_cr, and reports θ_E. A monotone-envelope rule is applied to ⟨Σ⟩(R) to ensure a stable last-crossing for θ_E on finite grids.

Solar-System behavior and PPN scope
- In the high-acceleration (screened) regime, μ → 1 and thus ν → 1, giving ξ → 1 and ρ_ph → 0. The local metric reduces to GR, so PPN parameters equal their GR values: γ = 1, β = 1, α1 = α2 = 0. The Cassini constraint |γ−1| ≲ 2.3×10^−5 is therefore satisfied by construction in the Solar System under the adopted screened subclass.
- Our manuscript confines claims to the weak-field, quasi-static regime for galaxies and the screened Solar System. Cosmology and strong-field regimes are outside scope of this derivation.

Relativistic scaffold (Φ = Ψ, c_T = 1)
- For completeness, we adopt a minimal covariant subclass (Jordan frame, matter minimally coupled) such that the quasi-static weak field has negligible anisotropic stress (Φ = Ψ) and the tensor speed satisfies c_T = 1 on the backgrounds of interest (GW170817-safe). The environment potential that encodes the same ξ appears additively in both potentials:

  Φ = Φ_b + φ_env ,  Ψ = Ψ_b + φ_env ,  with  φ_env ≡ (1/2) ln ξ .

- Photons follow null geodesics of g_{μν}, and lensing depends on Φ+Ψ, so Φ_W = (Φ+Ψ)/2 = Φ_b + φ_env. This matches the non-relativistic QUMOND construction via ρ_ph and ensures a single-theory mapping from ξ to both dynamics and lensing.

References (for bibliography)
- Milgrom, M. (2010). QUMOND: A quasi-linear formulation of MOND. MNRAS, 403, 886–895.
- Bekenstein, J.D. (2004). Relativistic gravitation theory for the modified Newtonian dynamics paradigm. Phys. Rev. D 70, 083509 (TeVeS exemplar).
- Will, C.M. (2014). The Confrontation between General Relativity and Experiment. Living Rev. Relativ.
- Bertotti, B., Iess, L., & Tortora, P. (2003). A test of general relativity using radio links with the Cassini spacecraft. Nature 425, 374–376.

