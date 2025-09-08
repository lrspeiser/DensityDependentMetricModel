# Appendix: Minimal covariant completion and weak-field mapping

Purpose
- Provide an explicit covariant subclass that (i) enforces c_T = 1, (ii) yields Φ = Ψ in the quasi-static weak field, and (iii) maps the environment scalar (encoded via ξ) to both dynamics and lensing consistently.
- This replaces any lensing-only scalars in manuscript figures; dynamics and lensing share the same parameters.

Model (restricted scalar–tensor subclass)
- Consider a shift-symmetric scalar ϕ with a conformal+disformal coupling to the matter frame that is screened in high-density/high-acceleration regions.
- Action (schematic, unit c=1):

  S = ∫ d^4x √−g [ M_P^2/2 R − 1/2 (∇ϕ)^2 − V(ϕ) ] + S_m[ A^2(ϕ, I) g_{μν} + B(ϕ, I) ∇_μ ϕ ∇_ν ϕ , ψ_m ]

  where I ≡ (∇ϕ)^2 / Λ^4 is a small, environment-modulated invariant and (A, B) are chosen so that:
  - c_T = 1 on relevant backgrounds (GW170817-safe; i.e., no beyond-Horndeski operators that shift tensor speed).
  - In the quasi-static weak field with screening active, anisotropic stress vanishes → Φ = Ψ.

Weak-field potentials and lensing
- In the weak field and under screening (Solar System), the PPN parameters reduce to GR: γ = 1, β = 1, α1 = α2 = 0.
- For galaxies, define the environment potential from the gating function ξ via

  φ_env(r) ≡ 1/2 ln ξ(r)

- Then to leading order in the weak field:
  Φ = Φ_b + a_env φ_env,   Ψ = Ψ_b + b_env φ_env
  with Φ_b ≈ Ψ_b the baryonic GR potentials and (a_env, b_env) → 0 under screening. We adopt a_env = b_env = 1 for manuscript baseline, consistent with Φ = Ψ and a single-theory prediction for dynamics and lensing.

- The Weyl (lensing) potential is Φ_W ≡ (Φ + Ψ)/2 = (Φ_b + Ψ_b)/2 + φ_env. Thus lensing responds to the same φ_env that boosts galaxy dynamics.

PPN and c_T
- With the above restricted subclass and screening in the Solar System, we recover GR PPN values. The theory forbids c_T ≠ 1 in the regimes considered, enforced by a guardrail in code.

Implementation mapping (code)
- theory/relativistic.py: provides evaluate_ppn (returns γ=1, β=1, α1=α2=0 under screening), check_c_T_guardrail, and weak-field helpers.
- theory/lensing_metric.py: computes ΔΣ(R) and θ_E from baryons + phantom via ξ (through φ_env) using Φ+Ψ.
- scripts/next_steps_from_run.py: use --metric-lensing-only (or --paper-build) to route lensing to Φ+Ψ predictions exclusively (no α_lens_ph).

Caveats and roadmap
- The action above is schematic to state existence and properties. A concrete restricted Horndeski/DHOST parameterization can be substituted to derive (γ, β, α1, α2) as explicit functions of parameters; the present manuscript uses the GR-limiting values under screening and a_env = b_env = 1 in galaxies.
- Cosmology is not treated here; background expansion and linear growth constraints can be addressed with the same subclass, keeping c_T = 1.

