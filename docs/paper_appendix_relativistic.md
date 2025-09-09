# Appendix: Minimal covariant completion and weak-field mapping

Purpose
- Provide an explicit covariant subclass that (i) enforces c_T = 1, (ii) yields Φ = Ψ in the quasi-static weak field, and (iii) maps the environment scalar (encoded via ξ) to both dynamics and lensing consistently.
- Replace any lensing-only scalars in manuscript figures; dynamics and lensing share the same parameters.

Model (restricted scalar–tensor subclass)
- Consider a shift-symmetric scalar ϕ with conformal/disformal couplings in the Jordan frame, screened in high-density/high-acceleration regions, with no operators that change the tensor speed c_T.
- Schematic action (c=1):

  S = ∫ d^4x √−g [ M_P^2/2 R − 1/2 (∇ϕ)^2 − V(ϕ) ] + S_m[ A^2(ϕ, I) g_{μν} + B(ϕ, I) ∇_μ ϕ ∇_ν ϕ , ψ_m ]

  where I ≡ (∇ϕ)^2 / Λ^4 is a small, environment-modulated invariant and (A, B) are chosen so that:
  - c_T = 1 on relevant backgrounds (GW170817-safe).
  - In the quasi-static weak field with screening active, anisotropic stress vanishes → Φ = Ψ.

Non-relativistic mapping (QUMOND → ξ)
- In the non-relativistic limit, we adopt the QUMOND modified Poisson equation (see docs/modified_poisson_qumond.md):

  ∇²Φ = ∇·[ ν(|∇Φ_b|/a0) ∇Φ_b ] ,   with  ∇²Φ_b = 4πG ρ_b.

- In spherical symmetry: g = ν(y) g_N, y ≡ g_N/a0, so V² = ξ V_bar² with ξ ≡ ν.
- For μ(x) = x/(1+x) ("simple" family), ν(y) = 1/2 + √(1/4 + 1/y), giving

  ξ(g_N) = 1/2 + √(1/4 + a0_eff/g_N) .

- Environmental gates enter as a0_eff = a0 [1 + ζ_env s_ρ W(T)]. In the limit ζ_env → 0 we recover the standard MOND boost.

Phantom density and lensing (no extra scalars)
- Rewriting QUMOND as standard Poisson with an effective density yields ρ_tot = ρ_b + ρ_ph with

  ρ_ph = (1/4πG) ∇·[(ν−1) ∇Φ_b] .

- On axisymmetric grids we use the equivalent identity

  ρ_ph = (ξ−1) ρ_b − (1/4πG) (∇ξ · ∇Φ_b) ,

  which we implement numerically (see scripts/next_steps_from_run.py) to project Σ_tot and solve ⟨Σ⟩(R_E) = Σ_cr for θ_E, applying a monotone-envelope stabilization for finite grids.

Weak-field potentials and lensing
- In the weak field with screening, define φ_env(r) ≡ (1/2) ln ξ(r) and write

  Φ = Φ_b + φ_env ,   Ψ = Ψ_b + φ_env ,

  ensuring Φ_W ≡ (Φ + Ψ)/2 = Φ_b + φ_env. Thus lensing responds to the same ξ that boosts galaxy dynamics.

PPN and c_T
- With the above subclass and screening in the Solar System, we recover GR PPN values: γ = 1, β = 1, α1 = α2 = 0. The code enforces a c_T guardrail (c_T = 1) and writes a PPN table under this assumption.

Implementation mapping (code)
- theory/relativistic.py: evaluate_ppn (GR values under screening), check_c_T_guardrail, weak-field helpers including φ_env = 1/2 ln ξ.
- scripts/next_steps_from_run.py: computes ρ_ph from ξ on (R,Z) grids, projects Σ_tot, solves θ_E, and writes ΔΣ/θ_E profiles. The Solar-System table and PPN export are also produced here.

Scope and limitations
- Claims are confined to the weak-field, quasi-static regime and the screened Solar System. Cosmology and strong-field tests lie outside the present scope.
- See docs/modified_poisson_qumond.md for derivations; the relativistic subclass here is minimal and designed to connect ξ consistently to Φ+Ψ with c_T = 1.

