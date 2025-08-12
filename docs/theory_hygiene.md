# Theory hygiene for Tidal Field Relativity (TFR)

Goal
- Provide a compact, reviewer-facing “phenomenology-only” theory note (≈2 pages) that addresses:
  - Bianchi identities and stress–energy conservation
  - Absence of ghosts/gradient instabilities in the weak field
  - Gravitational-wave speed c_T = 1 (GW170817 compliance)
  - Which equivalence principles are satisfied/violated
  - How ξ(ρ,T) is sourced (local functionals of baryons), and any composition dependence

Scope and strategy
- We present two equivalent, weak-field–focused ways to formalize TFR:
  1) Minimal effective-field rules (EFT rules) that guarantee the required properties without committing to a UV-complete Lagrangian.
  2) A concrete “minimal scalar + disformal-Jordan metric” action at leading order whose linearized limit reproduces our phenomenology and respects c_T = 1.
- Both are strictly weak-field and quasi-static (galaxy) limits; cosmology is not treated here beyond the statement of c_T = 1 and vanishing tensor modifications.

Notation
- Background metric g_{μν} with signature (−,+,+,+); G = Newton’s constant; c = 1.
- Baryonic stress–energy T^{(b)}_{μν} (stars/gas/radiation as appropriate).
- A screened scalar field φ_env encodes environmental response; we identify φ_env ≈ ½ ln ξ at the level of weak-field potentials.

Part A — Effective-field rules (weak field, quasi-static)
1) Diffeomorphism invariance and conservation
- All fields transform covariantly under diffeomorphisms. Field equations satisfy ∇_μ G^{μν} ≡ 0, hence ∇_μ (T^{(tot)})^{μν} = 0 automatically.
- Matter follows geodesics of an effective Jordan metric \tilde{g}_{μν} (defined below). Individual sector stress–energies can exchange via φ_env, but the total is covariantly conserved.

2) Tensor sector kept GR-like with c_T = 1
- Postulate: Quadratic action for tensor perturbations h_{ij} is identical to GR at the scales of interest:
  S_T^(2) = (M_Pl^2/8) ∫ d^4x a^3 [ (∂_t h_{ij})^2 − (∇ h_{ij})^2 ].
- No operators that modify tensor speed or introduce mass/instabilities are present to leading order (consistent with GW170817).

3) Scalar sector: stable, screened
- Scalar kinetic term has positive coefficient; small-sound-speed gradient term positive: Z_φ > 0, c_s^2 > 0.
- Screening rule: In high-density/high-T environments, the scalar response vanishes (φ_env → 0 and ∇φ_env → 0) sufficiently fast to satisfy Solar-System constraints.
- Allowed operators are restricted to those that (i) do not change c_T; (ii) do not introduce Ostrogradski ghosts (no higher-than-second-order equations without DHOST degeneracy conditions). In practice, we keep to canonical kinetic + potential + weak conformal/disformal matter couplings.

4) Effective metric seen by matter and photons (Jordan frame)
- Matter fields couple minimally to \tilde{g}_{μν}:
  \tilde{g}_{μν} = A^2(φ_env) g_{μν} + B(φ_env) ∂_μ φ_env ∂_ν φ_env.
- Weak-field potentials:
  g_tt = −(1 + 2Φ_b), g_ij = (1 − 2Ψ_b) δ_ij,
  \Rightarrow Φ_eff = Φ_b + a_env φ_env, Ψ_eff = Ψ_b + b_env φ_env, with a_env ≡ d ln A/dφ|_0 and b_env from the B-term at leading order.
- Dynamics: v_c^2 ≈ r ∂_r Φ_eff; lensing: α ∝ ∇_⊥ ∫ (Φ_eff + Ψ_eff) dz. This reproduces our phenomenology with α_env ∝ (a_env+b_env) ∇_⊥ ∫ φ_env dz.
- Solar-System safety follows from screening (φ_env ≈ 0 locally), suppressing PPN deviations (|γ−1|, Shapiro, time-delay) to within bounds.

5) Equivalence principles
- Weak Equivalence Principle (WEP): satisfied if all matter species couple universally to \tilde{g}_{μν} (no composition-dependent A or B). We impose universal couplings: A(φ_env) and B(φ_env) are the same for all standard-model sectors.
- Einstein Equivalence Principle (EEP): local Lorentz invariance and position invariance hold in the local freely falling frame of \tilde{g}_{μν}; non-minimal scalar couplings can induce tiny EEP violations, but our screened regime ensures they are below current bounds.
- Strong Equivalence Principle (SEP): violated, as in most scalar–tensor or modified-gravity theories; gravitational binding energy can source φ_env indirectly via ρ and tidal T, leading to effective SEP violations at the phenomenological level. This is consistent with galaxy-scale phenomenology and constrained in the Solar System by screening.

6) Sourcing of ξ(ρ,T)
- ξ(ρ,T) is a local functional of baryons via φ_env[ρ, T]: we adopt the mapping
  φ_env(x) ≈ ½ ln[1 + λ_max S_ρ(ρ(x)) W(T(x))],
  where ρ is the (suitably averaged) baryonic density and T is a scalar of the tidal tensor of the baryonic potential.
- In the EFT picture, this corresponds to φ_env solving a screened Poisson-type equation whose quasi-static solution is well approximated by the above local functional in disks. The functional uses only baryonic macroscopic fields (density and tidal invariant), not composition labels.

7) Composition dependence
- Because A(φ_env) and B(φ_env) are universal, test-body motion depends only on \tilde{g}_{μν} and is composition independent: no new composition-dependent forces at leading order.
- Sourcing via ρ and T could in principle inherit small composition dependence through equation-of-state differences, but in normal stellar/gas contexts these are negligible at the level of lensing/rotation data. Lab bounds are satisfied by screening.

Part B — Minimal weak-field action (one explicit construction)
We give a simple action that reproduces the above rules to leading order while preserving c_T = 1 and stability in the weak field:

S = ∫ d^4x √−g [ (M_Pl^2/2) R − ½ (∂φ)^2 − V(φ) ]
  + S_m[Ψ_m, \tilde{g}_{μν}],

with the Jordan metric
\tilde{g}_{μν} = e^{2 a_env φ} g_{μν} + b_env^2 Λ^{-4} ∂_μ φ ∂_ν φ,

where Λ is a high scale (≫ galaxy potentials in energy units) controlling the disformal strength. Conditions:
- Tensor speed: No Horndeski/Galileon X≡(∂φ)^2-dependent operators that modify c_T; thus c_T = 1.
- Stability: canonical kinetic term; choose V(φ) such that m_φ is large in high-density/high-T environments (chameleon/symmetron-like screening) or, effectively, such that the sourced solution yields φ ≈ 0 in Solar-System conditions. Small-field fluctuations have positive kinetic and gradient terms.
- Screening: Local effective mass m_eff(ρ,T) rises with ρ and/or T, forcing φ → 0 and suppressing ∇φ in dense/hot regions; on galactic outskirts, m_eff is small enough to allow a smooth profile matching φ_env ≈ ½ ln ξ.

Weak-field potentials and phenomenology
- Expand to first order in φ and weak gravity:
  Φ_eff = Φ_b + a_env φ, Ψ_eff = Ψ_b + b_env φ,
  reproducing v_c^2 and lensing α with the same (a_env+b_env) combination as in our predictor.
- Identify φ_env with the quasi-static solution of the scalar field equation in the disk geometry; in practice, our fits use the bounded local functional φ_env ≈ ½ ln ξ(ρ,T), which is the leading-order solution in a screened regime with slowly varying ρ and T.

Bianchi identities and conservation
- Diffeomorphism invariance of S ensures ∇_μ G^{μν}=0 and thus ∇_μ (T^{(φ)}_{μν} + T^{(m)}_{μν})=0. Energy–momentum can exchange between φ and matter via \tilde{g}_{μν}, but the total is conserved.

Parameter choices and constraints
- Set a_env ≈ 1 by convention (absorbed into φ normalization). Fit b_env from lensing; Solar-System screening requires |a_env φ|, |b_env φ| ≪ 10^{−5} locally.
- Choose Λ large so that disformal effects do not alter c_T or induce superluminality/causality issues in known regimes; practically, with φ gradients in galaxies, the B-term contributes only at the level needed for lensing while remaining screened in the Solar System.

PPN and GW constraints (qualitative summary)
- PPN γ−1 ∝ (b_env − a_env) φ in the weak field; with φ ≈ 0 locally by screening, Cassini is satisfied.
- GW170817 (c_T ≈ 1): ensured by zero tensor-sector modifications and negligible disformal backreaction for astrophysical sources; tensors propagate on g_{μν} with unit speed.

What is (and is not) claimed
- We do not present a full covariant completion beyond the above minimal scalar + disformal-Jordan structure. The weak-field EFT rules suffice for galaxy phenomenology and lensing tests while obeying the key consistency and observational constraints.
- Cosmology and non-linear structure formation are out of scope here; in particular, any potential background evolution of φ is assumed negligible for the galaxies studied.

Summary checklist
- Bianchi/conservation: guaranteed by diffeomorphism invariance; total ∇·T = 0.
- Ghost/gradient stability: canonical scalar; positive kinetic and c_s^2 in the weak field; no higher-derivative pathologies invoked.
- GW speed: c_T = 1 by construction (no tensor-sector operators).
- Equivalence principles: WEP satisfied (universal \tilde{g}_{μν}); EEP approximately satisfied in screened regimes; SEP violated (as expected in scalar–tensor), but safely screened in the Solar System.
- Sourcing and composition: ξ(ρ,T) sourced by local baryonic density and tidal invariant; universal couplings imply no detectable composition dependence at current sensitivities; screening protects lab/Solar-System bounds.

Pointers to implementation
- Lensing predictor and disformal phenomenology: tools/lensing_predict.py and docs/lensing.md.
- Environmental mapping used in fits: ξ(ρ,T) bounded functional with power-law or exponential S_ρ and tidal window W(T).
