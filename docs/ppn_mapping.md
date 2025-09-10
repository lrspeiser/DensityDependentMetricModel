# PPN mapping and Shapiro delay under screened, Φ=Ψ, c_T=1 subclass

Purpose
- Provide an explicit, auditable PPN derivation in the Solar limit for the gated model used in this repository.
- Spell out when the Cassini Shapiro-delay bound on |γ−1| constrains (or does not constrain) ε ≡ ξ − 1.
- Clarify how the “amplitude rescaling with γ=1” interpretation arises and when |ΔG/G| ≡ |ξ − 1| is a conservative tracer vis-à-vis Cassini.

Scope and assumptions (weak field, SI units)
- Metric in PPN (isotropic) gauge:
  g_00 = −1 + 2Φ/c^2 − 2β Φ^2/c^4 + O(c^{−6}),
  g_ij = (1 + 2γ Φ/c^2) δ_ij + O(c^{−4}).
- Solar System (screened) subclass adopted here:
  (i) tensor speed c_T = 1; (ii) negligible weak-field anisotropic stress so Φ = Ψ; (iii) matter minimally coupled (Jordan frame), so photons and massive bodies see the same metric.
- Baryonic Newtonian potential Φ_b solves ∇²Φ_b = 4π G ρ_b.
- Gating enters as an “environment” potential φ_env built from the same ξ used for galaxy dynamics:
  φ_env(x) ≡ (1/2) ln ξ(x).
  In SI, write Φ = Φ_b + c^2 φ_env and Ψ = Ψ_b + c^2 φ_env with Φ_b, Ψ_b in m^2 s^{−2}.
- Define ε ≡ ξ − 1; for Solar checks we require |ε| ≪ 1.

Result 1 — PPN parameters in the Solar limit
- Substitute Φ = Φ_b + c^2 φ_env and Ψ = Ψ_b + c^2 φ_env into the PPN ansatz. To first post-Newtonian (1PN) order:
  g_00 = −1 + 2(Φ_b + c^2 φ_env)/c^2 + O(c^{−4}),
  g_ij = [1 + 2(Φ_b + c^2 φ_env)/c^2] δ_ij + O(c^{−4}).
- Coefficients of Φ at 1PN are equal in g_00 and g_ij, hence
  γ ≡ Ψ/Φ = 1  and  β = 1,
  provided φ_env contributes additively to both potentials with the same coefficient (Φ=Ψ mapping) and no explicit U^2 term is introduced. Screening in the Solar System implies φ_env → 0 locally, so the exported PPN values are exactly the GR ones in this limit: γ = 1, β = 1, α1 = α2 = 0.
- Practical export in code: we report these GR values in nature_readiness/…/ppn_table.csv under the screened subclass.

Result 2 — Shapiro delay and amplitude rescaling
- For a static, spherically symmetric field, the one-way Shapiro delay for a light signal grazing the Sun is (PPN, isotropic coordinates):
  Δt = (1 + γ) GM_⊙/c^3 · ln[(r_E + r_R + R)/(r_E + r_R − R)],
  where r_E, r_R are heliocentric distances of emitter/receiver and R their Euclidean separation.
- If the weak-field potential is uniformly rescaled by (1 + ε) in the relevant Solar region (while γ = 1), this is equivalent to GM_⊙ → (1 + ε) GM_⊙ to 1PN order, giving
  Δt → (1 + γ)(1 + ε) GM_⊙/c^3 · ln(…).
- What Cassini actually constrains is the coefficient of the ln(…) term relative to the GM_⊙ value used in the orbit solution (ephemerides). Two regimes matter:
  A) Degenerate-amplitude regime (no Cassini bound on ε): If the same (1 + ε) rescales Solar dynamics used to determine GM_⊙ and the light-path potential entering Δt, then ε is absorbed into the fitted GM_⊙. Cassini’s γ measurement remains ≈ 1 independent of ε; the experiment does not constrain ε directly. Independent constraints on ε then come from ephemerides/ranging consistency across radii (see below).
  B) Non-degenerate regime (Cassini bounds ε): If GM_⊙ in the Cassini analysis is fixed from data insensitive to ε (or at radii where ε differs), while the light path experiences (1 + ε), then Cassini would infer an apparent γ_eff = (1 + ε) even if the true γ = 1. The published bound |γ − 1| ≲ 2.3 × 10^{−5} then implies |ε| ≲ 2.3 × 10^{−5} along the Cassini ray path near conjunction.

When does Cassini constrain ε in this model?
- The screened subclass in this repo is implemented such that ξ → 1 in (i) high-density/high-acceleration regions and (ii) deep vacuum, with enhancements confined to intermediate galactic environments. Around the Sun, both the solar interior (high ρ) and the low-density heliocentric vacuum are in regimes where ξ ≈ 1 by construction for the published gates. Under these conditions the light path sees ε ≈ 0 and regime A applies: Cassini constrains γ while ε is separately constrained by planetary dynamics (GM_⊙) and its radial constancy.
- To be conservative, we report |ΔG/G| ≡ |ξ − 1| as a “tracer” curve across 1–30 AU. Where γ = 1 identically and ξ ≈ 1, the Cassini γ bound does not directly limit the tracer; rather, the requirement is that the tracer stays ≪ 10^{−5} over the radii and timescales probed by ephemerides (and indeed our Solar plots show |ξ−1| ≪ 10^{−5} at Saturn for the published gates).
- If a future gate variant were to produce a non-uniform ε(r) in the inner Solar System, then two independent constraints apply: (i) ephemerides/ranging bound spatial variations of GM_⊙ (effectively ε(r)) at the 10^{−10}–10^{−11} level; (ii) Cassini’s γ bound would also project to |ε| ≲ 2.3×10^{−5} along the ray if GM_⊙ were held fixed from unaffected data.

Worked PPN sketch (algebra in SI)
1) Start with the line element (weak field):
   ds^2 = −[1 − 2(U/c^2) + 2β(U/c^2)^2] c^2 dt^2 + [1 + 2γ(U/c^2)] d\mathbf{x}^2.
   Here U has units of m^2 s^{−2} and solves ∇²U = 4π G ρ.
2) Adopt U = Φ_b + c^2 φ_env with φ_env = ½ ln ξ and |φ_env| ≪ 1 near the Sun.
   To 1PN, keep only terms linear in U:
   g_00 = −1 + 2(Φ_b + c^2 φ_env)/c^2 + O(c^{−4}),
   g_ij = [1 + 2(Φ_b + c^2 φ_env)/c^2] δ_ij + O(c^{−4}).
   Matching to the PPN template gives γ = 1 and (because no new U^2 coefficient is introduced) β = 1 in the screened Solar limit.
3) Shapiro delay derives from null geodesics in this metric; expanding to 1PN yields the standard expression above with γ entering only as the relative coefficient between spatial and temporal potentials. A uniform amplitude change U → (1 + ε) U with γ = 1 is indistinguishable from GM → (1 + ε) GM in the Shapiro term, leading to the degeneracy discussion.

Energy–momentum conservation and well-posedness (relativistic scaffold)
- Action-level origin (sketch). We embed the weak-field mapping in a minimal scalar–tensor subclass in the Jordan frame with c_T = 1:
  S = ∫ d^4x √−g [ (M_Pl^2/2) R − ½ (∇ϕ)^2 − V(ϕ) ] + S_m[Ψ_m, \tilde{g}_{μν}],
  with the effective Jordan metric \tilde{g}_{μν} = A^2(ϕ) g_{μν} + B(ϕ) ∂_μϕ ∂_νϕ.
  In the quasi-static weak field and under screening, ϕ sources φ_env ≈ ½ ln ξ and enters additively in Φ, Ψ with equal coefficients (Φ=Ψ). Matter couples minimally to \tilde{g}_{μν}, ensuring ∇_μ T^{μν}_{(tot)} = 0 by diffeomorphism invariance; photons follow null geodesics of \tilde{g}_{μν}.
- c_T = 1 and stability. We restrict operators so the tensor quadratic action matches GR (GW170817-safe). The scalar has positive kinetic term and sound speed c_s^2 > 0 in the weak field. No higher-derivative ghostly terms are invoked (Horndeski/DHOST-safe choices). Screening forces ϕ → 0 in Solar conditions, suppressing PPN deviations.
- What is implemented vs. assumed. The code implements the weak-field (QUMOND-like) mapping and uses ξ to compute galaxy dynamics and lensing via a single metric (Φ=Ψ). A full covariant completion is presented as a scaffold; we do not solve the scalar PDE in production runs. Claims are limited to the quasi-static weak field and screened Solar System.

Where in the repo
- docs/modified_poisson_qumond.md — non-relativistic derivation of ξ and “phantom” density used for lensing.
- docs/relativistic_scaffold.md — metric mapping, φ_env ≡ ½ ln ξ, and guardrails.
- theory/relativistic.py — weak-field helpers and PPN export under screening.
- nature_readiness/solar_system/cassini_shapiro.py — non-interactive Shapiro calculator (updated to use the standard formula and demonstrate the ε/γ degeneracy cases).

Reviewer checklist (what this establishes)
- In the Solar limit with Φ=Ψ and c_T=1, γ=β=1 to 1PN; the gate enters as an additive environment potential that vanishes under screening.
- Cassini constrains γ; it only constrains ε if amplitude rescaling is not absorbed into the GM used by ephemerides or if ε varies along the ray. In our published gates, ξ≈1 throughout the Solar regime, so Cassini’s γ bound is satisfied while |ΔG/G| serves as a conservative tracer.
