# Relativistic scaffold (assumptions for weak-field predictions)

This project adopts a minimal covariant subclass for the current manuscript build to ensure that dynamics and lensing are predicted from a single theory without any lensing-only scalars.

Assumptions (explicit):
- Tensor speed c_T = 1 (GW170817-safe) on the backgrounds of interest.
- No weak-field anisotropic stress in the quasi-static regime so that Φ = Ψ.

Implications used in code:
- See also docs/ppn_mapping.md for the explicit Solar‑System PPN and Shapiro mapping used to address Cassini.
- PPN coefficients in the screened/high-acceleration regime (Solar System): γ = 1, β = 1, α1 = 0, α2 = 0.
- Weak-field lensing uses Φ + Ψ, where Φ and Ψ are built from the same baryons plus the environment potential φ_env = 1/2 ln ξ used in dynamics (see appendix and QUMOND mapping).

Where this is enforced:
- docs/paper_appendix_relativistic.md: appendix with the schematic action, QUMOND→ξ mapping, phantom-density lensing link, and PPN/c_T statements that the manuscript cites.
- docs/modified_poisson_qumond.md: compact derivation of the modified Poisson equation, ξ boost, phantom density identity, and lensing mapping.
- theory/relativistic.py: weak-field helpers (Φ, Ψ, Φ+Ψ from ξ via φ_env), programmatic PPN export (GR values under screening), and c_T guardrail (must be 1).
- scripts/next_steps_from_run.py: computes ρ_ph from ξ on axisymmetric grids, projects Σ_tot, solves θ_E, and writes ΔΣ/θ_E profiles. The Solar-System ΔG/G and PPN tables are also produced here.

Scope and limitations:
- Claims are confined to the weak-field, quasi-static regime and to the screened Solar System. Cosmology and strong-field tests are out of scope for this manuscript build.
- For stacked galaxy–galaxy lensing comparisons (HSC/DES/KiDS), wire public ΔΣ(R) products or shear catalogs and compare against results/…/lensing_metric_stack.csv.
