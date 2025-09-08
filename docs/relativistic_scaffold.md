# Relativistic scaffold (assumptions for weak-field predictions)

This project adopts a minimal covariant subclass for the current manuscript build to ensure that dynamics and lensing are predicted from a single theory without any lensing-only scalars.

Assumptions (explicit):
- Tensor speed c_T = 1 (GW170817-safe) on the backgrounds of interest.
- No weak-field anisotropic stress in the quasi-static regime so that Φ = Ψ.

Implications used in code:
- PPN coefficients in the screened/high-acceleration regime (Solar System): γ = 1, β = 1, α1 = 0, α2 = 0.
- Weak-field lensing uses Φ + Ψ, where Φ and Ψ are built from the same baryons plus the environment potential φ_env = 1/2 ln ξ used in dynamics (see appendix).

Where this is enforced:
- docs/paper_appendix_relativistic.md: appendix with the schematic action, weak-field mapping, and PPN/c_T statements that the manuscript cites.
- theory/relativistic.py: explicit weak-field helpers (Φ, Ψ, Φ+Ψ from ξ via φ_env), programmatic PPN export (GR values under screening), and c_T guardrail (must be 1).
- theory/lensing_metric.py: metric-based ΔΣ(R) and deflection helpers from Φ_W = (Φ+Ψ)/2 (spherical symmetry for manuscript utilities).
- scripts/next_steps_from_run.py: pass --metric-lensing-only to compute θ_E and ΔΣ(R) directly from the metric prediction (no α_lens_ph or environment scaling in manuscript outputs). A per-lens ΔΣ profile is saved under results/next_steps/<run>/lensing_metric_profiles/ and a simple stack is written to results/next_steps/<run>/lensing_metric_stack.csv with a companion plot.
- Solar-System checks: ΔG/G table and plot are written alongside an optional PPN table (--write-ppn-table), with the Cassini bound annotated in the plot.

Caveats and roadmap:
- This scaffold encodes the relativistic subclass via assumptions to satisfy editorial requirements for a single-theory prediction path; a more explicit restricted Horndeski/DHOST parameterization can replace the schematic action and provide formula-level PPN dependencies. When that is in place, theory/relativistic.py should compute PPN from parameters and the guardrail can be strengthened beyond a hard assertion.
- For stacked galaxy–galaxy lensing comparisons (HSC/DES/KiDS), wire public ΔΣ(R) products or shear catalogs and compare against the generated ΔΣ(R) stack (results/…/lensing_metric_stack.csv).
