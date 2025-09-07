# Relativistic scaffold (assumptions for weak-field predictions)

This project adopts a minimal covariant subclass for the current manuscript build to ensure that dynamics and lensing are predicted from a single theory without any lensing-only scalars.

Assumptions (explicit):
- Tensor speed c_T = 1 (GW170817-safe) on the backgrounds of interest.
- No weak-field anisotropic stress in the quasi-static regime so that Φ = Ψ.

Implications used in code:
- PPN coefficients in the screened/high-acceleration regime (Solar System): γ = 1, β = 1, α1 = 0, α2 = 0.
- Weak-field lensing uses Φ + Ψ = 2Φ, and Φ is built from the same baryons plus the effective “phantom” mass implied by the xi(...) mapping used in dynamics.

Where this is enforced:
- theory/relativistic.py: programmatic PPN exporter and c_T guardrail; replace with explicit expressions if you adopt a concrete Lagrangian (e.g., a restricted Horndeski/DHOST subclass).
- scripts/next_steps_from_run.py: pass --metric-lensing-only to compute θ_E and ΔΣ(R) directly from the metric prediction (no α_lens_ph or environment scaling in manuscript outputs). A per-lens ΔΣ profile is saved under results/next_steps/<run>/lensing_metric_profiles/ and a simple stack is written to results/next_steps/<run>/lensing_metric_stack.csv with a companion plot.
- Solar-System checks: ΔG/G table and plot are written alongside an optional PPN table (--write-ppn-table), with the Cassini bound annotated in the plot.

Caveats and roadmap:
- This scaffold encodes the relativistic subclass via assumptions to satisfy editorial requirements for a single-theory prediction path; a full derivation of γ, β, α1, α2 from an action (with explicit screening) is planned. When that is in place, theory/relativistic.py should be updated to compute PPN from parameters and the guardrail can be strengthened beyond a hard assertion.
- For stacked galaxy–galaxy lensing comparisons (HSC/DES/KiDS), wire public ΔΣ(R) products or shear catalogs and compare against the generated ΔΣ(R) stack (results/…/lensing_metric_stack.csv).
