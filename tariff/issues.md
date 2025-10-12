# Open Issues, Tests, and Work Plan (Tariff Addendum)

This file tracks the outstanding theoretical and observational checks for the “energy‑tariff” cosmological addendum and the concrete actions to address them. None of these items alters the main paper’s galaxy‑scale results; they are required to elevate the cosmological add‑on to submission quality.

## Major Issues to Address Pre‑Submission

1. SN time‑dilation
   - Observation: Type Ia light curves exhibit (1+z) time dilation.
   - Requirement: A static tariff must either (a) reproduce the same time‑dilation through an independent mechanism, or (b) demonstrate that standardization pipelines do not bias the test in favor of (1+z) stretching.
   - Actions:
     - Reprocess a subset of SN light curves with explicit time‑domain fits under the tariff hypothesis.
     - Quantify goodness of fit vs standard stretch parameters and report Bayes factors.

2. CMB blackbody spectrum and spectral distortions
   - Observation: The CMB is a near‑perfect blackbody; FIRAS bounds are extremely tight.
   - Requirement: Any energy‑loss mechanism must preserve the CMB spectrum to FIRAS limits.
   - Actions:
     - Compute the transformation of a Planck spectrum under d ln E/dr = −k [ξ(r)−1] f_void(r).
     - If necessary, constrain allowed k(z)–ξ(z) evolution and include photon‑number effects so the equilibrium spectrum remains Planckian within FIRAS uncertainties.

3. Tolman surface‑brightness test and d_L–d_A duality
   - Observation: In expansion cosmology, surface brightness S ∝ (1+z)−4.
   - Requirement: In a static‑tariff framework, specify the luminosity‑distance mapping d_L = r (1+z)^p and test p jointly with deep‑imaging SB data.
   - Actions:
     - Add p to the tariff inference; compare to the latest Tolman measurements.
     - Assess Etherington’s duality implications under the tariff mapping.

4. BAO and cosmic chronometers
   - Observation: BAO provide a standard ruler; chronometers probe H(z).
   - Requirement: Derive the predicted z(r) and an “effective” H(z) under the tariff and confront BAO peak positions and chronometer datasets.
   - Actions:
     - Implement z(r) → H_eff(z) diagnostics; run against published BAO and CC compilations.

5. Large‑scale‑structure (LSS) consistency
   - Observation: LOS environments vary; our f_void(r) is a phenomenological proxy.
   - Requirement: Ensure the same gate ξ that modifies dynamics implies consistent void/filament behavior along SN sightlines.
   - Actions:
     - Ray‑trace through N‑body mocks to produce first‑principles f_void(r) and predict the correlation between SN residuals and LOS density contrast.
     - Test against tomographic LOS densities (DES/KiDS/SDSS).

6. Lensing time delays and strong‑lens cosmography
   - Observation: Time‑delay distances are sensitive to photon flight times.
   - Requirement: Determine whether the tariff changes photon travel times and whether that biases time‑delay distances.
   - Actions:
     - Compute Fermat‑potential integrals with/without tariff; compare to well‑measured systems.

7. Parameter identifiability and degeneracies
   - Observation: k and D_max are degenerate once the local slope (H0) is fixed; r0 and γ control tail shape.
   - Requirement: Report marginalized posteriors and degeneracies from joint fits.
   - Actions:
     - Perform joint fits to SNe + lensing + RAR; quantify posteriors for (k, D_max, r0, γ, p).

## Concrete Research Plan (Short‑Term)

A. Joint inference
- Fit (k, D_max, r0, γ, p) to Pantheon+ with explicit time‑dilation and K‑corrections; report Bayesian evidence vs ΛCDM.

B. Structure‑correlation test
- Cross‑correlate SN residuals with tomographic LOS densities; predict sign/amplitude from ray‑traced f_void(r).

C. CMB spectrum check
- Compute spectral distortions from the tariff model along a realistic thermal history; constrain k(z) if needed to satisfy FIRAS.

D. Lensing & dynamics coherence
- With the same ξ, re‑validate galaxy–galaxy lensing amplitudes and outer‑disk K_z while scanning the SN‑permitted band of (r0, γ).

E. Public release
- Package the tariff integrator, the parameter sweep (tariff/sweep_results.csv), and a reproducible notebook/figure scripts.

## Go/No‑Go Criterion for Submission

Green‑light once all are satisfied:
1) SN time‑dilation is matched without ad‑hoc fixes.
2) The CMB blackbody is preserved within FIRAS limits.
3) At least one environment‑dependence prediction (LOS correlation of SN residuals) is detected at >2σ.

