# A Density-Dependent Metric Modification with Enhanced Gravity in Low-Density Regimes

**Abstract:**
The flat rotation curves of galaxies continue to challenge the standard model of gravity, which assumes that visible baryonic matter alone cannot account for observed stellar velocities in galactic outskirts. Traditionally explained through dark matter halos, we propose an alternative: a **Density-Dependent Metric Model** in which **gravity strengthens** as baryonic matter density **decreases**. Unlike earlier formulations where gravity was suppressed in high-density regions and normalized elsewhere, our updated model shows a **gravity enhancement** in sparse regions such as the galactic outskirts. This revised framework better reproduces Milky Way rotation curves by allowing visible baryonic mass to exert more gravitational influence in low-density environments. We use dynamic nested sampling with over 2 million likelihood evaluations on 80,000 Gaia DR3 stars and confirm the model's physical plausibility and predictive accuracy. 

The entire implementation is available in the `DensityDependentMetricModel` GitHub repository. Key components include:
- `run_dynesty.py` – main driver for the dynamic nested sampler
- `density_metric2.py` – defines gravitational model including `xi_enhanced_bounded()`
- `data_io.py` – Gaia data loading and preprocessing
- `validate_ddmm.py` – multi-scale consistency checks (Solar System to CMB)

---

## 1. Introduction: Reframing the Rotation Curve Mystery

Galactic rotation curves do not decline as expected from Newtonian gravity applied to luminous matter. The standard fix — dark matter — remains unobserved in non-gravitational channels. Our revised density-dependent model introduces a fundamentally different mechanism: **spacetime becomes more responsive to baryonic mass when local densities are low**.

### 1.1. Smart Fabric Revisited: Gravity’s Adaptive Response

Imagine spacetime as a responsive medium. 
- In high-density zones, this fabric acts "normally," curving modestly under visible mass.
- In low-density zones, the fabric **stretches and reacts more intensely**, amplifying gravitational interactions.

This updated view is the inverse of screening models: **gravity doesn’t turn off in empty regions — it turns up.**

### 1.2. The Updated ξ(ρ) Hypothesis

We redefine the modulating function as:

$$
\xi(\rho) = 1 + A \cdot \left(\frac{\rho_c}{\rho}\right)^n
$$

Where:
- ξ > 1 in low-density zones (ρ ≪ ρ_c), enhancing gravity
- ξ → 1 in high-density zones (ρ ≫ ρ_c), recovering Newtonian gravity
- A and n determine the strength and sharpness of the enhancement

This shift reverses prior expectations. The outer galaxy no longer needs excess dark matter — **it needs more responsive gravity**.

Implemented in `density_metric2.py`:
```python
@nb.njit(cache=True)
def xi_enhanced_bounded(rho, rho_c, n, A=1.0):
    rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    xi = 1.0 + A * (rho_c / rho_arr)**n
    xi = np.minimum(xi, 5.0)  # Optional physical cap
    return xi if rho_arr.shape else xi[0]
```

---

## 2. Observations and Methodology

### 2.1. Gaia DR3 Star Sample
80,000 stars were selected based on quality metrics (RUWE, parallax S/N, velocity errors) and transformed into Galactocentric cylindrical coordinates.

### 2.2. Rotation Curve Construction
We computed circular velocities from stellar kinematics and modeled baryonic contributions from a 4-component mass model (thin/thick disk, bulge, gas).

### 2.3. Nested Sampling with Dynesty
We employed dynamic nested sampling (`run_dynesty.py`) to explore the posterior distribution of:
- Baryonic masses and scale lengths
- Critical density ρ_c
- Power-law index n
- Gravity amplification A

Physical plausibility checks were applied in real time to reject unphysical configurations. Sampling is resumed from checkpoint if interrupted.

```bash
python run_dynesty.py \
  --xi grav_color \
  --include_disk_thin --include_disk_thick --include_bulge --include_gas \
  --fit_xi_params --fit_disk_thin --fit_disk_thick --fit_bulge --fit_gas \
  --M_disk_thin_min 2e10 --h_z_thin_min 0.12 \
  --R_d_thick_max 10.0 --M_gas_max 6e10 \
  --sample_method rwalk --walks 50 --nlive_init 1500 \
  --dlogz_target 0.05 --maxcall 4000000 \
  --checkpoint_every 300 \
  --output_dir chains_gravcolor_run5
```

---

## 3. Results: Enhanced Gravity, Excellent Fit

### 3.1. Rotation Curve Fit
At R☉ = 8.122 kpc:
- v_Newton = 208.3 km/s
- ξ(R☉) = 1.839
- v_model = 282.4 km/s (vs ~280 km/s observed)

✅ 100% of posterior samples passed physical checks.

### 3.2. Posterior Structure
- Strong bimodality observed across multiple parameters
- Gravity enhancement universally preferred in the outskirts

### 3.3. Parameter Highlights
| Parameter | Median | ξ-enhanced Impact |
|----------|--------|--------------------|
| A        | ~1.2   | Sets boost factor  |
| ρ_c      | 2.7e8 M☉/kpc³ | Trigger scale |
| n        | ~2.0   | Controls sharpness |

---

## 4. Discussion: A Viable Alternative to Dark Matter

### 4.1. Enhanced Gravity as Phenomenological Driver
The model reproduces the flat rotation curve by **amplifying Newtonian gravity in low-density regions**, using only visible baryonic mass.

### 4.2. Predictive Success
- Accurate at Solar radius and beyond
- No need to assume missing mass

### 4.3. Theoretical Outlook
This behavior may emerge from:
- Quantum gravity effects
- Entropic gravity scenarios
- Modified field equations responsive to local matter content

---

## 5. Conclusions

- The **enhanced-gravity formulation of the Density-Dependent Metric Model** successfully explains Milky Way rotation curves without dark matter
- It introduces **a new regime of gravity**: one that strengthens in emptier space
- The model is physically valid, computationally tractable, and observationally precise

Future work will extend this framework to:
- SPARC galaxy catalog
- Weak lensing comparisons
- Theoretical derivations from emergent gravity frameworks

---

**Repository:** https://github.com/lrspeiser/DensityDependentMetricModel