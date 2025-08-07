# Balanced Screening Model for DDMM

## Overview

The Balanced Screening Model is a physically motivated enhancement to the Density-Dependent Metric Model (DDMM) that solves the critical deep space problem while maintaining compatibility with both Solar System constraints and galaxy rotation curves.

## The Deep Space Problem

Previous DDMM models (power law, gravitational color, etc.) suffered from a fundamental issue:
- In low-density regions (ρ → 0), the enhancement factor ξ → ∞
- This causes velocities to explode (3000+ km/s instead of 200-300 km/s)
- In deep space, this produces unphysical gravitational effects

## The Solution: Distance Screening

The Balanced Screening Model introduces distance-dependent screening to ensure proper behavior across all scales:

```
ξ(ρ,R) = 1 + A_max × density_factor(ρ) × screening_factor(R)
```

Where:
- `density_factor = (1 - ρ/ρ_c)^n` ensures ξ = 1 at solar density (Cassini constraint)
- `screening_factor = 0.5 × (1 + tanh((R_screen - R)/(0.3 × R_screen)))` ensures ξ → 1 in deep space
- `A_max` limits maximum enhancement (typically 2-3x, not 250x!)

## Key Physics

### 1. Gravitational Modification
The effective gravitational acceleration is:
```
g_eff = (GM/r²) × ξ(ρ,r)
```

Even if ξ is enhanced, the 1/r² term ensures g → 0 as r → ∞.

### 2. Three Regimes
- **Solar System (R ≈ 8 kpc, high ρ)**: ξ ≈ 1 (satisfies Cassini constraint)
- **Galaxy Edge (R ≈ 20-40 kpc, low ρ)**: ξ > 1 (explains rotation curves)  
- **Deep Space (R > 100 kpc)**: ξ → 1 (screening active, normal 1/r² falloff)

### 3. Redshift Compatibility
In standard GR, gravitational redshift in empty space → 0. In naive DDMM with ξ → ∞, redshift would diverge. The screening ensures redshift remains physical:
```
1 + z_grav = √(g₀₀(emit)/g₀₀(observe)) × √(ξ_emit/ξ_obs)
```
With screening, ξ remains bounded, preventing infinite redshift.

## Model Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `rho_c_solar_kpc3` | Critical density (solar value) | 1e7 - 1e9 | M_sun/kpc³ |
| `R_screen` | Screening radius | 30 - 80 | kpc |
| `n_exp` | Density exponent | 0.5 - 2.0 | - |
| `A_max` | Maximum enhancement | 1.5 - 3.0 | - |

## Running the Model

### Quick Test
```bash
python test_balanced_with_deep_space.py
```

### Full Gaia Dataset (144,000 stars)
```bash
python runners/run_dynesty_single.py \
  --xi balanced_screening \
  --nlive 500 \
  --maxcall 10000000 \
  --dlogz_target 0.01 \
  --max_sample_gaia 144000
```

### With Proper Solar Density
When running, ensure `rho_c_solar_kpc3` is set to actual solar density (~1e8 M_sun/kpc³), not arbitrary high values:

```python
params = {
    'M_thin_disk_solar': 3e10,
    'R_thin_disk_kpc': 3.0,
    'hz_thin_disk_kpc': 0.3,
    # ... other mass parameters ...
    'rho_c_solar_kpc3': 1e8,  # CRITICAL: Use solar density!
    'R_screen': 50.0,
    'n_exp': 1.0,
    'A_max': 2.0
}
```

## Expected Results

### Velocity Profile
| R (kpc) | v_Newton (km/s) | v_DDMM (km/s) | Enhancement |
|---------|-----------------|---------------|-------------|
| 8 | 164 | 164 | 1.00x (Cassini) |
| 15 | 120 | 206 | 1.72x |
| 25 | 93 | 159 | 1.71x |
| 50 | 66 | 93 | 1.41x |
| 100 | 46 | 46 | 1.00x (screened) |

### Model Comparison
- **LogZ vs GR**: Expected ΔLogZ > 0 if DDMM is preferred
- **Efficiency**: ~1-2% with 500 live points
- **Runtime**: 4-8 hours for full Gaia dataset

## Advantages Over Previous Models

1. **Physical Velocities**: 150-300 km/s instead of 3000+ km/s
2. **Cassini Compliance**: ξ = 1.0 at solar density by construction
3. **Deep Space Safety**: Screening ensures normal 1/r² behavior
4. **Bounded Enhancement**: Maximum 2-3x, not unbounded growth
5. **Cosmological Compatibility**: Preserves standard physics at large scales

## Implementation Details

The model is implemented in `core/density_metric_cupy.py` as `xi_balanced_screening_cupy()` with full GPU acceleration via CuPy. Parameter bounds are configured in `runners/run_dynesty_cupy.py` under the `'balanced_screening'` case.

## Validation Tests

The model passes comprehensive tests:
1. ✓ Galaxy rotation curves (150-300 km/s)
2. ✓ Cassini constraint (|ξ-1| < 10⁻⁵ at solar position)
3. ✓ Deep space limit (ξ → 1 as R → ∞)
4. ✓ Velocity falloff (v ∝ 1/√r in deep space)
5. ✓ No numerical instabilities or NaN values

## Citation

If using this model, please cite:
- Original DDMM paper: [citation]
- Balanced Screening implementation: This repository

## Future Improvements

- Adaptive screening radius based on galaxy properties
- Anisotropic screening for disk vs halo regions
- Connection to cosmological parameters at z > 0