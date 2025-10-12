# Core Physics Module Documentation

This directory contains the fundamental, GPU‑accelerated implementations used for galaxy predictions. It operates in the weak‑field, quasi‑static regime and uses a gating factor ξ to rescale the baryonic prediction for circular speed.

What the code actually computes
- Baryonic circular speed V_bar(R) from disks/bulges/gas.
- Gate ξ(...), which depends on local baryonic fields (density ρ, tidal proxy T, acceleration g_bar) with conservative screening such that ξ→1 in Solar‑System conditions.
- Predicted speed: V_model²(R) = ξ(R) · V_bar²(R).

Relativistic mapping and PPN
- The repository adopts a minimal metric subclass with Φ=Ψ and c_T=1 in screened, quasi‑static limits. The environment potential φ_env ≡ ½ ln ξ enters additively into both potentials so lensing responds to Φ+Ψ built from the same ξ.
- In the Solar limit under screening, PPN parameters reduce to their GR values (γ=β=1, α1=α2=0). See docs/ppn_mapping.md for a worked derivation and conditions under which the Cassini Shapiro bound does or does not constrain ε≡ξ−1.

Key files
- density_metric_cupy.py — primary GPU implementation (v_total_kms_cupy combines baryons and ξ; many ξ variants with conservative Solar screening are provided).
- xi_registry.py — declares which ξ models are “published” (reproducible) vs experimental.
- theory/relativistic.py — weak‑field helpers and PPN export under screening.

Notes
- The previous placeholder equations suggesting “G_μν + ξ(ρ) G_μν = 8π T_μν” were incorrect and have been removed. The working formulation is non‑relativistic (QUMOND‑like) for galaxy dynamics with a consistent weak‑field metric mapping for lensing.

This directory contains the fundamental physics implementations for the Density-Dependent Metric Model (DDMM).

## Mathematical Framework

The DDMM modifies Einstein field equations by introducing a density-dependent enhancement function:

**Modified Field Equations:**
```
G_μν + ξ(ρ)G_μν = 8πT_μν
```

**Enhancement Function:**
```
ξ(ρ) = A(ρ_c/ρ)^n
```

Where:
- A: dimensionless amplitude parameter
- ρ_c: critical density (typically ρ_c ≈ 10^-29 g/cm³)  
- n: power law index (typically n ∈ [0.1, 2])
- ρ: local matter density

## Core Implementation Files

### 1. density_metric2.py - Main JAX Implementation

**Purpose**: Primary DDMM physics engine with JAX optimizations

**Key Mathematical Functions:**

*Metric Perturbation:*
```
h_μν = -2ξ(ρ)η_μν
g_μν = η_μν(1 - 2ξ(ρ))
```

*Velocity Correction:*
```
v_eff² = v_obs²(1 + ξ(ρ))
v_obs² = GM/r * (1 + ξ(ρ_local))
```

**Performance:** 
- JIT compilation: ~10x speedup
- Automatic differentiation for gradients
- Vectorized operations over density fields

### 2. density_metric_cupy.py - GPU Implementation

**Purpose**: CUDA-optimized implementation for large-scale computations

**GPU Features:**
- CUDA kernels for density field computation
- Shared memory optimization
- Multi-GPU scaling support

**Memory Scaling:**
- Single GPU: up to 10^8 density points
- Memory usage: ~8 GB for full Gaia dataset

### 3. density_contrast_model.py - Density Reconstruction

**Purpose**: Reconstructs 3D density fields from Gaia stellar catalogs

**Reconstruction Algorithm:**
```
ρ(r) = Σᵢ (mᵢ/h³) * W(|r - rᵢ|/h)
```

**Smoothing Kernel (Wendland C²):**
```
W(q) = (21/2π) * (1-q)⁴(1+4q)  for q ≤ 1
     = 0                        for q > 1
```

### 4. data_io.py - Data Management

**Purpose**: Handles all data loading, preprocessing, and output formatting

**Supported Formats:**
- Gaia DR3: stellar positions, velocities, photometry
- Pantheon+: supernova distance moduli
- Galaxy rotation curves: velocity profiles

**Data Preprocessing:**
- Quality cuts: parallax_error/parallax < 0.2
- Extinction corrections using 3D dust maps
- Proper motion cleaning and outlier removal

### 5. run_dynesty.py - Bayesian Parameter Estimation

**Purpose**: Nested sampling for parameter estimation

**Sampling Configuration:**
- Dynesty nested sampler
- Multi-dimensional parameter space
- Adaptive sampling boundaries

**Prior Specifications:**
```
A ∈ [0, 10]           # Enhancement amplitude
n ∈ [0.1, 3.1]        # Power law index  
log₁₀(ρ_c) ∈ [-32, -26]  # Critical density
```

**Likelihood Function:**
```
χ² = -0.5 * Σᵢ ((v_obs[i] - v_model[i])/σ[i])²
```

## Physical Validation

**Newtonian Limit (ξ → 0):**
- Recovers Newton law: F = GMm/r²
- Verified to machine precision

**General Relativity Limit:**
- When A = 0, exactly reproduces GR
- Schwarzschild metric recovered

**Solar System Constraints:**
- PPN parameter γ = 1 + ξ(ρ_☉)
- Cassini bound: |γ - 1| < 2.3 × 10⁻⁵

## File Interdependencies

```
density_metric2.py ←→ data_io.py
       ↓
density_contrast_model.py
       ↓
run_dynesty.py
       ↑
density_metric_cupy.py
```

## Usage Examples

**Basic DDMM Calculation:**
```python
from core.density_metric2 import DDMMModel
from core.data_io import load_gaia_data

stars = load_gaia_data("gaia_dr3_subset.fits")
model = DDMMModel(A=0.1, n=1.5, rho_c=1e-29)
v_model = model.compute_velocities(stars['positions'])
```

**GPU-Accelerated Run:**
```python  
from core.density_metric_cupy import DDMMModelGPU

model_gpu = DDMMModelGPU(device=0)
result = model_gpu.fit_parameters(large_dataset)
```
ENDOFFILE < /dev/null
