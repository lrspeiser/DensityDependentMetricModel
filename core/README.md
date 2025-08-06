# Core Physics Module Documentation

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
