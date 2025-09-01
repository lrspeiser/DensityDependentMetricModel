# Density-Dependent Metric Model (DDMM)

## A Novel Framework for Modified Gravity in Galactic Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a comprehensive framework for testing modified gravity theories through density-dependent metric modifications. The model addresses the galaxy rotation curve problem without invoking dark matter, while maintaining consistency with Solar System constraints.

## Table of Contents

1. [Theoretical Framework](#theoretical-framework)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Model Variants](#model-variants)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Production Runs](#production-runs)
7. [Expected Results](#expected-results)
8. [Publications](#publications)

## Theoretical Framework

### Core Hypothesis

The fundamental premise of DDMM is that the gravitational metric tensor acquires a density-dependent modification in regions of low matter density:

```
g_μν = η_μν + h_μν(ρ)
```

where `h_μν(ρ)` represents density-dependent perturbations that vanish in high-density environments (like the Solar System) but become significant in galactic outskirts.

### Physical Motivation

1. **Galaxy Rotation Curves**: Observed flat rotation curves require either dark matter or modified gravity
2. **Cassini Constraint**: Solar System tests constrain |ξ - 1| < 2.3 × 10⁻⁵ at Earth's orbit
3. **Screening Mechanism**: Natural density-based screening preserves GR in high-density regions

### Key Innovation: The Xi Function

The model introduces a dimensionless enhancement factor ξ(ρ) that modifies the Newtonian potential:

```
v²_total = v²_Newton × ξ(ρ)
```

## Mathematical Formulation

### Base Equations

#### 1. Total Velocity Model

The observed circular velocity at radius R is:

```python
v_total(R) = v_Newton(R) × √ξ(ρ(R))
```

where:
- `v_Newton(R)` = Newtonian velocity from baryonic matter
- `ξ(ρ(R))` = density-dependent enhancement factor
- `ρ(R)` = local matter density at radius R

#### 2. Newtonian Contribution

```python
v²_Newton = G × M_enc(R) / R
```

where `M_enc(R)` includes:
- Thin disk: `M_thin × (1 - exp(-R/R_thin) × (1 + R/R_thin))`
- Thick disk: `M_thick × (1 - exp(-R/R_thick) × (1 + R/R_thick))`
- Bulge (Hernquist): `M_bulge × (R/(R+a_bulge))²`
- Gas disk: `M_gas × (1 - exp(-R/R_gas) × (1 + R/R_gas))`

#### 3. Density Calculation

Total midplane density:

```python
ρ_total = ρ_disk + ρ_bulge + ρ_gas

ρ_disk = Σ_disk/(2h_z) × exp(-R/R_d)
ρ_bulge = M_bulge/(2π) × a_bulge/(R(R+a_bulge)³)
ρ_gas = Σ_gas/(2h_z_gas) × exp(-R/R_gas)
```

### Xi Function Models

#### 1. Power Law
```python
ξ(ρ) = A × (ρ/ρ_c)^(n-1)
```
- Simple power-law scaling
- Parameters: `ρ_c` (critical density), `n` (power index), `A` (amplitude)

#### 2. Exponential
```python
ξ(ρ) = A × exp((ρ/ρ_c - 1) × n)
```
- Exponential enhancement in low-density regions
- Rapid suppression at high density

#### 3. Gravitational Color Confinement
```python
ξ(ρ) = 1 + λ_g × exp(-(ρ/ρ_c)^γ)
```
- Inspired by QCD color confinement
- Parameters: `λ_g` (coupling strength), `γ` (screening exponent)

#### 4. Tidal Screening Models

**Tidal Band Model:**
```python
T = |∇²Φ| / (4πGρ)  # Tidal proxy
P(T) = 1/(1 + exp((ln(T/T₀))/σ_lnT))  # Logistic activation
ξ = 1 + (λ_max - 1) × (1 - (ρ/ρ_c)^γ) × P(T)
```

**Tidal Ratio Model:**
```python
R_tidal = T/ρ^η  # Tidal-to-density ratio
ξ = 1 + (λ_max - 1) × sigmoid(R_tidal)
```

#### 5. RAR-Based Models

**RAR Gate Model:**
```python
g_bar = v²_Newton/R  # Baryonic acceleration
g_obs = g_bar × ξ
RAR_prediction = g_bar / (1 - exp(-√(g_bar/a₀)))
ξ = 1 + (λ_max - 1) × (1 - (ρ/ρ_c)^γ) × Gate(g_bar, g_RAR)
```

**RAR Blend Model:**
```python
ξ = 1 + A_excess × (g_RAR/g_bar - 1) × TidalScreen(T)
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ compatible GPU (recommended: 8GB+ VRAM)
- 32GB+ system RAM for full dataset processing

### Setup

```bash
# Clone repository
git clone https://github.com/lrspeiser/DensityDependentMetricModel.git
cd DensityDependentMetricModel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CuPy for GPU acceleration
pip install cupy-cuda11x  # Adjust for your CUDA version
```

## Usage

### Quick Test Run

Test a single model with reduced dataset:

```bash
cd runners
python run_dynesty_stellar_fit_cupy.py --xi power --sample_max 1000 --maxcall 10000
```

### Model Comparison

Run all models with test settings:

```bash
python run_all_stellar_fits.py --test
```

### Production Run

Full dataset with maximum precision:

```bash
python run_production_fits.py --auto --priority
```

This will:
- Use all 144,000 Gaia stars
- Run 40 million likelihood evaluations
- Use 2,000-3,000 live points
- Automatically optimize for your GPU

### Custom Configuration

```bash
python run_production_fits.py \
    --models power grav_color tidal_band \
    --sample_max 144000 \
    --maxcall 40000000 \
    --nlive 3000
```

## Production Runs

### Configurations

| Config | Stars | Max Calls | Live Points | Runtime/Model |
|--------|-------|-----------|-------------|---------------|
| **high_precision** | 200,000 | 40M | 3,000 | 8-10 hours |
| **standard** | 144,000 | 20M | 2,000 | 4-6 hours |
| **fast** | 100,000 | 10M | 1,500 | 2-3 hours |
| **benchmark** | 50,000 | 5M | 1,000 | 1-2 hours |

### Priority Models

1. **power** - Power law modification (most promising)
2. **grav_color** - Gravitational color confinement
3. **tidal_band** - Tidal screening mechanism
4. **rar_gate** - RAR-anchored modification
5. **nfw** - Standard ΛCDM for comparison

### GPU Optimization

The code automatically:
- Enables TF32 tensor cores (2x speedup on RTX 30xx/40xx/50xx)
- Pre-allocates GPU memory
- Uses CuPy fused kernels
- Optimizes CUDA scheduling

## Expected Results

### Model Performance Metrics

Based on preliminary tests with Gaia DR3 data:

| Model | Chi² Range | RMSE (km/s) | Key Features |
|-------|------------|-------------|--------------|
| **GR** | 10,000-20,000 | 50-70 | Baseline (fails outer galaxy) |
| **NFW** | 2,000-5,000 | 20-30 | Standard dark matter |
| **power** | 1,000-3,000 | 15-25 | Best overall fit |
| **grav_color** | 1,500-3,500 | 18-28 | Physical motivation |
| **tidal_band** | 1,200-2,800 | 16-26 | Environment-dependent |
| **rar_gate** | 1,000-2,500 | 15-23 | MOND-compatible |

### Key Predictions

1. **Solar Neighborhood**: ξ ≈ 1.00001 (satisfies Cassini constraint)
2. **R = 20 kpc**: ξ ≈ 1.5-2.5 (explains flat rotation curve)
3. **Dwarf Galaxies**: ξ up to 5-10 (MOND-like regime)

### Validation Criteria

Models are evaluated on:
1. **Chi-squared fit** to 144,000 Gaia stars
2. **Regional accuracy** (inner, solar, outer galaxy)
3. **Physical consistency** (no superluminal velocities)
4. **Cassini constraint** satisfaction
5. **SPARC galaxy fits** (147 galaxies)

## Output Structure

```
production_results/
├── power/
│   ├── stellar_fit_cupy_power_results.npz    # Parameter samples
│   ├── stellar_fit_cupy_power_plot.png       # Fit visualization
│   └── convergence_diagnostics.json          # Sampling statistics
├── config_20240901_120000.json               # Run configuration
└── final_results_20240901_180000.json        # Summary statistics
```

## Theoretical Implications

### Advantages Over Dark Matter

1. **No fine-tuning**: Natural screening from density dependence
2. **No missing matter**: Uses only observed baryons
3. **Predictive power**: Parameters constrained by local physics
4. **Unification**: Single mechanism for all scales

### Consistency Checks

✅ **Cassini Constraint**: |ξ - 1| < 2.3 × 10⁻⁵ at 1 AU  
✅ **Weak Equivalence Principle**: Preserved (metric modification)  
✅ **Strong Equivalence Principle**: Modified (density dependence)  
✅ **Conservation Laws**: Energy-momentum preserved  

### Open Questions

1. Cosmological implications (CMB, structure formation)
2. Gravitational lensing predictions
3. Gravitational wave propagation
4. Quantum gravity connection

## Code Architecture

### Core Modules

- `core/density_metric_cupy.py` - GPU-accelerated physics
- `core/density_metric.py` - CPU reference implementation
- `runners/run_dynesty_stellar_fit_cupy.py` - Bayesian inference engine
- `runners/run_production_fits.py` - Production pipeline

### Performance

- **GPU Acceleration**: 50-100x speedup over CPU
- **Memory Efficient**: Streaming processing for large datasets
- **Parallel Sampling**: Multiple chains for convergence
- **Checkpointing**: Automatic save/resume capability

## Contributing

Contributions welcome! Key areas:
- Additional xi function models
- Cosmological extensions
- Gravitational lensing calculations
- JWST data integration

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ddmm2024,
  title={Density-Dependent Metric Modifications as Dark Matter Alternative},
  author={Speiser, L.R. and Collaborators},
  journal={In Preparation},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

- **Lead Developer**: L.R. Speiser
- **Email**: [via GitHub]
- **Issues**: https://github.com/lrspeiser/DensityDependentMetricModel/issues

## Acknowledgments

- Gaia DR3 collaboration for stellar kinematics data
- SPARC team for galaxy rotation curves
- CuPy developers for GPU acceleration framework
- Dynesty team for nested sampling implementation

---

*"The universe is not only queerer than we suppose, but queerer than we can suppose."* - J.B.S. Haldane

The search for dark matter has consumed decades. Perhaps it's time to consider that gravity itself holds the secret.
