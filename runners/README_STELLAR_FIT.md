# Stellar-Focused Dynesty Fitting

## Overview

The modified `run_dynesty_stellar_fit.py` script changes the optimization approach from maximizing Bayesian evidence to directly fitting the observed stellar velocity data from Gaia DR3. This addresses the issue where models with high Bayesian evidence (logZ) may not actually fit the rotation curve well.

## Key Changes from Original `run_dynesty.py`

### 1. **Likelihood Function Redesign**
- **Focus on Chi-squared**: The primary metric is now chi-squared residuals between model and observed velocities
- **Regional Weighting**: Different galaxy regions are weighted to ensure good fits everywhere:
  - Solar neighborhood (7.5-8.5 kpc): Requires excellent fit
  - Inner galaxy (< 5 kpc): Standard weighting
  - Outer galaxy (> 15 kpc): Higher weight to ensure models capture the flat rotation curve

### 2. **Shape Matching**
The likelihood now includes bonuses/penalties for matching the expected rotation curve shape:
- Rising velocity in the inner galaxy (2-5 kpc)
- Peak around solar radius (8 kpc) with velocity ~220 km/s
- Flattening or slow decline in outer galaxy (> 12 kpc)

### 3. **Direct Stellar Data Fitting**
```python
# Core change: Direct chi-squared minimization
residuals = (v_data_jax - v_model) / sigma_data_jax
chi2_total = jnp.sum(residuals**2)
log_L_base = -0.5 * chi2_total
```

### 4. **Regional Quality Metrics**
The code tracks RMSE (root mean square error) in different regions:
- Solar RMSE: Fit quality at Sun's location
- Inner RMSE: Fit quality in bulge/inner disk
- Outer RMSE: Fit quality in outer disk where dark matter effects dominate

### 5. **Adaptive Penalties**
- Poor fits trigger quadratic penalties proportional to the deviation
- Solar neighborhood fits are prioritized with stronger penalties
- Outer galaxy velocities must stay within physical bounds (150-300 km/s)

## Usage

### Basic Run
```bash
python runners/run_dynesty_stellar_fit.py --xi power
```

### With Custom Settings
```bash
python runners/run_dynesty_stellar_fit.py \
    --xi grav_color \
    --nlive_init 1000 \
    --maxcall 2000000 \
    --output_dir stellar_fits_enhanced \
    --fit_disk_thin \
    --fit_disk_thick
```

### Fitting Multiple Components
```bash
python runners/run_dynesty_stellar_fit.py \
    --xi power \
    --fit_xi_params \
    --fit_disk_thin \
    --fit_disk_thick \
    --fit_bulge \
    --M_gas_fixed 3.0e10
```

## Parameters

### Xi Function Types
- `power`: Power-law density dependence (default)
- `exponential`: Exponential density dependence
- `grav_color`: Gravitational color model
- `tanh`: Hyperbolic tangent transition
- `arctan`: Arctangent transition

### Key Options
- `--nlive_init`: Number of live points (default: 500, auto-scales with dimensions)
- `--maxcall`: Maximum likelihood evaluations (default: 1,000,000)
- `--disable_cassini_penalty`: Turn off Solar System constraints for galaxy-only fits

### Component Fitting Flags
- `--fit_xi_params`: Fit the modified gravity parameters (default: True)
- `--fit_disk_thin`: Fit thin disk parameters
- `--fit_disk_thick`: Fit thick disk parameters
- `--fit_bulge`: Fit bulge parameters
- `--fit_gas`: Fit gas disk parameters

## Output

The script generates:

1. **Results file** (`stellar_fit_{xi}_results.npz`):
   - Best-fit parameters
   - Sample chains with weights
   - Chi-squared and RMSE values
   - Regional fit statistics

2. **Diagnostic plot** (`stellar_fit_{xi}_rotation_curve.png`):
   - Top panel: Observed vs. model velocities
   - Bottom panel: Residuals with RMSE bounds

3. **Log file** with detailed fitting progress

## Comparison with Original Approach

| Aspect | Original `run_dynesty.py` | Stellar-Focused Version |
|--------|---------------------------|------------------------|
| **Primary Goal** | Maximize Bayesian evidence | Minimize velocity residuals |
| **Optimization Target** | log(Z) with prior volume | Chi-squared to stellar data |
| **Convergence Criterion** | Evidence convergence (dlogz) | Fit quality (RMSE) |
| **Best Model Selection** | Highest evidence | Lowest chi-squared |
| **Regional Weighting** | Uniform or evidence-based | Explicit regional priorities |

## Expected Improvements

1. **Better Rotation Curve Fits**: Models will match observed velocities more closely
2. **Physical Realism**: Shape constraints ensure physically plausible curves
3. **Clear Metrics**: Chi-squared and RMSE provide intuitive fit quality measures
4. **Regional Balance**: Good fits across all galactic radii, not just high-density regions

## Recommended Workflow

1. **Run GR Baseline First**:
   ```bash
   python runners/run_dynesty_stellar_fit.py --xi power --fit_xi_params --disable_cassini_penalty
   ```
   Note the chi-squared and RMSE values.

2. **Test Modified Gravity Models**:
   ```bash
   python runners/run_dynesty_stellar_fit.py --xi grav_color --fit_xi_params
   ```

3. **Compare Results**:
   - Check if chi-squared improves
   - Verify RMSE decreases, especially in outer galaxy
   - Examine residual plots for systematic trends

4. **Fine-tune with Component Fitting**:
   If needed, allow baryonic parameters to vary:
   ```bash
   python runners/run_dynesty_stellar_fit.py --xi grav_color --fit_xi_params --fit_disk_thin --fit_disk_thick
   ```

## Troubleshooting

### High Chi-squared Values
- Check if stellar data is properly loaded
- Verify uncertainty estimates are reasonable
- Consider relaxing penalty thresholds

### Poor Outer Galaxy Fits
- May need to adjust xi function parameters bounds
- Consider different xi function types
- Check if gas component mass is reasonable

### Slow Convergence
- Increase `nlive_init` for better exploration
- Reduce `maxcall` for faster initial tests
- Use `--disable_cassini_penalty` for galaxy-only fits

## Technical Notes

### Memory Usage
The stellar-focused version loads all Gaia data into memory and converts to JAX arrays. Ensure sufficient RAM for large datasets.

### GPU Acceleration
If available, JAX will automatically use GPU. Monitor with:
```bash
nvidia-smi  # For NVIDIA GPUs
```

### Parallel Processing
Currently single-threaded for simplicity. For parallel execution, modify the pool parameter in DynamicNestedSampler.

## Citation

If you use this stellar-focused fitting approach, please cite:
- The original Gaia DR3 data release
- The dynesty package for nested sampling
- Your paper describing the density-dependent metric model
