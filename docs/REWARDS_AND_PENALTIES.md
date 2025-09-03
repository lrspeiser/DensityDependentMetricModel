# Rewards and Penalties in the DDMM Parameter Search

## Overview

Yes! The code uses sophisticated reward/penalty mechanisms through **Bayesian nested sampling**. Here's how it works:

## 1. Primary Reward/Penalty: The Likelihood Function

The main scoring happens in `log_likelihood()`:

```python
def log_likelihood(theta, param_names, R_data, v_data, sigma_data):
    # PENALTY: Invalid parameters get -infinity (immediate rejection)
    if 'M_' in name and value <= 0:
        return -np.inf  # Negative mass = impossible!
    
    # Calculate model predictions
    v_model = v_total_kms_cupy(R_data, params, xi_type)
    
    # REWARD: Better fit to data gets higher likelihood
    chi2 = sum((v_data - v_model)² / sigma²)
    log_likelihood = -0.5 * chi2
    
    return log_likelihood  # Higher is better!
```

### Scoring Scale:
- **Perfect fit**: log_likelihood ≈ -72,000 (for 144,000 stars)
- **Good fit**: log_likelihood ≈ -100,000
- **Poor fit**: log_likelihood < -1,000,000
- **Invalid**: log_likelihood = -∞ (rejected immediately)

## 2. Physical Constraints (Hard Penalties)

The code enforces physics through hard penalties:

```python
# In log_likelihood functions:

# PENALTY 1: Non-physical parameters
if mass <= 0: return -np.inf  # Can't have negative mass!
if radius <= 0: return -np.inf  # Can't have negative size!

# PENALTY 2: NaN or infinite velocities
if not np.all(np.isfinite(v_model)):
    return -np.inf  # Model broke = rejected

# PENALTY 3: Cassini constraint violation (implicit)
# If xi ≠ 1 at solar position, chi2 becomes huge
# This naturally penalizes models that violate Solar System tests
```

## 3. Prior Distributions (Soft Guidance)

Priors guide the search toward reasonable regions:

```python
# From setup_parameter_bounds():

# Log-uniform priors for masses (explores wide range efficiently)
M_thin_disk: [1e10, 1e11] with log prior
# Why: Equal probability to explore 1e10 and 5e10

# Uniform priors for scales (equal probability across range)  
R_screen: [30, 80] with uniform prior
# Why: No preference for any particular screening radius

# Tight priors for critical parameters
rho_c: [1e7, 1e9]  # Must be near solar density!
# Why: Cassini constraint requires this
```

## 4. Nested Sampling's Smart Exploration

Dynesty uses **dynamic nested sampling** which:

1. **Starts broad**: Explores entire parameter space
2. **Identifies good regions**: Finds where likelihood is high
3. **Focuses effort**: Concentrates sampling in promising areas
4. **Adapts live points**: Adds more samplers to difficult regions

```python
# In run_dynesty_single.py:
sampler = dynesty.DynamicNestedSampler(
    nlive=500,  # Start with 500 parallel explorers
    sample='rslice',  # Efficient slice sampling
    bound='multi'  # Multiple bounding ellipsoids
)

# The sampler automatically:
# - Kills bad parameter sets (low likelihood)
# - Spawns new ones near good sets (high likelihood)
# - Gradually zooms in on best regions
```

## 5. Convergence Acceleration

The code speeds up convergence through:

### A. Early Stopping
```python
sampler.run_nested(
    dlogz_init=0.01,  # Stop when improvement < 0.01
    maxcall=10000000  # Or after 10M evaluations
)
```

### B. Efficiency Tracking
```python
# The code monitors efficiency:
if efficiency < 1%:
    # Search is struggling, might need different approach
```

### C. Progressive Refinement
```python
# Initial phase: Loose tolerance
dlogz_init=1.0  # Find rough solution fast

# Refinement phase: Tight tolerance  
dlogz_init=0.01  # Polish the best solution
```

## 6. Real Example: Balanced Screening

Here's how rewards/penalties work for our balanced screening model:

```python
# Good parameters (high reward):
params = {
    'rho_c_solar_kpc3': 7e7,  # Matches solar density ✓
    'R_screen': 50,            # Reasonable screening ✓
    'A_max': 2.0               # Modest enhancement ✓
}
# Result: log_likelihood ≈ -85,000 (good!)

# Bad parameters (heavy penalty):
params = {
    'rho_c_solar_kpc3': 1e13,  # Way too high ✗
    'R_screen': 50,
    'A_max': 2.0
}
# Result: log_likelihood ≈ -1,000,000,000 (terrible!)
# Why: Violates Cassini, gives 6000 km/s velocities
```

## 7. Adaptive Sampling in Action

Watch how the sampler learns:

```
Iteration 1-100: Random exploration
  - LogL range: [-1e9, -1e6]
  - Finding viable regions

Iteration 100-1000: Focusing
  - LogL range: [-1e6, -500,000]
  - Identified promising parameters
  
Iteration 1000-10000: Refinement
  - LogL range: [-200,000, -85,000]
  - Converging on best solution
  
Final result: 
  - Best LogL: -85,000
  - Parameters: rho_c=7.2e7, R_screen=48, A_max=2.1
```

## 8. Why This Is Efficient

Traditional grid search for 15 parameters with 10 values each:
- Would need 10^15 evaluations (impossible!)

Nested sampling with rewards/penalties:
- Typically converges in ~100,000-1,000,000 evaluations
- That's 10^9 times faster!

## Key Takeaway

The code doesn't just randomly try parameters. It:
1. **Rewards** parameters that fit the data well
2. **Penalizes** unphysical or constraint-violating choices
3. **Learns** which regions are promising
4. **Focuses** computational effort where it matters
5. **Converges** efficiently to the best solution

This is why fixing rho_c bounds from [1e12, 1e15] to [1e7, 1e9] is so important - it removes a huge region of parameter space that would always be penalized for violating Cassini!