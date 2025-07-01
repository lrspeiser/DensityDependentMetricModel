#!/usr/bin/env python3
"""
debug_dynesty_pipeline.py - Find where 80k stars became 570 stars
"""
import sys
import numpy as np
from pathlib import Path

# Add the modules to path
sys.path.append('.')

print("DEBUGGING DYNESTY DATA PIPELINE")
print("="*60)

# Step 1: Simulate the exact data loading that dynesty uses
print("1. TESTING DATA LOADING (as dynesty does it):")
try:
    from data_io import load_gaia
    gaia_data_dict = load_gaia(sample_max=80000)
    
    if gaia_data_dict:
        R_data = gaia_data_dict["R_kpc"]
        v_data = gaia_data_dict["v_obs"] 
        sigma_data = gaia_data_dict["sigma_v"]
        
        print(f"  ✅ Loaded: {len(R_data):,} stars")
        print(f"  ✅ R range: {np.min(R_data):.2f} - {np.max(R_data):.2f} kpc")
        print(f"  ✅ v range: {np.min(v_data):.1f} - {np.max(v_data):.1f} km/s")
        print(f"  ✅ σ range: {np.min(sigma_data):.1f} - {np.max(sigma_data):.1f} km/s")
    else:
        print("  ❌ Failed to load gaia data")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Error in data loading: {e}")
    sys.exit(1)

# Step 2: Check for any NaN/inf values that might be filtered
print(f"\n2. DATA QUALITY CHECKS:")
n_finite_R = np.sum(np.isfinite(R_data))
n_finite_v = np.sum(np.isfinite(v_data))
n_finite_sigma = np.sum(np.isfinite(sigma_data))

print(f"  Finite R values: {n_finite_R:,} / {len(R_data):,}")
print(f"  Finite v values: {n_finite_v:,} / {len(v_data):,}")
print(f"  Finite σ values: {n_finite_sigma:,} / {len(sigma_data):,}")

# Check for extreme values
n_reasonable_R = np.sum((R_data > 0.01) & (R_data < 100))
n_reasonable_v = np.sum((v_data > 0) & (v_data < 1000))
n_reasonable_sigma = np.sum((sigma_data > 0) & (sigma_data < 200))

print(f"  Reasonable R (0.01-100 kpc): {n_reasonable_R:,} / {len(R_data):,}")
print(f"  Reasonable v (0-1000 km/s): {n_reasonable_v:,} / {len(v_data):,}")
print(f"  Reasonable σ (0-200 km/s): {n_reasonable_sigma:,} / {len(sigma_data):,}")

# Step 3: Test model calculation with default parameters
print(f"\n3. TESTING MODEL CALCULATION:")
try:
    from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP
    
    # Use default parameters similar to what dynesty would start with
    test_params = {
        'include_disk_thin': True,
        'include_disk_thick': False,
        'include_bulge': False,
        'include_gas': False,
        'include_bulge_density': False,
        'M_disk_thin_solar': 4e10,  # Default value
        'R_d_thin_kpc': 2.5,        # Default value
        'h_z_thin_kpc': 0.3,        # Default value
        'rho_c_solar_kpc3': 1e7,    # Default value
        'n_exp': 1.5                 # Default value
    }
    
    print(f"  Testing with default parameters...")
    v_newton = v_baryon_total_newtonian_kms(R_data, test_params)
    rho_mid = rho_baryon_total_midplane_solar_kpc3(R_data, test_params)
    
    print(f"  ✅ v_newton calculated: {len(v_newton):,} values")
    print(f"    Range: {np.min(v_newton):.1f} - {np.max(v_newton):.1f} km/s")
    
    print(f"  ✅ ρ_mid calculated: {len(rho_mid):,} values") 
    print(f"    Range: {np.min(rho_mid):.2e} - {np.max(rho_mid):.2e} M☉/kpc³")
    
    # Test xi calculation
    xi_func = XI_FUNCTION_MAP['power']
    xi_values = xi_func(rho_mid, test_params['rho_c_solar_kpc3'], test_params['n_exp'])
    
    if hasattr(xi_values, '__len__'):
        print(f"  ✅ ξ calculated: {len(xi_values):,} values")
        print(f"    Range: {np.min(xi_values):.3f} - {np.max(xi_values):.3f}")
        
        # Check for problematic xi values
        n_finite_xi = np.sum(np.isfinite(xi_values))
        n_positive_xi = np.sum(xi_values > 0)
        n_reasonable_xi = np.sum((xi_values > 0) & (xi_values <= 10))
        
        print(f"    Finite ξ: {n_finite_xi:,}")
        print(f"    Positive ξ: {n_positive_xi:,}")
        print(f"    Reasonable ξ (0-10): {n_reasonable_xi:,}")
        
        if n_reasonable_xi < len(xi_values) * 0.9:
            print(f"    ⚠️  Many ξ values are problematic!")
    
    # Test final model velocity
    v_model = v_newton * np.sqrt(np.maximum(xi_values, 0.0))
    n_finite_v_model = np.sum(np.isfinite(v_model))
    
    print(f"  ✅ v_model calculated: {n_finite_v_model:,} finite values")
    print(f"    Range: {np.nanmin(v_model):.1f} - {np.nanmax(v_model):.1f} km/s")
    
except Exception as e:
    print(f"  ❌ Error in model calculation: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Check if there are spatial cuts in run_dynesty.py
print(f"\n4. CHECKING FOR HIDDEN SPATIAL CUTS:")
print(f"  Looking for potential filters that might reduce sample size...")

# Check radial distribution
r_bins = np.linspace(0, 25, 26)
r_hist, _ = np.histogram(R_data, bins=r_bins)
print(f"  Radial distribution:")
for i in range(len(r_bins)-1):
    if r_hist[i] > 0:
        print(f"    {r_bins[i]:.1f}-{r_bins[i+1]:.1f} kpc: {r_hist[i]:,} stars")

# Check if there might be |b| < 30° cut or similar
if 'z_kpc' in gaia_data_dict:
    z_data = gaia_data_dict['z_kpc']
    n_lowz = np.sum(np.abs(z_data) < 1.0)  # Stars close to plane
    print(f"  Stars with |z| < 1 kpc: {n_lowz:,} / {len(z_data):,}")

# Step 5: Simulate likelihood calculation
print(f"\n5. TESTING LIKELIHOOD CALCULATION:")
try:
    # Simulate what happens in the likelihood function
    residuals = v_data - v_model
    sigma_safe = np.maximum(sigma_data, 1e-9)
    chi2_terms = (residuals / sigma_safe)**2
    
    # Check for extreme chi2 values that might cause filtering
    n_finite_chi2 = np.sum(np.isfinite(chi2_terms))
    n_reasonable_chi2 = np.sum(chi2_terms < 1000)  # Arbitrary threshold
    
    print(f"  Finite χ² terms: {n_finite_chi2:,} / {len(chi2_terms):,}")
    print(f"  Reasonable χ² (< 1000): {n_reasonable_chi2:,} / {len(chi2_terms):,}")
    
    if n_reasonable_chi2 < len(chi2_terms) * 0.5:
        print(f"    ⚠️  Many χ² values are extreme!")
        print(f"    This might cause numerical issues in likelihood")
        
    log_L = -0.5 * np.sum(chi2_terms + np.log(2 * np.pi * sigma_safe**2))
    print(f"  Total log-likelihood: {log_L:.2f}")
    
except Exception as e:
    print(f"  ❌ Error in likelihood calculation: {e}")

# Step 6: Check the actual results file
print(f"\n6. CHECKING DYNESTY RESULTS:")
results_file = Path("chains_dynesty/dynesty_mw_power_DTf_samples.npz")
if results_file.exists():
    try:
        data = np.load(results_file)
        samples = data['samples']
        print(f"  ✅ Results file found")
        print(f"  ✅ Samples shape: {samples.shape}")
        print(f"  ✅ Number of parameters fitted: {samples.shape[1]}")
        print(f"  ✅ Number of posterior samples: {samples.shape[0]}")
        
        if 'weights' in data:
            weights = data['weights']
            eff_samples = np.sum(weights)**2 / np.sum(weights**2)
            print(f"  ✅ Effective samples: {eff_samples:.0f}")
            
    except Exception as e:
        print(f"  ❌ Error reading results: {e}")
else:
    print(f"  ❌ Results file not found")

print(f"\n" + "="*60)
print("SUMMARY:")
print("The data loads correctly (80k stars) but somewhere in the")
print("dynesty pipeline, it gets reduced to 570 stars.")
print("\nPossible causes:")
print("1. Hidden filtering in likelihood function")
print("2. Spatial cuts applied during dynesty run")
print("3. Numerical issues causing star rejection")
print("4. Memory limitations causing sampling")
print("5. Bug in parameter passing to model functions")