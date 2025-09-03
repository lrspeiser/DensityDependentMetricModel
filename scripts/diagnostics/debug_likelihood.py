#!/usr/bin/env python3
"""
Debug script to test likelihood evaluation directly.
"""

import sys
import numpy as np
import cupy as cp
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import functions
from core.density_metric_cupy import v_total_kms_cupy, DEFAULT_DTYPE

def test_likelihood_evaluation():
    """Test the likelihood evaluation with different parameter sets."""
    
    print("Testing likelihood evaluation...")
    
    # Create mock data
    np.random.seed(42)
    n_stars = 100
    R_data = np.random.uniform(2, 25, n_stars)
    v_true = 220 * np.sqrt(1 - np.exp(-R_data/3))
    sigma_data = np.ones(n_stars) * 10
    v_data = v_true + np.random.normal(0, sigma_data)
    
    # Transfer to GPU
    R_data_gpu = cp.asarray(R_data, dtype=DEFAULT_DTYPE)
    v_data_gpu = cp.asarray(v_data, dtype=DEFAULT_DTYPE)
    sigma_data_gpu = cp.asarray(sigma_data, dtype=DEFAULT_DTYPE)
    
    print(f"Data shape: {len(R_data)} stars")
    print(f"R range: {R_data.min():.1f} - {R_data.max():.1f} kpc")
    print(f"v range: {v_data.min():.1f} - {v_data.max():.1f} km/s")
    
    # Fixed baryonic parameters
    fixed_params = {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True
    }
    
    # Test different xi types
    test_cases = [
        ('gr', {}, False),
        ('power', {'rho_c_solar_kpc3': 1e8, 'n_exp': 2.0, 'A': 1.0}, False),
        ('tidal_band', {'rho_c': 1e7, 'gamma': 3.0, 'lambda_max': 4.0, 
                       'T0': 10.0, 'sigma_lnT': 0.8, 'wmin': 0.02}, True),
    ]
    
    for xi_type, extra_params, allow_exp in test_cases:
        print(f"\n--- Testing {xi_type} ---")
        
        # Combine parameters
        params = fixed_params.copy()
        params.update(extra_params)
        if allow_exp:
            params['allow_experimental'] = True
        
        print(f"Parameters: {extra_params}")
        
        try:
            # Calculate model velocities
            v_model_gpu = v_total_kms_cupy(R_data_gpu, params, xi_type=xi_type)
            
            # Check for NaN or inf
            if cp.any(cp.isnan(v_model_gpu)):
                print("  ERROR: Model contains NaN values!")
                continue
            if cp.any(cp.isinf(v_model_gpu)):
                print("  ERROR: Model contains inf values!")
                continue
            
            # Calculate chi-squared
            residuals_gpu = (v_data_gpu - v_model_gpu) / sigma_data_gpu
            chi2_gpu = cp.sum(residuals_gpu**2)
            chi2 = float(chi2_gpu)
            
            # Calculate log-likelihood
            log_L = -0.5 * chi2
            
            print(f"  v_model range: {float(cp.min(v_model_gpu)):.1f} - {float(cp.max(v_model_gpu)):.1f} km/s")
            print(f"  Chi²: {chi2:.1f}")
            print(f"  Log(L): {log_L:.1f}")
            
            if np.isfinite(log_L):
                print("  ✓ Likelihood is valid!")
            else:
                print("  ✗ Likelihood is invalid!")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

def test_dynesty_interface():
    """Test the Dynesty interface directly."""
    print("\n\n=== Testing Dynesty Interface ===\n")
    
    from dynesty import DynamicNestedSampler
    
    # Create mock data
    np.random.seed(42)
    n_stars = 100
    R_data = np.random.uniform(2, 25, n_stars)
    v_true = 220 * np.sqrt(1 - np.exp(-R_data/3))
    sigma_data = np.ones(n_stars) * 10
    v_data = v_true + np.random.normal(0, sigma_data)
    
    # Transfer to GPU
    R_data_gpu = cp.asarray(R_data, dtype=DEFAULT_DTYPE)
    v_data_gpu = cp.asarray(v_data, dtype=DEFAULT_DTYPE)
    sigma_data_gpu = cp.asarray(sigma_data, dtype=DEFAULT_DTYPE)
    
    # Fixed parameters
    fixed_params = {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True
    }
    
    # Test with power law (simpler case)
    param_names = ['rho_c_solar_kpc3', 'n_exp', 'A']
    param_bounds = [(1e6, 1e10), (0.5, 4.0), (0.1, 5.0)]
    ndim = len(param_names)
    
    def log_likelihood(theta):
        """Likelihood function for Dynesty."""
        params = fixed_params.copy()
        for i, name in enumerate(param_names):
            params[name] = theta[i]
        
        try:
            # Calculate model
            v_model_gpu = v_total_kms_cupy(R_data_gpu, params, xi_type='power')
            
            # Check for invalid values
            if cp.any(cp.isnan(v_model_gpu)) or cp.any(cp.isinf(v_model_gpu)):
                return -np.inf
            
            # Calculate chi-squared
            residuals_gpu = (v_data_gpu - v_model_gpu) / sigma_data_gpu
            chi2 = float(cp.sum(residuals_gpu**2))
            
            return -0.5 * chi2
            
        except Exception as e:
            print(f"Likelihood error with theta={theta}: {e}")
            return -np.inf
    
    def prior_transform(u):
        """Prior transform for Dynesty."""
        params = np.zeros(ndim)
        for i, (low, high) in enumerate(param_bounds):
            if low > 0 and high/low > 100:  # Log scale
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10**(log_low + u[i] * (log_high - log_low))
            else:  # Linear scale
                params[i] = low + u[i] * (high - low)
        return params
    
    # Test random points
    print("Testing random parameter points...")
    n_test = 10
    for i in range(n_test):
        u = np.random.random(ndim)
        theta = prior_transform(u)
        log_L = log_likelihood(theta)
        print(f"  Point {i+1}: theta={theta}, log(L)={log_L:.1f}")
        if np.isfinite(log_L):
            print("    ✓ Valid likelihood!")
        else:
            print("    ✗ Invalid likelihood!")
    
    # Test with Dynesty
    print("\nInitializing Dynesty sampler...")
    try:
        sampler = DynamicNestedSampler(
            log_likelihood,
            prior_transform,
            ndim,
            nlive=10,
            bound='multi',
            sample='auto'
        )
        
        print("Running short test...")
        sampler.run_nested(
            nlive_init=10,
            maxcall=100,
            print_progress=False,
            dlogz_init=0.5
        )
        
        print("✓ Dynesty test successful!")
        
    except Exception as e:
        print(f"✗ Dynesty test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_likelihood_evaluation()
    test_dynesty_interface()
