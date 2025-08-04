#!/usr/bin/env python3
"""
Test script to verify that the blob structure is consistent in the likelihood function.
"""

import numpy as np
import jax
import jax.numpy as jnp
from run_dynesty import log_likelihood_dynesty
import argparse

def test_blob_consistency():
    """Test that all return paths in log_likelihood_dynesty return consistent blob structures."""
    
    # Create mock arguments
    args = argparse.Namespace()
    args.fit_target = 'milkyway'
    args.fit_disk_reparameterized = False
    args.include_disk_thin = True
    args.include_disk_thick = False
    args.include_bulge = False
    args.include_gas = False
    args.disable_cassini_penalty = True
    args.disable_rho_c_penalty = False
    args.gamma_fixed = 2.7
    args.lambda_g_fixed = 8.0
    args.rho_c_fixed = 5e8
    
    # Create mock data
    R_data = jnp.array([5.0, 8.0, 15.0, 20.0])
    v_data = jnp.array([200.0, 220.0, 180.0, 160.0])
    sigma_data = jnp.array([10.0, 10.0, 10.0, 10.0])
    
    # Create mock parameter info
    param_info = [
        {'name': 'rho_c_solar_kpc3', 'is_fitted': True, 'current_val': 5e8},
        {'name': 'gamma_exp', 'is_fitted': False, 'current_val': 2.7},
        {'name': 'lambda_g', 'is_fitted': False, 'current_val': 8.0},
        {'name': 'M_disk_thin_solar', 'is_fitted': True, 'current_val': 5e10},
        {'name': 'R_d_thin_kpc', 'is_fitted': True, 'current_val': 3.0},
        {'name': 'h_z_thin_kpc', 'is_fitted': True, 'current_val': 0.3},
    ]
    
    fitted_names = ['rho_c_solar_kpc3', 'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc']
    
    print("Testing blob consistency...")
    
    # Test 1: Valid parameters
    try:
        theta_valid = np.array([5e8, 5e10, 3.0, 0.3])
        logL, blob = log_likelihood_dynesty(
            theta_valid, fitted_names, args, param_info, 
            R_data, v_data, sigma_data, 'grav_color'
        )
        print(f"✓ Valid parameters: logL = {logL:.2f}, blob length = {len(blob)}")
        print(f"  Blob: {blob}")
    except Exception as e:
        print(f"✗ Valid parameters failed: {e}")
    
    # Test 2: Invalid parameters (should trigger early return)
    try:
        theta_invalid = np.array([np.inf, 5e10, 3.0, 0.3])
        logL, blob = log_likelihood_dynesty(
            theta_invalid, fitted_names, args, param_info, 
            R_data, v_data, sigma_data, 'grav_color'
        )
        print(f"✓ Invalid parameters: logL = {logL:.2f}, blob length = {len(blob)}")
        print(f"  Blob: {blob}")
    except Exception as e:
        print(f"✗ Invalid parameters failed: {e}")
    
    # Test 3: Very invalid parameters (should trigger another early return)
    try:
        theta_very_invalid = np.array([1e20, 5e10, 3.0, 0.3])  # Very high rho_c
        logL, blob = log_likelihood_dynesty(
            theta_very_invalid, fitted_names, args, param_info, 
            R_data, v_data, sigma_data, 'grav_color'
        )
        print(f"✓ Very invalid parameters: logL = {logL:.2f}, blob length = {len(blob)}")
        print(f"  Blob: {blob}")
    except Exception as e:
        print(f"✗ Very invalid parameters failed: {e}")
    
    print("\nBlob consistency test completed!")

if __name__ == "__main__":
    test_blob_consistency() 