#!/usr/bin/env python3
"""
Test script for the comprehensive baryonic model
"""

import numpy as np
import cupy as cp
from pathlib import Path
import sys

# Add the current directory to the path to import our modules
sys.path.append('.')

from density_metric_cupy import v_total_kms_cupy, v_baryon_comprehensive_kms_cupy, volume_density_comprehensive_solar_kpc3_cupy

def test_comprehensive_model():
    """Test the comprehensive baryonic model"""
    
    print("="*60)
    print("TESTING COMPREHENSIVE BARYONIC MODEL")
    print("="*60)
    
    # Test parameters (typical Milky Way values)
    test_params = {
        'M_thin_disk_solar': 5e10,      # 50 billion solar masses
        'R_thin_disk_kpc': 3.0,         # 3 kpc scale length
        'hz_thin_disk_kpc': 0.3,        # 0.3 kpc scale height
        
        'M_thick_disk_solar': 5e9,      # 5 billion solar masses
        'R_thick_disk_kpc': 3.5,        # 3.5 kpc scale length
        'hz_thick_disk_kpc': 0.8,       # 0.8 kpc scale height
        
        'M_bulge_solar': 1e10,          # 10 billion solar masses
        'R_bulge_kpc': 0.5,             # 0.5 kpc scale length
        
        'M_gas_solar': 1e10,            # 10 billion solar masses
        'R_gas_kpc': 7.0,               # 7 kpc scale length
        'hz_gas_kpc': 0.1               # 0.1 kpc scale height
    }
    
    # Test radii
    R_test = np.array([1.0, 3.0, 5.0, 8.0, 12.0, 15.0])  # kpc
    
    print(f"\nTest Parameters:")
    for key, value in test_params.items():
        print(f"  {key}: {value:.2e}")
    
    print(f"\nTest Radii: {R_test} kpc")
    
    # Test comprehensive baryonic velocity
    print(f"\n1. Testing Comprehensive Baryonic Velocity")
    v_baryon = v_baryon_comprehensive_kms_cupy(R_test, test_params)
    print(f"   Baryonic velocities: {v_baryon}")
    
    # Test comprehensive density
    print(f"\n2. Testing Comprehensive Density")
    rho_total = volume_density_comprehensive_solar_kpc3_cupy(R_test, test_params)
    print(f"   Total densities: {rho_total}")
    
    # Test GR model (should be same as baryonic)
    print(f"\n3. Testing GR Model (xi=gr)")
    v_gr = v_total_kms_cupy(R_test, test_params, xi_type='gr')
    print(f"   GR velocities: {v_gr}")
    
    # Test power law model
    print(f"\n4. Testing Power Law Model (xi=power)")
    power_params = test_params.copy()
    power_params.update({
        'rho_c_solar_kpc3': 1e8,
        'n_exp': 2.0,
        'A_xi': 1.0
    })
    v_power = v_total_kms_cupy(R_test, power_params, xi_type='power')
    print(f"   Power law velocities: {v_power}")
    
    # Calculate total baryonic mass
    total_mass = sum([test_params[key] for key in test_params.keys() if 'M_' in key and 'solar' in key])
    print(f"\n5. Total Baryonic Mass: {total_mass:.2e} M_☉")
    
    # Check if velocities are reasonable
    print(f"\n6. Velocity Analysis:")
    print(f"   Min velocity: {np.min(v_gr):.1f} km/s")
    print(f"   Max velocity: {np.max(v_gr):.1f} km/s")
    print(f"   Velocity at 8 kpc: {v_gr[R_test == 8.0][0]:.1f} km/s")
    
    # Check if GR and baryonic are identical
    if np.allclose(v_gr, v_baryon, rtol=1e-10):
        print(f"   ✓ GR model correctly equals baryonic model")
    else:
        print(f"   ✗ GR model differs from baryonic model")
    
    # Check if power law enhances velocities
    enhancement = v_power / v_gr
    print(f"\n7. Power Law Enhancement:")
    for i, R in enumerate(R_test):
        print(f"   R = {R:.1f} kpc: {enhancement[i]:.3f}x")
    
    print(f"\n" + "="*60)
    print(f"TEST COMPLETE")
    print(f"="*60)
    
    return True

def test_parameter_bounds():
    """Test parameter bounds from run_dynesty_cupy.py"""
    
    print(f"\n" + "="*60)
    print(f"TESTING PARAMETER BOUNDS")
    print(f"="*60)
    
    # Import the setup function
    from run_dynesty_cupy import setup_parameter_bounds
    
    # Test GR baseline bounds
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds('gr')
    
    print(f"\nGR Baseline Parameters ({len(param_names)}):")
    for i, name in enumerate(param_names):
        log_prior = "log" if use_log_prior[i] else "linear"
        print(f"  {name}: [{bounds_low[i]:.2e}, {bounds_high[i]:.2e}] ({log_prior} prior)")
    
    # Test other models
    for xi_type in ['enhanced', 'power', 'grav_color']:
        param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(xi_type)
        print(f"\n{xi_type.title()} Model Parameters ({len(param_names)}):")
        for i, name in enumerate(param_names):
            log_prior = "log" if use_log_prior[i] else "linear"
            print(f"  {name}: [{bounds_low[i]:.2e}, {bounds_high[i]:.2e}] ({log_prior} prior)")
    
    print(f"\n" + "="*60)
    print(f"BOUNDS TEST COMPLETE")
    print(f"="*60)

if __name__ == "__main__":
    test_comprehensive_model()
    test_parameter_bounds() 