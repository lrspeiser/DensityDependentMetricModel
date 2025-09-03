#!/usr/bin/env python3
"""
Test script to debug CuPy model issues.
"""

import sys
import numpy as np
import cupy as cp
from pathlib import Path

# Import CuPy model functions
from core.density_metric_cupy import (
    v_total_kms_cupy,
    DEFAULT_DTYPE
)

def test_model(xi_type, allow_experimental=False):
    """Test a specific xi model."""
    print(f"\nTesting {xi_type}...")
    
    # Create test data
    R_test = cp.array([5.0, 8.0, 12.0, 20.0], dtype=DEFAULT_DTYPE)
    
    # Fixed baryonic parameters
    params = {
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
    
    # Add model-specific parameters
    if xi_type == 'power':
        params.update({
            'rho_c_solar_kpc3': 1e8,
            'n_exp': 2.0,
            'A': 1.0
        })
    elif xi_type == 'tidal_band':
        params.update({
            'rho_c': 1e7,
            'gamma': 3.0,
            'lambda_max': 4.0,
            'T0': 10.0,
            'sigma_lnT': 0.8,
            'wmin': 0.02
        })
    elif xi_type == 'rar_gate':
        params.update({
            'a0_m_s2': 1.2e-10,
            'gamma_exp': 3.0,
            'lambda_max': 3.0,
            'T0': 10.0,
            'sigma_lnT': 0.8,
            'wmin': 0.02
        })
    
    try:
        # Call the model
        v_model = v_total_kms_cupy(R_test, params, xi_type=xi_type, allow_experimental=allow_experimental)
        
        # Convert to CPU and print
        v_cpu = cp.asnumpy(v_model)
        print(f"  Success! Velocities: {v_cpu}")
        return True
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Testing CuPy stellar models...")
    
    # Test basic models
    test_model('gr')
    test_model('power')
    # test_model('grav_color')
    
    # Test experimental models
    test_model('tidal_band', allow_experimental=True)
    # test_model('rar_gate', allow_experimental=True)
    
    print("\nAll tests completed!")

if __name__ == '__main__':
    main()
