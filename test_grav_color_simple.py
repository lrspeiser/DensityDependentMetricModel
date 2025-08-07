#!/usr/bin/env python3
"""Simple test of grav_color_void_safe model."""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import v_total_kms_cupy

# Test parameters - these worked in previous runs
params = {
    'M_thin_disk_solar': 3e10,
    'R_thin_disk_kpc': 3.0,
    'hz_thin_disk_kpc': 0.3,
    'M_thick_disk_solar': 3e9,
    'R_thick_disk_kpc': 4.0,
    'hz_thick_disk_kpc': 0.8,
    'M_bulge_solar': 3e9,
    'R_bulge_kpc': 1.0,
    'M_gas_solar': 3e9,
    'R_gas_kpc': 7.0,
    'hz_gas_kpc': 0.2,
    'rho_c_solar_kpc3': 1e13,
    'gamma_exp': 2.5,
    'lambda_g': 8.0
}

# Test at a few radii
R_test = cp.array([8.0, 10.0, 15.0, 20.0], dtype=cp.float32)  # kpc

print("Testing grav_color_void_safe model...")
print("=" * 50)

try:
    # Compute velocities
    v_model = v_total_kms_cupy(R_test, params, xi_type='grav_color_void_safe')
    
    print("Success! Model velocities:")
    for r, v in zip(R_test.get(), v_model.get()):
        print(f"  R = {r:5.1f} kpc: v = {v:6.2f} km/s")
    
    # Check if reasonable
    v_min, v_max = float(cp.min(v_model).get()), float(cp.max(v_model).get())
    print(f"\nVelocity range: [{v_min:.2f}, {v_max:.2f}] km/s")
    
    if v_min < 0:
        print("WARNING: Negative velocities!")
    if v_max > 500:
        print("WARNING: Very high velocities!")
    if not cp.all(cp.isfinite(v_model)):
        print("WARNING: Non-finite velocities!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()