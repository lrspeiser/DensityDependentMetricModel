#!/usr/bin/env python3
"""Debug the balanced screening model to understand why velocities are too high."""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    xi_balanced_screening_cupy,
    volume_density_comprehensive_solar_kpc3_cupy
)

# Test parameters
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
    'R_screen': 50.0,
    'n_exp': 1.0,
    'A_max': 2.0
}

# Test at solar position
R_test = cp.array([8.0], dtype=cp.float32)

# Calculate density at solar position
rho_solar = volume_density_comprehensive_solar_kpc3_cupy(R_test, params)

print("Debugging Balanced Screening Model")
print("=" * 50)
print(f"At R = 8 kpc (Solar position):")
print(f"  Density: {float(rho_solar[0].get()):.2e} M_sun/kpc^3")
print(f"  Critical density: {params['rho_c_solar_kpc3']:.2e} M_sun/kpc^3")
print(f"  Density ratio: {float(rho_solar[0].get())/params['rho_c_solar_kpc3']:.6f}")

# Calculate xi
xi_solar = xi_balanced_screening_cupy(
    rho_solar,
    params['rho_c_solar_kpc3'],
    R_test,
    params['R_screen'],
    params['n_exp'],
    params['A_max']
)

print(f"  Xi value: {float(xi_solar[0].get()):.6f}")

# The problem: if rho_solar < rho_c, we get enhancement at solar position!
# This violates Cassini constraint

print("\nProblem identified:")
if float(rho_solar[0].get()) < params['rho_c_solar_kpc3']:
    print(f"  Solar density ({float(rho_solar[0].get()):.2e}) < Critical density ({params['rho_c_solar_kpc3']:.2e})")
    print("  This causes enhancement at solar position, violating Cassini!")
    print("\nSolution: Set rho_c to match actual solar density")
    
    # Fix: Use actual solar density
    rho_c_fixed = float(rho_solar[0].get())
    print(f"\nUsing rho_c = {rho_c_fixed:.2e} M_sun/kpc^3")
    
    xi_fixed = xi_balanced_screening_cupy(
        rho_solar,
        rho_c_fixed,
        R_test,
        params['R_screen'],
        params['n_exp'],
        params['A_max']
    )
    
    print(f"Fixed Xi value: {float(xi_fixed[0].get()):.6f}")
    
    # Test at different radii with fixed rho_c
    print("\nTesting with corrected rho_c:")
    print("-" * 40)
    
    R_test_array = cp.array([5, 8, 10, 15, 20, 25, 30, 40, 50], dtype=cp.float32)
    rho_test = volume_density_comprehensive_solar_kpc3_cupy(R_test_array, params)
    
    xi_test = xi_balanced_screening_cupy(
        rho_test,
        rho_c_fixed,
        R_test_array,
        params['R_screen'],
        params['n_exp'],
        params['A_max']
    )
    
    print("R (kpc) | Density    | rho/rho_c | Xi")
    print("-" * 40)
    for r, rho, xi in zip(R_test_array.get(), rho_test.get(), xi_test.get()):
        print(f"{r:7.1f} | {rho:.2e} | {rho/rho_c_fixed:.4f} | {xi:.4f}")