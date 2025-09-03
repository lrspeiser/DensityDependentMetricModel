#!/usr/bin/env python3
"""Check the Newtonian velocities before enhancement."""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import v_total_kms_cupy

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
    'rho_c_solar_kpc3': 7.18e7,
    'R_screen': 50.0,
    'n_exp': 1.0,
    'A_max': 2.0
}

R_test = cp.array([8.0, 15.0, 25.0], dtype=cp.float32)

print("Velocity Breakdown")
print("=" * 60)

# Test with different xi types
for xi_type in ['gr', 'balanced_screening']:
    v_model = v_total_kms_cupy(R_test, params, xi_type=xi_type)
    
    print(f"\n{xi_type.upper()}:")
    print("R (kpc) | v_model (km/s)")
    print("-" * 30)
    for r, v in zip(R_test.get(), v_model.get()):
        print(f"{r:7.1f} | {v:14.1f}")

# The issue: Total mass
total_mass = params['M_thin_disk_solar'] + params['M_thick_disk_solar'] + params['M_bulge_solar'] + params['M_gas_solar']
print(f"\nTotal baryonic mass: {total_mass:.1e} M_sun")
print(f"Milky Way total mass (typical): ~6e10 M_sun")

if total_mass > 1e11:
    print("WARNING: Mass is too high!")
    
# Calculate what velocity we'd expect at R=8 kpc with correct mass
M_milky_way = 6e10  # Solar masses
G = 4.302e-6  # (km/s)^2 kpc/M_sun
v_expected = np.sqrt(G * M_milky_way / 8.0)
print(f"\nExpected v at R=8 kpc with M=6e10: {v_expected:.1f} km/s")

# What we're getting
v_actual = float(v_model[0].get())
print(f"Actual v from model: {v_actual:.1f} km/s")
print(f"Ratio: {v_actual/v_expected:.1f}x too high!")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("The velocities are too high because:")
print("1. The base Newtonian velocity is already too high")
print("2. This gets multiplied by xi (1.0-2.9)")
print("3. Result: velocities 10-20x too high")
print("\nSOLUTION: The optimizer needs to find lower masses!")
print("But if the bounds force high masses, it can't converge.")