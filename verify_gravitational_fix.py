#!/usr/bin/env python3
"""
Verify that the gravitational constant fix resolves the velocity issues.
"""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import v_total_kms_cupy

print("=" * 70)
print("VERIFYING GRAVITATIONAL CONSTANT FIX")
print("=" * 70)

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
    'rho_c_solar_kpc3': 7e7,
    'R_screen': 50.0,
    'n_exp': 1.0,
    'A_max': 2.0
}

# Test at various radii
R_test = cp.array([5, 8, 10, 15, 20, 25, 30], dtype=cp.float32)

print("\n1. Testing with GR (no enhancement):")
print("-" * 50)
v_gr = v_total_kms_cupy(R_test, params, xi_type='gr')

print("R (kpc) | v_GR (km/s) | Status")
print("-" * 40)
for r, v in zip(R_test.get(), v_gr.get()):
    status = "OK" if 100 < v < 300 else "WARNING"
    print(f"{r:7.1f} | {v:11.1f} | {status}")

print("\n2. Testing with balanced screening:")
print("-" * 50)
v_balanced = v_total_kms_cupy(R_test, params, xi_type='balanced_screening')

print("R (kpc) | v_model (km/s) | Status")
print("-" * 40)
for r, v in zip(R_test.get(), v_balanced.get()):
    status = "OK" if 100 < v < 400 else "WARNING"
    print(f"{r:7.1f} | {v:14.1f} | {status}")

# Calculate what the old (wrong) velocities would have been
print("\n3. Comparison with old (wrong) constant:")
print("-" * 50)
print("The gravitational constant was 4.302e-3 (wrong)")
print("Now it's 4.302e-6 (correct)")
print("This is a factor of 1000 difference!")
print("\nExpected reduction in velocity: sqrt(1000) ≈ 31.6x")

v_max_new = float(cp.max(v_balanced).get())
v_max_old_estimate = v_max_new * 31.6

print(f"\nOld max velocity (estimate): {v_max_old_estimate:.0f} km/s")
print(f"New max velocity (actual):   {v_max_new:.0f} km/s")

# Final verdict
print("\n" + "=" * 70)
print("VERDICT:")
print("-" * 70)

if v_max_new < 500:
    print("✓ SUCCESS: Velocities are now in a realistic range!")
    print("  The gravitational constant fix has resolved the issue.")
else:
    print("✗ PROBLEM: Velocities are still too high.")
    print("  Further investigation needed.")

print("\nNext steps:")
print("1. Re-run the balanced screening model with full Gaia data")
print("2. The optimizer should now find reasonable parameters")
print("3. LogZ values should improve dramatically")
print("=" * 70)