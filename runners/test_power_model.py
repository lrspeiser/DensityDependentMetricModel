#!/usr/bin/env python3
"""
test_power_model.py - Direct test of the power model with CuPy.

This script tests the power model xi function and velocity calculation directly.
"""

import numpy as np
import cupy as cp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CuPy model functions
from core.density_metric_cupy import (
    v_baryon_total_newtonian_kms_cupy,
    volume_density_total_midplane_solar_kpc3_cupy,
    xi_power_law_cupy,
    DEFAULT_DTYPE
)

print("Testing power model xi function...")

# Test parameters
R_test = cp.array([2.0, 5.0, 8.0, 12.0, 20.0], dtype=DEFAULT_DTYPE)
rho_c = 1e8
n_exp = 2.7
A = 1.5

# Calculate density at test points
M_disk = 4.0e10
R_d = 2.6
hz_disk = 0.3
M_bulge = 1.2e10
R_b = 0.7
M_gas = 3.0e10
R_gas = 7.0
hz_gas = 0.15

print(f"\nTest radii: {R_test}")

# Calculate density
rho = volume_density_total_midplane_solar_kpc3_cupy(
    R_test, M_disk, R_d, hz_disk, M_bulge, R_b, True, M_gas, R_gas, hz_gas, True
)
print(f"Densities: {rho}")

# Calculate xi
xi = xi_power_law_cupy(rho, rho_c, n_exp, A)
print(f"Xi values: {xi}")

# Check for invalid values
if cp.any(cp.isnan(xi)):
    print("ERROR: Xi contains NaN values!")
elif cp.any(cp.isinf(xi)):
    print("ERROR: Xi contains inf values!")
elif cp.any(xi < 0):
    print("ERROR: Xi contains negative values!")
else:
    print("Xi values look valid")

# Calculate Newtonian velocity
v_newton = v_baryon_total_newtonian_kms_cupy(
    R_test, M_disk, R_d, M_bulge, R_b, True, M_gas, R_gas, True
)
print(f"\nNewtonian velocities: {v_newton}")

# Calculate modified velocity
v_modified = v_newton * cp.sqrt(cp.maximum(xi, 0.0))
print(f"Modified velocities: {v_modified}")

# Test with extreme parameters
print("\n--- Testing with extreme parameters ---")
rho_extreme = cp.array([1e5, 1e8, 1e10, 1e12], dtype=DEFAULT_DTYPE)
print(f"Extreme densities: {rho_extreme}")

xi_extreme = xi_power_law_cupy(rho_extreme, rho_c, n_exp, A)
print(f"Xi values for extreme densities: {xi_extreme}")

# Test parameter sensitivity
print("\n--- Testing parameter sensitivity ---")
test_rho = 1e9
print(f"Test density: {test_rho}")

for test_n in [0.5, 1.0, 2.0, 3.0, 4.0]:
    xi_test = xi_power_law_cupy(cp.array([test_rho], dtype=DEFAULT_DTYPE), rho_c, test_n, A)
    print(f"  n={test_n}: xi={float(xi_test[0]):.4f}")

print("\n--- Testing A parameter ---")
for test_A in [0.1, 0.5, 1.0, 2.0, 5.0]:
    xi_test = xi_power_law_cupy(cp.array([test_rho], dtype=DEFAULT_DTYPE), rho_c, n_exp, test_A)
    print(f"  A={test_A}: xi={float(xi_test[0]):.4f}")

print("\nTest complete!")
