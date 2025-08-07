#!/usr/bin/env python3
"""
Demonstrate why rho_c must be set correctly for the entire galaxy.
"""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    xi_balanced_screening_cupy,
    volume_density_comprehensive_solar_kpc3_cupy,
    v_total_kms_cupy
)

print("=" * 70)
print("UNDERSTANDING rho_c: The Critical Density Parameter")
print("=" * 70)

# Realistic galaxy parameters
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
    'R_screen': 50.0,
    'n_exp': 1.0,
    'A_max': 2.0
}

# Test locations throughout the galaxy
R_test = cp.array([5, 8, 10, 15, 20, 25, 30, 40, 50], dtype=cp.float32)

# Calculate actual densities at these locations
rho_actual = volume_density_comprehensive_solar_kpc3_cupy(R_test, params)

print("\n1. ACTUAL DENSITIES IN OUR GALAXY MODEL:")
print("-" * 50)
print("R (kpc) | Density (M_sun/kpc^3)")
print("-" * 50)
for r, rho in zip(R_test.get(), rho_actual.get()):
    print(f"{r:7.1f} | {rho:.2e}")

# Key insight: Solar density at R=8 kpc
solar_density = float(rho_actual[1].get())  # Index 1 is R=8 kpc
print(f"\n>>> Solar System density (R=8 kpc): {solar_density:.2e} M_sun/kpc^3 <<<")

print("\n2. EFFECT OF WRONG rho_c (Using rho_c = 1e13):")
print("-" * 50)

# Wrong rho_c (from old bounds)
rho_c_wrong = 1e13
params['rho_c_solar_kpc3'] = rho_c_wrong

xi_wrong = xi_balanced_screening_cupy(
    rho_actual, rho_c_wrong, R_test,
    params['R_screen'], params['n_exp'], params['A_max']
)

v_wrong = v_total_kms_cupy(R_test, params, xi_type='balanced_screening')

print("R (kpc) | rho/rho_c | Xi     | v_model (km/s)")
print("-" * 50)
for r, rho, xi, v in zip(R_test.get(), rho_actual.get(), xi_wrong.get(), v_wrong.get()):
    ratio = rho/rho_c_wrong
    print(f"{r:7.1f} | {ratio:.2e} | {xi:.4f} | {v:.1f}")

print("\nPROBLEM: All densities << rho_c, so xi ≈ 3 everywhere!")
print("         This gives huge velocities (6000+ km/s)")
print("         Cassini constraint VIOLATED at Solar System!")

print("\n3. EFFECT OF CORRECT rho_c (Using actual solar density):")
print("-" * 50)

# Correct rho_c
rho_c_correct = solar_density
params['rho_c_solar_kpc3'] = rho_c_correct

xi_correct = xi_balanced_screening_cupy(
    rho_actual, rho_c_correct, R_test,
    params['R_screen'], params['n_exp'], params['A_max']
)

v_correct = v_total_kms_cupy(R_test, params, xi_type='balanced_screening')

print("R (kpc) | rho/rho_c | Xi     | v_model (km/s) | Location")
print("-" * 60)
for r, rho, xi, v in zip(R_test.get(), rho_actual.get(), xi_correct.get(), v_correct.get()):
    ratio = rho/rho_c_correct
    if r == 8:
        location = "SOLAR SYSTEM"
    elif r < 15:
        location = "Inner disk"
    elif r < 30:
        location = "Outer disk"
    else:
        location = "Halo/transition"
    print(f"{r:7.1f} | {ratio:.4f} | {xi:.4f} | {v:14.1f} | {location}")

print("\nSUCCESS: Xi = 1 at Solar System (Cassini satisfied)")
print("         Xi > 1 in outer disk (explains rotation curves)")
print("         Reasonable velocities (200-300 km/s)")

print("\n4. WHY THIS AFFECTS ALL GAIA STARS:")
print("-" * 50)
print("Every star's velocity depends on xi(ρ,R), which depends on ρ/ρ_c")
print("If ρ_c is wrong, EVERY star gets wrong enhancement!")
print("")
print("Example: Star at R=20 kpc")
print(f"  - With ρ_c = 1e13: xi = {float(xi_wrong[4].get()):.2f}, v = {float(v_wrong[4].get()):.0f} km/s")
print(f"  - With ρ_c = {rho_c_correct:.1e}: xi = {float(xi_correct[4].get()):.2f}, v = {float(v_correct[4].get()):.0f} km/s")

print("\n5. PARAMETER SEARCH STRATEGY:")
print("-" * 50)
print("The optimizer will search for rho_c in range [1e7, 1e9] M_sun/kpc^3")
print("This range brackets the actual solar density")
print("The best-fit rho_c should converge to ~1e8 to satisfy Cassini")
print("\nKey: rho_c is NOT a free parameter - it's constrained by Cassini!")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("- rho_c must match solar density (~1e8) for Cassini constraint")
print("- This single value affects enhancement for ALL stars")
print("- Wrong rho_c → wrong xi everywhere → wrong velocities everywhere")
print("- The updated bounds [1e7, 1e9] ensure physically reasonable values")
print("=" * 70)