#!/usr/bin/env python3
"""
Simple diagnostic to test xi functions directly.
"""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    xi_elastic_strain_cupy,
    xi_hookean_potential_cupy,
    xi_tension_field_cupy,
    xi_power_law_cupy
)

print("Testing Xi Functions")
print("="*60)

# Test density range
rho_test = cp.logspace(2, 12, 20)  # From 1e2 to 1e12 M_sun/kpc^3
R_test = cp.ones_like(rho_test) * 8.0  # Solar radius

# Test each function
print("\n1. POWER LAW (known working)")
print("-"*40)
try:
    rho_c = 1e15
    n = 1.2
    xi = xi_power_law_cupy(rho_test, rho_c, n)
    xi_np = cp.asnumpy(xi)
    print(f"  Min xi: {np.min(xi_np):.3f}")
    print(f"  Max xi: {np.max(xi_np):.3f}")
    print(f"  At rho=1e9: xi = {xi_np[np.argmin(np.abs(cp.asnumpy(rho_test) - 1e9))]:.6f}")
    print("  Status: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n2. ELASTIC STRAIN")
print("-"*40)
try:
    params = {
        'relaxation_scale': 1.0,
        'strain_critical': 10.0,
        'k_elastic': 0.5,
        'rho_solar': 1e9
    }
    rho_c = 1e8
    xi = xi_elastic_strain_cupy(rho_test, rho_c, params)
    xi_np = cp.asnumpy(xi)
    print(f"  Min xi: {np.min(xi_np):.3f}")
    print(f"  Max xi: {np.max(xi_np):.3f}")
    print(f"  At rho=1e9: xi = {xi_np[np.argmin(np.abs(cp.asnumpy(rho_test) - 1e9))]:.6f}")
    
    # Check for problems
    if np.any(np.isnan(xi_np)):
        print("  WARNING: Contains NaN values!")
    if np.any(np.isinf(xi_np)):
        print("  WARNING: Contains Inf values!")
    if np.any(xi_np < 0):
        print("  WARNING: Contains negative xi values (unphysical)!")
    if np.max(xi_np) > 1e6:
        print(f"  WARNING: Extremely large xi values (max = {np.max(xi_np):.2e})!")
    
    print("  Status: CHECK WARNINGS" if np.max(xi_np) > 100 else "  Status: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n3. HOOKEAN")
print("-"*40)
try:
    params = {
        'k_spacetime': 0.1,
        'rho_equilibrium': 1e9,
        'stress_break': 100.0
    }
    xi = xi_hookean_potential_cupy(rho_test, R_test, params)
    xi_np = cp.asnumpy(xi)
    print(f"  Min xi: {np.min(xi_np):.3f}")
    print(f"  Max xi: {np.max(xi_np):.3f}")
    print(f"  At rho=1e9: xi = {xi_np[np.argmin(np.abs(cp.asnumpy(rho_test) - 1e9))]:.6f}")
    
    # Check for problems
    if np.any(np.isnan(xi_np)):
        print("  WARNING: Contains NaN values!")
    if np.any(np.isinf(xi_np)):
        print("  WARNING: Contains Inf values!")
    if np.any(xi_np < 0):
        print("  WARNING: Contains negative xi values (unphysical)!")
    if np.max(xi_np) > 1e6:
        print(f"  WARNING: Extremely large xi values (max = {np.max(xi_np):.2e})!")
    
    print("  Status: CHECK WARNINGS" if np.max(xi_np) > 100 else "  Status: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n4. TENSION FIELD")
print("-"*40)
try:
    params = {
        'rho_relaxation': 1e8,
        'tension_max': 5.0,
        'R_snap': 25.0
    }
    xi = xi_tension_field_cupy(rho_test, R_test, params)
    xi_np = cp.asnumpy(xi)
    print(f"  Min xi: {np.min(xi_np):.3f}")
    print(f"  Max xi: {np.max(xi_np):.3f}")
    print(f"  At rho=1e9: xi = {xi_np[np.argmin(np.abs(cp.asnumpy(rho_test) - 1e9))]:.6f}")
    
    # Check for problems
    if np.any(np.isnan(xi_np)):
        print("  WARNING: Contains NaN values!")
    if np.any(np.isinf(xi_np)):
        print("  WARNING: Contains Inf values!")
    if np.any(xi_np < 0):
        print("  WARNING: Contains negative xi values (unphysical)!")
    if np.max(xi_np) > 1e6:
        print(f"  WARNING: Extremely large xi values (max = {np.max(xi_np):.2e})!")
    
    print("  Status: CHECK WARNINGS" if np.max(xi_np) > 100 else "  Status: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60)
print("SOLAR SYSTEM CONSTRAINT CHECK")
print("="*60)
print("For Solar System tests to pass, |xi - 1| < 1e-5 at rho ~ 1e9 M_sun/kpc^3")
print("\nActual values at solar density:")
solar_idx = np.argmin(np.abs(cp.asnumpy(rho_test) - 1e9))

for name, func, params in [
    ("Power Law", lambda r: xi_power_law_cupy(r, 1e15, 1.2), {}),
    ("Elastic Strain", lambda r: xi_elastic_strain_cupy(r, 1e8, {'relaxation_scale': 1.0, 'strain_critical': 10.0, 'k_elastic': 0.5, 'rho_solar': 1e9}), {}),
    ("Hookean", lambda r: xi_hookean_potential_cupy(r, R_test, {'k_spacetime': 0.1, 'rho_equilibrium': 1e9, 'stress_break': 100.0}), {}),
    ("Tension Field", lambda r: xi_tension_field_cupy(r, R_test, {'rho_relaxation': 1e8, 'tension_max': 5.0, 'R_snap': 25.0}), {})
]:
    try:
        xi = func(rho_test)
        xi_solar = cp.asnumpy(xi)[solar_idx]
        enhancement = xi_solar - 1.0
        passes = abs(enhancement) < 1e-5
        print(f"  {name:15s}: xi - 1 = {enhancement:+.2e} {'✓ PASS' if passes else '✗ FAIL'}")
    except:
        print(f"  {name:15s}: ERROR")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("1. Models with xi >> 1 at solar density will fail Solar System tests")
print("2. Models with extremely large xi values will cause numerical issues")
print("3. Check if parameter bounds allow xi to be close to 1 at high densities")