#!/usr/bin/env python3
"""Check what xi values are being produced."""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    xi_balanced_screening_cupy,
    volume_density_comprehensive_solar_kpc3_cupy
)

# Test at various radii
R_test = cp.array([5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30], dtype=cp.float32)

# Parameters
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
}

# Calculate densities
rho = volume_density_comprehensive_solar_kpc3_cupy(R_test, params)

# Test with calibrated rho_c
rho_c = 7.18e7  # From actual solar density
R_screen = 50.0
n_exp = 1.0
A_max = 2.0

xi = xi_balanced_screening_cupy(rho, rho_c, R_test, R_screen, n_exp, A_max)

print("Xi Enhancement Analysis")
print("=" * 60)
print("R (kpc) | Density     | rho/rho_c | Xi     | Enhancement")
print("-" * 60)

for i in range(len(R_test)):
    r = float(R_test[i].get())
    d = float(rho[i].get())
    ratio = d / rho_c
    x = float(xi[i].get())
    enh = (x - 1) * 100
    
    marker = ""
    if r == 8:
        marker = " <- SOLAR (should be ~1.0)"
    elif x > 2.5:
        marker = " <- Too high?"
    
    print(f"{r:7.1f} | {d:.3e} | {ratio:9.4f} | {x:.4f} | {enh:+7.1f}%{marker}")

print("\nProblem Analysis:")
if float(xi[3].get()) > 1.1:  # Index 3 is R=8 kpc
    print("! Cassini violation: Xi at solar position should be 1.0")
    print("  This means rho_c doesn't match the actual solar density")
    
print("\nVelocity Impact:")
print("If xi ~ 3 everywhere, velocities will be sqrt(3) ~ 1.73x too high")
print("But we're seeing 20x too high, suggesting the mass model is also wrong")