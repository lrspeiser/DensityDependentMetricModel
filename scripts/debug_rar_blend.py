#!/usr/bin/env python3
"""
Debug RAR Blend model to see what xi enhancement values it's producing.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.density_metric_cupy import v_total_kms_cupy
import cupy as cp

# Load RAR Blend best parameters
data = np.load('runs/rar_blend_20250823_211648/posterior_samples.npz', allow_pickle=True)
best_idx = np.argmax(data['logl'])
params = dict(zip(data['param_names'], data['samples'][best_idx]))
params['allow_experimental'] = True

print("RAR Blend Parameters:")
print(f"  a0_m_s2: {params['a0_m_s2']:.3e}")
print(f"  A_excess: {params['A_excess']:.6f}")
print(f"  lambda_cap: {params['lambda_cap']:.3f}")
print(f"  T0: {params['T0']:.3f}")
print(f"  sigma_lnT: {params['sigma_lnT']:.3f}")
print(f"  wmin: {params['wmin']:.4f}")

# Test at different radii
test_R = np.array([5, 8, 10, 12, 15, 20])
print("\nTesting enhancement at different radii:")

# First compute with RAR Blend
v_rar = v_total_kms_cupy(cp.asarray(test_R), params, xi_type='rar_blend')
v_rar_np = cp.asnumpy(v_rar)

# Now compute with GR (xi=1)
params_gr = dict(params)
v_gr = v_total_kms_cupy(cp.asarray(test_R), params_gr, xi_type='gr')
v_gr_np = cp.asnumpy(v_gr)

# Compute effective xi
xi_eff = (v_rar_np / v_gr_np)**2

print("\nR(kpc) | v_GR(km/s) | v_RAR(km/s) | xi_eff | Enhancement(%)")
print("-" * 65)
for i, r in enumerate(test_R):
    enhancement = (xi_eff[i] - 1) * 100
    print(f"{r:6.1f} | {v_gr_np[i]:10.1f} | {v_rar_np[i]:11.1f} | {xi_eff[i]:6.3f} | {enhancement:13.1f}")

# Check the RAR function directly
print("\n" + "="*65)
print("Direct xi calculation check:")

# Constants
ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14

# Compute baryonic properties at test radii
from core.density_metric_cupy import v_baryon_comprehensive_kms_cupy
v_bar = v_baryon_comprehensive_kms_cupy(cp.asarray(test_R), params)
v_bar_np = cp.asnumpy(v_bar)

# Compute g_bar
g_bar = ACC_M_S2_PER_KMS2_PER_KPC * v_bar_np**2 / test_R

# Compute RAR function D_RAR
a0 = params['a0_m_s2']
x = np.sqrt(g_bar / a0)
D_RAR = 1 / (1 - np.exp(-x))
excess = D_RAR - 1

# Compute tidal T
T = v_bar_np**2 / test_R**2
T0 = params['T0']
sigma_lnT = params['sigma_lnT']
wmin = params['wmin']

# Tidal window
u = (np.log(T) - np.log(T0)) / sigma_lnT
W = wmin + (1 - wmin) * np.exp(-0.5 * u**2)

# Final xi
A_excess = params['A_excess']
lambda_cap = params['lambda_cap']
xi_calc = 1 + A_excess * excess * W
xi_calc = np.clip(xi_calc, 1, 1 + lambda_cap)

print("\nR(kpc) | g_bar/a0 | D_RAR | excess | W(T) | xi_calc | xi_max")
print("-" * 75)
for i, r in enumerate(test_R):
    print(f"{r:6.1f} | {(g_bar[i]/a0):8.3f} | {D_RAR[i]:5.2f} | {excess[i]:6.2f} | {W[i]:4.3f} | {xi_calc[i]:7.4f} | {1+lambda_cap:.3f}")

print(f"\nProblem diagnosed: A_excess = {A_excess:.6f} is too small!")
print(f"Even with large RAR excess (~{np.max(excess):.1f}) and lambda_cap = {lambda_cap:.1f},")
print(f"the product A_excess * excess * W ≈ {A_excess * np.max(excess) * 1:.6f}")
print(f"This means xi ≈ 1 + {A_excess * np.max(excess):.6f} ≈ 1.000")
