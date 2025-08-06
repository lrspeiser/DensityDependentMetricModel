#!/usr/bin/env python3
"""
adjust.py - A diagnostic script to test how parameter adjustments affect the DDMM rotation curve.
"""
import numpy as np

# --- IMPORTANT: UPDATE THIS PATH ---
DYNESTY_RESULTS_FILE = 'chains_dynesty/mw_grav_color_DTf_DKf_Bf_Gf_20250716/dynesty_mw_grav_color_Bf_DTf_DKf_Gf_results_samples_FIXED.npz'
# -----------------------------------

# Import the functions we know exist
from density_metric2 import (
    v_baryon_total_newtonian_kms,
    rho_baryon_total_midplane_solar_kpc3,
    XI_FUNCTION_MAP
)

# 1. Load current parameters
print(f"Loading dynesty results from: {DYNESTY_RESULTS_FILE}")
data = np.load(DYNESTY_RESULTS_FILE, allow_pickle=True)
param_names = list(data.get('param_names', data.get('paramnames')))
weights = np.exp(data['logwt'] - data['logz'][-1]) if 'logwt' in data else np.ones(len(data['samples'])) / len(data['samples'])
median_params = np.average(data['samples'], weights=weights, axis=0)
params_base = dict(zip(param_names, median_params))

# 2. Add the necessary boolean flags for the calculation functions
params_base['include_disk_thin'] = 'M_disk_thin_solar' in params_base
params_base['include_disk_thick'] = 'M_disk_thick_solar' in params_base
params_base['include_bulge'] = 'M_bulge_solar' in params_base
params_base['include_gas'] = 'M_gas_solar' in params_base

# 3. Define the correct way to calculate the DDMM velocity
def calculate_ddmm_velocity(r_kpc, params):
    """Calculates the DDMM velocity using the correct formula."""
    v_newton = v_baryon_total_newtonian_kms(r_kpc, params)
    rho = rho_baryon_total_midplane_solar_kpc3(r_kpc, params)
    xi_func = XI_FUNCTION_MAP['power']
    
    n_key = 'gamma_exp' if 'gamma_exp' in params else 'n_exp'
    A_key = 'lambda_g' if 'lambda_g' in params else 'A'
    
    xi = xi_func(rho, params['rho_c_solar_kpc3'], params[n_key], params.get(A_key, 1.0))
    xi = np.minimum(xi, 5.0)
    
    return v_newton * np.sqrt(xi)

# --- Start of Analysis ---
print("\nCurrent parameters:")
A_key = 'lambda_g' if 'lambda_g' in params_base else 'A'
n_key = 'gamma_exp' if 'gamma_exp' in params_base else 'n_exp'

print(f"  A ({A_key}) = {params_base.get(A_key):.2f}")
print(f"  n ({n_key}) = {params_base.get(n_key):.2f}")
print(f"  rho_c = {params_base['rho_c_solar_kpc3']:.2e}")

# Calculate what scaling we need
v_target = 231  # Data at R=8 kpc
v_current = calculate_ddmm_velocity(np.array([8.0]), params_base)[0]
scaling_needed = (v_target / v_current)**2
print(f"\nModel V at 8kpc: {v_current:.1f} km/s")
print(f"Target V at 8kpc: {v_target:.1f} km/s")
print(f"Need to scale total ξ by a factor of: {scaling_needed:.3f}")

# --- Test Different Adjustments ---
print("\n--- Testing Adjustments to Achieve Target ---")
test_radii = np.array([6, 8, 10, 12, 15])

# Option 1: Adjust 'A' (lambda_g)
# The enhancement is proportional to A, so to scale xi, we scale A by the same factor.
params_adj1 = params_base.copy()
params_adj1[A_key] = params_base[A_key] * scaling_needed

# Option 2: Adjust 'n' (gamma_exp)
# A more complex relationship, but generally decreasing n decreases xi. Let's try a guess.
params_adj2 = params_base.copy()
params_adj2[n_key] = params_base[n_key] * 0.7 # Guess: reduce by 30%

# Option 3: Adjust 'rho_c'
# Enhancement is prop to (rho_c/rho)^n. To decrease xi, we increase rho_c.
# (rho_c_new/rho_c_old)^n = scaling_needed => rho_c_new = rho_c_old * scaling_needed^(1/n)
rho_c_scaling = scaling_needed**(1 / params_base[n_key])
params_adj3 = params_base.copy()
params_adj3['rho_c_solar_kpc3'] = params_base['rho_c_solar_kpc3'] / rho_c_scaling # Note: inverse relationship in formula

# Test each adjustment
for i, params_test in enumerate([params_adj1, params_adj2, params_adj3]):
    v_test = calculate_ddmm_velocity(test_radii, params_test)
    print(f"\nOption {i+1}: V at 8 kpc = {v_test[1]:.1f} km/s")
    if i == 0: print(f"  (Scaled {A_key} by {scaling_needed:.2f})")
    elif i == 1: print(f"  (Scaled {n_key} by 0.7)")
    else: print(f"  (Scaled rho_c by {1/rho_c_scaling:.2f})")