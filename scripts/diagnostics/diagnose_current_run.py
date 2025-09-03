#!/usr/bin/env python3
"""
Diagnose why the balanced screening model is getting poor likelihood values.
"""

import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    v_total_kms_cupy,
    volume_density_comprehensive_solar_kpc3_cupy
)
from core.data_io import process_gaia_data
import pandas as pd

print("=" * 70)
print("DIAGNOSING BALANCED SCREENING MODEL PERFORMANCE")
print("=" * 70)

# Load a sample of actual Gaia data
print("\n1. Loading actual Gaia data sample...")
cache_file = "external_data/gaia_sky_slices/all_sky_gaia.csv"
gaia_df = pd.read_csv(cache_file)
if 'R_kpc' not in gaia_df.columns:
    gaia_df = process_gaia_data(gaia_df)

# Sample 100 stars for quick testing
sample_size = 100
sample_indices = np.random.choice(len(gaia_df), sample_size, replace=False)
R_sample = gaia_df.iloc[sample_indices]['R_kpc'].values
v_observed = gaia_df.iloc[sample_indices]['v_obs'].values
sigma_obs = gaia_df.iloc[sample_indices]['sigma_v'].values

print(f"Sampled {sample_size} stars")
print(f"R range: {R_sample.min():.1f} - {R_sample.max():.1f} kpc")
print(f"v_obs range: {v_observed.min():.1f} - {v_observed.max():.1f} km/s")

# Convert to CuPy
R_cupy = cp.array(R_sample, dtype=cp.float32)
v_obs_cupy = cp.array(v_observed, dtype=cp.float32)
sigma_cupy = cp.array(sigma_obs, dtype=cp.float32)

print("\n2. Testing different parameter configurations...")
print("-" * 70)

# Test configurations
test_configs = [
    {
        'name': 'Original (rho_c too high)',
        'params': {
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
            'rho_c_solar_kpc3': 1e13,  # WAY too high!
            'R_screen': 50.0,
            'n_exp': 1.0,
            'A_max': 2.0
        }
    },
    {
        'name': 'Auto-calibrated rho_c',
        'params': None  # Will calculate
    },
    {
        'name': 'Conservative (A_max=1.5)',
        'params': None  # Will calculate with A_max=1.5
    }
]

# Calculate actual density at R=8 kpc for calibration
R_solar = cp.array([8.0], dtype=cp.float32)
base_params = test_configs[0]['params'].copy()
rho_solar = volume_density_comprehensive_solar_kpc3_cupy(R_solar, base_params)
rho_solar_value = float(rho_solar[0].get())
print(f"\nActual density at R=8 kpc: {rho_solar_value:.2e} M_sun/kpc^3")

# Fill in auto-calibrated configs
test_configs[1]['params'] = base_params.copy()
test_configs[1]['params']['rho_c_solar_kpc3'] = rho_solar_value

test_configs[2]['params'] = base_params.copy()
test_configs[2]['params']['rho_c_solar_kpc3'] = rho_solar_value
test_configs[2]['params']['A_max'] = 1.5

print("\n3. Running tests...")
print("-" * 70)
print("Config Name               | Chi2/DOF | LogL       | v_mean  | v_max   | Status")
print("-" * 70)

for config in test_configs:
    params = config['params']
    
    # Calculate model velocities
    try:
        v_model = v_total_kms_cupy(R_cupy, params, xi_type='balanced_screening')
        
        # Calculate chi-squared
        chi2 = cp.sum(((v_obs_cupy - v_model) / sigma_cupy)**2)
        chi2_value = float(chi2.get())
        chi2_per_dof = chi2_value / sample_size
        logl = -0.5 * chi2_value
        
        # Get velocity statistics
        v_mean = float(cp.mean(v_model).get())
        v_max = float(cp.max(v_model).get())
        v_min = float(cp.min(v_model).get())
        
        # Status
        if v_max > 1000:
            status = "BAD: v too high!"
        elif chi2_per_dof > 100:
            status = "BAD: poor fit"
        elif chi2_per_dof > 10:
            status = "POOR"
        elif chi2_per_dof > 2:
            status = "OK"
        else:
            status = "GOOD"
        
        print(f"{config['name']:25s} | {chi2_per_dof:8.1f} | {logl:10.1f} | {v_mean:7.1f} | {v_max:7.1f} | {status}")
        
    except Exception as e:
        print(f"{config['name']:25s} | ERROR: {str(e)[:40]}")

print("\n4. Parameter recommendations:")
print("-" * 70)

# Calculate recommended bounds
rho_c_recommended = rho_solar_value
rho_c_min = rho_c_recommended * 0.5
rho_c_max = rho_c_recommended * 2.0

print(f"rho_c should be in range [{rho_c_min:.2e}, {rho_c_max:.2e}]")
print(f"Current bounds in code: [1e7, 1e9]")

if rho_c_recommended < 1e7 or rho_c_recommended > 1e9:
    print("WARNING: Current bounds don't include the optimal rho_c!")
    print(f"RECOMMENDATION: Update bounds to [{rho_c_min:.2e}, {rho_c_max:.2e}]")
else:
    print("Current bounds are appropriate.")

print("\n5. Quick parameter space scan:")
print("-" * 70)

# Scan A_max values
print("A_max | Best Chi2/DOF | Notes")
print("-" * 40)

for A_max_test in [1.2, 1.5, 2.0, 2.5, 3.0]:
    params_test = base_params.copy()
    params_test['rho_c_solar_kpc3'] = rho_solar_value
    params_test['A_max'] = A_max_test
    
    v_model = v_total_kms_cupy(R_cupy, params_test, xi_type='balanced_screening')
    chi2 = cp.sum(((v_obs_cupy - v_model) / sigma_cupy)**2)
    chi2_per_dof = float(chi2.get()) / sample_size
    
    v_max = float(cp.max(v_model).get())
    if v_max > 500:
        notes = f"v_max={v_max:.0f} km/s (too high)"
    elif v_max < 150:
        notes = f"v_max={v_max:.0f} km/s (too low)"
    else:
        notes = f"v_max={v_max:.0f} km/s (good)"
    
    print(f"{A_max_test:5.1f} | {chi2_per_dof:13.1f} | {notes}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
print("\nKey findings:")
print("1. If velocities are >1000 km/s, rho_c is likely too high")
print("2. The model needs rho_c calibrated to actual solar density")
print("3. A_max may need to be lower (1.5-2.0) for realistic velocities")
print("4. Check that parameter bounds include the optimal values")