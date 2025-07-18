#!/usr/bin/env python3
"""
check.py - A quick diagnostic script to test DDMM model parameters against Gaia data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# --- IMPORTANT: UPDATE THESE TWO PATHS ---
DYNESTY_RESULTS_FILE = 'chains_dynesty/mw_grav_color_DTf_DKf_Bf_Gf_20250716/dynesty_mw_grav_color_Bf_DTf_DKf_Gf_results_samples_FIXED.npz'
GAIA_SLICES_DIRECTORY = 'gaia_sky_slices'
# -----------------------------------------

# Import the functions we know exist
from density_metric2 import (
    v_baryon_total_newtonian_kms,
    rho_baryon_total_midplane_solar_kpc3,
    XI_FUNCTION_MAP
)

def load_gaia_slices(cache_dir: str) -> Optional[pd.DataFrame]:
    """Helper to load and combine cached Gaia data slices."""
    cache_path = Path(cache_dir)
    slice_files = list(cache_path.glob("processed_*.parquet"))
    if not slice_files:
        print(f"❌ Error: No processed .parquet files found in '{cache_dir}'.")
        return None
    df_list = [pd.read_parquet(f) for f in slice_files]
    return pd.concat(df_list, ignore_index=True)

# 1. Load your model parameters
print(f"Loading dynesty results from: {DYNESTY_RESULTS_FILE}")
data = np.load(DYNESTY_RESULTS_FILE, allow_pickle=True)
param_names = list(data.get('param_names', data.get('paramnames')))
weights = np.exp(data['logwt'] - data['logz'][-1]) if 'logwt' in data else np.ones(len(data['samples'])) / len(data['samples'])
median_params = np.average(data['samples'], weights=weights, axis=0)
params = dict(zip(param_names, median_params))

# 2. Add the necessary boolean flags for the calculation functions
print("Configuring model components...")
params['include_disk_thin'] = 'M_disk_thin_solar' in params
params['include_disk_thick'] = 'M_disk_thick_solar' in params
params['include_bulge'] = 'M_bulge_solar' in params
params['include_gas'] = 'M_gas_solar' in params

# 3. Define the correct way to calculate the DDMM velocity
def calculate_ddmm_velocity(r_kpc, params):
    """Calculates the DDMM velocity using the correct formula."""
    v_newton = v_baryon_total_newtonian_kms(r_kpc, params)
    rho = rho_baryon_total_midplane_solar_kpc3(r_kpc, params)
    xi_func = XI_FUNCTION_MAP['power']
    
    # Use robust keys
    n_key = 'gamma_exp' if 'gamma_exp' in params else 'n_exp'
    A_key = 'lambda_g' if 'lambda_g' in params else 'A'
    
    xi = xi_func(rho, params['rho_c_solar_kpc3'], params[n_key], params.get(A_key, 1.0))
    xi = np.minimum(xi, 5.0)
    
    return v_newton * np.sqrt(xi)

# 4. Compute model at a few test radii
print("\n--- Model Predictions ---")
test_radii = np.array([6, 8, 10, 12, 15])
v_model = calculate_ddmm_velocity(test_radii, params)

for r, v in zip(test_radii, v_model):
    print(f"Radius = {r:2d} kpc  |  Predicted V_rot = {v:.1f} km/s")
print("(Goal is ~220-240 km/s for a flat rotation curve)")

# 5. Load your actual data and check it
print("\n--- Comparison with Data ---")
gaia_df = load_gaia_slices(GAIA_SLICES_DIRECTORY)

if gaia_df is not None:
    # Use the correct key names
    r_data_key = 'R_kpc'
    v_data_key = 'v_obs'
    
    # Check what velocities your data actually has near the Sun
    mask = (gaia_df[r_data_key] > 7.5) & (gaia_df[r_data_key] < 8.5)
    
    if mask.sum() > 0:
        median_v = np.median(gaia_df[v_data_key][mask])
        std_v = np.std(gaia_df[v_data_key][mask])
        print(f"Actual data velocities at R ~ 8 kpc: {median_v:.1f} ± {std_v:.1f} km/s")
        
        # Compare model to data at R=8kpc
        v_model_sun = calculate_ddmm_velocity(np.array([8.0]), params)[0]
        print(f"Model prediction at R = 8 kpc:       {v_model_sun:.1f} km/s")
        print(f"Model is currently overshooting by:   {v_model_sun - median_v:.1f} km/s")
    else:
        print("Could not find data near R=8kpc to compare against.")
else:
    print("Could not load Gaia data to perform comparison.")