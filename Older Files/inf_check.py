#!/usr/bin/env python3
"""
Diagnose likelihood issues using your existing cached data
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('.')
import logging
logger = logging.getLogger("run_dynesty")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


print("DIAGNOSING LIKELIHOOD ISSUES WITH EXISTING DATA")
print("=" * 60)

# Load your existing processed data directly
print("\n1. Loading your cached Gaia data...")
try:
    # Try your new cache location first
    df = pd.read_parquet('gaia_cache/gaia_disk_processed.parquet')
    print(f"   ✅ Loaded from gaia_cache/gaia_disk_processed.parquet")
except:
    try:
        # Try old cache location
        df = pd.read_parquet('gaia_query_cache_DR3_processed_for_fit.parquet')
        print(f"   ✅ Loaded from gaia_query_cache_DR3_processed_for_fit.parquet")
    except:
        print("   ❌ No cached data found!")
        sys.exit(1)

print(f"   Total stars: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# Convert to arrays for testing
R_data = df['R_kpc'].values[:1000]  # Just 1000 for testing
v_data = df['v_obs'].values[:1000]
sigma_data = df['sigma_v'].values[:1000]

print(f"\n   Test sample: {len(R_data)} stars")
print(f"   R range: {R_data.min():.1f} - {R_data.max():.1f} kpc")
print(f"   v range: {v_data.min():.1f} - {v_data.max():.1f} km/s")
print(f"   sigma_v range: {sigma_data.min():.1f} - {sigma_data.max():.1f} km/s")

# Check for data issues
print("\n2. Data quality checks:")
n_finite_v = np.sum(np.isfinite(v_data))
n_positive_sigma = np.sum(sigma_data > 0)
print(f"   Finite velocities: {n_finite_v}/{len(v_data)}")
print(f"   Positive errors: {n_positive_sigma}/{len(sigma_data)}")

if sigma_data.min() <= 0:
    print(f"   ⚠️  WARNING: Found non-positive errors! Min = {sigma_data.min()}")
if np.any(~np.isfinite(v_data)):
    print(f"   ⚠️  WARNING: Found non-finite velocities!")

# Now test the likelihood function
print("\n3. Testing likelihood calculation...")

from run_dynesty import get_param_labels_and_bounds
import run_dynesty
run_dynesty.logger = logger
import argparse

# Set up args matching your run
args = argparse.Namespace()
args.fit_target = 'milkyway'
args.xi = 'power'
args.use_gp_surrogate = False

# All components
args.include_disk_thin = True
args.include_disk_thick = True  
args.include_bulge = True
args.include_gas = True
args.fit_xi_params = True
args.fit_disk_thin = True
args.fit_disk_thick = True
args.fit_bulge = True
args.fit_gas = True

# Initial values (reasonable for MW)
args.rho_c_fixed = 1e8
args.n_exp_fixed = 1.0
args.M_disk_thin_fixed = 5e10
args.R_d_thin_fixed = 3.0
args.h_z_thin_fixed = 0.3
args.M_disk_thick_fixed = 1e10
args.R_d_thick_fixed = 4.0
args.h_z_thick_fixed = 1.0
args.M_bulge_fixed = 1e10
args.a_bulge_fixed = 0.7
args.M_gas_fixed = 1e10
args.R_d_gas_fixed = 8.0
args.h_z_gas_fixed = 0.2

# Get parameter configuration
print("\n4. Getting parameter configuration...")
try:
    fitted_p_names, _, p0_guess, p_low, p_high, _ = get_param_labels_and_bounds(args)
    print(f"   ✅ Got {len(fitted_p_names)} parameters")
    
    # Check if all_param_info_list was set
    if hasattr(args, 'all_param_info_list'):
        print(f"   ✅ all_param_info_list is set with {len(args.all_param_info_list)} items")
    else:
        print(f"   ❌ all_param_info_list is NOT set - this will cause -inf!")
        
except Exception as e:
    print(f"   ❌ Error in get_param_labels_and_bounds: {e}")
    import traceback
    traceback.print_exc()

# Test the likelihood directly
print("\n5. Testing log_likelihood_dynesty directly...")
from run_dynesty import log_likelihood_dynesty

# Build the arguments tuple
all_param_info = getattr(args, 'all_param_info_list', [])
logl_args = (fitted_p_names, args, all_param_info,
             R_data, v_data, sigma_data, 'power', None)

print(f"   Arguments prepared:")
print(f"   - fitted_param_names: {len(fitted_p_names)} params")
print(f"   - all_param_info_list is None: {all_param_info is None}")
print(f"   - all_param_info_list length: {len(all_param_info) if all_param_info else 'N/A'}")
print(f"   - Data lengths: R={len(R_data)}, v={len(v_data)}, sigma={len(sigma_data)}")

try:
    log_L, blob = log_likelihood_dynesty(p0_guess, *logl_args)
    print(f"\n   RESULT: log_L = {log_L}")
    if isinstance(blob, list) and len(blob) > 0:
        print(f"   RMS from blob: {blob[0]:.1f} km/s")
    
    if log_L == -np.inf:
        print("\n   ❌ Got -inf! Trying to understand why...")
        
        # Test model calculation directly
        print("\n6. Testing model components directly...")
        from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, xi_power_law
        
        # Build full parameter dict
        param_dict = dict(zip(fitted_p_names, p0_guess))
        # Add fixed parameters from all_param_info_list
        if all_param_info:
            for p_info in all_param_info:
                if not p_info.get('is_fitted', True):
                    param_dict[p_info['name']] = p_info['current_val']
        
        # Add component flags
        param_dict.update({
            'include_disk_thin': True,
            'include_disk_thick': True,
            'include_bulge': True,
            'include_gas': True,
            'include_bulge_density': True
        })
        
        print(f"\n   Total parameters in dict: {len(param_dict)}")
        print(f"   Total mass = {sum(v for k,v in param_dict.items() if 'M_' in k and 'solar' in k):.2e} M_sun")
        
        # Test each component
        try:
            v_newton = v_baryon_total_newtonian_kms(R_data[:10], param_dict)
            print(f"   v_Newton at R={R_data[0]:.1f} kpc: {v_newton[0]:.1f} km/s")
        except Exception as e:
            print(f"   ❌ Error in v_newton: {e}")
            
        try:
            rho = rho_baryon_total_midplane_solar_kpc3(R_data[:10], param_dict)
            print(f"   rho at R={R_data[0]:.1f} kpc: {rho[0]:.2e} M_sun/kpc^3")
        except Exception as e:
            print(f"   ❌ Error in rho: {e}")
            
        try:
            if 'rho' in locals():
                xi = xi_power_law(rho, param_dict.get('rho_c_solar_kpc3', 1e8), 
                                 param_dict.get('n_exp', 1.0))
                print(f"   xi at R={R_data[0]:.1f} kpc: {xi[0]:.3f}")
        except Exception as e:
            print(f"   ❌ Error in xi: {e}")
            
except Exception as e:
    print(f"\n   ❌ Exception in log_likelihood_dynesty: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("Most likely issues:")
print("1. all_param_info_list is None or not properly set")
print("2. Physical plausibility check still active somewhere")
print("3. Parameter bounds creating impossible combinations")
print("4. Error in model calculation (check component flags)")