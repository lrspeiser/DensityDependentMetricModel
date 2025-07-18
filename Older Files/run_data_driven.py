#!/usr/bin/env python3
'''Modified run script that lets data drive the fit'''

import sys
import subprocess

# First ensure we have fresh data
print("Checking for fresh data...")
import os
if not os.path.exists('gaia_query_cache_DR3_processed_for_fit.parquet'):
    print("No data found! Run 'python get_fresh_data.py' first")
    sys.exit(1)

# Now run with minimal constraints
cmd = [
    "python", "run_dynesty.py",
    
    # Fit everything
    "--fit_xi_params",
    "--fit_disk_thin", "--fit_disk_thick", "--fit_bulge", "--fit_gas",
    "--include_disk_thin", "--include_disk_thick", "--include_bulge", "--include_gas",
    
    # Start with very broad priors (let data constrain)
    "--M_disk_thin_fixed", "5e10",  # Start conservative
    "--R_d_thin_fixed", "3.0",
    "--h_z_thin_fixed", "0.3",
    
    "--M_disk_thick_fixed", "1e10",
    "--R_d_thick_fixed", "4.0", 
    "--h_z_thick_fixed", "1.0",
    
    "--M_bulge_fixed", "1e10",
    "--a_bulge_kpc", "0.7",
    
    "--M_gas_fixed", "1e10",
    "--R_d_gas_kpc", "8.0",
    "--h_z_gas_kpc", "0.2",
    
    # Broad xi priors
    "--rho_c_fixed", "1e8",
    "--n_exp_fixed", "1.0",
    
    # Sampling settings
    "--nlive_init", "1500",
    "--maxcall", "5000000",
    "--num_threads", "1",  # Start with 1 to avoid multiprocessing issues
    
    # Use the fresh data
    "--max_sample_gaia", "200000",
    
    # Output
    "--output_dir", "chains_data_driven",
    "--enable_dashboard"
]

print("\nStarting data-driven run...")
print(" ".join(cmd))
subprocess.run(cmd)
