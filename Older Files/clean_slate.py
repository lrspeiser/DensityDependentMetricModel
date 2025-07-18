#!/usr/bin/env python3
"""
Set up a proper data-driven analysis without premature constraints
"""

import os
import subprocess

# Step 1: Clear ALL old data
print("🧹 Step 1: Clearing all old cached data...")
cache_files = [
    "gaia_query_cache_DR3_raw.csv",
    "gaia_query_cache_DR3_processed_for_fit.parquet",
    "gaia_query_cache_DR3_raw_enhanced.csv",
    "gaia_query_cache_DR3_processed_enhanced.parquet"
]

for f in cache_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"   Removed {f}")

# Step 2: Get fresh, comprehensive data
print("\n📡 Step 2: Getting fresh comprehensive Gaia data...")
print("   This will query 300,000 stars across all galactic regions")

data_script = """
from data_io import load_gaia

# Force new comprehensive query
gaia_data = load_gaia(
    sample_max=300000,
    force_new_query_gaia=True,  # Force fresh query
    force_reprocess_raw=True,   # Force reprocessing
    use_enhanced_query=True,    # Use enhanced quality cuts
    validate_data=True
)

if gaia_data:
    print(f"\\n✅ Successfully loaded {len(gaia_data['R_kpc']):,} stars")
    print(f"   R range: {gaia_data['R_kpc'].min():.1f} - {gaia_data['R_kpc'].max():.1f} kpc")
    print(f"   <v> = {gaia_data['v_obs'].mean():.1f} ± {gaia_data['v_obs'].std():.1f} km/s")
    
    # Check radial coverage
    import numpy as np
    R_bins = [0, 5, 8, 10, 15, 20, 30]
    print("\\n   Radial coverage:")
    for i in range(len(R_bins)-1):
        mask = (gaia_data['R_kpc'] >= R_bins[i]) & (gaia_data['R_kpc'] < R_bins[i+1])
        print(f"   [{R_bins[i]:2d},{R_bins[i+1]:2d}) kpc: {mask.sum():6d} stars")
"""

with open('get_fresh_data.py', 'w') as f:
    f.write(data_script)

print("\nRun this to get fresh data:")
print("python get_fresh_data.py")

# Step 3: Create modified run script without physical checks
print("\n📝 Step 3: Creating data-driven run script...")

run_script = """#!/usr/bin/env python3
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

print("\\nStarting data-driven run...")
print(" ".join(cmd))
subprocess.run(cmd)
"""

with open('run_data_driven.py', 'w') as f:
    f.write(run_script)
os.chmod('run_data_driven.py', 0o755)

print("\n✅ Setup complete!")
print("\n📋 STEPS TO RUN:")
print("1. First get fresh data:     python get_fresh_data.py")
print("2. Then run the analysis:    python run_data_driven.py")
print("\nThis will let your DATA determine what's physical, not hardcoded constraints!")

# Step 4: Show how to monitor without preconceptions
monitor_script = """
# Monitor what the data is telling us
import pandas as pd
import numpy as np

# Load the actual data being fitted
data = pd.read_parquet('gaia_query_cache_DR3_processed_for_fit.parquet')

print(f"Data characteristics:")
print(f"  N_stars: {len(data):,}")
print(f"  R range: {data['R_kpc'].min():.1f} - {data['R_kpc'].max():.1f} kpc")
print(f"  <v(R=8kpc)>: {data[np.abs(data['R_kpc']-8)<0.5]['v_obs'].mean():.1f} km/s")
print(f"\\nLet the data speak!")
"""

with open('check_data_characteristics.py', 'w') as f:
    f.write(monitor_script)