#!/usr/bin/env python3
"""
run_gr_baseline.py - Pure Newtonian/GR baseline with observational baryon parameters
No dark matter, no fitting - just reality check
"""

import subprocess
import sys

# Observational parameters from literature
# (Bland-Hawthorn & Gerhard 2016, Bovy 2017, etc.)
observational_params = {
    # Thin disk (main stellar disk)
    'M_disk_thin_fixed': 5.0e10,    # McMillan 2017
    'R_d_thin_fixed': 2.6,           # Bovy & Rix 2013
    'h_z_thin_fixed': 0.3,           # Juric et al. 2008
    
    # Thick disk  
    'M_disk_thick_fixed': 1.0e10,    # Bland-Hawthorn & Gerhard 2016
    'R_d_thick_fixed': 3.6,          # Robin et al. 2014
    'h_z_thick_fixed': 0.9,          # Juric et al. 2008
    
    # Bulge
    'M_bulge_fixed': 1.4e10,         # Portail et al. 2017
    'a_bulge_fixed': 0.5,            # Cao et al. 2013
    
    # Gas (HI + H2 + He)
    'M_gas_fixed': 1.5e10,           # Kalberla & Dedes 2008 + factor for He
    'R_d_gas_fixed': 7.0,            # Kalberla & Kerp 2009
    'h_z_gas_fixed': 0.15,           # Nakanishi & Sofue 2016
    
    # Xi parameters (not used for GR, but needed by code)
    'rho_c_fixed': 1e13,
    'n_exp_fixed': 1.5,
    'A_fixed': 1.0,
}

# Build command
cmd = [
    'python', 'run_dynesty.py',
    '--xi', 'gr',  # This uses xi_gr_baseline function
    '--output_dir', 'chains_gr_baseline_observational',
    '--include_disk_thin',
    '--include_disk_thick', 
    '--include_bulge',
    '--include_gas',
    # DO NOT FIT ANYTHING - all parameters fixed
    '--nlive_init', '1000',
    '--maxcall', '1000000',
    '--dlogz_target', '0.01'
]

# Add all fixed parameters
for param, value in observational_params.items():
    cmd.extend([f'--{param.replace("_", "-")}', str(value)])

print("Running GR baseline with observational parameters...")
print("Command:", ' '.join(cmd))
subprocess.run(cmd)