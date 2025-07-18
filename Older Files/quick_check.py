# Quick test to see what's happening
import numpy as np
from run_dynesty import check_physical_plausibility

# Test parameters (your initial values)
test_params = np.array([
    5e8,    # rho_c
    1.0,    # n_exp
    6e10,   # M_disk_thin
    2.6,    # R_d_thin
    0.3,    # h_z_thin
    1.2e10, # M_disk_thick
    3.6,    # R_d_thick
    0.9,    # h_z_thick
    2e10,   # M_bulge
    0.5,    # a_bulge
    1.5e10, # M_gas
    7.0,    # R_d_gas
    0.15    # h_z_gas
])

param_names = ['rho_c_solar_kpc3', 'n_exp', 'M_disk_thin_solar', 'R_d_thin_kpc', 
               'h_z_thin_kpc', 'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
               'M_bulge_solar', 'a_bulge_kpc', 'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc']

# Create a minimal args object
class Args:
    include_bulge = True
    include_disk_thin = True
    include_disk_thick = True
    include_gas = True

is_valid, reason = check_physical_plausibility(test_params, param_names, Args())
print(f"Valid: {is_valid}, Reason: {reason}")

# Check total mass
total_mass = 6e10 + 1.2e10 + 2e10 + 1.5e10
print(f"Total mass: {total_mass:.2e} M_sun")