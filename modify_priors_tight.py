# This modifies run_dynesty.py to use tight priors around your best values
import fileinput
import sys

# Your best-fit parameters
best_params = {
    'rho_c_solar_kpc3': {'val': 1.335e8, 'low': 1.0e8, 'high': 1.7e8},
    'n_exp': {'val': 0.889, 'low': 0.85, 'high': 0.92},
    'M_disk_thin_solar': {'val': 1.493e11, 'low': 1.3e11, 'high': 1.7e11},
    'R_d_thin_kpc': {'val': 1.502, 'low': 1.48, 'high': 1.52},
    'h_z_thin_kpc': {'val': 0.596, 'low': 0.5, 'high': 0.7},
    'M_disk_thick_solar': {'val': 6.634e10, 'low': 6.0e10, 'high': 7.3e10},
    'R_d_thick_kpc': {'val': 5.98, 'low': 5.8, 'high': 6.2},
    'h_z_thick_kpc': {'val': 0.645, 'low': 0.55, 'high': 0.75},
    'M_bulge_solar': {'val': 3.782e10, 'low': 3.3e10, 'high': 4.3e10},
    'a_bulge_kpc': {'val': 0.102, 'low': 0.095, 'high': 0.11},
    'M_gas_solar': {'val': 1.978e10, 'low': 1.7e10, 'high': 2.3e10},
    'R_d_gas_kpc': {'val': 9.54, 'low': 8.5, 'high': 10.5},
    'h_z_gas_kpc': {'val': 0.055, 'low': 0.045, 'high': 0.065},
}

print("Modified prior bounds in MW_MULTI_COMP_PARAM_CONFIG")
