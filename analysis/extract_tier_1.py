# Save this as save_tier1_for_tier2.py
import numpy as np

# Create a simplified NPZ file from Tier 1 results
tier1_params = {
    'rho_c_solar_kpc3': 1.695e6,
    'A': 2.925,
    'n_exp': 0.533,
    'M_disk_thin_solar': 8.302e10,
    'R_d_thin_kpc': 2.963,
    'h_z_thin_kpc': 0.322,
    'M_disk_thick_solar': 6.847e9,
    'R_d_thick_kpc': 5.836,
    'h_z_thick_kpc': 0.885,
    'M_bulge_solar': 1.291e10,
    'a_bulge_kpc': 1.116,
    'M_gas_solar': 1.626e10,
    'R_d_gas_kpc': 9.838,
    'h_z_gas_kpc': 0.399
}

param_names = list(tier1_params.keys())
param_values = np.array([tier1_params[p] for p in param_names])

# Create fake samples (just the best-fit repeated)
samples = np.tile(param_values, (100, 1))
weights = np.ones(100) / 100

np.savez('tier1_best_fit.npz',
         samples=samples,
         weights=weights,
         param_names=param_names)

print("Saved tier1_best_fit.npz")