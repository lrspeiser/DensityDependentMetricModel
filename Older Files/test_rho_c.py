import numpy as np
from density_metric2 import *
from data_io import load_gaia

# Load data
data = load_gaia(sample_max=1000)

# Test different configurations
configs = [
    {"name": "Original", "rho_c": 1e8, "lambda_g": 8.0},
    {"name": "Higher rho_c", "rho_c": 1e9, "lambda_g": 8.0},
    {"name": "Lower lambda", "rho_c": 1e8, "lambda_g": 2.0},
    {"name": "Balanced", "rho_c": 5e8, "lambda_g": 4.0},
]

params = {
    'M_disk_thin_solar': 5e10,
    'R_d_thin_kpc': 2.6,
    'h_z_thin_kpc': 0.3,
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'gamma': 2.7
}

for config in configs:
    params['rho_c_solar_kpc3'] = config['rho_c']
    params['lambda_g'] = config['lambda_g']
    
    # Calculate at a few radii
    R_test = np.array([5, 8, 12])
    v_newton = v_baryon_total_newtonian_kms(R_test, params)
    rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)
    xi = xi_gravitational_color(rho, config['rho_c'], 2.7, config['lambda_g'])
    v_model = v_newton * np.sqrt(xi)
    
    print(f"\n{config['name']} (ρ_c={config['rho_c']:.0e}, λ={config['lambda_g']}):")
    print("R (kpc) | v_Newton | ξ | v_model")
    for i in range(len(R_test)):
        print(f"{R_test[i]:6.1f} | {v_newton[i]:8.1f} | {xi[i]:.3f} | {v_model[i]:7.1f}")