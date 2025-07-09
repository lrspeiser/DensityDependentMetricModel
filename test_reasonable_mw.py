import numpy as np
from density_metric2 import *

# Literature-based reasonable MW model
params_reasonable = {
    'M_disk_thin_solar': 5e10,
    'R_d_thin_kpc': 2.5,
    'h_z_thin_kpc': 0.3,
    'M_disk_thick_solar': 1e10,
    'R_d_thick_kpc': 3.5,
    'h_z_thick_kpc': 0.9,
    'include_disk_thin': True,
    'include_disk_thick': True,
    'include_bulge': False,
    'include_gas': False
}

# Calculate rotation curve
R_arr = np.linspace(1, 20, 100)
v_newton = v_baryon_total_newtonian_kms(R_arr, params_reasonable)

print("Reasonable MW model test:")
print(f"Total mass: {(params_reasonable['M_disk_thin_solar'] + params_reasonable['M_disk_thick_solar']):.1e} M_sun")
print(f"v_newton at R_sun: {v_newton[40]:.1f} km/s (need ~220 km/s)")
print(f"Boost needed: {220/v_newton[40]:.1f}×")
print(f"This requires <ξ> = {(220/v_newton[40])**2:.1f}")

# Check K_z with these parameters
from scipy.integrate import quad

M_total = params_reasonable['M_disk_thin_solar'] + params_reasonable['M_disk_thick_solar']
# Approximate as single disk for K_z
h_z_eff = 0.4  # Effective scale height
Sigma_solar = M_total / (2 * np.pi * 2.5**2) * np.exp(-8.122/2.5)
Kz_model = 2 * np.pi * G_ASTRO_UNITS * Sigma_solar * (1 - np.exp(-1.1/h_z_eff)) / 1000

print(f"\nK_z check:")
print(f"  Model K_z: {Kz_model:.3e} (km/s)²/pc")
print(f"  Observed: 2.3e-3 (km/s)²/pc")
print(f"  Ratio: {Kz_model/2.3e-3:.1f}")
