import numpy as np
import matplotlib.pyplot as plt
from density_metric2 import *

# Test parameters
params = {
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

R_arr = np.linspace(0.5, 20, 100)

# Calculate for standard model
rho_arr = rho_baryon_total_midplane_solar_kpc3(R_arr, params)
xi_standard = xi_power_law(rho_arr, 1e9, 1.5)

# Calculate for nonlocal model - need enclosed mass
M_enclosed = np.zeros_like(R_arr)
for i, R in enumerate(R_arr):
    # Approximate enclosed mass
    M_enclosed[i] = params['M_disk_thin_solar'] * (1 - np.exp(-R/params['R_d_thin_kpc'])) + \
                    params['M_disk_thick_solar'] * (1 - np.exp(-R/params['R_d_thick_kpc']))

xi_nonlocal_arr = xi_nonlocal(rho_arr, M_enclosed, R_arr, rho_c=1e8, M_c=5e10)

# Calculate for anisotropic model
xi_aniso_radial = xi_anisotropic(rho_arr, direction='radial', rho_c_rad=5e8)
xi_aniso_vertical = xi_anisotropic(rho_arr, direction='vertical', rho_c_vert=1e7)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Xi profiles
ax1.plot(R_arr, xi_standard, 'b-', label='Standard power law', linewidth=2)
ax1.plot(R_arr, xi_nonlocal_arr, 'r--', label='Non-local', linewidth=2)
ax1.plot(R_arr, xi_aniso_radial, 'g-.', label='Anisotropic (radial)', linewidth=2)
ax1.plot(R_arr, xi_aniso_vertical, 'm:', label='Anisotropic (vertical)', linewidth=2)
ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax1.set_xlabel('R (kpc)')
ax1.set_ylabel('ξ')
ax1.set_title('Comparison of ξ Models')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.5)

# Resulting rotation curves
v_newton = v_baryon_total_newtonian_kms(R_arr, params)
v_standard = v_newton * np.sqrt(xi_standard)
v_nonlocal = v_newton * np.sqrt(xi_nonlocal_arr)
v_aniso_rad = v_newton * np.sqrt(xi_aniso_radial)

ax2.plot(R_arr, v_newton, 'k--', label='Newton', linewidth=2)
ax2.plot(R_arr, v_standard, 'b-', label='Standard ξ', linewidth=2)
ax2.plot(R_arr, v_nonlocal, 'r--', label='Non-local ξ', linewidth=2)
ax2.plot(R_arr, v_aniso_rad, 'g-.', label='Anisotropic ξ', linewidth=2)
ax2.axhline(220, color='gray', linestyle=':', label='Target at R☉')
ax2.set_xlabel('R (kpc)')
ax2.set_ylabel('v (km/s)')
ax2.set_title('Rotation Curves with Different ξ Models')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('new_xi_models_test.png', dpi=150)
plt.show()

# Test K_z for each model
print("\nK_z predictions:")
print("="*60)

# Surface density at R_sun
Sigma_total = (params['M_disk_thin_solar'] / (2*np.pi*params['R_d_thin_kpc']**2) * 
               np.exp(-8.122/params['R_d_thin_kpc']) +
               params['M_disk_thick_solar'] / (2*np.pi*params['R_d_thick_kpc']**2) * 
               np.exp(-8.122/params['R_d_thick_kpc']))

rho_solar = rho_baryon_total_midplane_solar_kpc3(np.array([8.122]), params)[0]

# Standard model
xi_std = xi_power_law(rho_solar, 1e9, 1.5)[0]
Kz_std = 2 * np.pi * G_ASTRO_UNITS * Sigma_total * xi_std * (1 - np.exp(-1.1/0.4)) / 1000
print(f"Standard model: K_z = {Kz_std:.3e} (km/s)²/pc, ratio = {Kz_std/2.3e-3:.0f}")

# Anisotropic vertical
xi_vert = xi_anisotropic(rho_solar, direction='vertical', rho_c_vert=1e7)[0]
Kz_aniso = 2 * np.pi * G_ASTRO_UNITS * Sigma_total * xi_vert * (1 - np.exp(-1.1/0.4)) / 1000
print(f"Anisotropic (vertical): K_z = {Kz_aniso:.3e} (km/s)²/pc, ratio = {Kz_aniso/2.3e-3:.0f}")

# Check if anisotropic can work
xi_rad_solar = xi_anisotropic(rho_solar, direction='radial', rho_c_rad=5e8)[0]
print(f"\nAt R_sun: xi_radial = {xi_rad_solar:.3f}, xi_vertical = {xi_vert:.3f}")
print(f"Suppression factor for K_z: {xi_vert/xi_rad_solar:.3f}")
