import numpy as np
import matplotlib.pyplot as plt
from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, xi_gravitational_color

# Set up parameters dict with your actual values
params = {
    'M_disk_thin_solar': 7.320e+10,
    'R_d_thin_kpc': 3.465,
    'h_z_thin_kpc': 0.6036,
    'M_disk_thick_solar': 1.439e+10,
    'R_d_thick_kpc': 7.404,
    'h_z_thick_kpc': 1.431,
    'M_bulge_solar': 9.774e+09,
    'a_bulge_kpc': 1.470,
    'M_gas_solar': 4.617e+10,
    'R_d_gas_kpc': 7.712,
    'h_z_gas_kpc': 0.06846,
    'include_disk_thin': True,
    'include_disk_thick': True,
    'include_bulge': True,
    'include_gas': True,
    'include_bulge_density': True,
    'rho_c_solar_kpc3': 1e8,
    'gamma_exp': 2.5,
    'lambda_g': 3.0
}

# Calculate curves
R = np.linspace(1, 30, 200)
v_newton = v_baryon_total_newtonian_kms(R, params)
rho = rho_baryon_total_midplane_solar_kpc3(R, params)
xi = xi_gravitational_color(rho, params['rho_c_solar_kpc3'], params['gamma_exp'], params['lambda_g'])
v_modified = v_newton * np.sqrt(xi)

# Create figure with multiple panels
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Rotation curves
ax1 = axes[0, 0]
ax1.plot(R, v_newton, 'b-', label='Newtonian', linewidth=2)
ax1.plot(R, v_modified, 'r-', label='Modified (grav color)', linewidth=2)
ax1.fill_between(R, v_newton, v_modified, alpha=0.3, color='red', label='Enhancement')
ax1.set_xlabel('R (kpc)')
ax1.set_ylabel('v (km/s)')
ax1.set_title('Rotation Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 300)

# Panel 2: Density profile
ax2 = axes[0, 1]
ax2.semilogy(R, rho, 'g-', linewidth=2)
ax2.axhline(params['rho_c_solar_kpc3'], color='red', linestyle='--', label=f'ρ_c = {params["rho_c_solar_kpc3"]:.1e}')
ax2.set_xlabel('R (kpc)')
ax2.set_ylabel('ρ (M☉/kpc³)')
ax2.set_title('Midplane Density')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)

# Panel 3: Xi enhancement factor
ax3 = axes[1, 0]
ax3.plot(R, xi, 'purple', linewidth=2)
ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Newtonian (ξ=1)')
ax3.set_xlabel('R (kpc)')
ax3.set_ylabel('ξ')
ax3.set_title(f'Gravitational Enhancement (γ={params["gamma_exp"]}, λ_g={params["lambda_g"]})')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 30)
ax3.set_ylim(0.9, max(xi)*1.1)

# Panel 4: Enhancement percentage
ax4 = axes[1, 1]
enhancement_pct = (np.sqrt(xi) - 1) * 100
ax4.plot(R, enhancement_pct, 'orange', linewidth=2)
ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
ax4.set_xlabel('R (kpc)')
ax4.set_ylabel('Velocity Enhancement (%)')
ax4.set_title('Percentage Enhancement in Rotation Velocity')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 30)

plt.suptitle('Gravitational Color Model Results', fontsize=16)
plt.tight_layout()
plt.savefig('grav_color_analysis.png', dpi=150)
plt.show()

# Print key values at solar radius
R_sun = 8.122
v_newton_sun = v_baryon_total_newtonian_kms(np.array([R_sun]), params)[0]
rho_sun = rho_baryon_total_midplane_solar_kpc3(np.array([R_sun]), params)[0]
xi_sun = xi_gravitational_color(rho_sun, params['rho_c_solar_kpc3'], params['gamma_exp'], params['lambda_g'])[0]
v_modified_sun = v_newton_sun * np.sqrt(xi_sun)

print(f"\nAt Solar Radius (R = {R_sun} kpc):")
print(f"  Density: {rho_sun:.2e} M☉/kpc³")
print(f"  ρ/ρ_c: {rho_sun/params['rho_c_solar_kpc3']:.2f}")
print(f"  ξ: {xi_sun:.3f}")
print(f"  v_Newton: {v_newton_sun:.1f} km/s")
print(f"  v_modified: {v_modified_sun:.1f} km/s")
print(f"  Enhancement: {(np.sqrt(xi_sun)-1)*100:.1f}%")

# Find where maximum enhancement occurs
xi_max_idx = np.argmax(xi)
R_max = R[xi_max_idx]
print(f"\nMaximum enhancement at R = {R_max:.1f} kpc:")
print(f"  ξ_max = {xi[xi_max_idx]:.3f}")
print(f"  Velocity enhancement: {(np.sqrt(xi[xi_max_idx])-1)*100:.1f}%")