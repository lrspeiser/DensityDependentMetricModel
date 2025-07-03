# compare_models_so_far.py
import numpy as np
import matplotlib.pyplot as plt

# Plot how xi functions differ at key radii
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Key radii
R_key = [2, 5, 8, 15, 20]  # kpc
rho_key = [1e9, 5e8, 1e8, 5e7, 1e7]  # Approximate densities at these radii

# Single disk
xi_single = [1/(1+(rho/1.64e9)**1.56) for rho in rho_key]

# Thin+Thick  
xi_thick = [1/(1+(rho/1.86e8)**0.724) for rho in rho_key]

ax1.plot(R_key, xi_single, 'bo-', linewidth=2, markersize=10, label='Single disk')
ax1.plot(R_key, xi_thick, 'ro-', linewidth=2, markersize=10, label='Thin+Thick')
ax1.set_xlabel('R (kpc)', fontsize=14)
ax1.set_ylabel('ξ(ρ(R))', fontsize=14)
ax1.set_title('Effective ξ at Different Radii', fontsize=16)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show rotation curve impact
v_newton = 220  # km/s, approximate
ax2.plot(R_key, [v_newton * np.sqrt(xi) for xi in xi_single], 'bo-', linewidth=2, markersize=10, label='Single disk')
ax2.plot(R_key, [v_newton * np.sqrt(xi) for xi in xi_thick], 'ro-', linewidth=2, markersize=10, label='Thin+Thick')
ax2.set_xlabel('R (kpc)', fontsize=14)
ax2.set_ylabel('v_circ (km/s)', fontsize=14)
ax2.set_title('Impact on Rotation Curve', fontsize=16)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xi_radial_impact.png', dpi=300)
plt.show()

# Print interpretations
print("Key insight:")
print(f"At R=8 kpc: ξ_single = {xi_single[2]:.3f}, ξ_thick = {xi_thick[2]:.3f}")
print(f"Thin+Thick model has {'stronger' if xi_thick[2] < xi_single[2] else 'weaker'} density dependence at solar radius")
