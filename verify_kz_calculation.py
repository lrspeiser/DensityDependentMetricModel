#!/usr/bin/env python3
"""
Verify K_z calculation is correct.
K_z force comes from Poisson equation integrated vertically.
"""
import numpy as np
from scipy.integrate import quad
from density_metric2 import G_ASTRO_UNITS

def calculate_Kz_exact(z_kpc, M_disk, R_d, h_z, R=8.122):
    """
    Calculate K_z using exact integration of Poisson equation.
    
    K_z(R,z) = 4πG ∫[0 to z] ρ(R,z') dz'
    
    For exponential disk:
    ρ(R,z) = ρ_0 exp(-R/R_d) exp(-|z|/h_z)
    where ρ_0 = M_disk / (4π R_d² h_z)
    """
    # Midplane density
    rho_0 = M_disk / (4 * np.pi * R_d**2 * h_z)
    
    # Density at this R
    rho_R = rho_0 * np.exp(-R/R_d)
    
    # Integrate from 0 to z
    def integrand(zp):
        return rho_R * np.exp(-abs(zp)/h_z)
    
    integral, _ = quad(integrand, 0, z_kpc)
    
    # K_z in (km/s)²/kpc
    Kz_kpc = 4 * np.pi * G_ASTRO_UNITS * integral
    
    # Convert to (km/s)²/pc
    Kz_pc = Kz_kpc / 1000
    
    return Kz_pc, rho_R

# Test with your parameters
M_disk = 1.269e11
R_d = 4.138
h_z = 0.595

print("K_z calculation verification:")
print("="*50)

# Calculate at different heights
z_values = [0.5, 1.0, 1.1, 1.5, 2.0]
for z in z_values:
    Kz, rho = calculate_Kz_exact(z, M_disk, R_d, h_z)
    print(f"z = {z:.1f} kpc: K_z = {Kz:.3e} (km/s)²/pc")

# Compare with observations
Kz_obs = 2.3e-3  # at z = 1.1 kpc
Kz_model, _ = calculate_Kz_exact(1.1, M_disk, R_d, h_z)
print(f"\nAt z = 1.1 kpc:")
print(f"  Model: {Kz_model:.3e} (km/s)²/pc")
print(f"  Observed: {Kz_obs:.3e} (km/s)²/pc")
print(f"  Ratio: {Kz_model/Kz_obs:.0f}")

# What if we include xi suppression?
xi_at_Rsun = 0.95  # Your value
Kz_with_xi = Kz_model * xi_at_Rsun
print(f"\nWith ξ = {xi_at_Rsun:.2f} suppression:")
print(f"  Modified K_z: {Kz_with_xi:.3e} (km/s)²/pc")
print(f"  Still too large by: {Kz_with_xi/Kz_obs:.0f}×")
