#!/usr/bin/env python3
"""Create correct DDMM parameters file for validation"""

import json

# Your actual fitted parameters from the dynesty analysis
params = {
    # Core DDMM parameters
    "rho_c_solar_kpc3": 1.02e8,      # Critical density
    "n_exp": 3.174,                   # This is your gamma_exp value!
    
    # Disk components
    "M_disk_thin_solar": 8.51e10,
    "R_d_thin_kpc": 6.271,
    "h_z_thin_kpc": 0.445,
    
    "M_disk_thick_solar": 1.90e10,
    "R_d_thick_kpc": 1.800,
    "h_z_thick_kpc": 0.065,
    
    # Bulge
    "M_bulge_solar": 3.71e10,
    "a_bulge_kpc": 14.377,
    
    # Gas disk
    "M_gas_solar": 5.69e9,
    "R_d_gas_kpc": 8.0,      # Typical value if not fitted
    "h_z_gas_kpc": 0.15,     # Typical value if not fitted
    
    # Component flags
    "include_disk_thin": True,
    "include_disk_thick": True,
    "include_bulge": True,
    "include_gas": True,
    
    # Xi function type
    "xi_type": "power",
    
    # Additional parameters if using grav_color model
    "lambda_g": 0.911,       # From your fit
    "A": 30.0                 # Default amplitude
}

# Save as JSON
with open("ddmm_params_correct.json", "w") as f:
    json.dump(params, f, indent=2)

print("Created ddmm_params_correct.json")
print(f"\nKey parameters:")
print(f"  ρ_c = {params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
print(f"  n = {params['n_exp']:.3f} (was incorrectly {85149895554.13:.0f}!)")
print(f"  Total baryon mass = {params['M_disk_thin_solar'] + params['M_disk_thick_solar'] + params['M_bulge_solar'] + params['M_gas_solar']:.2e} M☉")

# Quick check of what ξ should be at different densities
print("\nExpected ξ values with correct parameters:")
densities = [1e-10, 1e2, 1e6, 1e8, 1e24]
names = ["Orbit", "Solar System", "Galaxy", "Critical", "Lab"]

for rho, name in zip(densities, names):
    if params['xi_type'] == 'power':
        xi = 3.0 if rho <= params['rho_c_solar_kpc3'] else (rho / params['rho_c_solar_kpc3'])**(-params['n_exp'])
    print(f"  ξ({name:12s} ρ={rho:.0e}) = {xi:.6f}")
    