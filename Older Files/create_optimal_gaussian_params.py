#!/usr/bin/env python3
"""
Create parameter file with optimal Gaussian xi values
"""
import json

# Optimal parameters from tuning
optimal_params = {
    "xi_function": "gaussian",
    "fit_xi_params": True,
    
    # Optimal Gaussian xi parameters
    "rho_c": 0.316,    # Peak at ~0.3 M☉/kpc³
    "n_exp": 0.70,     # Moderate width
    "A": 5.0,          # Strong enhancement (6x total at peak)
    
    # Standard MW parameters
    "M_disk_thin_solar": 5e10,
    "R_d_thin_kpc": 2.6,
    "h_z_thin_kpc": 0.3,
    
    "M_disk_thick_solar": 0.15e10,
    "R_d_thick_kpc": 3.6,
    "h_z_thick_kpc": 0.9,
    
    "M_bulge_solar": 1.5e10,
    "a_bulge_kpc": 0.5,
    
    "M_gas_solar": 0.5e10,
    "R_d_gas_kpc": 7.0,
    "h_z_gas_kpc": 0.15,
    
    # Include components
    "include_disk_thin": True,
    "include_disk_thick": True,
    "include_bulge": True,
    "include_gas": True,
    "include_bulge_density": True,
    
    # Fitting flags
    "fit_disk_thin": True,
    "fit_disk_thick": True,
    "fit_bulge": True,
    "fit_gas": False
}

# Save optimal parameters
with open('gaussian_params_optimal.json', 'w') as f:
    json.dump(optimal_params, f, indent=2)

print("Created gaussian_params_optimal.json with:")
print(f"  rho_c = {optimal_params['rho_c']}")
print(f"  n_exp = {optimal_params['n_exp']}")
print(f"  A = {optimal_params['A']}")
print("\nThis configuration gives:")
print("  ξ(halo=0.01) ≈ 1.50")
print("  ξ(galaxy=0.5) ≈ 5.80")
print("  ξ(solar=100) ≈ 1.01")
print("  ξ(stellar=1e6) ≈ 1.00")

# Also create a parameter file for dynesty with fixed xi parameters
dynesty_params = optimal_params.copy()
dynesty_params["fit_xi_params"] = False  # Don't fit xi params

with open('gaussian_params_fixed_xi.json', 'w') as f:
    json.dump(dynesty_params, f, indent=2)

print("\nAlso created gaussian_params_fixed_xi.json with xi parameters fixed")
print("(for faster dynesty runs when you just want to fit MW components)")

print("\nNext steps:")
print("1. Validate: python3 validate_ddmm.py gaussian_params_optimal.json --output_dir validation_gaussian_optimal")
print("2. Run dynesty (quick): python3 run_dynesty.py --params_file gaussian_params_fixed_xi.json --nlive_init 300")
print("3. Run dynesty (full): python3 run_dynesty.py --params_file gaussian_params_optimal.json --nlive_init 500")