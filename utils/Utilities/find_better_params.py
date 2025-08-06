#!/usr/bin/env python3
"""
find_better_params.py - Find parameters that produce ~220 km/s at R=8 kpc
"""
import numpy as np
from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP

# Test at solar radius
R_test = np.array([8.0])

# Base parameters
params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'h_z_thin_kpc': 0.3
}

print("Testing different parameter combinations to achieve v ≈ 220 km/s at R = 8 kpc\n")

# Test 1: Original parameters
print("Test 1: Your original parameters")
params.update({
    'rho_c_solar_kpc3': 9e8,
    'n_exp': 1.6,
    'M_disk_thin_solar': 4e10,
    'R_d_thin_kpc': 2.5
})
v_newton = v_baryon_total_newtonian_kms(R_test, params)[0]
rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)[0]
xi = XI_FUNCTION_MAP['power'](rho, params['rho_c_solar_kpc3'], params['n_exp'])
v_total = v_newton * np.sqrt(xi)
print(f"  M_disk = {params['M_disk_thin_solar']:.1e}, R_d = {params['R_d_thin_kpc']}")
print(f"  v_newton = {v_newton:.1f} km/s")
print(f"  ρ = {rho:.1e} M☉/kpc³, ξ = {xi:.2f}")
print(f"  v_total = {v_total:.1f} km/s (need 220)\n")

# Test 2: Higher disk mass
print("Test 2: Higher disk mass")
params.update({
    'M_disk_thin_solar': 7e10,
    'R_d_thin_kpc': 2.5
})
v_newton = v_baryon_total_newtonian_kms(R_test, params)[0]
rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)[0]
xi = XI_FUNCTION_MAP['power'](rho, params['rho_c_solar_kpc3'], params['n_exp'])
v_total = v_newton * np.sqrt(xi)
print(f"  M_disk = {params['M_disk_thin_solar']:.1e}, R_d = {params['R_d_thin_kpc']}")
print(f"  v_newton = {v_newton:.1f} km/s")
print(f"  ρ = {rho:.1e} M☉/kpc³, ξ = {xi:.2f}")
print(f"  v_total = {v_total:.1f} km/s\n")

# Test 3: Larger scale radius
print("Test 3: Larger scale radius")
params.update({
    'M_disk_thin_solar': 5e10,
    'R_d_thin_kpc': 3.5
})
v_newton = v_baryon_total_newtonian_kms(R_test, params)[0]
rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)[0]
xi = XI_FUNCTION_MAP['power'](rho, params['rho_c_solar_kpc3'], params['n_exp'])
v_total = v_newton * np.sqrt(xi)
print(f"  M_disk = {params['M_disk_thin_solar']:.1e}, R_d = {params['R_d_thin_kpc']}")
print(f"  v_newton = {v_newton:.1f} km/s")
print(f"  ρ = {rho:.1e} M☉/kpc³, ξ = {xi:.2f}")
print(f"  v_total = {v_total:.1f} km/s\n")

# Test 4: Adjust xi parameters for stronger boost
print("Test 4: Lower rho_c for stronger xi boost")
params.update({
    'rho_c_solar_kpc3': 1e8,  # Lower rho_c
    'n_exp': 1.6,
    'M_disk_thin_solar': 5e10,
    'R_d_thin_kpc': 3.0
})
v_newton = v_baryon_total_newtonian_kms(R_test, params)[0]
rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)[0]
xi = XI_FUNCTION_MAP['power'](rho, params['rho_c_solar_kpc3'], params['n_exp'])
v_total = v_newton * np.sqrt(xi)
print(f"  M_disk = {params['M_disk_thin_solar']:.1e}, R_d = {params['R_d_thin_kpc']}")
print(f"  ρ_c = {params['rho_c_solar_kpc3']:.1e}, n = {params['n_exp']}")
print(f"  v_newton = {v_newton:.1f} km/s")
print(f"  ρ = {rho:.1e} M☉/kpc³, ξ = {xi:.2f}")
print(f"  v_total = {v_total:.1f} km/s\n")

# Test 5: Nearly Newtonian
print("Test 5: Nearly Newtonian (high rho_c)")
params.update({
    'rho_c_solar_kpc3': 1e10,  # Very high rho_c makes xi ≈ 1
    'n_exp': 0.1,
    'M_disk_thin_solar': 8e10,  # Need more mass for Newtonian
    'R_d_thin_kpc': 3.0
})
v_newton = v_baryon_total_newtonian_kms(R_test, params)[0]
rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)[0]
xi = XI_FUNCTION_MAP['power'](rho, params['rho_c_solar_kpc3'], params['n_exp'])
v_total = v_newton * np.sqrt(xi)
print(f"  M_disk = {params['M_disk_thin_solar']:.1e}, R_d = {params['R_d_thin_kpc']}")
print(f"  ρ_c = {params['rho_c_solar_kpc3']:.1e}, n = {params['n_exp']}")
print(f"  v_newton = {v_newton:.1f} km/s")
print(f"  ρ = {rho:.1e} M☉/kpc³, ξ = {xi:.2f}")
print(f"  v_total = {v_total:.1f} km/s\n")

print("=" * 60)
print("RECOMMENDATION: Use parameters from the test that gives v_total ≈ 220 km/s")
print("Then run dynesty with those fixed values.")