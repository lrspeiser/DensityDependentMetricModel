#!/usr/bin/env python3
"""
test_velocities.py - Simple script to find good parameters
"""
import numpy as np
from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP

# Test at solar radius
R_test = np.array([8.0])

# Base parameters
base_params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'h_z_thin_kpc': 0.3
}

def test_params(M_disk, R_d, rho_c, n_exp):
    """Test a parameter combination"""
    params = base_params.copy()
    params.update({
        'M_disk_thin_solar': M_disk,
        'R_d_thin_kpc': R_d,
        'rho_c_solar_kpc3': rho_c,
        'n_exp': n_exp
    })
    
    # Calculate velocities
    v_newton = v_baryon_total_newtonian_kms(R_test, params)
    rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)
    
    # Get xi - handle array output
    xi_func = XI_FUNCTION_MAP['power']
    xi_result = xi_func(rho, rho_c, n_exp)
    
    # Convert to scalars
    v_n = float(v_newton[0]) if hasattr(v_newton, '__len__') else float(v_newton)
    rho_val = float(rho[0]) if hasattr(rho, '__len__') else float(rho)
    xi_val = float(xi_result[0]) if hasattr(xi_result, '__len__') else float(xi_result)
    
    v_total = v_n * np.sqrt(xi_val)
    
    return v_n, rho_val, xi_val, v_total

print("Finding parameters that give v ≈ 220 km/s at R = 8 kpc")
print("="*60)

# Test cases
test_cases = [
    # (M_disk, R_d, rho_c, n_exp, description)
    (4e10, 2.5, 9e8, 1.6, "Your original"),
    (6e10, 2.5, 9e8, 1.6, "Higher mass"),
    (7e10, 2.5, 9e8, 1.6, "Even higher mass"),
    (5e10, 3.0, 9e8, 1.6, "Larger scale radius"),
    (5e10, 3.0, 1e8, 1.6, "Lower rho_c"),
    (6e10, 3.0, 5e8, 1.2, "Balanced"),
    (8e10, 3.0, 1e10, 0.1, "Nearly Newtonian"),
    (5.5e10, 3.2, 3e8, 1.4, "Tuned #1"),
    (6.5e10, 3.0, 2e8, 1.3, "Tuned #2"),
]

results = []
for M_disk, R_d, rho_c, n_exp, desc in test_cases:
    v_n, rho_val, xi_val, v_total = test_params(M_disk, R_d, rho_c, n_exp)
    results.append((desc, M_disk, R_d, rho_c, n_exp, v_n, xi_val, v_total))
    
    print(f"\n{desc}:")
    print(f"  M_disk = {M_disk:.1e} M☉, R_d = {R_d:.1f} kpc")
    print(f"  ρ_c = {rho_c:.1e} M☉/kpc³, n = {n_exp:.1f}")
    print(f"  v_newton = {v_n:.1f} km/s, ξ = {xi_val:.2f}")
    print(f"  v_total = {v_total:.1f} km/s {'✓' if abs(v_total - 220) < 10 else ''}")

print("\n" + "="*60)
print("BEST MATCHES (within 10 km/s of 220):")
for desc, M_disk, R_d, rho_c, n_exp, v_n, xi_val, v_total in results:
    if abs(v_total - 220) < 10:
        print(f"\n{desc}: v = {v_total:.1f} km/s")
        print(f"  --M_disk_thin_fixed {M_disk:.0e} --R_d_thin_fixed {R_d:.1f} \\")
        print(f"  --rho_c_fixed {rho_c:.0e} --n_exp_fixed {n_exp:.1f}")

if not any(abs(r[-1] - 220) < 10 for r in results):
    print("\nNone found! Try adjusting parameters further.")
    print("Your model needs ~40% more velocity.")