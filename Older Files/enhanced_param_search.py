#!/usr/bin/env python3
"""
enhanced_param_search.py - More aggressive parameter search to hit 220 km/s
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

def test_params(M_disk, R_d, rho_c, n_exp, h_z=0.3):
    """Test a parameter combination"""
    params = base_params.copy()
    params.update({
        'M_disk_thin_solar': M_disk,
        'R_d_thin_kpc': R_d,
        'rho_c_solar_kpc3': rho_c,
        'n_exp': n_exp,
        'h_z_thin_kpc': h_z
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

print("ENHANCED PARAMETER SEARCH")
print("Target: v_total ≈ 220 km/s at R = 8 kpc")
print("="*70)

# Strategy 1: Much higher disk masses
print("\n1. HIGHER DISK MASSES:")
high_mass_cases = [
    (9e10, 3.0, 1e9, 1.0, "Very high mass, weak xi"),
    (1e11, 3.0, 2e9, 1.2, "Extremely high mass"),
    (1.2e11, 3.5, 5e8, 1.5, "Ultra high mass, large Rd"),
    (8e10, 2.8, 1e9, 0.5, "High mass, very weak xi"),
]

for M_disk, R_d, rho_c, n_exp, desc in high_mass_cases:
    v_n, rho_val, xi_val, v_total = test_params(M_disk, R_d, rho_c, n_exp)
    print(f"  {desc}:")
    print(f"    M={M_disk:.1e}, Rd={R_d:.1f}, ρc={rho_c:.1e}, n={n_exp:.1f}")
    print(f"    v_newton={v_n:.1f}, ξ={xi_val:.3f}, v_total={v_total:.1f} km/s")
    if abs(v_total - 220) < 5:
        print(f"    *** EXCELLENT MATCH! ***")
    elif abs(v_total - 220) < 15:
        print(f"    ** Good match **")
    print()

# Strategy 2: Nearly Newtonian (xi ≈ 1)
print("\n2. NEARLY NEWTONIAN APPROACH (ξ ≈ 1):")
newtonian_cases = [
    (7.5e10, 3.0, 1e10, 0.1, "Almost pure Newtonian"),
    (8e10, 3.2, 5e9, 0.2, "Newtonian with slight xi"),
    (7.8e10, 3.1, 2e10, 0.1, "Pure Newtonian tuned"),
]

for M_disk, R_d, rho_c, n_exp, desc in newtonian_cases:
    v_n, rho_val, xi_val, v_total = test_params(M_disk, R_d, rho_c, n_exp)
    print(f"  {desc}:")
    print(f"    M={M_disk:.1e}, Rd={R_d:.1f}, ρc={rho_c:.1e}, n={n_exp:.1f}")
    print(f"    v_newton={v_n:.1f}, ξ={xi_val:.3f}, v_total={v_total:.1f} km/s")
    if abs(v_total - 220) < 5:
        print(f"    *** EXCELLENT MATCH! ***")
    elif abs(v_total - 220) < 15:
        print(f"    ** Good match **")
    print()

# Strategy 3: Thick disk to reduce density
print("\n3. THICKER DISK (lower midplane density):")
thick_cases = [
    (8e10, 3.0, 5e8, 1.5, 0.5, "Thick disk, lower ρ"),
    (9e10, 3.2, 3e8, 1.8, 0.6, "Very thick disk"),
    (7.5e10, 2.8, 4e8, 1.3, 0.45, "Moderately thick"),
]

for M_disk, R_d, rho_c, n_exp, h_z, desc in thick_cases:
    v_n, rho_val, xi_val, v_total = test_params(M_disk, R_d, rho_c, n_exp, h_z)
    print(f"  {desc}:")
    print(f"    M={M_disk:.1e}, Rd={R_d:.1f}, hz={h_z:.1f}, ρc={rho_c:.1e}, n={n_exp:.1f}")
    print(f"    v_newton={v_n:.1f}, ξ={xi_val:.3f}, v_total={v_total:.1f} km/s")
    print(f"    midplane ρ={rho_val:.2e} M☉/kpc³")
    if abs(v_total - 220) < 5:
        print(f"    *** EXCELLENT MATCH! ***")
    elif abs(v_total - 220) < 15:
        print(f"    ** Good match **")
    print()

# Strategy 4: Grid search around promising region
print("\n4. FINE-TUNED GRID SEARCH:")
print("Searching around promising high-mass region...")

best_params = None
best_v = 0
best_diff = float('inf')

# Grid search
masses = np.linspace(9e10, 1.3e11, 8)
R_ds = [2.8, 3.0, 3.2, 3.5]
rho_cs = [1e8, 3e8, 5e8, 8e8, 1e9, 2e9]
n_exps = [0.5, 0.8, 1.0, 1.2, 1.5]

print("Testing grid (this may take a moment)...")
count = 0
good_matches = []

for M in masses:
    for Rd in R_ds:
        for rho_c in rho_cs:
            for n in n_exps:
                v_n, rho_val, xi_val, v_total = test_params(M, Rd, rho_c, n)
                diff = abs(v_total - 220)
                count += 1
                
                if diff < best_diff:
                    best_diff = diff
                    best_params = (M, Rd, rho_c, n, v_n, xi_val, v_total)
                
                if diff < 10:  # Within 10 km/s
                    good_matches.append((M, Rd, rho_c, n, v_n, xi_val, v_total, diff))

print(f"Tested {count} combinations.")
print(f"\nBEST OVERALL MATCH:")
if best_params:
    M, Rd, rho_c, n, v_n, xi_val, v_total = best_params
    print(f"  M_disk = {M:.2e} M☉")
    print(f"  R_d = {Rd:.1f} kpc") 
    print(f"  ρ_c = {rho_c:.1e} M☉/kpc³")
    print(f"  n = {n:.1f}")
    print(f"  → v_newton = {v_n:.1f} km/s, ξ = {xi_val:.3f}")
    print(f"  → v_total = {v_total:.1f} km/s (diff = {best_diff:.1f})")
    
    print(f"\n  Dynesty command:")
    print(f"  python run_dynesty.py --fit_xi_params --fit_disk_thin \\")
    print(f"    --M_disk_thin_fixed {M:.1e} --R_d_thin_fixed {Rd:.1f} \\")
    print(f"    --rho_c_fixed {rho_c:.1e} --n_exp_fixed {n:.1f} \\")
    print(f"    --max_sample_gaia 80000 --nlive_init 1000")

print(f"\nGOOD MATCHES (within 10 km/s):")
if good_matches:
    good_matches.sort(key=lambda x: x[-1])  # Sort by difference
    for i, (M, Rd, rho_c, n, v_n, xi_val, v_total, diff) in enumerate(good_matches[:5]):
        print(f"\n  Match #{i+1}: v = {v_total:.1f} km/s (±{diff:.1f})")
        print(f"    M={M:.2e}, Rd={Rd:.1f}, ρc={rho_c:.1e}, n={n:.1f}")
        print(f"    ξ={xi_val:.3f} (Newtonian boost: {1/np.sqrt(xi_val):.2f}x)")
else:
    print("  No matches within 10 km/s found.")
    print("  Consider:")
    print("  1. Even higher disk masses (1.4e11+ M☉)")
    print("  2. Adding bulge/thick disk components")  
    print("  3. Different xi function (logistic instead of power)")
    print("  4. Checking if Freeman velocity calculation is correct")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")