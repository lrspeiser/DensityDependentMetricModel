#!/usr/bin/env python3
"""
diagnose_high_rmse_fixed.py - Fixed diagnostic script for gravitational color model
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import your modules
from data_io import load_gaia
from density_metric2 import (
    v_baryon_total_newtonian_kms, 
    rho_baryon_total_midplane_solar_kpc3,
    xi_power_law,
    xi_gravitational_color,
    XI_FUNCTION_MAP,
    R_SUN_KPC
)
from run_dynesty import get_param_labels_and_bounds, PHYSICAL_BOUNDS

def diagnose_data_quality(gaia_data):
    """Check data for common issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    R = gaia_data['R_kpc']
    v = gaia_data['v_obs']
    sigma = gaia_data['sigma_v']
    
    print(f"Number of stars: {len(R)}")
    print(f"R range: [{R.min():.2f}, {R.max():.2f}] kpc")
    print(f"v range: [{v.min():.1f}, {v.max():.1f}] km/s")
    print(f"sigma range: [{sigma.min():.1f}, {sigma.max():.1f}] km/s")
    
    # Check for outliers
    v_median = np.median(v)
    v_mad = np.median(np.abs(v - v_median))
    outliers = np.abs(v - v_median) > 5 * v_mad
    n_outliers = np.sum(outliers)
    
    print(f"\nVelocity statistics:")
    print(f"  Median: {v_median:.1f} km/s")
    print(f"  MAD: {v_mad:.1f} km/s")
    print(f"  Outliers (>5 MAD): {n_outliers} ({n_outliers/len(v)*100:.1f}%)")
    
    if n_outliers > 0:
        print(f"  Outlier velocities: {v[outliers][:10]}...")  # Show first 10
    
    # Check if velocities are reasonable for MW
    expected_v_range = (150, 250)  # km/s
    v_in_range = (v > expected_v_range[0]) & (v < expected_v_range[1])
    print(f"\nVelocities in expected MW range {expected_v_range}: {np.sum(v_in_range)/len(v)*100:.1f}%")
    
    # Plot velocity distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.hist(v, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(v_median, color='red', linestyle='--', label=f'Median: {v_median:.1f}')
    plt.axvspan(expected_v_range[0], expected_v_range[1], alpha=0.2, color='green', label='Expected MW range')
    plt.xlabel('v (km/s)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Velocity Distribution')
    
    plt.subplot(132)
    plt.scatter(R, v, alpha=0.5, s=10)
    plt.xlabel('R (kpc)')
    plt.ylabel('v (km/s)')
    plt.title('Rotation Curve Data')
    plt.axhspan(expected_v_range[0], expected_v_range[1], alpha=0.2, color='green')
    
    plt.subplot(133)
    plt.scatter(R, sigma, alpha=0.5, s=10)
    plt.xlabel('R (kpc)')
    plt.ylabel('σ_v (km/s)')
    plt.title('Velocity Errors')
    
    plt.tight_layout()
    plt.savefig('data_diagnostics.png', dpi=150)
    print("\nSaved data diagnostics plot to: data_diagnostics.png")
    
    return n_outliers > 0.1 * len(v)  # True if >10% outliers

def test_model_at_typical_params(gaia_data, args):
    """Test model with typical MW parameters"""
    print("\n" + "="*60)
    print("MODEL TEST WITH TYPICAL PARAMETERS")
    print("="*60)
    
    # Set typical MW parameters
    typical_params = {
        'rho_c_solar_kpc3': 1e8,  # Changed to typical for grav_color
        'gamma': 2.7,  # Theory value
        'lambda_g': 8.0,  # Theory value
        'n_exp': 1.5,  # For fallback
        'M_disk_thin_solar': 5e10,
        'R_d_thin_kpc': 2.6,
        'h_z_thin_kpc': 0.3,
        'M_disk_thick_solar': 1.5e10,
        'R_d_thick_kpc': 4.0,
        'h_z_thick_kpc': 0.9,
        'M_bulge_solar': 1.5e10,
        'a_bulge_kpc': 0.7,
        'M_gas_solar': 3e10,
        'R_d_gas_kpc': 7.0,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': args.include_disk_thin,
        'include_disk_thick': args.include_disk_thick,
        'include_bulge': args.include_bulge,
        'include_gas': args.include_gas,
        'include_bulge_density': args.include_bulge
    }
    
    print("\nTypical parameters:")
    for key, val in typical_params.items():
        if isinstance(val, bool):
            print(f"  {key}: {val}")
        elif isinstance(val, float) and val > 1e6:
            print(f"  {key}: {val:.2e}")
        else:
            print(f"  {key}: {val}")
    
    # Calculate model velocities
    R_data = gaia_data['R_kpc']
    v_newton = v_baryon_total_newtonian_kms(R_data, typical_params)
    rho_midplane = rho_baryon_total_midplane_solar_kpc3(R_data, typical_params)
    
    # Apply xi modification based on xi type
    if args.xi == 'grav_color':
        print("\nUsing gravitational color model")
        xi_vals = xi_gravitational_color(
            rho_midplane, 
            typical_params['rho_c_solar_kpc3'], 
            typical_params['gamma'],
            typical_params['lambda_g']
        )
    else:
        xi_func = XI_FUNCTION_MAP.get(args.xi, xi_power_law)
        xi_vals = xi_func(rho_midplane, typical_params['rho_c_solar_kpc3'], typical_params['n_exp'])
    
    v_model = v_newton * np.sqrt(xi_vals)
    
    # Calculate RMSE
    v_obs = gaia_data['v_obs']
    rmse = np.sqrt(np.mean((v_model - v_obs)**2))
    
    print(f"\nModel predictions:")
    print(f"  v_Newton range: [{v_newton.min():.1f}, {v_newton.max():.1f}] km/s")
    print(f"  ξ range: [{xi_vals.min():.3f}, {xi_vals.max():.3f}]")
    print(f"  v_model range: [{v_model.min():.1f}, {v_model.max():.1f}] km/s")
    print(f"  RMSE: {rmse:.1f} km/s")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(R_data, v_obs, alpha=0.3, s=10, label='Data')
    
    # Sort for smooth line
    sort_idx = np.argsort(R_data)
    plt.plot(R_data[sort_idx], v_newton[sort_idx], 'b-', label='Newton', linewidth=2)
    plt.plot(R_data[sort_idx], v_model[sort_idx], 'r-', label=f'Model (ξ×Newton)', linewidth=2)
    
    plt.xlabel('R (kpc)')
    plt.ylabel('v (km/s)')
    plt.legend()
    plt.title(f'Model Test (RMSE = {rmse:.1f} km/s)')
    plt.ylim(0, 350)  # Set reasonable y-limits
    
    plt.subplot(122)
    plt.scatter(R_data, xi_vals, alpha=0.5, s=10)
    plt.xlabel('R (kpc)')
    plt.ylabel('ξ')
    plt.title(f'Xi function ({args.xi})')
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Newtonian')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_test_typical.png', dpi=150)
    print("\nSaved model test plot to: model_test_typical.png")
    
    # Test at solar radius specifically
    print(f"\nAt Solar radius (R = {R_SUN_KPC:.1f} kpc):")
    v_newton_solar = v_baryon_total_newtonian_kms(np.array([R_SUN_KPC]), typical_params)[0]
    rho_solar = rho_baryon_total_midplane_solar_kpc3(np.array([R_SUN_KPC]), typical_params)[0]
    
    if args.xi == 'grav_color':
        xi_solar = xi_gravitational_color(
            rho_solar, 
            typical_params['rho_c_solar_kpc3'], 
            typical_params['gamma'],
            typical_params['lambda_g']
        )[0]
    else:
        xi_func = XI_FUNCTION_MAP.get(args.xi, xi_power_law)
        xi_solar = xi_func(rho_solar, typical_params['rho_c_solar_kpc3'], typical_params['n_exp'])[0]
    
    v_model_solar = v_newton_solar * np.sqrt(xi_solar)
    
    print(f"  v_Newton: {v_newton_solar:.1f} km/s")
    print(f"  ρ(R☉,z=0): {rho_solar:.2e} M☉/kpc³")
    print(f"  ξ(R☉): {xi_solar:.3f}")
    print(f"  v_model: {v_model_solar:.1f} km/s")
    
    return rmse

def test_xi_behavior(args):
    """Test xi function behavior"""
    print("\n" + "="*60)
    print(f"XI FUNCTION BEHAVIOR TEST ({args.xi})")
    print("="*60)
    
    # Test density range
    rho_test = np.logspace(4, 10, 100)  # 1e4 to 1e10 M☉/kpc³
    rho_c_test = 1e8
    
    if args.xi == 'grav_color':
        gamma_test = 2.7
        lambda_test = 8.0
        xi_vals = xi_gravitational_color(rho_test, rho_c_test, gamma_test, lambda_test)
        print(f"\nGravitational color parameters:")
        print(f"  ρ_c = {rho_c_test:.1e} M☉/kpc³")
        print(f"  γ = {gamma_test}")
        print(f"  λ = {lambda_test}")
    else:
        n_test = 1.5
        xi_func = XI_FUNCTION_MAP.get(args.xi, xi_power_law)
        xi_vals = xi_func(rho_test, rho_c_test, n_test)
        print(f"\nXi function: {args.xi}")
        print(f"Parameters: ρ_c = {rho_c_test:.1e} M☉/kpc³, n = {n_test}")
    
    print(f"Xi range: [{xi_vals.min():.3f}, {xi_vals.max():.3f}]")
    
    # Check behavior at key densities
    test_points = [
        (1e4, "Very low density"),
        (1e6, "Low density (voids)"),
        (1e7, "Medium-low density"),
        (1e8, "ρ = ρ_c"),
        (1e9, "High density"),
        (1e10, "Very high density")
    ]
    
    print("\nXi at key densities:")
    for rho, desc in test_points:
        if args.xi == 'grav_color':
            xi = xi_gravitational_color(rho, rho_c_test, gamma_test, lambda_test)[0]
        else:
            xi = xi_func(rho, rho_c_test, n_test)[0]
        print(f"  ρ = {rho:.1e} ({desc}): ξ = {xi:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.loglog(rho_test, xi_vals)
    plt.axvline(rho_c_test, color='r', linestyle='--', label='ρ_c')
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Newtonian')
    plt.xlabel('ρ (M☉/kpc³)')
    plt.ylabel('ξ')
    plt.title(f'Xi function: {args.xi}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.1, 10)  # Set reasonable limits
    
    plt.subplot(122)
    # Show velocity modification
    v_base = 220  # km/s
    v_modified = v_base * np.sqrt(xi_vals)
    plt.semilogx(rho_test, v_modified)
    plt.axvline(rho_c_test, color='r', linestyle='--', label='ρ_c')
    plt.axhline(v_base, color='k', linestyle='--', alpha=0.5, label='Newtonian')
    plt.xlabel('ρ (M☉/kpc³)')
    plt.ylabel('v_modified (km/s)')
    plt.title(f'Velocity modification (v_base = {v_base} km/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xi_function_test_grav_color.png', dpi=150)
    print("\nSaved xi function test to: xi_function_test_grav_color.png")

def main():
    parser = argparse.ArgumentParser(description="Diagnose high RMSE issues")
    parser.add_argument('--max_sample_gaia', type=int, default=20000)
    parser.add_argument('--xi', type=str, default='grav_color', 
                       choices=['power', 'logistic', 'enhanced', 'mond', 'exp_enhance', 'grav_color'])
    parser.add_argument('--include_bulge', action='store_true', default=False)
    parser.add_argument('--include_disk_thin', action='store_true', default=True)
    parser.add_argument('--include_disk_thick', action='store_true', default=False)
    parser.add_argument('--include_gas', action='store_true', default=False)
    
    args = parser.parse_args()
    
    print("="*60)
    print("HIGH RMSE DIAGNOSTIC TOOL - GRAVITATIONAL COLOR MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading Gaia data...")
    gaia_data = load_gaia(
        sample_max=args.max_sample_gaia,
        force_new_query_gaia=False,
        force_reprocess_raw=False,
        processed_cache_filename="gaia_cache/gaia_query_cache_DR3_processed_for_fit.parquet"
    )
    
    if gaia_data is None:
        print("ERROR: Failed to load data!")
        return
    
    # Run diagnostics
    has_outliers = diagnose_data_quality(gaia_data)
    
    if has_outliers:
        print("\n⚠️  WARNING: Data has many outliers. Consider cleaning or using robust fitting.")
    
    # Test model
    rmse = test_model_at_typical_params(gaia_data, args)
    
    if rmse > 100:
        print(f"\n⚠️  WARNING: Even with typical parameters, RMSE = {rmse:.1f} km/s")
        print("   This suggests a fundamental issue with:")
        print("   1. Data quality or units")
        print("   2. Model configuration")
        print("   3. Xi function behavior")
    
    # Test xi function
    test_xi_behavior(args)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if rmse > 50:
        print("\n🔴 HIGH RMSE DETECTED. Likely causes:")
        print("1. Need more mass components (bulge, thick disk, gas)")
        print("2. ρ_c might need adjustment (try 1e7 - 1e9 range)")
        print("3. Single thin disk might be too simple")
        print(f"4. Current model components: thin={args.include_disk_thin}, thick={args.include_disk_thick}, bulge={args.include_bulge}, gas={args.include_gas}")
    else:
        print("\n🟢 Model seems reasonable with typical parameters.")
        print("   RMSE < 50 km/s is good for MW rotation curve fitting")

if __name__ == "__main__":
    main()