#!/usr/bin/env python3
"""
check_xi_contribution.py - Check if xi is contributing to the fit
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from density_metric2 import v_total_kms, v_baryon_total_newtonian_kms, XI_FUNCTION_MAP, R_SUN_KPC

def check_xi_contribution(output_dir="chains_dynesty"):
    """Check how much xi is contributing to the rotation curve"""
    
    print("=== CHECKING XI CONTRIBUTION ===\n")
    
    # Try to load run configuration
    config_file = Path(output_dir) / "run_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("Run configuration:")
        print(f"  xi type: {config.get('xi', 'unknown')}")
        print(f"  fit_xi_params: {config.get('fit_xi_params', False)}")
        if 'rho_c_fixed' in config:
            print(f"  rho_c_fixed: {config['rho_c_fixed']:.2e}")
        if 'n_exp_fixed' in config:
            print(f"  n_exp_fixed: {config['n_exp_fixed']}")
    
    # Current best-fit parameters from your output
    params = {
        # Baryonic parameters
        'M_disk_thin_solar': 8.87e10,
        'R_d_thin_kpc': 3.54,
        'h_z_thin_kpc': 0.28,
        'M_disk_thick_solar': 2.29e10,
        'R_d_thick_kpc': 5.13,
        'h_z_thick_kpc': 1.19,
        'M_bulge_solar': 6.08e9,
        'a_bulge_kpc': 1.70,
        'M_gas_solar': 5.55e10,
        'R_d_gas_kpc': 5.51,
        'h_z_gas_kpc': 0.25,
        # Include flags
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
        # Xi parameters - THESE MIGHT BE MISSING OR FIXED!
        'rho_c_solar_kpc3': 1e13,  # You need to check what value is actually being used
        'n_exp': 1.5,
        'A': 1.0
    }
    
    # Check rotation curves
    R_test = np.linspace(1, 30, 200)
    
    # Newtonian velocity
    v_newton = v_baryon_total_newtonian_kms(R_test, params)
    
    # Modified velocity (with xi)
    v_total = v_total_kms(R_test, params, xi_type='power')  # Adjust xi_type as needed
    
    # Calculate xi enhancement
    xi_values = (v_total / np.maximum(v_newton, 1e-10))**2
    
    # Values at solar radius
    idx_solar = np.argmin(np.abs(R_test - R_SUN_KPC))
    v_newton_solar = v_newton[idx_solar]
    v_total_solar = v_total[idx_solar]
    xi_solar = xi_values[idx_solar]
    
    print(f"\nAt Solar Radius (R = {R_SUN_KPC} kpc):")
    print(f"  v_newton = {v_newton_solar:.1f} km/s")
    print(f"  v_total = {v_total_solar:.1f} km/s")
    print(f"  xi = {xi_solar:.3f}")
    print(f"  Enhancement = {(xi_solar - 1) * 100:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top: Rotation curves
    ax1.plot(R_test, v_newton, 'b-', linewidth=2, label='Newtonian (baryons only)')
    ax1.plot(R_test, v_total, 'r-', linewidth=2, label=f'Modified (with ξ)')
    ax1.axvline(R_SUN_KPC, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(220, color='gray', linestyle=':', alpha=0.5, label='MW observed')
    ax1.set_ylabel('v_circ [km/s]')
    ax1.set_ylim(0, 300)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Rotation Curves: Checking ξ Contribution')
    
    # Bottom: Xi enhancement
    ax2.plot(R_test, xi_values, 'g-', linewidth=2)
    ax2.axvline(R_SUN_KPC, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(1.0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('R [kpc]')
    ax2.set_ylabel('ξ (enhancement factor)')
    ax2.set_ylim(0.8, 2.0)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with key info
    textstr = f'Total Baryonic Mass: {(params["M_disk_thin_solar"] + params["M_disk_thick_solar"] + params["M_bulge_solar"] + params["M_gas_solar"]):.2e} M☉\n'
    textstr += f'ξ at R☉: {xi_solar:.3f}\n'
    textstr += f'v(R☉): {v_total_solar:.1f} km/s'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('xi_contribution_check.png', dpi=150)
    print(f"\n✅ Saved plot to xi_contribution_check.png")
    
    # Diagnosis
    print("\n=== DIAGNOSIS ===")
    if xi_solar < 1.05:
        print("❌ Xi is providing almost NO enhancement!")
        print("   This explains why the sampler needs so much baryonic mass")
        print("   Possible issues:")
        print("   1. Xi parameters might be fixed at values that give no enhancement")
        print("   2. rho_c might be too high compared to disk density")
        print("   3. Xi function might not be implemented correctly")
    elif xi_solar < 1.2:
        print("⚠️  Xi enhancement is weak (~10-20%)")
        print("   The model needs significant baryonic mass to compensate")
    else:
        print("✓ Xi is providing reasonable enhancement")
    
    print("\nRECOMMENDATIONS:")
    if params['M_disk_thin_solar'] > 7e10:
        print("1. The thin disk mass is unrealistically high")
        print("   Either increase the upper bound or check why xi isn't helping")
    print("2. Check if --fit_xi_params was used in the run")
    print("3. Try running with explicit xi parameters:")
    print("   --rho_c_fixed 1e8 --n_exp_fixed 1.0")

if __name__ == "__main__":
    check_xi_contribution()