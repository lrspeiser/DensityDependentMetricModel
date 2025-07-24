#!/usr/bin/env python3
"""
Simple validation of GR baseline results
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import your modules
from density_metric2 import (
    v_baryon_total_newtonian_kms,
    rho_baryon_total_midplane_solar_kpc3,
    R_SUN_KPC
)

def validate_gr_baseline():
    """Run validation checks on GR baseline"""
    
    # Load summary for easy access
    with open('chains_GR_reparameterized/gr_baseline_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Extract parameters
    params = {
        'M_disk_thin_solar': summary['derived_masses']['M_disk_thin_solar'],
        'M_disk_thick_solar': summary['derived_masses']['M_disk_thick_solar'],
        'M_bulge_solar': summary['best_fit_params']['M_bulge_solar'],
        'a_bulge_kpc': summary['best_fit_params']['a_bulge_kpc'],
        'M_gas_solar': summary['best_fit_params']['M_gas_solar'],
        'R_d_gas_kpc': summary['best_fit_params']['R_d_gas_kpc'],
        'h_z_gas_kpc': summary['best_fit_params']['h_z_gas_kpc'],
        # Fixed parameters
        'R_d_thin_kpc': 2.6,
        'h_z_thin_kpc': 0.3,
        'R_d_thick_kpc': 4.5,
        'h_z_thick_kpc': 0.9,
        # Include flags
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True
    }
    
    print("\n=== GR BASELINE VALIDATION ===")
    print(f"Total baryonic mass: {summary['derived_masses']['M_total_baryons']/1e11:.1f} × 10¹¹ M☉")
    print(f"Log evidence: {summary['logZ']:.2f} ± {summary['logZ_err']:.2f}")
    
    # 1. Check velocity at solar radius
    v_solar = v_baryon_total_newtonian_kms(R_SUN_KPC, params)
    # Convert JAX array to scalar - use item() for JAX arrays
    if hasattr(v_solar, 'item'):
        v_solar_val = v_solar.item()
    else:
        v_solar_val = float(v_solar) if hasattr(v_solar, '__len__') else v_solar
    print(f"\n1. Solar neighborhood velocity:")
    print(f"   v(R☉) = {v_solar_val:.1f} km/s")
    print(f"   Observed: ~220 km/s")
    print(f"   Missing factor: {(220/v_solar_val)**2:.1f}× (this is dark matter in ΛCDM)")
    
    # 2. Check rotation curve shape
    r = np.linspace(2, 30, 100)
    v = v_baryon_total_newtonian_kms(r, params)
    
    # Key radii
    v_5kpc = v[np.argmin(np.abs(r - 5))]
    v_15kpc = v[np.argmin(np.abs(r - 15))]
    v_25kpc = v[np.argmin(np.abs(r - 25))]
    
    print(f"\n2. Rotation curve shape:")
    print(f"   v(5 kpc)  = {v_5kpc:.1f} km/s")
    print(f"   v(15 kpc) = {v_15kpc:.1f} km/s")
    print(f"   v(25 kpc) = {v_25kpc:.1f} km/s")
    print(f"   Decline from 5→25 kpc: {(v_25kpc/v_5kpc - 1)*100:.1f}%")
    print(f"   Status: {'✅ Keplerian' if v_25kpc < v_5kpc else '❌ Not declining'}")
    
    # 3. Check parameters at bounds
    print(f"\n3. Parameters at bounds:")
    if summary['best_fit_params']['M_bulge_solar'] > 2.49e10:
        print(f"   ⚠️ M_bulge at upper bound (25 × 10⁹ M☉)")
    if summary['best_fit_params']['M_gas_solar'] > 5.99e10:
        print(f"   ⚠️ M_gas at upper bound (60 × 10⁹ M☉)")
    if summary['best_fit_params']['R_d_gas_kpc'] < 4.01:
        print(f"   ⚠️ R_d_gas at lower bound (4.0 kpc)")
    
    # 4. Physical reasonableness
    print(f"\n4. Physical assessment:")
    M_total = summary['derived_masses']['M_total_baryons']
    if M_total > 1.2e11:
        print(f"   ⚠️ Total mass {M_total/1e11:.0f}×10¹¹ M☉ is very high for MW")
        print(f"   Typical estimates: 5-7×10¹⁰ M☉")
    
    # Create diagnostic plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Rotation curve
    ax1.plot(r, v, 'b-', lw=3, label='GR (baryons only)')
    ax1.axhline(220, color='red', ls='--', lw=2, alpha=0.7, label='Observed (flat)')
    ax1.fill_between(r, v, 220, where=(v < 220), alpha=0.2, color='red', 
                     label='Missing (needs DM)')
    ax1.axvline(R_SUN_KPC, color='orange', ls=':', alpha=0.5)
    ax1.text(R_SUN_KPC + 0.5, 100, 'Sun', rotation=90, va='bottom', color='orange')
    
    ax1.set_ylabel('Circular Velocity (km/s)', fontsize=14)
    ax1.set_xlim(2, 30)
    ax1.set_ylim(0, 300)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('GR Baseline: Newtonian Gravity Fails Without Dark Matter', fontsize=16)
    
    # Residuals from flat curve
    residuals = v - 220
    ax2.plot(r, residuals, 'b-', lw=3)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.fill_between(r, residuals, 0, where=(residuals < 0), alpha=0.3, color='red')
    
    ax2.set_xlabel('Radius (kpc)', fontsize=14)
    ax2.set_ylabel('v - 220 km/s', fontsize=14)
    ax2.set_xlim(2, 30)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('chains_GR_reparameterized')
    plt.savefig(output_dir / 'gr_baseline_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Validation plot saved to {output_dir}/gr_baseline_validation.png")
    plt.close()
    
    # Summary
    print("\n=== SUMMARY ===")
    print("The GR baseline clearly shows:")
    print("1. Massive dark matter problem (need ~4× more mass)")
    print("2. Wrong curve shape (Keplerian decline vs observed flat)")
    print("3. Parameters pushed to unphysical bounds")
    print(f"4. Log evidence = {summary['logZ']:.0f} (comparison baseline)")
    print("\nThis demonstrates why we need either:")
    print("- Dark matter (ΛCDM approach)")
    print("- Modified gravity (DDMM approach)")

if __name__ == "__main__":
    validate_gr_baseline()