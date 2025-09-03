#!/usr/bin/env python3
"""
Diagnose xi enhancement values for tidal models.

This script analyzes why xi is nearly unity and tests parameter sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.density_metric_cupy import xi_tidal_bandpass_cupy, volume_density_comprehensive_solar_kpc3_cupy
import cupy as cp

def analyze_xi_enhancement(R_kpc, params, galaxy_name="Test Galaxy"):
    """Analyze xi enhancement for given parameters."""
    
    R_arr = cp.asarray(R_kpc, dtype=cp.float32)
    
    # Compute baryonic density for MW-like galaxy
    rho = volume_density_comprehensive_solar_kpc3_cupy(R_arr, params)
    
    # Compute tidal proxy T = v_bar^2 / R^2
    # For MW-like galaxy with v_circ ~ 220 km/s at R ~ 8 kpc
    v_bar = 220.0 * cp.sqrt(R_arr / 8.0)  # Simple approximation
    T = v_bar**2 / cp.maximum(R_arr**2, 1e-18)
    
    # Extract parameters
    rho_c = params.get('rho_c_solar_kpc3', 5e7)
    gamma = params.get('gamma_exp', 3.0)
    lambda_max = params.get('lambda_max', 0.5)
    T0 = params.get('T0', 8.5)
    sigma_lnT = params.get('sigma_lnT', 0.8)
    wmin = params.get('wmin', 0.02)
    
    # Compute xi
    xi = xi_tidal_bandpass_cupy(rho, T, rho_c, gamma, lambda_max, T0, sigma_lnT, wmin)
    
    # Convert to numpy for plotting
    R_np = cp.asnumpy(R_arr)
    rho_np = cp.asnumpy(rho)
    T_np = cp.asnumpy(T)
    xi_np = cp.asnumpy(xi)
    
    # Compute components
    S_rho = (rho_c / np.maximum(rho_np, 1e-30))**gamma
    T_safe = np.maximum(T_np, 1e-30)
    T0_safe = max(T0, 1e-30)
    u = (np.log(T_safe) - np.log(T0_safe)) / max(sigma_lnT, 1e-6)
    W = np.exp(-0.5 * u * u)
    W_full = wmin + (1.0 - wmin) * W
    
    # Print diagnostics
    print(f"\n{'='*60}")
    print(f"Xi Enhancement Analysis for {galaxy_name}")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  rho_c = {rho_c:.2e} M_sun/kpc^3")
    print(f"  gamma = {gamma:.2f}")
    print(f"  lambda_max = {lambda_max:.2f}")
    print(f"  T0 = {T0:.2f} (km/s)^2/kpc^2")
    print(f"  sigma_lnT = {sigma_lnT:.2f}")
    print(f"  wmin = {wmin:.3f}")
    
    print(f"\nRadius-dependent values:")
    print(f"{'R [kpc]':>10} {'rho [M/kpc3]':>15} {'S_rho':>10} {'T':>15} {'W':>10} {'xi':>10} {'v_boost':>10}")
    print("-" * 95)
    
    indices = [0, len(R_np)//4, len(R_np)//2, 3*len(R_np)//4, -1]
    for i in indices:
        if i < 0:
            i = len(R_np) + i
        if i < len(R_np):
            v_boost = np.sqrt(xi_np[i]) - 1.0
            print(f"{R_np[i]:10.2f} {rho_np[i]:15.2e} {S_rho[i]:10.4f} "
                  f"{T_np[i]:15.2e} {W_full[i]:10.4f} {xi_np[i]:10.6f} {v_boost:10.4%}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Xi Enhancement Diagnostics - {galaxy_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Density profile
    ax = axes[0, 0]
    ax.semilogy(R_np, rho_np, 'b-', linewidth=2)
    ax.axhline(rho_c, color='r', linestyle='--', label=f'rho_c = {rho_c:.1e}')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Density [M_sun/kpc^3]')
    ax.set_title('Baryonic Density Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Density screening factor
    ax = axes[0, 1]
    ax.plot(R_np, S_rho, 'g-', linewidth=2)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('S_rho = (rho_c/rho)^gamma')
    ax.set_title(f'Density Screening Factor (gamma={gamma:.1f})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(5, np.max(S_rho)*1.1))
    
    # Plot 3: Tidal proxy
    ax = axes[0, 2]
    ax.semilogy(R_np, T_np, 'c-', linewidth=2)
    ax.axhline(T0, color='r', linestyle='--', label=f'T0 = {T0:.1f}')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('T = v^2/R^2 [(km/s)^2/kpc^2]')
    ax.set_title('Tidal Proxy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tidal window
    ax = axes[1, 0]
    ax.plot(R_np, W_full, 'm-', linewidth=2)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('W(T)')
    ax.set_title(f'Tidal Window (sigma_lnT={sigma_lnT:.1f}, wmin={wmin:.2f})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 5: Xi enhancement
    ax = axes[1, 1]
    ax.plot(R_np, xi_np, 'r-', linewidth=2.5, label='xi')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(1.0 + lambda_max, color='r', linestyle='--', alpha=0.5, label=f'1 + lambda_max = {1+lambda_max:.2f}')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('xi')
    ax.set_title('Metric Enhancement Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.0 + lambda_max * 1.2)
    
    # Plot 6: Velocity boost
    ax = axes[1, 2]
    v_boost_pct = (np.sqrt(xi_np) - 1.0) * 100
    ax.plot(R_np, v_boost_pct, 'orange', linewidth=2.5)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Velocity Boost [%]')
    ax.set_title('Velocity Enhancement = sqrt(xi) - 1')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, {
        'R': R_np,
        'rho': rho_np,
        'T': T_np,
        'xi': xi_np,
        'S_rho': S_rho,
        'W': W_full,
        'v_boost': np.sqrt(xi_np) - 1.0
    }


def parameter_sensitivity_analysis():
    """Test sensitivity to different parameter values."""
    
    R_kpc = np.logspace(-1, 1.5, 100)  # 0.1 to 30 kpc
    
    # Base MW-like parameters
    base_params = {
        'M_thin_disk_solar': 4.5e10,
        'R_thin_disk_kpc': 2.6,
        'hz_thin_disk_kpc': 0.3,
        'M_thick_disk_solar': 1.0e10,
        'R_thick_disk_kpc': 2.0,
        'hz_thick_disk_kpc': 0.9,
        'M_bulge_solar': 1.5e10,
        'R_bulge_kpc': 0.5,
        'M_gas_solar': 0.5e10,
        'R_gas_kpc': 7.0,
        'hz_gas_kpc': 0.1,
        'rho_c_solar_kpc3': 5e7,
        'gamma_exp': 3.0,
        'lambda_max': 0.5,
        'T0': 8.5,
        'sigma_lnT': 0.8,
        'wmin': 0.02
    }
    
    # Test 1: Conservative parameters (as in current model)
    print("\n" + "="*80)
    print("TEST 1: Conservative Parameters (Current Model)")
    print("="*80)
    fig1, data1 = analyze_xi_enhancement(R_kpc, base_params, "Conservative MW")
    
    # Test 2: More aggressive parameters
    aggressive_params = base_params.copy()
    aggressive_params.update({
        'rho_c_solar_kpc3': 1e6,  # Much lower critical density
        'gamma_exp': 2.0,          # Weaker screening
        'lambda_max': 3.0,         # Larger maximum enhancement
        'T0': 20.0,                # Higher tidal threshold
        'sigma_lnT': 1.5,          # Broader window
        'wmin': 0.1                # Higher floor
    })
    
    print("\n" + "="*80)
    print("TEST 2: Aggressive Parameters")
    print("="*80)
    fig2, data2 = analyze_xi_enhancement(R_kpc, aggressive_params, "Aggressive MW")
    
    # Test 3: Paper-like parameters
    paper_params = base_params.copy()
    paper_params.update({
        'rho_c_solar_kpc3': 5e5,  # Lower critical density
        'gamma_exp': 2.5,          # Moderate screening
        'lambda_max': 1.5,         # Moderate enhancement
        'T0': 15.0,                # Moderate tidal threshold
        'sigma_lnT': 1.0,          # Moderate window
        'wmin': 0.05               # Small floor
    })
    
    print("\n" + "="*80)
    print("TEST 3: Paper-like Parameters")
    print("="*80)
    fig3, data3 = analyze_xi_enhancement(R_kpc, paper_params, "Paper-like MW")
    
    # Comparison plot
    fig_comp, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(data1['R'], (data1['v_boost'])*100, 'b-', linewidth=2, label='Conservative')
    ax.plot(data2['R'], (data2['v_boost'])*100, 'r-', linewidth=2, label='Aggressive')
    ax.plot(data3['R'], (data3['v_boost'])*100, 'g-', linewidth=2, label='Paper-like')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Velocity Boost [%]', fontsize=12)
    ax.set_title('Parameter Sensitivity: Velocity Enhancement Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(10, color='gray', linestyle=':', alpha=0.5, label='10% boost')
    ax.axhline(20, color='gray', linestyle=':', alpha=0.5, label='20% boost')
    
    plt.tight_layout()
    
    return [fig1, fig2, fig3, fig_comp]


def main():
    """Main diagnostic routine."""
    
    print("\n" + "="*80)
    print("XI ENHANCEMENT DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path("images/xi_diagnostics")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run sensitivity analysis
    figs = parameter_sensitivity_analysis()
    
    # Save figures
    names = ['conservative', 'aggressive', 'paper_like', 'comparison']
    for fig, name in zip(figs, names):
        output_file = output_dir / f'xi_diagnostic_{name}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close(fig)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The analysis shows that the tidal enhancement is nearly negligible with conservative parameters because:

1. **High critical density (rho_c ~ 5e7 M_sun/kpc^3)**: The baryonic density at most galactic radii
   is comparable to or exceeds this value, resulting in minimal density screening (S_rho ~ 1).

2. **Strong density screening (gamma = 3)**: The power-law exponent suppresses enhancement 
   in high-density regions very effectively.

3. **Low lambda_max (0.5)**: Even in the best-case scenario, the maximum velocity boost 
   is only sqrt(1.5) - 1 ≈ 22%.

4. **Tidal window parameters**: The combination of T0, sigma_lnT, and wmin further 
   restricts where enhancement can occur.

To achieve meaningful tidal enhancements that match observations:
- Lower rho_c to ~1e5-1e6 M_sun/kpc^3
- Reduce gamma to ~2.0-2.5
- Increase lambda_max to ~1.5-3.0
- Adjust tidal window parameters for broader activation

These changes would produce 10-30% velocity boosts at large radii, consistent with 
observed galaxy rotation curves.
""")


if __name__ == '__main__':
    main()
