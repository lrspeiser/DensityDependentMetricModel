#!/usr/bin/env python3
"""
Plot SPARC galaxies with enhanced tidal model using better-tuned parameters.

This script:
1. Uses actual SPARC baryon components for GR baseline
2. Applies tidal model with physically plausible parameters that produce meaningful enhancements
3. Compares with NFW dark matter model
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy, BASE_M_L_3_6_MICRON_DISK
from models.er_sparc import v_bar_from_components
from models.nfw import v_model_nfw
import cupy as cp
from core.density_metric_cupy import xi_tidal_bandpass_cupy

def compute_enhanced_tidal_velocity(
    R_kpc: np.ndarray,
    v_bar: np.ndarray,
    rho_midplane: np.ndarray,
    params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tidal enhanced velocity with tuned parameters.
    
    Returns:
        (v_tidal, xi) - enhanced velocity and xi factor
    """
    R_cp = cp.asarray(R_kpc, dtype=cp.float32)
    v_bar_cp = cp.asarray(v_bar, dtype=cp.float32)
    rho_cp = cp.asarray(rho_midplane, dtype=cp.float32)
    
    # Compute tidal proxy T = v_bar^2 / R^2
    T = v_bar_cp**2 / cp.maximum(R_cp**2, 1e-18)
    
    # Extract parameters with better defaults for meaningful enhancement
    rho_c = params.get('rho_c_solar_kpc3', 1e6)  # Lower critical density
    gamma = params.get('gamma_exp', 2.2)          # Weaker screening
    lambda_max = params.get('lambda_max', 2.0)    # Higher max enhancement
    T0 = params.get('T0', 25.0)                   # Adjusted tidal threshold
    sigma_lnT = params.get('sigma_lnT', 1.2)      # Broader window
    wmin = params.get('wmin', 0.08)               # Higher floor
    
    # Compute xi enhancement
    xi = xi_tidal_bandpass_cupy(rho_cp, T, rho_c, gamma, lambda_max, T0, sigma_lnT, wmin)
    
    # Apply enhancement to velocity
    v_tidal = v_bar_cp * cp.sqrt(cp.maximum(xi, 1.0))
    
    # Convert back to numpy
    return cp.asnumpy(v_tidal), cp.asnumpy(xi)


def fit_nfw_simple(R: np.ndarray, v_obs: np.ndarray, v_bar: np.ndarray, 
                   e_v: np.ndarray) -> Tuple[float, float]:
    """Simple NFW fitting using grid search."""
    from scipy.optimize import minimize
    
    def chi2(params):
        V200, c = params
        v_model = v_model_nfw(R, v_bar, V200, c)
        weights = 1.0 / np.maximum(e_v, 1.0)**2
        return np.sum(weights * (v_obs - v_model)**2)
    
    # Initial guess and bounds
    x0 = [150.0, 10.0]
    bounds = [(40.0, 400.0), (2.0, 40.0)]
    
    try:
        result = minimize(chi2, x0, method='L-BFGS-B', bounds=bounds)
        return result.x[0], result.x[1]
    except:
        return 150.0, 10.0  # Default values if fitting fails


def plot_galaxy_comparison(
    galaxy_id: str,
    sparc_dir: Path,
    output_dir: Path,
    tidal_params: Optional[Dict] = None,
    show_components: bool = True
) -> None:
    """Create comparison plot for a single SPARC galaxy."""
    
    # Load galaxy data
    data = load_single_sparc_galaxy(galaxy_id, sparc_dir=str(sparc_dir))
    if data is None:
        print(f"Failed to load {galaxy_id}")
        return
    
    # Extract data arrays
    R = np.asarray(data['R_kpc'], dtype=float)
    V_obs = np.asarray(data['V_obs'], dtype=float)
    e_V = np.asarray(data['e_V_obs'], dtype=float)
    V_gas = np.asarray(data['V_gas_comp_kms'], dtype=float)
    V_disk = np.asarray(data['V_disk_comp_kms'], dtype=float)
    V_bulge = np.asarray(data['V_bulge_comp_kms'], dtype=float)
    
    # Get midplane densities
    rho_star_mid = np.asarray(data['rho_star_mid_Msun_kpc3_baseML'], dtype=float)
    rho_gas_mid = np.asarray(data['rho_gas_mid_Msun_kpc3'], dtype=float)
    rho_midplane = rho_star_mid + rho_gas_mid
    
    # Use default M/L ratios
    ups_disk = 0.5
    ups_bulge = 0.7
    
    # Compute baryon curve (GR prediction)
    v_bar = v_bar_from_components(R, V_gas, V_disk, V_bulge, ups_disk, ups_bulge)
    
    # Compute tidal enhanced curve
    if tidal_params is None:
        tidal_params = {}  # Will use defaults in compute_enhanced_tidal_velocity
    
    v_tidal, xi = compute_enhanced_tidal_velocity(R, v_bar, rho_midplane, tidal_params)
    
    # Fit NFW model
    V200, c = fit_nfw_simple(R, V_obs, v_bar, e_V)
    v_nfw = v_model_nfw(R, v_bar, V200, c)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    ax1.errorbar(R, V_obs, yerr=e_V, fmt='ko', markersize=5, 
                capsize=3, label='Observed', alpha=0.8, zorder=5)
    
    # Model curves
    ax1.plot(R, v_bar, 'b--', linewidth=2.5, label='GR (baryons only)', zorder=3)
    ax1.plot(R, v_nfw, 'g:', linewidth=2.5, label=f'NFW (V200={V200:.0f}, c={c:.1f})', zorder=2)
    ax1.plot(R, v_tidal, 'r-', linewidth=2.5, label='Enhanced Tidal Model', zorder=4)
    
    # Show components if requested
    if show_components:
        ax1.plot(R, V_gas, 'c-', linewidth=0.8, alpha=0.3, label='Gas')
        ax1.plot(R, V_disk, 'm-', linewidth=0.8, alpha=0.3, label='Disk')
        if np.any(V_bulge > 0):
            ax1.plot(R, V_bulge, 'y-', linewidth=0.8, alpha=0.3, label='Bulge')
    
    ax1.set_ylabel('Circular Velocity [km/s]', fontsize=12)
    ax1.set_title(f'{galaxy_id}: Rotation Curve Comparison (Enhanced Tidal Model)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, R.max() * 1.1)
    ax1.set_ylim(0, max(V_obs.max(), v_tidal.max()) * 1.15)
    
    # Xi enhancement panel
    ax2.plot(R, xi, 'r-', linewidth=2.5, label='ξ factor')
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(1.0 + tidal_params.get('lambda_max', 2.0), 
                color='r', linestyle='--', alpha=0.3,
                label=f"Max: 1+λ={1.0 + tidal_params.get('lambda_max', 2.0):.1f}")
    
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('ξ Enhancement', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, R.max() * 1.1)
    ax2.set_ylim(0.95, max(1.5, xi.max() * 1.1))
    
    # Add velocity boost percentage on right axis
    ax2_right = ax2.twinx()
    ax2_right.set_ylabel('Velocity Boost [%]', fontsize=11, color='orange')
    v_boost_pct = (np.sqrt(xi) - 1.0) * 100
    ax2_right.plot(R, v_boost_pct, 'orange', linewidth=2, alpha=0.7)
    ax2_right.tick_params(axis='y', labelcolor='orange')
    ax2_right.set_ylim(-2.5, max(25, v_boost_pct.max() * 1.1))
    
    # Add RMS errors as text
    res_gr = V_obs - v_bar
    res_nfw = V_obs - v_nfw
    res_tidal = V_obs - v_tidal
    
    rms_gr = np.sqrt(np.mean(res_gr**2))
    rms_nfw = np.sqrt(np.mean(res_nfw**2))
    rms_tidal = np.sqrt(np.mean(res_tidal**2))
    
    # Calculate chi-squared values
    chi2_gr = np.sum((res_gr / np.maximum(e_V, 1.0))**2)
    chi2_nfw = np.sum((res_nfw / np.maximum(e_V, 1.0))**2)
    chi2_tidal = np.sum((res_tidal / np.maximum(e_V, 1.0))**2)
    
    text_str = (f'RMS Error:\n'
                f'GR: {rms_gr:.1f} km/s\n'
                f'NFW: {rms_nfw:.1f} km/s\n'
                f'Tidal: {rms_tidal:.1f} km/s\n\n'
                f'χ²/dof:\n'
                f'GR: {chi2_gr/len(R):.2f}\n'
                f'NFW: {chi2_nfw/len(R):.2f}\n'
                f'Tidal: {chi2_tidal/len(R):.2f}')
    
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'{galaxy_id}_enhanced_tidal.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)


def batch_plot_galaxies(
    galaxy_ids: List[str],
    sparc_dir: Path,
    output_dir: Path,
    tidal_params_sets: Optional[Dict[str, Dict]] = None
) -> None:
    """Generate plots for multiple galaxies with different parameter sets."""
    
    if tidal_params_sets is None:
        # Default parameter sets
        tidal_params_sets = {
            'moderate': {
                'rho_c_solar_kpc3': 5e6,
                'gamma_exp': 2.5,
                'lambda_max': 1.5,
                'T0': 20.0,
                'sigma_lnT': 1.0,
                'wmin': 0.05
            },
            'enhanced': {
                'rho_c_solar_kpc3': 1e6,
                'gamma_exp': 2.2,
                'lambda_max': 2.0,
                'T0': 25.0,
                'sigma_lnT': 1.2,
                'wmin': 0.08
            },
            'aggressive': {
                'rho_c_solar_kpc3': 5e5,
                'gamma_exp': 2.0,
                'lambda_max': 3.0,
                'T0': 30.0,
                'sigma_lnT': 1.5,
                'wmin': 0.1
            }
        }
    
    for param_set_name, params in tidal_params_sets.items():
        print(f"\n{'='*60}")
        print(f"Parameter Set: {param_set_name}")
        print(f"{'='*60}")
        
        # Create subdirectory for this parameter set
        set_dir = output_dir / param_set_name
        set_dir.mkdir(exist_ok=True, parents=True)
        
        for galaxy_id in galaxy_ids:
            print(f"Processing {galaxy_id}...")
            try:
                plot_galaxy_comparison(
                    galaxy_id, 
                    sparc_dir, 
                    set_dir, 
                    tidal_params=params,
                    show_components=True
                )
            except Exception as e:
                print(f"  Error: {e}")


def main():
    """Main routine to generate enhanced tidal model plots."""
    
    print("\n" + "="*80)
    print("SPARC GALAXY PLOTS WITH ENHANCED TIDAL MODEL")
    print("="*80)
    
    # Setup paths
    sparc_dir = Path("external_data/Rotmod_LTG")
    output_dir = Path("images/sparc_enhanced_tidal")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Select galaxies to plot
    # Focus on well-studied galaxies with good data quality
    galaxy_ids = [
        'NGC3198',  # Classic case
        'NGC2403',  # M81 group
        'NGC2841',  # Well-studied
        'NGC5055',  # M63
        'NGC6946',  # Fireworks galaxy
        'DDO154',   # Dwarf galaxy
        'NGC3621',  # Field spiral
        'NGC2976',  # M81 group dwarf
        'IC2574',   # M81 group irregular
        'NGC0300'   # Sculptor group
    ]
    
    # Check which galaxies are available
    available_galaxies = []
    for gid in galaxy_ids:
        test_file = sparc_dir / f"{gid}_rotmod.dat"
        if test_file.exists():
            available_galaxies.append(gid)
        else:
            print(f"Warning: {gid} data not found")
    
    if not available_galaxies:
        print("No galaxy data found! Please check SPARC data directory.")
        return
    
    print(f"\nFound {len(available_galaxies)} galaxies: {', '.join(available_galaxies)}")
    
    # Generate plots with different parameter sets
    batch_plot_galaxies(available_galaxies, sparc_dir, output_dir)
    
    # Also generate a summary comparison for NGC3198 with all three parameter sets
    print("\n" + "="*60)
    print("Generating NGC3198 parameter comparison...")
    print("="*60)
    
    if 'NGC3198' in available_galaxies:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load data once
        data = load_single_sparc_galaxy('NGC3198', sparc_dir=str(sparc_dir))
        R = np.asarray(data['R_kpc'], dtype=float)
        V_obs = np.asarray(data['V_obs'], dtype=float)
        e_V = np.asarray(data['e_V_obs'], dtype=float)
        V_gas = np.asarray(data['V_gas_comp_kms'], dtype=float)
        V_disk = np.asarray(data['V_disk_comp_kms'], dtype=float)
        V_bulge = np.asarray(data['V_bulge_comp_kms'], dtype=float)
        rho_midplane = (np.asarray(data['rho_star_mid_Msun_kpc3_baseML'], dtype=float) + 
                       np.asarray(data['rho_gas_mid_Msun_kpc3'], dtype=float))
        
        v_bar = v_bar_from_components(R, V_gas, V_disk, V_bulge, 0.5, 0.7)
        
        param_sets = [
            ('Moderate', {'rho_c_solar_kpc3': 5e6, 'gamma_exp': 2.5, 'lambda_max': 1.5,
                         'T0': 20.0, 'sigma_lnT': 1.0, 'wmin': 0.05}),
            ('Enhanced', {'rho_c_solar_kpc3': 1e6, 'gamma_exp': 2.2, 'lambda_max': 2.0,
                         'T0': 25.0, 'sigma_lnT': 1.2, 'wmin': 0.08}),
            ('Aggressive', {'rho_c_solar_kpc3': 5e5, 'gamma_exp': 2.0, 'lambda_max': 3.0,
                           'T0': 30.0, 'sigma_lnT': 1.5, 'wmin': 0.1})
        ]
        
        for ax, (name, params) in zip(axes, param_sets):
            v_tidal, xi = compute_enhanced_tidal_velocity(R, v_bar, rho_midplane, params)
            
            ax.errorbar(R, V_obs, yerr=e_V, fmt='ko', markersize=4, alpha=0.7, label='Observed')
            ax.plot(R, v_bar, 'b--', linewidth=2, label='GR')
            ax.plot(R, v_tidal, 'r-', linewidth=2.5, label='Tidal')
            
            ax.set_xlabel('R [kpc]')
            ax.set_ylabel('V [km/s]')
            ax.set_title(f'{name} Parameters')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add max boost text
            max_boost = (np.sqrt(xi).max() - 1.0) * 100
            ax.text(0.98, 0.02, f'Max boost: {max_boost:.1f}%', 
                   transform=ax.transAxes, ha='right', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        fig.suptitle('NGC3198: Tidal Model Parameter Sensitivity', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comparison_file = output_dir / 'NGC3198_parameter_comparison.png'
        fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {comparison_file}")
        plt.close(fig)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
