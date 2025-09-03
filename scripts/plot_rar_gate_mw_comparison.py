#!/usr/bin/env python3
"""
Plot RAR gate model against GR and NFW for the Milky Way rotation curve.

Uses the best-fit parameters from the converged RAR gate run.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cupy as cp
from core.density_metric_cupy import (
    v_baryon_comprehensive_kms_cupy,
    v_total_kms_cupy,
    volume_density_comprehensive_solar_kpc3_cupy
)

def load_best_params(run_dir):
    """Load best-fit parameters from RAR gate run."""
    
    # Try loading from best_params_info.json first
    best_params_file = run_dir / "best_params_info.json"
    if best_params_file.exists():
        with open(best_params_file, 'r') as f:
            data = json.load(f)
            if 'best_params' in data and 'param_names' in data:
                # Convert list to dictionary
                params_list = data['best_params']
                param_names = data['param_names']
                return dict(zip(param_names, params_list))
            return data.get('best_params', data)
    
    # Fallback to run_summary_enhanced.json
    summary_file = run_dir / "run_summary_enhanced.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            return data.get('parameter_estimates', {}).get('best_fit', {})
    
    raise FileNotFoundError(f"No parameter files found in {run_dir}")


def compute_nfw_velocity(R_kpc, v_bar, M_200=1.5e12, c=12, R_200=230):
    """Compute NFW dark matter halo rotation curve."""
    G_NEWTON = 4.301e-6  # km^2 kpc / (M_sun s^2)
    Rs = R_200 / c  # Scale radius
    
    R_arr = np.asarray(R_kpc)
    
    def M_enc(r):
        x = r / Rs
        return M_200 * (np.log(1 + x) - x/(1 + x)) / (np.log(1 + c) - c/(1 + c))
    
    M_enclosed = np.array([M_enc(r) if r > 0 else 0 for r in R_arr])
    
    # Avoid division by zero
    R_safe = np.maximum(R_arr, 1e-6)
    v_dm = np.sqrt(G_NEWTON * M_enclosed / R_safe)
    
    # Total velocity = quadrature sum of baryons and dark matter
    v_total = np.sqrt(v_bar**2 + v_dm**2)
    
    return v_total


def plot_rotation_curves(params, output_dir):
    """Create rotation curve comparison plot."""
    
    # Define radial range from 1 to 30 kpc
    R_kpc = np.logspace(0, np.log10(30), 100)  # 1 to 30 kpc
    
    # Also add specific integer distances for annotation
    R_integers = np.array([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
    
    # Convert to CuPy arrays
    R_cp = cp.asarray(R_kpc, dtype=cp.float32)
    
    # 1. Compute GR (baryons only) curve
    v_gr_cp = v_baryon_comprehensive_kms_cupy(R_cp, params)
    v_gr = cp.asnumpy(v_gr_cp)
    
    # 2. Compute RAR gate model curve
    # Need to ensure allow_experimental is set for RAR gate
    params_rar = params.copy()
    params_rar['allow_experimental'] = True
    v_rar_cp = v_total_kms_cupy(R_cp, params_rar, xi_type='rar_gate')
    v_rar = cp.asnumpy(v_rar_cp)
    
    # 3. Compute NFW model curve
    # Use typical MW halo parameters
    v_nfw = compute_nfw_velocity(R_kpc, v_gr, M_200=1.5e12, c=12, R_200=230)
    
    # Compute velocities at integer distances
    R_int_cp = cp.asarray(R_integers, dtype=cp.float32)
    v_gr_int = cp.asnumpy(v_baryon_comprehensive_kms_cupy(R_int_cp, params))
    v_rar_int = cp.asnumpy(v_total_kms_cupy(R_int_cp, params_rar, xi_type='rar_gate'))
    v_nfw_int = compute_nfw_velocity(R_integers, v_gr_int, M_200=1.5e12, c=12, R_200=230)
    
    # Create figure with two panels
    fig = plt.figure(figsize=(14, 10))
    
    # Main plot
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot curves
    ax1.plot(R_kpc, v_gr, 'b--', linewidth=2.5, label='GR (baryons only)', alpha=0.8)
    ax1.plot(R_kpc, v_nfw, 'g:', linewidth=2.5, label='GR + NFW dark matter', alpha=0.8)
    ax1.plot(R_kpc, v_rar, 'r-', linewidth=3, label='RAR gate model', alpha=0.9)
    
    # Mark integer distances
    ax1.scatter(R_integers, v_gr_int, color='blue', s=50, zorder=5, alpha=0.7)
    ax1.scatter(R_integers, v_nfw_int, color='green', s=50, zorder=5, alpha=0.7)
    ax1.scatter(R_integers, v_rar_int, color='red', s=60, zorder=5, alpha=0.9)
    
    # Add vertical lines at key distances
    for R in [8, 15, 25]:  # Solar radius and outer regions
        ax1.axvline(R, color='gray', linestyle=':', alpha=0.3)
        ax1.text(R, ax1.get_ylim()[0] + 5, f'{R} kpc', 
                ha='center', va='bottom', fontsize=9, color='gray')
    
    # Add solar position marker
    ax1.axvline(8.5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(8.5, ax1.get_ylim()[1] - 10, '☉', 
            ha='center', va='top', fontsize=16, color='orange')
    
    ax1.set_xlabel('Radius [kpc]', fontsize=12)
    ax1.set_ylabel('Circular Velocity [km/s]', fontsize=12)
    ax1.set_title('Milky Way Rotation Curve: RAR Gate vs GR vs NFW', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 31)
    ax1.set_ylim(0, max(v_rar.max(), v_nfw.max()) * 1.1)
    
    # Difference plot
    ax2 = plt.subplot(2, 1, 2)
    
    # Compute differences from GR
    diff_rar = v_rar - v_gr
    diff_nfw = v_nfw - v_gr
    
    ax2.plot(R_kpc, diff_nfw, 'g:', linewidth=2.5, label='NFW - GR', alpha=0.8)
    ax2.plot(R_kpc, diff_rar, 'r-', linewidth=3, label='RAR gate - GR', alpha=0.9)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Mark integer distances
    diff_rar_int = v_rar_int - v_gr_int
    diff_nfw_int = v_nfw_int - v_gr_int
    ax2.scatter(R_integers, diff_nfw_int, color='green', s=50, zorder=5, alpha=0.7)
    ax2.scatter(R_integers, diff_rar_int, color='red', s=60, zorder=5, alpha=0.9)
    
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('ΔV [km/s]', fontsize=12)
    ax2.set_title('Velocity Enhancement over GR', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 31)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'rar_gate_mw_comparison.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Also create a data table
    print("\n" + "="*80)
    print("VELOCITY COMPARISON TABLE")
    print("="*80)
    print(f"{'R [kpc]':>8} {'GR [km/s]':>12} {'NFW [km/s]':>12} {'RAR [km/s]':>12} "
          f"{'NFW-GR':>10} {'RAR-GR':>10} {'RAR/GR':>8}")
    print("-"*80)
    
    for i, R in enumerate(R_integers):
        print(f"{R:8.0f} {v_gr_int[i]:12.1f} {v_nfw_int[i]:12.1f} {v_rar_int[i]:12.1f} "
              f"{diff_nfw_int[i]:10.1f} {diff_rar_int[i]:10.1f} "
              f"{v_rar_int[i]/v_gr_int[i]:8.3f}")
    
    # Compute enhancement factors
    print("\n" + "="*80)
    print("ENHANCEMENT ANALYSIS")
    print("="*80)
    
    # Compute xi values for RAR gate
    rho_cp = volume_density_comprehensive_solar_kpc3_cupy(R_cp, params)
    v_bar_sq = v_gr_cp**2
    T = v_bar_sq / cp.maximum(R_cp**2, 1e-18)
    
    # RAR gate specific calculation
    ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14
    gbar_m_s2 = ACC_M_S2_PER_KMS2_PER_KPC * cp.maximum(T, 0.0) * cp.maximum(R_cp, 1e-12)
    a0 = params.get('a0_m_s2', 5e-11)
    gamma = params.get('gamma_exp', 5.0)
    x = gbar_m_s2 / max(a0, 1e-30)
    Sg = 1.0 / (1.0 + cp.power(x, gamma))
    
    # Convert to numpy for analysis
    Sg_np = cp.asnumpy(Sg)
    gbar_np = cp.asnumpy(gbar_m_s2)
    
    print(f"\nAt R = 8.5 kpc (Solar position):")
    idx_solar = np.argmin(np.abs(R_kpc - 8.5))
    print(f"  g_bar = {gbar_np[idx_solar]:.2e} m/s²")
    print(f"  g_bar/a0 = {gbar_np[idx_solar]/a0:.2f}")
    print(f"  Suppression factor S_g = {Sg_np[idx_solar]:.4f}")
    print(f"  Velocity: GR = {v_gr[idx_solar]:.1f}, RAR = {v_rar[idx_solar]:.1f} km/s")
    print(f"  Enhancement: {(v_rar[idx_solar]/v_gr[idx_solar] - 1)*100:.1f}%")
    
    print(f"\nAt R = 20 kpc (Outer disk):")
    idx_20 = np.argmin(np.abs(R_kpc - 20))
    print(f"  g_bar = {gbar_np[idx_20]:.2e} m/s²")
    print(f"  g_bar/a0 = {gbar_np[idx_20]/a0:.2f}")
    print(f"  Suppression factor S_g = {Sg_np[idx_20]:.4f}")
    print(f"  Velocity: GR = {v_gr[idx_20]:.1f}, RAR = {v_rar[idx_20]:.1f} km/s")
    print(f"  Enhancement: {(v_rar[idx_20]/v_gr[idx_20] - 1)*100:.1f}%")
    
    return fig


def main():
    """Main routine."""
    
    print("\n" + "="*80)
    print("RAR GATE MODEL - MILKY WAY ROTATION CURVE COMPARISON")
    print("="*80)
    
    # Load best-fit parameters from RAR gate run
    run_dir = Path("runs/rar_gate_from_best_20250820_185422")
    print(f"\nLoading parameters from: {run_dir}")
    
    try:
        params = load_best_params(run_dir)
        print("✓ Parameters loaded successfully")
        
        # Print key RAR gate parameters
        print("\nRAR Gate Parameters:")
        print(f"  a0 = {params.get('a0_m_s2', 0)*1e10:.3f} × 10^-10 m/s²")
        print(f"  gamma_exp = {params.get('gamma_exp', 0):.3f}")
        print(f"  lambda_max = {params.get('lambda_max', 0):.3f}")
        print(f"  T0 = {params.get('T0', 0):.1f}")
        
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return
    
    # Create output directory
    output_dir = Path("images/rar_gate_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate comparison plot
    print("\nGenerating rotation curve comparison...")
    try:
        fig = plot_rotation_curves(params, output_dir)
        print("\n✓ Plot generation complete")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
