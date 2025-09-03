#!/usr/bin/env python3
"""
Enhanced plot of RAR gate model vs GR vs NFW with additional annotations and data.
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
    best_params_file = run_dir / "best_params_info.json"
    if best_params_file.exists():
        with open(best_params_file, 'r') as f:
            data = json.load(f)
            if 'best_params' in data and 'param_names' in data:
                params_list = data['best_params']
                param_names = data['param_names']
                return dict(zip(param_names, params_list))
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
    R_safe = np.maximum(R_arr, 1e-6)
    v_dm = np.sqrt(G_NEWTON * M_enclosed / R_safe)
    v_total = np.sqrt(v_bar**2 + v_dm**2)
    return v_total


def create_enhanced_plot(params, output_dir):
    """Create enhanced comparison plot with better styling."""
    
    # Define radial range
    R_kpc = np.logspace(0, np.log10(30), 200)  # More points for smoother curves
    R_integers = np.array([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
    
    # Convert to CuPy arrays
    R_cp = cp.asarray(R_kpc, dtype=cp.float32)
    
    # Compute curves
    v_gr_cp = v_baryon_comprehensive_kms_cupy(R_cp, params)
    v_gr = cp.asnumpy(v_gr_cp)
    
    params_rar = params.copy()
    params_rar['allow_experimental'] = True
    v_rar_cp = v_total_kms_cupy(R_cp, params_rar, xi_type='rar_gate')
    v_rar = cp.asnumpy(v_rar_cp)
    
    v_nfw = compute_nfw_velocity(R_kpc, v_gr, M_200=1.5e12, c=12, R_200=230)
    
    # Integer distance values
    R_int_cp = cp.asarray(R_integers, dtype=cp.float32)
    v_gr_int = cp.asnumpy(v_baryon_comprehensive_kms_cupy(R_int_cp, params))
    v_rar_int = cp.asnumpy(v_total_kms_cupy(R_int_cp, params_rar, xi_type='rar_gate'))
    v_nfw_int = compute_nfw_velocity(R_integers, v_gr_int, M_200=1.5e12, c=12, R_200=230)
    
    # Create figure with custom style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3 subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    
    # Main rotation curve plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot main curves with better styling
    ax1.plot(R_kpc, v_gr, 'b-', linewidth=3, label='GR (Baryons Only)', alpha=0.9)
    ax1.plot(R_kpc, v_nfw, 'g--', linewidth=3, label='GR + NFW Dark Matter', alpha=0.9)
    ax1.plot(R_kpc, v_rar, 'r-', linewidth=3.5, label='RAR Gate Model', alpha=1.0)
    
    # Add shaded regions for different galactic zones
    ax1.axvspan(0, 3, alpha=0.1, color='yellow', label='Bulge dominated')
    ax1.axvspan(3, 10, alpha=0.1, color='cyan', label='Disk dominated')
    ax1.axvspan(10, 30, alpha=0.1, color='purple', label='Halo dominated')
    
    # Mark special radii
    ax1.axvline(8.5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(8.5, 290, 'Solar Position', rotation=90, va='bottom', ha='right', 
             fontsize=10, color='orange', fontweight='bold')
    
    # Scatter points at integer distances
    ax1.scatter(R_integers, v_gr_int, color='blue', s=80, zorder=5, alpha=0.8, 
                marker='o', edgecolors='darkblue', linewidth=1.5)
    ax1.scatter(R_integers, v_nfw_int, color='green', s=80, zorder=5, alpha=0.8,
                marker='s', edgecolors='darkgreen', linewidth=1.5)
    ax1.scatter(R_integers, v_rar_int, color='red', s=100, zorder=6, alpha=0.9,
                marker='D', edgecolors='darkred', linewidth=2)
    
    ax1.set_xlabel('Galactocentric Radius [kpc]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Circular Velocity [km/s]', fontsize=14, fontweight='bold')
    ax1.set_title('Milky Way Rotation Curve: Comparing Gravitational Models', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, 
               shadow=True, ncol=2)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xlim(0, 31)
    ax1.set_ylim(0, 300)
    
    # Velocity difference plot
    ax2 = fig.add_subplot(gs[1, 0])
    diff_rar = v_rar - v_gr
    diff_nfw = v_nfw - v_gr
    
    ax2.fill_between(R_kpc, 0, diff_nfw, color='green', alpha=0.3, label='NFW enhancement')
    ax2.fill_between(R_kpc, 0, diff_rar, color='red', alpha=0.3, label='RAR enhancement')
    ax2.plot(R_kpc, diff_nfw, 'g--', linewidth=2.5, alpha=0.9)
    ax2.plot(R_kpc, diff_rar, 'r-', linewidth=2.5, alpha=1.0)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('ΔV [km/s]', fontsize=12)
    ax2.set_title('Velocity Enhancement over GR', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(-10, 120)
    
    # Percentage enhancement plot
    ax3 = fig.add_subplot(gs[2, 0])
    pct_rar = (v_rar / v_gr - 1) * 100
    pct_nfw = (v_nfw / v_gr - 1) * 100
    
    ax3.plot(R_kpc, pct_nfw, 'g--', linewidth=2.5, label='NFW', alpha=0.9)
    ax3.plot(R_kpc, pct_rar, 'r-', linewidth=2.5, label='RAR Gate', alpha=1.0)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Mark 10% and 50% enhancement levels
    ax3.axhline(10, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax3.text(29, 10, '10%', va='bottom', ha='right', fontsize=9, color='gray')
    ax3.text(29, 50, '50%', va='bottom', ha='right', fontsize=9, color='gray')
    
    ax3.set_xlabel('Radius [kpc]', fontsize=12)
    ax3.set_ylabel('Enhancement [%]', fontsize=12)
    ax3.set_title('Percentage Velocity Enhancement', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 31)
    ax3.set_ylim(-5, 80)
    
    # Data table
    ax4 = fig.add_subplot(gs[1:, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['R [kpc]', 'GR', 'NFW', 'RAR', 'RAR/GR'])
    for i, R in enumerate([1, 5, 8.5, 10, 15, 20, 25, 30]):
        idx = np.argmin(np.abs(R_kpc - R))
        table_data.append([
            f'{R:.1f}',
            f'{v_gr[idx]:.1f}',
            f'{v_nfw[idx]:.1f}',
            f'{v_rar[idx]:.1f}',
            f'{v_rar[idx]/v_gr[idx]:.3f}'
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            if j == 0:  # R column
                table[(i, j)].set_facecolor('#E0E0E0')
            elif j == 1:  # GR column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 2:  # NFW column
                table[(i, j)].set_facecolor('#E8F5E9')
            elif j == 3:  # RAR column
                table[(i, j)].set_facecolor('#FFEBEE')
            else:  # Ratio column
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax4.set_title('Velocity Values [km/s]', fontsize=12, fontweight='bold', pad=20)
    
    # Add model parameters text
    params_text = (
        f"RAR Gate Parameters:\n"
        f"a₀ = {params.get('a0_m_s2', 0)*1e10:.2f}×10⁻¹⁰ m/s²\n"
        f"γ = {params.get('gamma_exp', 0):.2f}\n"
        f"λ_max = {params.get('lambda_max', 0):.3f}\n"
        f"T₀ = {params.get('T0', 0):.1f}\n\n"
        f"NFW Parameters:\n"
        f"M₂₀₀ = 1.5×10¹² M☉\n"
        f"c = 12\n"
        f"R₂₀₀ = 230 kpc"
    )
    
    fig.text(0.75, 0.15, params_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Complete Analysis: GR vs Dark Matter vs RAR Gate Model', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'rar_gate_enhanced_comparison.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved enhanced plot: {output_file}")
    
    return fig


def main():
    """Main routine."""
    print("\n" + "="*80)
    print("ENHANCED RAR GATE MODEL VISUALIZATION")
    print("="*80)
    
    # Load parameters
    run_dir = Path("runs/rar_gate_from_best_20250820_185422")
    params = load_best_params(run_dir)
    
    # Create output directory
    output_dir = Path("images/rar_gate_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate enhanced plot
    print("\nGenerating enhanced visualization...")
    fig = create_enhanced_plot(params, output_dir)
    
    print("\n✓ Enhanced plot generation complete")
    print("="*80)


if __name__ == '__main__':
    main()
