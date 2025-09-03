#!/usr/bin/env python3
"""
Generate comparison plots between tidal_band and rar_blend model runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_run_data(run_dir):
    """Load results from a run directory."""
    run_path = Path(run_dir)
    
    # Load NPZ file
    npz_file = run_path / "posterior_samples.npz"
    if not npz_file.exists():
        print(f"Warning: {npz_file} not found")
        return None
        
    data = np.load(npz_file)
    
    # Load enhanced summary if available
    summary_file = run_path / "run_summary_enhanced.json"
    summary = None
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return {
        'samples': data.get('samples'),
        'logl': data.get('logl'),
        'logz': data.get('logz'),
        'param_names': data.get('param_names'),
        'xi_type': str(data.get('xi_type', 'unknown')),
        'summary': summary
    }

def plot_evidence_comparison():
    """Create evidence comparison bar plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Evidence values
    models = ['GR\n(baseline)', 'Tidal Band', 'RAR Blend']
    logz_values = [-1490897.53, -534265.34, -519396.52]
    colors = ['gray', 'blue', 'red']
    
    # Absolute evidence
    ax1.bar(models, logz_values, color=colors, alpha=0.7)
    ax1.set_ylabel('log(Z)', fontsize=12)
    ax1.set_title('Absolute Model Evidence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis with scientific notation
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Delta log Z relative to GR
    delta_logz = [0, 956632.19, 971501.01]
    ax2.bar(models, delta_logz, color=colors, alpha=0.7)
    ax2.set_ylabel('Δlog(Z) vs GR', fontsize=12)
    ax2.set_title('Evidence Improvement over GR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (model, val) in enumerate(zip(models, delta_logz)):
        if val > 0:
            ax2.text(i, val, f'+{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Evidence Comparison: GR vs Tidal Band vs RAR Blend', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_parameter_comparison(tidal_data, rar_data):
    """Compare key parameters between models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Parameters to compare
    params_to_compare = [
        ('Total Baryonic Mass', None, [1.42e10, 2.93e10], 'M☉'),
        ('Max Enhancement (ξ_max)', None, [1.503, 6.67], ''),
        ('T₀', 'T0', None, '(km/s)²/kpc²'),
        ('σ_lnT', 'sigma_lnT', None, ''),
        ('w_min', 'wmin', None, ''),
        ('Efficiency', None, [2.80, 2.83], '%')
    ]
    
    for idx, (title, param_name, manual_values, unit) in enumerate(params_to_compare):
        ax = axes[idx]
        
        if manual_values:
            values = manual_values
        else:
            # Extract from data
            tidal_idx = list(tidal_data['param_names']).index(param_name)
            rar_idx = list(rar_data['param_names']).index(param_name)
            
            tidal_samples = tidal_data['samples'][:, tidal_idx]
            rar_samples = rar_data['samples'][:, rar_idx]
            
            # Plot histograms
            ax.hist(tidal_samples, bins=50, alpha=0.5, label='Tidal Band', 
                   color='blue', density=True)
            ax.hist(rar_samples, bins=50, alpha=0.5, label='RAR Blend', 
                   color='red', density=True)
            
            values = [np.median(tidal_samples), np.median(rar_samples)]
        
        if manual_values:
            # Bar plot for manual values
            x = np.arange(2)
            ax.bar(x, values, color=['blue', 'red'], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(['Tidal Band', 'RAR Blend'])
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2e}' if v > 1000 else f'{v:.3f}', 
                       ha='center', va='bottom')
        
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        if unit:
            ax.set_ylabel(f'{unit}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if not manual_values:
            ax.legend()
    
    plt.suptitle('Parameter Comparison: Tidal Band vs RAR Blend', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_enhancement_profiles():
    """Plot xi enhancement profiles for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Density range
    rho = np.logspace(4, 10, 1000)  # M☉/kpc³
    
    # Tidal band enhancement (simplified, at peak T)
    rho_c_tidal = 5.01e7
    gamma_tidal = 3.0
    lambda_max_tidal = 0.503
    S_rho_tidal = 1.0 / (1.0 + (rho/rho_c_tidal)**gamma_tidal)
    xi_tidal = 1.0 + lambda_max_tidal * S_rho_tidal
    
    # RAR blend (simplified, showing max possible)
    lambda_cap_rar = 6.67
    # Simplified representation - actual depends on g_bar
    xi_rar_max = np.ones_like(rho) * lambda_cap_rar
    xi_rar_typical = 1.0 + (lambda_cap_rar - 1.0) * np.exp(-rho/1e8)
    
    # Plot tidal band
    ax1.semilogx(rho, xi_tidal, 'b-', linewidth=2, label='Tidal Band')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(rho, 1, xi_tidal, alpha=0.3, color='blue')
    ax1.set_xlabel('Density ρ (M☉/kpc³)', fontsize=12)
    ax1.set_ylabel('Enhancement Factor ξ', fontsize=12)
    ax1.set_title('Tidal Band Enhancement Profile', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.9, 2])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add Solar System marker
    ax1.axvline(x=1e29, color='orange', linestyle=':', alpha=0.7, label='Solar System')
    
    # Plot RAR blend
    ax2.semilogx(rho, xi_rar_typical, 'r-', linewidth=2, label='RAR Blend (typical)')
    ax2.semilogx(rho, xi_rar_max, 'r--', linewidth=1, alpha=0.5, label='RAR Cap')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(rho, 1, xi_rar_typical, alpha=0.3, color='red')
    ax2.set_xlabel('Density ρ (M☉/kpc³)', fontsize=12)
    ax2.set_ylabel('Enhancement Factor ξ', fontsize=12)
    ax2.set_title('RAR Blend Enhancement Profile', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.9, 8])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add Solar System marker
    ax2.axvline(x=1e29, color='orange', linestyle=':', alpha=0.7, label='Solar System')
    
    plt.suptitle('Enhancement Factor Comparison: Density Dependence', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def main():
    """Main function to generate all comparison plots."""
    print("=" * 70)
    print("TIDAL BAND vs RAR BLEND COMPARISON")
    print("=" * 70)
    
    # Load data
    print("\nLoading run data...")
    tidal_dir = "runs/tidal_band_from_best_20250820_185242"
    rar_dir = "runs/rar_blend_20250823_211648"
    
    tidal_data = load_run_data(tidal_dir)
    rar_data = load_run_data(rar_dir)
    
    if not tidal_data or not rar_data:
        print("Error: Could not load run data")
        return
    
    # Create output directory
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\n1. Generating evidence comparison...")
    fig1 = plot_evidence_comparison()
    fig1.savefig(output_dir / "evidence_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir}/evidence_comparison.png")
    
    print("\n2. Generating parameter comparison...")
    fig2 = plot_parameter_comparison(tidal_data, rar_data)
    fig2.savefig(output_dir / "parameter_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir}/parameter_comparison.png")
    
    print("\n3. Generating enhancement profiles...")
    fig3 = plot_enhancement_profiles()
    fig3.savefig(output_dir / "enhancement_profiles.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir}/enhancement_profiles.png")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print("\nTidal Band Model:")
    print(f"  - log(Z): -534,265.34")
    print(f"  - Total baryons: 1.42×10¹⁰ M☉")
    print(f"  - Max enhancement: ~1.5×")
    print(f"  - Runtime: 64.7 hours")
    
    print("\nRAR Blend Model:")
    print(f"  - log(Z): -519,396.52")
    print(f"  - Total baryons: 2.93×10¹⁰ M☉")
    print(f"  - Max enhancement: ~6.7×")
    print(f"  - Runtime: 31.8 hours")
    
    print("\nWinner: RAR Blend")
    print("  - Better evidence: Δlog(Z) = +14,869")
    print("  - More realistic baryon mass")
    print("  - Greater flexibility in enhancement")
    print("  - Faster convergence")
    
    print("\n" + "=" * 70)
    print("All plots saved to comparison_plots/")
    plt.show()

if __name__ == "__main__":
    main()
