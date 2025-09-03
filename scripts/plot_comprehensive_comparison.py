#!/usr/bin/env python
"""
Generate comprehensive publication-quality plots comparing all models for the Milky Way,
including GR, NFW, tidal_band, and rar_gate models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_gaia_data, process_gaia_data
from rotation_curves import v_total_kms
from xi_models import xi_models

# Set publication quality
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100

def load_model_params(run_dir):
    """Load model parameters from a run directory."""
    # Try to load from run_summary_enhanced.json first
    summary_file = run_dir / 'run_summary_enhanced.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            if 'parameter_estimates' in data and 'best_fit' in data['parameter_estimates']:
                params = data['parameter_estimates']['best_fit']
                logz = data['evidence_metrics']['logz'] if 'evidence_metrics' in data else None
                return params, logz
    
    # Fall back to best_params_info.json
    params_file = run_dir / 'best_params_info.json'
    if params_file.exists():
        with open(params_file, 'r') as f:
            data = json.load(f)
            params = data.get('params', {})
            logz = data.get('logZ', None)
            return params, logz
    
    return None, None

def main():
    print("="*70)
    print("GENERATING COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    # Create output directory
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load and process Gaia data
    print("\n1. Loading Gaia DR3 data...")
    gaia_file = Path('external_data/gaia_sky_slices/all_sky_gaia.csv')
    
    if not gaia_file.exists():
        print(f"   ERROR: Gaia data file not found at {gaia_file}")
        return
    
    # Load and process data
    df = pd.read_csv(gaia_file)
    print(f"   Loaded {len(df)} stars from Gaia DR3")
    
    # Process to get physical quantities
    df_processed = process_gaia_data(df)
    
    # Bin the data
    R_min, R_max = 5.0, 16.0
    bin_width = 0.5
    R_bins = np.arange(R_min, R_max + bin_width, bin_width)
    R_centers = (R_bins[:-1] + R_bins[1:]) / 2
    
    binned_data = []
    for i in range(len(R_bins) - 1):
        mask = (df_processed['R_kpc'] >= R_bins[i]) & (df_processed['R_kpc'] < R_bins[i+1])
        if mask.sum() > 10:  # Need at least 10 stars per bin
            v_mean = df_processed.loc[mask, 'vtan_norm'].mean()
            v_std = df_processed.loc[mask, 'vtan_norm'].std()
            n_stars = mask.sum()
            binned_data.append({
                'R_kpc': R_centers[i],
                'v_mean': v_mean,
                'v_std': v_std,
                'v_err': v_std / np.sqrt(n_stars),
                'n_stars': n_stars
            })
    
    binned_df = pd.DataFrame(binned_data)
    print(f"   Binned into {len(binned_df)} radial bins")
    
    # Define models to plot
    models = [
        ('gr_20250812_113949', 'GR (baryons)', 'blue', '-'),
        ('nfw_20250812_114008', 'NFW dark matter', 'green', '--'),
        ('tidal_band_20250812_115416', 'Tidal Band', 'orange', ':'),
        ('rar_gate_from_best_20250820_185422', 'RAR Gate', 'red', '-')
    ]
    
    # Load model parameters
    print("\n2. Loading model parameters...")
    model_data = {}
    for run_dir, label, _, _ in models:
        run_path = Path('runs') / run_dir
        if run_path.exists():
            params, logz = load_model_params(run_path)
            if params:
                model_data[label] = {'params': params, 'logz': logz}
                print(f"   ✓ {label}: logZ = {logz:.1f}")
            else:
                print(f"   ✗ {label}: Could not load parameters")
        else:
            print(f"   ✗ {label}: Run directory not found")
    
    # Create the plot
    print("\n3. Creating rotation curve comparison plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})
    
    # Plot Gaia data
    ax1.errorbar(binned_df['R_kpc'], binned_df['v_mean'], 
                 yerr=binned_df['v_err'], fmt='o', color='black', 
                 markersize=4, capsize=3, label='Gaia DR3 data', zorder=10)
    
    # Plot model curves
    R_plot = np.linspace(5, 16, 100)
    residuals = {}
    
    for run_dir, label, color, linestyle in models:
        if label in model_data:
            params = model_data[label]['params']
            
            # Determine xi_type from run directory name
            if 'gr' in run_dir:
                xi_type = 'gr'
            elif 'nfw' in run_dir:
                xi_type = 'nfw'
            elif 'tidal_band' in run_dir:
                xi_type = 'tidal_band'
            elif 'rar_gate' in run_dir:
                xi_type = 'rar_gate'
            else:
                continue
            
            # Check if xi_type is available
            if xi_type not in xi_models:
                print(f"   Warning: xi_type '{xi_type}' not available")
                continue
            
            try:
                # Calculate rotation curve
                v_model = np.array([v_total_kms(R, params, xi_type=xi_type) for R in R_plot])
                ax1.plot(R_plot, v_model, color=color, linestyle=linestyle, 
                        linewidth=2, label=label)
                
                # Calculate residuals for binned data
                v_at_data = np.array([v_total_kms(R, params, xi_type=xi_type) 
                                     for R in binned_df['R_kpc']])
                residuals[label] = binned_df['v_mean'].values - v_at_data
                
            except Exception as e:
                print(f"   Error plotting {label}: {e}")
    
    # Format main plot
    ax1.set_ylabel('Circular Velocity (km/s)', fontsize=11)
    ax1.set_ylim([180, 260])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.set_title('Milky Way Rotation Curve: Model Comparison', fontsize=12, fontweight='bold')
    
    # Plot residuals
    for label, resid in residuals.items():
        for run_dir, model_label, color, linestyle in models:
            if model_label == label:
                ax2.plot(binned_df['R_kpc'], resid, 'o-', color=color, 
                        markersize=3, linewidth=1, alpha=0.7, label=label)
                break
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=11)
    ax2.set_ylabel('Residuals (km/s)', fontsize=11)
    ax2.set_ylim([-20, 20])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_dir / 'comprehensive_model_comparison'
    fig.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_file}.pdf', bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}.[png/pdf]")
    
    # Create comparison table
    print("\n4. Creating model comparison table...")
    print("\n" + "="*80)
    print(f"{'Model':<20} {'log(Z)':<15} {'Δlog(Z) vs GR':<15} {'χ²/dof':<10}")
    print("-"*80)
    
    # Calculate chi-squared for each model
    gr_logz = model_data.get('GR (baryons)', {}).get('logz', 0)
    
    for label in ['GR (baryons)', 'NFW dark matter', 'Tidal Band', 'RAR Gate']:
        if label in model_data:
            logz = model_data[label]['logz']
            delta_logz = logz - gr_logz if gr_logz else 0
            
            # Calculate chi-squared if we have residuals
            if label in residuals:
                chi2 = np.sum(residuals[label]**2 / binned_df['v_err'].values**2)
                dof = len(binned_df) - 10  # Approximate number of parameters
                chi2_dof = chi2 / dof
            else:
                chi2_dof = np.nan
            
            print(f"{label:<20} {logz:<15.1f} {delta_logz:<15.0f} {chi2_dof:<10.2f}")
    
    print("="*80)
    
    # Save statistics to file
    stats_file = output_dir / 'comprehensive_model_statistics.csv'
    stats_data = []
    for label in model_data:
        logz = model_data[label]['logz']
        delta_logz = logz - gr_logz if gr_logz else 0
        stats_data.append({
            'Model': label,
            'log(Z)': logz,
            'Δlog(Z) vs GR': delta_logz
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False)
    print(f"\n   ✓ Statistics saved to {stats_file}")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
