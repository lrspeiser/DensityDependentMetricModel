#!/usr/bin/env python3
"""
Generate publication-quality Milky Way plots using ONLY real data:
1. Process actual Gaia DR3 data
2. Use actual run results (no fallbacks!)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from core.data_io import process_gaia_data

def load_and_bin_gaia_data():
    """Load actual Gaia DR3 data and compute binned velocities."""
    
    # Check if binned data already exists
    binned_file = Path("data/mw_binned_velocities.csv")
    if binned_file.exists():
        print(f"  Loading existing binned data from {binned_file}")
        return pd.read_csv(binned_file)
    
    # Load the actual Gaia data
    gaia_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
    if not gaia_file.exists():
        raise FileNotFoundError(f"ERROR: Gaia data not found at {gaia_file}")
    
    print(f"Loading Gaia data from {gaia_file}...")
    df = pd.read_csv(gaia_file)
    print(f"  Loaded {len(df)} stars")
    
    # Process the Gaia data to get physical coordinates
    print("  Processing Gaia data to compute physical coordinates...")
    df = process_gaia_data(df)
    print(f"  Processed: R_kpc range = {df['R_kpc'].min():.1f} - {df['R_kpc'].max():.1f} kpc")
    
    # Filter to 5-16 kpc range as specified in paper
    df_filtered = df[(df['R_kpc'] >= 5) & (df['R_kpc'] <= 16)]
    print(f"  After filtering to 5-16 kpc: {len(df_filtered)} stars")
    
    # Use the v_obs column which already contains the circular velocity
    if 'v_obs' not in df_filtered.columns:
        raise ValueError("v_obs column not found in processed Gaia data")
    
    # Bin the data in 0.5 kpc bins as specified in paper
    R_bins = np.arange(5, 16.5, 0.5)
    R_centers = (R_bins[:-1] + R_bins[1:]) / 2
    
    # Compute binned statistics
    binned_data = []
    
    for i in range(len(R_bins)-1):
        mask = (df_filtered['R_kpc'] >= R_bins[i]) & (df_filtered['R_kpc'] < R_bins[i+1])
        stars_in_bin = df_filtered[mask]
        
        if len(stars_in_bin) > 10:  # Need at least 10 stars
            v_mean = stars_in_bin['v_obs'].mean()
            v_std = stars_in_bin['sigma_v'].mean() if 'sigma_v' in stars_in_bin.columns else stars_in_bin['v_obs'].std()
            n_stars = len(stars_in_bin)
            
            # Standard error
            v_err = v_std / np.sqrt(n_stars)
            
            binned_data.append({
                'R_kpc': R_centers[i],
                'v_mean': v_mean,
                'v_std': v_std,
                'v_err': v_err,
                'n_stars': n_stars
            })
    
    binned_df = pd.DataFrame(binned_data)
    
    # Save binned data for future use
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    binned_df.to_csv(output_dir / "mw_binned_velocities.csv", index=False)
    print(f"  Saved binned data to {output_dir}/mw_binned_velocities.csv")
    
    return binned_df

def load_run_results(run_dir):
    """Load model results from a run directory."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"  Warning: {run_path} not found")
        return None
    
    # Load from run_summary_enhanced.json
    summary_file = run_path / "run_summary_enhanced.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            
            # Extract parameters
            params = None
            if 'parameter_estimates' in data and 'best_fit' in data['parameter_estimates']:
                params = data['parameter_estimates']['best_fit']
            
            # Extract evidence
            logz = None
            if 'evidence_metrics' in data and 'logz' in data['evidence_metrics']:
                logz = data['evidence_metrics']['logz']
            
            # Get xi_type from metadata
            xi_type = data.get('metadata', {}).get('xi_type', 'unknown')
            
            return {
                'params': params,
                'xi_type': xi_type,
                'logz': logz
            }
    
    return None

def compute_rotation_curve(R_kpc, model_data):
    """Compute rotation curve for a model."""
    if model_data is None or model_data['params'] is None:
        return None
    
    try:
        # Import physics module
        from core.density_metric_cupy import v_total_kms_cupy
        import cupy as cp
        
        # Convert to CuPy array
        R_cp = cp.asarray(R_kpc, dtype=cp.float32)
        
        # Add allow_experimental flag for RAR models
        params = dict(model_data['params'])
        if model_data['xi_type'] in ['rar_gate', 'rar_blend']:
            params['allow_experimental'] = True
        
        # Compute velocities
        v_model = v_total_kms_cupy(R_cp, params, xi_type=model_data['xi_type'])
        
        # Convert back to numpy
        return cp.asnumpy(v_model)
        
    except Exception as e:
        print(f"  Error computing rotation curve for {model_data['xi_type']}: {e}")
        return None

def main():
    """Generate plots with ONLY real data."""
    print("=" * 70)
    print("GENERATING MILKY WAY PLOTS WITH REAL DATA ONLY")
    print("=" * 70)
    
    # Load and bin Gaia data
    print("\n1. Processing Gaia DR3 data...")
    binned_df = load_and_bin_gaia_data()
    print(f"  Binned into {len(binned_df)} radial bins")
    
    # Load model results from actual runs
    print("\n2. Loading model results from actual runs...")
    
    models = {
        'GR (baryons)': load_run_results("runs/gr_20250812_113949"),
        'NFW (dark matter)': load_run_results("runs/nfw_20250812_114008"),
        'Tidal Band': load_run_results("runs/tidal_band_from_best_20250820_185242"),
        'RAR Gate': load_run_results("runs/rar_gate_from_best_20250820_185422"),
        'RAR Blend': load_run_results("runs/rar_blend_20250823_211648")
    }
    
    # Report what we found
    print("\nModel evidence values:")
    for name, data in models.items():
        if data and data['logz'] is not None:
            print(f"  {name:20s}: logZ = {data['logz']:12.1f}")
        else:
            print(f"  {name:20s}: No evidence found")
    
    # Calculate relative evidence
    gr_logz = models['GR (baryons)']['logz'] if models['GR (baryons)'] else None
    if gr_logz:
        print("\nΔlogZ relative to GR:")
        for name, data in models.items():
            if data and data['logz'] is not None and name != 'GR (baryons)':
                delta = data['logz'] - gr_logz
                print(f"  {name:20s}: Δ = {delta:+12.1f}")
    
    # Create plot
    print("\n3. Creating rotation curve comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Plot Gaia data
    ax1.errorbar(binned_df['R_kpc'], binned_df['v_mean'],
                yerr=binned_df['v_err'],
                fmt='o', color='black', markersize=5,
                capsize=3, alpha=0.8, 
                label=f"Gaia DR3 ({len(binned_df)} bins)",
                zorder=10)
    
    # Extended R range for model curves
    R_model = np.linspace(5, 16, 100)
    
    # Define colors and styles
    styles = {
        'GR (baryons)': ('blue', '--', 2),
        'NFW (dark matter)': ('green', '-.', 2),
        'Tidal Band': ('orange', '-', 2.5),
        'RAR Gate': ('red', '-', 2.5),
        'RAR Blend': ('purple', ':', 2)
    }
    
    # Plot model curves and calculate residuals
    residuals = {}
    
    for name, model_data in models.items():
        if model_data and model_data['params']:
            # Compute model curve
            v_model = compute_rotation_curve(R_model, model_data)
            
            if v_model is not None:
                color, linestyle, linewidth = styles.get(name, ('gray', '-', 1))
                
                # Plot curve
                label = name
                if model_data['logz']:
                    label += f" (logZ={model_data['logz']:.0f})"
                
                ax1.plot(R_model, v_model,
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        label=label, alpha=0.8)
                
                # Calculate residuals at data points
                v_at_data = compute_rotation_curve(binned_df['R_kpc'].values, model_data)
                if v_at_data is not None:
                    residuals[name] = binned_df['v_mean'].values - v_at_data
    
    # Format main plot
    ax1.set_ylabel('Circular Velocity (km/s)', fontsize=12)
    ax1.set_ylim([180, 260])
    ax1.set_xlim([4.5, 16.5])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
    ax1.set_title('Milky Way Rotation Curve: Real Data and Model Fits', 
                 fontsize=14, fontweight='bold')
    
    # Solar position
    ax1.axvline(x=8.5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(8.5, 185, 'Sun', ha='center', fontsize=10, color='orange')
    
    # Plot residuals
    for name, resid in residuals.items():
        if name in ['GR (baryons)', 'Tidal Band', 'RAR Gate']:  # Select key models
            color, _, _ = styles.get(name, ('gray', '-', 1))
            ax2.plot(binned_df['R_kpc'], resid, 'o-',
                    color=color, markersize=4, linewidth=1,
                    alpha=0.7, label=name)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax2.set_ylabel('Residuals (km/s)', fontsize=12)
    ax2.set_xlim([4.5, 16.5])
    ax2.set_ylim([-30, 30])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "mw_real_data_comparison"
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_file}.pdf", bbox_inches='tight')
    print(f"\n  Saved to {output_file}.[png/pdf]")
    
    # Create summary statistics table
    print("\n4. Summary Statistics")
    print("=" * 70)
    print(f"{'Model':<25} {'log(Z)':<15} {'Δlog(Z) vs GR':<15}")
    print("-" * 70)
    
    if gr_logz:
        for name, data in models.items():
            if data and data['logz'] is not None:
                delta = data['logz'] - gr_logz if name != 'GR (baryons)' else 0
                print(f"{name:<25} {data['logz']:<15.1f} {delta:<+15.0f}")
    
    print("=" * 70)
    print("\nDONE! All values shown are from actual run data.")
    
    plt.show()

if __name__ == "__main__":
    main()
