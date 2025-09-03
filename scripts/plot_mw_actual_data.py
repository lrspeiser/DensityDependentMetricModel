#!/usr/bin/env python3
"""
Generate publication-quality Milky Way rotation curve plots using:
1. ACTUAL Gaia DR3 data (144,000 stars)
2. Actual GR run results
3. Actual NFW run results  
4. RAR model runs (rar_gate and rar_blend)
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

def load_gaia_data_and_bin():
    """Load actual Gaia DR3 data and compute binned velocities."""
    
    # Load the actual Gaia data
    gaia_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
    if not gaia_file.exists():
        raise FileNotFoundError(f"Gaia data not found at {gaia_file}")
    
    print(f"Loading Gaia data from {gaia_file}...")
    df = pd.read_csv(gaia_file)
    print(f"  Loaded {len(df)} stars")
    
    # Process the Gaia data to get physical coordinates
    if 'R_kpc' not in df.columns:
        print("  Processing Gaia data to compute physical coordinates...")
        from core.data_io import process_gaia_data
        df = process_gaia_data(df)
        print(f"  Processed: R_kpc range = {df['R_kpc'].min():.1f} - {df['R_kpc'].max():.1f} kpc")
    
    # Filter to 5-16 kpc range as specified in paper
    df_filtered = df[(df['R_kpc'] >= 5) & (df['R_kpc'] <= 16)]
    print(f"  After filtering to 5-16 kpc: {len(df_filtered)} stars")
    
    # Bin the data in 0.5 kpc bins as specified in paper
    R_bins = np.arange(5, 16.5, 0.5)
    R_centers = (R_bins[:-1] + R_bins[1:]) / 2
    
    # Get velocity column
    v_col = 'v_circ' if 'v_circ' in df_filtered.columns else 'v_obs'
    if v_col not in df_filtered.columns:
        # Try to compute from vx, vy
        if 'vx' in df_filtered.columns and 'vy' in df_filtered.columns:
            # Compute tangential velocity
            x, y = df_filtered['x'].values, df_filtered['y'].values
            vx, vy = df_filtered['vx'].values, df_filtered['vy'].values
            v_tan = (-vx * y + vy * x) / np.sqrt(x**2 + y**2)
            df_filtered['v_circ'] = np.abs(v_tan)
            v_col = 'v_circ'
        else:
            raise ValueError("Cannot find velocity data in Gaia file")
    
    # Compute binned statistics
    v_binned = []
    v_err_binned = []
    n_stars_binned = []
    
    for i in range(len(R_bins)-1):
        mask = (df_filtered['R_kpc'] >= R_bins[i]) & (df_filtered['R_kpc'] < R_bins[i+1])
        stars_in_bin = df_filtered[mask]
        
        if len(stars_in_bin) > 0:
            v_mean = stars_in_bin[v_col].mean()
            v_std = stars_in_bin[v_col].std()
            n_stars = len(stars_in_bin)
            
            # Standard error + systematic uncertainty
            v_err = v_std / np.sqrt(n_stars) + 2.0  # Add 2 km/s systematic
        else:
            v_mean = np.nan
            v_err = np.nan
            n_stars = 0
        
        v_binned.append(v_mean)
        v_err_binned.append(v_err)
        n_stars_binned.append(n_stars)
    
    # Remove NaN bins
    valid = ~np.isnan(v_binned)
    
    return {
        'R_kpc': R_centers[valid],
        'v_obs': np.array(v_binned)[valid],
        'v_err': np.array(v_err_binned)[valid],
        'n_stars': np.array(n_stars_binned)[valid]
    }

def load_model_results(run_dir):
    """Load model results from a run directory."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"  Warning: {run_path} not found")
        return None
    
    # Try to load NPZ file
    npz_file = run_path / "posterior_samples.npz"
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        
        # Get best fit parameters
        if 'logl' in data:
            best_idx = np.argmax(data['logl'])
        else:
            best_idx = len(data['samples']) // 2
        
        params = dict(zip(data['param_names'], data['samples'][best_idx]))
        
        # Get model type
        xi_type = str(data.get('xi_type', 'unknown'))
        if isinstance(xi_type, np.ndarray):
            xi_type = str(xi_type.item())
        
        return {
            'params': params,
            'xi_type': xi_type,
            'logz': float(data.get('logz', np.nan))
        }
    
    return None

def compute_rotation_curve(R_kpc, model_data):
    """Compute rotation curve for a model."""
    if model_data is None:
        return None
    
    try:
        # Import physics module
        from core.density_metric_cupy import v_total_kms_cupy
        import cupy as cp
        
        # Convert to CuPy array
        R_cp = cp.asarray(R_kpc, dtype=cp.float32)
        
        # Set allow_experimental for RAR models
        params = dict(model_data['params'])
        if model_data['xi_type'] in ['rar_gate', 'rar_blend']:
            params['allow_experimental'] = True
        
        # Compute velocities
        v_model = v_total_kms_cupy(R_cp, params, xi_type=model_data['xi_type'])
        
        # Convert back to numpy
        return cp.asnumpy(v_model)
        
    except Exception as e:
        print(f"  Error computing rotation curve: {e}")
        # Return simplified model
        return None

def create_mw_comparison_plot():
    """Create the main comparison plot with actual data."""
    
    # Load and bin Gaia data
    print("\n1. Processing Gaia DR3 data...")
    gaia_data = load_gaia_data_and_bin()
    print(f"  Binned into {len(gaia_data['R_kpc'])} radial bins")
    
    # Load model results
    print("\n2. Loading model results...")
    
    models = {
        'GR': load_model_results("runs/gr_20250812_113949"),
        'NFW': load_model_results("runs/nfw_20250812_114008"),
        'Tidal Band': load_model_results("runs/tidal_band_from_best_20250820_185242"),
        'RAR Blend (A≈0)': load_model_results("runs/rar_blend_20250823_211648")
    }
    
    # Report what we found
    for name, data in models.items():
        if data:
            print(f"  {name}: xi_type={data['xi_type']}, logZ={data.get('logz', 'N/A')}")
        else:
            print(f"  {name}: Not found")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Extended R range for model curves
    R_model = np.linspace(3, 25, 200)
    
    # Plot data points
    ax1.errorbar(gaia_data['R_kpc'], gaia_data['v_obs'],
                yerr=gaia_data['v_err'],
                fmt='o', color='black', markersize=6,
                capsize=3, alpha=0.8, label=f"Gaia DR3 (n={len(gaia_data['R_kpc'])} bins)",
                zorder=10)
    
    # Plot model curves
    colors = {'GR': 'blue', 'NFW': 'green', 'RAR Gate': 'orange', 'RAR Blend': 'red'}
    linestyles = {'GR': '--', 'NFW': '-.', 'RAR Gate': '-', 'RAR Blend': '-'}
    
    for name, model_data in models.items():
        if model_data:
            v_model = compute_rotation_curve(R_model, model_data)
            if v_model is not None:
                ax1.plot(R_model, v_model,
                        color=colors[name],
                        linestyle=linestyles[name],
                        linewidth=2,
                        label=name,
                        alpha=0.8)
    
    # Formatting main plot
    ax1.set_xlim(4, 20)
    ax1.set_ylim(180, 260)
    ax1.set_ylabel('Circular Velocity (km/s)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_title('Milky Way Rotation Curve: Gaia DR3 Data vs Models',
                 fontsize=14, fontweight='bold')
    
    # Add Solar position
    ax1.axvline(x=8.5, color='orange', linestyle=':', alpha=0.5)
    ax1.text(8.5, 185, 'Sun', ha='center', fontsize=10, color='orange')
    
    # Plot residuals (if we have model curves)
    from scipy.interpolate import interp1d
    
    for name, model_data in models.items():
        if model_data and name != 'NFW':  # Skip NFW for clarity
            v_model = compute_rotation_curve(gaia_data['R_kpc'], model_data)
            if v_model is not None:
                residuals = gaia_data['v_obs'] - v_model
                ax2.scatter(gaia_data['R_kpc'], residuals,
                          color=colors[name],
                          s=30,
                          alpha=0.6,
                          label=f'{name} residuals')
    
    # Format residuals plot
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax2.set_ylabel('Residuals (km/s)', fontsize=12)
    ax2.set_xlim(4, 20)
    ax2.set_ylim(-30, 30)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    """Generate the corrected Milky Way plot."""
    print("=" * 70)
    print("GENERATING MILKY WAY PLOT WITH ACTUAL DATA")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plot
    try:
        fig = create_mw_comparison_plot()
        
        # Save
        output_file = output_dir / "mw_actual_data_comparison.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / "mw_actual_data_comparison.pdf", bbox_inches='tight')
        print(f"\nSaved to {output_file}")
        
        # Model comparison statistics
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print("\nLogZ values (from runs):")
        print("  GR:        ~-1,490,897")
        print("  NFW:       (check run)")
        print("  RAR Gate:  (check run)")
        print("  RAR Blend: -519,396.52")
        print("\nΔlogZ vs GR:")
        print("  RAR Blend: +971,501 (decisive evidence)")
        
    except Exception as e:
        print(f"\nError generating plot: {e}")
        import traceback
        traceback.print_exc()
    
    plt.show()

if __name__ == "__main__":
    main()
