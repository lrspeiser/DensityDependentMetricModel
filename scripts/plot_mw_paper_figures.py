#!/usr/bin/env python3
"""
Generate publication-quality Milky Way rotation curve plots showing:
1. Actual stellar data points with error bars
2. GR (baryons-only) prediction
3. NFW dark matter model
4. Tidal_band model
5. RAR_blend model
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

# Import physics modules
try:
    from core.density_metric_cupy import v_total_kms_cupy
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available, using simplified models")

def load_gaia_data():
    """Load the binned Gaia DR3 data used in the fits."""
    # Try to load from the standard location
    data_file = Path("data/mw_binned_velocities.csv")
    if not data_file.exists():
        # Try alternative location
        data_file = Path("external_data/mw_binned_velocities.csv")
    
    if not data_file.exists():
        raise FileNotFoundError(f"ERROR: No Gaia data found at {data_file}. Run data processing first!")
    
    df = pd.read_csv(data_file)
    return {
        'R_kpc': df['R_kpc'].values,
        'v_obs': df['v_circ'].values if 'v_circ' in df else df['v_obs'].values,
        'v_err': df['sigma_v'].values if 'sigma_v' in df else df['v_err'].values,
        'n_stars': df['n_stars'].values if 'n_stars' in df else np.ones(len(df))
    }

def load_model_params(run_dir):
    """Load best-fit parameters and evidence from a run directory."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Warning: Run directory {run_path} not found")
        return None, None
    
    # Try to load from enhanced summary first
    summary_file = run_path / "run_summary_enhanced.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            params = None
            logz = None
            
            # Get parameters
            if 'parameter_estimates' in summary and 'best_fit' in summary['parameter_estimates']:
                params = summary['parameter_estimates']['best_fit']
            elif 'best_params' in summary:
                params = summary['best_params']
            
            # Get evidence
            if 'evidence_metrics' in summary and 'logz' in summary['evidence_metrics']:
                logz = summary['evidence_metrics']['logz']
            elif 'logZ' in summary:
                logz = summary['logZ']
            
            return params, logz
    
    # Try NPZ file
    npz_file = run_path / "posterior_samples.npz"
    if npz_file.exists():
        data = np.load(npz_file)
        samples = data['samples']
        logl = data['logl'] if 'logl' in data else None
        param_names = data['param_names']
        
        if logl is not None:
            best_idx = np.argmax(logl)
        else:
            # Use median as best
            best_idx = len(samples) // 2
        
        params = dict(zip(param_names, samples[best_idx]))
        logz = data.get('logz', None)
        return params, logz
    
    return None, None

def compute_gr_curve(R_kpc, baryon_params=None):
    """Compute GR (Newtonian) rotation curve."""
    if baryon_params is None:
        # Default MW baryon parameters
        baryon_params = {
            'M_disk': 5e10,  # M_sun
            'R_d': 3.0,       # kpc
            'M_bulge': 1e10,
            'R_b': 0.5
        }
    
    # Simple exponential disk + bulge model
    M_disk = baryon_params.get('M_disk', 5e10)
    R_d = baryon_params.get('R_d', 3.0)
    M_bulge = baryon_params.get('M_bulge', 1e10)
    R_b = baryon_params.get('R_b', 0.5)
    
    # Disk contribution (simplified)
    v_disk_sq = 4.302e-6 * M_disk * R_kpc / R_d**2 * (1 - np.exp(-R_kpc/R_d))
    
    # Bulge contribution (Hernquist)
    v_bulge_sq = 4.302e-6 * M_bulge / (R_kpc + R_b)
    
    return np.sqrt(v_disk_sq + v_bulge_sq)

def compute_nfw_curve(R_kpc, V200=200, c=10):
    """Compute NFW dark matter halo rotation curve."""
    # NFW parameters
    R200 = V200 / 10  # Approximate R200 in kpc
    Rs = R200 / c
    
    # NFW circular velocity
    x = R_kpc / Rs
    v_nfw = V200 * np.sqrt((np.log(1 + x) - x/(1 + x)) / (x * (np.log(1 + c) - c/(1 + c))))
    
    return v_nfw

def compute_model_curve(R_kpc, params, xi_type):
    """Compute model rotation curve using actual physics."""
    if not CUPY_AVAILABLE:
        # Simplified model
        v_gr = compute_gr_curve(R_kpc)
        if xi_type == 'tidal_band':
            # Simple enhancement
            xi = 1 + 0.5 * np.exp(-R_kpc/10)
            return v_gr * np.sqrt(xi)
        elif xi_type == 'rar_blend':
            # Stronger enhancement
            xi = 1 + 2.0 * np.exp(-R_kpc/15)
            return v_gr * np.sqrt(xi)
        else:
            return v_gr
    
    # Use actual physics model
    R_cp = cp.asarray(R_kpc, dtype=cp.float32)
    v_model = v_total_kms_cupy(R_cp, params, xi_type=xi_type)
    return cp.asnumpy(v_model)

def create_mw_rotation_plot():
    """Create the main Milky Way rotation curve comparison plot."""
    # Load Gaia data
    gaia_data = load_gaia_data()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Extended R range for models
    R_model = np.linspace(1, 30, 200)
    
    # Plot 1: Rotation curves
    # Gaia data points with error bars
    ax1.errorbar(gaia_data['R_kpc'], gaia_data['v_obs'], 
                yerr=gaia_data['v_err'],
                fmt='o', color='black', markersize=6, 
                capsize=3, alpha=0.7, label='Gaia DR3 data')
    
    # GR (baryons only)
    v_gr = compute_gr_curve(R_model)
    ax1.plot(R_model, v_gr, 'b--', linewidth=2, 
            label='GR (baryons only)', alpha=0.8)
    
    # NFW dark matter
    v_nfw_halo = compute_nfw_curve(R_model, V200=220, c=12)
    v_nfw_total = np.sqrt(v_gr**2 + v_nfw_halo**2)
    ax1.plot(R_model, v_nfw_total, 'g-.', linewidth=2, 
            label='ΛCDM (NFW halo)', alpha=0.8)
    
    # Load model results if available
    tidal_params, tidal_logz = load_model_params("runs/tidal_band_from_best_20250820_185242")
    rar_params, rar_logz = load_model_params("runs/rar_blend_20250823_211648")
    gr_params, gr_logz = load_model_params("runs/gr_20250812_113949")
    nfw_params, nfw_logz = load_model_params("runs/nfw_20250812_114008")
    
    if tidal_params:
        try:
            v_tidal = compute_model_curve(R_model, tidal_params, 'tidal_band')
            ax1.plot(R_model, v_tidal, 'c-', linewidth=2.5, 
                    label='Tidal Band (TFR)', alpha=0.9)
        except Exception as e:
            print(f"Error computing tidal band curve: {e}")
    
    if rar_params:
        try:
            v_rar = compute_model_curve(R_model, rar_params, 'rar_blend')
            ax1.plot(R_model, v_rar, 'r-', linewidth=2.5, 
                    label='RAR Blend', alpha=0.9)
        except Exception as e:
            print(f"Error computing RAR blend curve: {e}")
    
    # Formatting
    ax1.set_xlim(3, 25)
    ax1.set_ylim(150, 280)
    ax1.set_ylabel('Circular Velocity (km/s)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_title('Milky Way Rotation Curve: Data vs Models', 
                 fontsize=14, fontweight='bold')
    
    # Add solar position marker
    ax1.axvline(x=8.5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(8.5, 155, 'Sun', ha='center', fontsize=10, color='orange')
    
    # Plot 2: Residuals
    # Interpolate models to data points
    from scipy.interpolate import interp1d
    
    R_data = gaia_data['R_kpc']
    v_data = gaia_data['v_obs']
    v_err = gaia_data['v_err']
    
    # GR residuals
    v_gr_interp = interp1d(R_model, v_gr, kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    res_gr = v_data - v_gr_interp(R_data)
    ax2.errorbar(R_data, res_gr, yerr=v_err, 
                fmt='o', color='blue', markersize=4, alpha=0.6, 
                label='GR residuals')
    
    # NFW residuals
    v_nfw_interp = interp1d(R_model, v_nfw_total, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
    res_nfw = v_data - v_nfw_interp(R_data)
    ax2.errorbar(R_data + 0.1, res_nfw, yerr=v_err,
                fmt='s', color='green', markersize=4, alpha=0.6,
                label='NFW residuals')
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax2.set_ylabel('Residuals (km/s)', fontsize=12)
    ax2.set_xlim(3, 25)
    ax2.set_ylim(-40, 40)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_summary_statistics():
    """Generate summary statistics table from actual run data."""
    # Load actual evidence values from runs
    _, gr_logz = load_model_params("runs/gr_20250812_113949")
    _, nfw_logz = load_model_params("runs/nfw_20250812_114008")
    _, tidal_logz = load_model_params("runs/tidal_band_from_best_20250820_185242")
    _, rar_logz = load_model_params("runs/rar_blend_20250823_211648")
    
    # If we don't have evidence, return empty table
    if gr_logz is None:
        print("Warning: Could not load evidence values from runs")
        return pd.DataFrame()
    
    stats = {
        'Model': ['GR (baryons)', 'ΛCDM (NFW)', 'Tidal Band', 'RAR Blend'],
        'log(Z)': [
            gr_logz if gr_logz else 'N/A',
            nfw_logz if nfw_logz else 'N/A',
            tidal_logz if tidal_logz else 'N/A',
            rar_logz if rar_logz else 'N/A'
        ]
    }
    
    # Calculate relative evidence
    if gr_logz:
        stats['Δlog(Z) vs GR'] = [
            0,
            (nfw_logz - gr_logz) if nfw_logz else 'N/A',
            (tidal_logz - gr_logz) if tidal_logz else 'N/A',
            (rar_logz - gr_logz) if rar_logz else 'N/A'
        ]
    
    df = pd.DataFrame(stats)
    return df

def main():
    """Generate all publication figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES FOR MILKY WAY")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("paper_figures")
    output_dir.mkdir(exist_ok=True)
    
    # Generate main rotation curve plot
    print("\n1. Creating Milky Way rotation curve comparison...")
    fig1 = create_mw_rotation_plot()
    fig1.savefig(output_dir / "mw_rotation_curves_comparison.png", 
                dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / "mw_rotation_curves_comparison.pdf", 
                bbox_inches='tight')
    print(f"   Saved to {output_dir}/mw_rotation_curves_comparison.[png/pdf]")
    
    # Generate statistics table
    print("\n2. Creating model comparison statistics...")
    stats_df = create_summary_statistics()
    print("\n" + stats_df.to_string(index=False))
    stats_df.to_csv(output_dir / "model_statistics.csv", index=False)
    stats_df.to_latex(output_dir / "model_statistics.tex", index=False)
    print(f"\n   Saved to {output_dir}/model_statistics.[csv/tex]")
    
    print("\n" + "=" * 70)
    print("Publication figures generated successfully!")
    print("Files saved in paper_figures/")
    
    plt.show()

if __name__ == "__main__":
    main()
