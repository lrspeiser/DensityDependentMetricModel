#!/usr/bin/env python3
"""
analyze_gaia_distribution.py - Analyze Gaia star distribution for academic paper

This script analyzes the Gaia data used in the DDMM runs to provide
detailed statistics about the star distribution, radial bins, and
data quality for academic paper documentation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional - removed for compatibility
from datetime import datetime

def process_gaia_data_for_analysis(df_raw):
    """Process raw Gaia data to get the columns used in DDMM runs."""
    
    print("Processing raw Gaia data to calculate Galactocentric coordinates...")
    
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord, Galactocentric, CylindricalDifferential
    except ImportError:
        print("ERROR: astropy not available. Cannot process Gaia data.")
        return None
    
    # Define the Sun's Galactocentric frame (same as used in DDMM runs)
    R0_KPC = 8.122  # Distance from Sun to Galactic center
    ZSUN_KPC = 0.0208  # Height of Sun above Galactic plane
    VSUN_KMS = [11.1, 232.24, 7.25]  # Solar motion [U, V, W]
    
    gc_frame = Galactocentric(
        galcen_distance=R0_KPC * u.kpc,
        z_sun=ZSUN_KPC * u.kpc,
        galcen_v_sun=VSUN_KMS * u.km/u.s
    )
    
    # Create Astropy SkyCoord object from the raw Gaia data
    coords = SkyCoord(
        ra=df_raw['ra'].values * u.deg,
        dec=df_raw['dec'].values * u.deg,
        distance=(1000 / df_raw['parallax'].values) * u.pc,
        pm_ra_cosdec=df_raw['pmra'].values * u.mas/u.yr,
        pm_dec=df_raw['pmdec'].values * u.mas/u.yr,
        radial_velocity=df_raw['radial_velocity'].values * u.km/u.s,
        frame='icrs'
    )
    
    # Transform to Galactocentric coordinates
    galcen_coords = coords.transform_to(gc_frame)
    cylindrical_velocities = galcen_coords.velocity.represent_as(
        CylindricalDifferential, galcen_coords.data
    )
    
    # Calculate the tangential velocity (v_phi)
    v_phi_kms = (galcen_coords.cylindrical.rho * cylindrical_velocities.d_phi).to(
        u.km/u.s, equivalencies=u.dimensionless_angles()
    ).value
    
    # The observed rotation velocity is the absolute value of v_phi
    v_obs_kms = np.abs(v_phi_kms)
    
    # Calculate the cylindrical coordinates
    R_kpc = galcen_coords.cylindrical.rho.to(u.kpc).value
    
    # Error propagation
    dist_kpc = coords.distance.to(u.kpc).value
    pm_error_kms = np.sqrt(df_raw['pmra_error']**2 + df_raw['pmdec_error']**2) * dist_kpc * 4.74047
    
    sigma_v = np.sqrt(
        df_raw['radial_velocity_error'].fillna(0)**2 + 
        pm_error_kms.fillna(0)**2
    )
    
    # Apply error bounds (same as in DDMM runs)
    MIN_VELOCITY_ERROR_KMS = 1.0
    MAX_VELOCITY_ERROR_KMS = 100.0
    sigma_v = np.clip(sigma_v, MIN_VELOCITY_ERROR_KMS, MAX_VELOCITY_ERROR_KMS)
    
    # Create processed DataFrame
    df_processed = pd.DataFrame({
        'R_kpc': R_kpc,
        'v_obs': v_obs_kms,
        'sigma_v': sigma_v,
        'z_kpc': galcen_coords.z.to(u.kpc).value,
        'v_R_kms': cylindrical_velocities.d_rho.to(u.km/u.s).value,
        'v_z_kms': cylindrical_velocities.d_z.to(u.km/u.s).value
    })
    
    print(f"✓ Successfully processed {len(df_processed)} stars")
    return df_processed

def analyze_gaia_distribution():
    """Analyze the Gaia star distribution used in DDMM runs."""
    
    print("="*80)
    print("GAIA STAR DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the raw Gaia data
    gaia_file = Path("gaia_sky_slices/all_sky_gaia.csv")
    
    if not gaia_file.exists():
        print(f"ERROR: Gaia data file not found: {gaia_file}")
        return None
    
    print(f"Loading raw Gaia data from: {gaia_file}")
    df_raw = pd.read_csv(gaia_file)
    
    print(f"Total stars loaded: {len(df_raw):,}")
    print(f"DataFrame shape: {df_raw.shape}")
    print(f"Raw columns: {df_raw.columns.tolist()}")
    print()
    
    # Process the data to get the columns used in DDMM runs
    df = process_gaia_data_for_analysis(df_raw)
    if df is None:
        return None
    
    # Check for required columns
    required_cols = ["R_kpc", "v_obs", "sigma_v"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns after processing: {missing_cols}")
        return None
    
    # Basic statistics
    print("BASIC STATISTICS:")
    print("-" * 40)
    for col in required_cols:
        print(f"{col}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Std:  {df[col].std():.3f}")
        print(f"  Min:  {df[col].min():.3f}")
        print(f"  Max:  {df[col].max():.3f}")
        print(f"  Median: {df[col].median():.3f}")
        print()
    
    # Radial distribution analysis
    print("RADIAL DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    # Create radial bins (0-1, 1-2, ..., 15-16 kpc)
    bins = np.arange(0, 16.1, 1.0)
    bin_labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]
    
    counts, bin_edges = np.histogram(df['R_kpc'], bins=bins)
    
    print("Stars per 1 kpc radial bin:")
    for i, (label, count) in enumerate(zip(bin_labels, counts)):
        print(f"  {label} kpc: {count:,} stars")
    
    print()
    print(f"Total stars in 0-16 kpc range: {counts.sum():,}")
    print(f"Stars beyond 16 kpc: {len(df[df['R_kpc'] > 16]):,}")
    print()
    
    # Velocity distribution analysis
    print("VELOCITY DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    v_bins = np.arange(0, 400, 20)  # 0-400 km/s in 20 km/s bins
    v_counts, v_edges = np.histogram(df['v_obs'], bins=v_bins)
    
    print("Stars per 20 km/s velocity bin:")
    for i, (start, end) in enumerate(zip(v_edges[:-1], v_edges[1:])):
        print(f"  {start:.0f}-{end:.0f} km/s: {v_counts[i]:,} stars")
    
    print()
    print(f"Velocity range: {df['v_obs'].min():.1f} - {df['v_obs'].max():.1f} km/s")
    print(f"Mean velocity: {df['v_obs'].mean():.1f} ± {df['v_obs'].std():.1f} km/s")
    print()
    
    # Data quality analysis
    print("DATA QUALITY ANALYSIS:")
    print("-" * 40)
    
    # Check for NaN/inf values
    for col in required_cols:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum()
        print(f"{col}:")
        print(f"  NaN values: {n_nan:,}")
        print(f"  Inf values: {n_inf:,}")
        print(f"  Valid values: {len(df) - n_nan - n_inf:,}")
        print()
    
    # Uncertainty analysis
    print("UNCERTAINTY ANALYSIS:")
    print("-" * 40)
    
    sigma_stats = df['sigma_v'].describe()
    print("Velocity uncertainty (sigma_v) statistics:")
    print(f"  Mean: {sigma_stats['mean']:.2f} km/s")
    print(f"  Std:  {sigma_stats['std']:.2f} km/s")
    print(f"  Min:  {sigma_stats['min']:.2f} km/s")
    print(f"  Max:  {sigma_stats['max']:.2f} km/s")
    print(f"  25%:  {sigma_stats['25%']:.2f} km/s")
    print(f"  50%:  {sigma_stats['50%']:.2f} km/s")
    print(f"  75%:  {sigma_stats['75%']:.2f} km/s")
    print()
    
    # Create summary for academic paper
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "total_stars": len(df),
        "radial_range_kpc": [float(df['R_kpc'].min()), float(df['R_kpc'].max())],
        "velocity_range_kms": [float(df['v_obs'].min()), float(df['v_obs'].max())],
        "mean_velocity_kms": float(df['v_obs'].mean()),
        "velocity_std_kms": float(df['v_obs'].std()),
        "radial_distribution": {
            "bin_edges": bins.tolist(),
            "bin_labels": bin_labels,
            "star_counts": counts.tolist()
        },
        "velocity_distribution": {
            "bin_edges": v_edges.tolist(),
            "star_counts": v_counts.tolist()
        },
        "data_quality": {
            "nan_counts": {col: int(df[col].isna().sum()) for col in required_cols},
            "inf_counts": {col: int(np.isinf(df[col]).sum()) for col in required_cols}
        },
        "uncertainty_stats": {
            "mean_sigma_v": float(sigma_stats['mean']),
            "std_sigma_v": float(sigma_stats['std']),
            "median_sigma_v": float(sigma_stats['50%'])
        }
    }
    
    # Save summary to JSON
    output_file = Path("gaia_distribution_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_file}")
    print()
    
    # Create plots
    create_distribution_plots(df, summary)
    
    return summary

def create_distribution_plots(df, summary):
    """Create visualization plots for the Gaia distribution."""
    
    # Set up the plotting style
    plt.style.use('default')
    # sns.set_palette("husl")  # Optional - removed for compatibility
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gaia Star Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Radial distribution
    ax1 = axes[0, 0]
    bins = np.arange(0, 16.1, 1.0)
    ax1.hist(df['R_kpc'], bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Galactocentric Radius (kpc)')
    ax1.set_ylabel('Number of Stars')
    ax1.set_title('Radial Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity distribution
    ax2 = axes[0, 1]
    v_bins = np.arange(0, 400, 20)
    ax2.hist(df['v_obs'], bins=v_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Observed Velocity (km/s)')
    ax2.set_ylabel('Number of Stars')
    ax2.set_title('Velocity Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity vs Radius scatter
    ax3 = axes[1, 0]
    # Sample a subset for visibility
    sample_size = min(10000, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    ax3.scatter(df.iloc[sample_indices]['R_kpc'], 
               df.iloc[sample_indices]['v_obs'], 
               alpha=0.3, s=1)
    ax3.set_xlabel('Galactocentric Radius (kpc)')
    ax3.set_ylabel('Observed Velocity (km/s)')
    ax3.set_title('Velocity vs Radius (10k sample)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty distribution
    ax4 = axes[1, 1]
    ax4.hist(df['sigma_v'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Velocity Uncertainty (km/s)')
    ax4.set_ylabel('Number of Stars')
    ax4.set_title('Velocity Uncertainty Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = Path("gaia_distribution_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to: {plot_file}")
    plt.close()

def print_academic_summary(summary):
    """Print a formatted summary suitable for academic papers."""
    
    print("="*80)
    print("ACADEMIC PAPER SUMMARY")
    print("="*80)
    print()
    
    print("DATA SET CHARACTERISTICS:")
    print(f"• Total number of stars: {summary['total_stars']:,}")
    print(f"• Radial coverage: {summary['radial_range_kpc'][0]:.1f} - {summary['radial_range_kpc'][1]:.1f} kpc")
    print(f"• Velocity coverage: {summary['velocity_range_kms'][0]:.1f} - {summary['velocity_range_kms'][1]:.1f} km/s")
    print(f"• Mean velocity: {summary['mean_velocity_kms']:.1f} ± {summary['velocity_std_kms']:.1f} km/s")
    print()
    
    print("RADIAL DISTRIBUTION (1 kpc bins):")
    for i, (label, count) in enumerate(zip(summary['radial_distribution']['bin_labels'], 
                                         summary['radial_distribution']['star_counts'])):
        if count > 0:
            print(f"• {label} kpc: {count:,} stars")
    print()
    
    print("DATA QUALITY:")
    total_nan = sum(summary['data_quality']['nan_counts'].values())
    total_inf = sum(summary['data_quality']['inf_counts'].values())
    print(f"• Stars with valid data: {summary['total_stars'] - total_nan - total_inf:,}")
    print(f"• Mean velocity uncertainty: {summary['uncertainty_stats']['mean_sigma_v']:.2f} km/s")
    print(f"• Median velocity uncertainty: {summary['uncertainty_stats']['median_sigma_v']:.2f} km/s")
    print()

if __name__ == "__main__":
    summary = analyze_gaia_distribution()
    if summary:
        print_academic_summary(summary) 