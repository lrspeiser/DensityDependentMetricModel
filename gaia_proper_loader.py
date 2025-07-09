#!/usr/bin/env python3
"""
gaia_proper_loader.py - Download and validate Gaia data with proper radial distribution
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CylindricalDifferential
from astroquery.gaia import Gaia
import time

# Set longer timeout
if hasattr(Gaia, 'TIMEOUT'):
    Gaia.TIMEOUT = 900
elif hasattr(Gaia, 'tap'):
    if hasattr(Gaia.tap, 'timeout'):
        Gaia.tap.timeout = 900

# Galactocentric frame parameters
R0_KPC = 8.122 * u.kpc
ZSUN_KPC = 0.025 * u.kpc
VSUN_KMS = CartesianDifferential([11.1, 245.6, 7.25] * u.km/u.s)


def query_gaia_disk_stars(n_stars_per_bin=5000, r_bins=None, cache_file="gaia_disk_stars_raw.parquet"):
    """
    Query Gaia for disk stars with good radial coverage.
    
    Parameters:
    -----------
    n_stars_per_bin : int
        Target number of stars per radial bin
    r_bins : list
        Radial bins in kpc. Default: [4, 6, 8, 10, 12, 15, 20]
    cache_file : str
        Cache filename
    """
    if r_bins is None:
        r_bins = [4, 6, 8, 10, 12, 15, 20]
    
    cache_path = Path(cache_file)
    if cache_path.exists():
        print(f"💾 Loading cached raw data from {cache_file}")
        return pd.read_parquet(cache_path)
    
    print("📡 Querying Gaia for disk stars with good radial coverage...")
    print(f"   Target: {n_stars_per_bin} stars in each of {len(r_bins)-1} radial bins")
    
    all_results = []
    
    # Query different sky regions to get various distances
    # Use galactic coordinates to target disk
    l_regions = [
        (30, 60),    # Anti-center direction
        (60, 90),    
        (90, 120),
        (180, 210),  # Toward center (but avoiding bulge)
        (210, 240),
        (240, 270),
        (270, 300),
        (300, 330)
    ]
    
    for l_min, l_max in l_regions:
        print(f"\n   Querying l = {l_min}°-{l_max}°...")
        
        # Adjust distance range based on longitude
        # Stars toward anti-center are more distant
        if l_min < 180:
            dist_min, dist_max = 500, 15000  # parsecs
        else:
            dist_min, dist_max = 200, 8000
        
        query = f"""
        SELECT TOP {n_stars_per_bin}
            source_id, ra, dec, l, b,
            parallax, parallax_error,
            pmra, pmra_error, pmdec, pmdec_error,
            radial_velocity, radial_velocity_error,
            phot_g_mean_mag, ruwe
        FROM gaiadr3.gaia_source
        WHERE l BETWEEN {l_min} AND {l_max}
          AND b BETWEEN -10 AND 10
          AND parallax > {1000.0/dist_max}
          AND parallax < {1000.0/dist_min}
          AND parallax_over_error > 5
          AND pmra IS NOT NULL AND pmdec IS NOT NULL
          AND pmra_error < 1 AND pmdec_error < 1
          AND radial_velocity IS NOT NULL
          AND ruwe < 1.6
          AND radial_velocity_error < 30
          AND phot_g_mean_mag < 17
        ORDER BY random_index
        """
        
        try:
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            df_region = results.to_pandas()
            print(f"      Got {len(df_region)} stars")
            all_results.append(df_region)
        except Exception as e:
            print(f"      Failed: {e}")
            continue
    
    if not all_results:
        raise RuntimeError("Failed to get any Gaia data!")
    
    # Combine all results
    df_combined = pd.concat(all_results, ignore_index=True)
    print(f"\n✅ Total raw stars: {len(df_combined)}")
    
    # Save raw data
    df_combined.to_parquet(cache_path)
    print(f"💾 Saved raw data to {cache_path}")
    
    return df_combined


def process_to_galactocentric(df_raw, cache_file="gaia_disk_stars_processed.parquet"):
    """
    Convert raw Gaia data to galactocentric coordinates and validate distribution.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        print(f"💾 Loading cached processed data from {cache_file}")
        df_proc = pd.read_parquet(cache_path)
        if validate_radial_distribution(df_proc):
            return df_proc
        else:
            print("   Cached data failed validation, reprocessing...")
    
    print("\n🌌 Converting to galactocentric coordinates...")
    
    # Set up coordinate frame
    gc_frame = Galactocentric(
        galcen_distance=R0_KPC,
        z_sun=ZSUN_KPC,
        galcen_v_sun=VSUN_KMS
    )
    
    # Convert coordinates
    coords_icrs = SkyCoord(
        ra=df_raw['ra'].values * u.deg,
        dec=df_raw['dec'].values * u.deg,
        distance=(1000.0 / df_raw['parallax'].values) * u.pc,
        pm_ra_cosdec=df_raw['pmra'].values * u.mas/u.yr,
        pm_dec=df_raw['pmdec'].values * u.mas/u.yr,
        radial_velocity=df_raw['radial_velocity'].values * u.km/u.s,
        frame='icrs'
    )
    
    coords_gc = coords_icrs.transform_to(gc_frame)
    
    # Extract cylindrical coordinates
    R_kpc = coords_gc.cylindrical.rho.to(u.kpc).value
    z_kpc = coords_gc.z.to(u.kpc).value
    
    # Get velocities in cylindrical coordinates
    cyl_diff = coords_gc.velocity.represent_as(CylindricalDifferential, coords_gc.data)
    v_R = cyl_diff.d_rho.to(u.km/u.s).value
    v_phi = (coords_gc.cylindrical.rho * cyl_diff.d_phi).to(u.km/u.s, u.dimensionless_angles()).value
    v_z = cyl_diff.d_z.to(u.km/u.s).value
    
    # Calculate velocity errors (simplified)
    distance_kpc = coords_icrs.distance.to(u.kpc).value
    pm_err = np.sqrt(df_raw['pmra_error']**2 + df_raw['pmdec_error']**2)
    v_tan_err = 4.74 * distance_kpc * pm_err  # km/s
    v_err = np.sqrt(df_raw['radial_velocity_error']**2 + v_tan_err**2)
    v_err = np.clip(v_err, 5, 50)  # Reasonable bounds
    
    # Create processed dataframe
    df_processed = pd.DataFrame({
        'source_id': df_raw['source_id'],
        'R_kpc': R_kpc,
        'z_kpc': z_kpc,
        'v_R': v_R,
        'v_phi': v_phi,
        'v_z': v_z,
        'v_obs': np.abs(v_phi),  # Use absolute tangential velocity
        'sigma_v': v_err,
        'l': df_raw['l'],
        'b': df_raw['b']
    })
    
    # Quality cuts
    mask = (
        (df_processed['R_kpc'] > 3) & (df_processed['R_kpc'] < 20) &
        (np.abs(df_processed['z_kpc']) < 1.0) &  # Disk stars
        (df_processed['v_obs'] > 50) & (df_processed['v_obs'] < 350) &
        (df_processed['sigma_v'] < 50)
    )
    
    df_processed = df_processed[mask].copy()
    print(f"   After quality cuts: {len(df_processed)} stars")
    
    # Validate distribution
    if not validate_radial_distribution(df_processed):
        print("⚠️  WARNING: Processed data has poor radial distribution!")
    
    # Save
    df_processed.to_parquet(cache_path)
    print(f"💾 Saved processed data to {cache_path}")
    
    return df_processed


def validate_radial_distribution(df, min_per_bin=20, r_bins=None):
    """
    Validate that data has good radial coverage.
    """
    if r_bins is None:
        r_bins = [3, 5, 7, 9, 11, 13, 15, 20]
    
    print("\n📊 Validating radial distribution:")
    
    counts, _ = np.histogram(df['R_kpc'], bins=r_bins)
    
    valid = True
    for i, count in enumerate(counts):
        print(f"   R = {r_bins[i]:.0f}-{r_bins[i+1]:.0f} kpc: {count} stars", end="")
        if count < min_per_bin:
            print(" ⚠️  Too few!")
            valid = False
        else:
            print(" ✓")
    
    # Also check concentration
    r_std = df['R_kpc'].std()
    print(f"   R std dev: {r_std:.2f} kpc", end="")
    if r_std < 2.0:
        print(" ⚠️  Too concentrated!")
        valid = False
    else:
        print(" ✓")
    
    return valid


def create_balanced_sample(df, n_per_bin=100, r_bins=None):
    """
    Create a balanced sample with roughly equal stars per radial bin.
    """
    if r_bins is None:
        r_bins = [3, 5, 7, 9, 11, 13, 15, 20]
    
    print(f"\n⚖️  Creating balanced sample ({n_per_bin} stars per bin)...")
    
    samples = []
    for i in range(len(r_bins)-1):
        mask = (df['R_kpc'] >= r_bins[i]) & (df['R_kpc'] < r_bins[i+1])
        df_bin = df[mask]
        
        if len(df_bin) >= n_per_bin:
            df_sample = df_bin.sample(n=n_per_bin, random_state=42)
        else:
            df_sample = df_bin
        
        samples.append(df_sample)
        print(f"   R = {r_bins[i]}-{r_bins[i+1]} kpc: {len(df_sample)} stars")
    
    df_balanced = pd.concat(samples, ignore_index=True)
    print(f"   Total: {len(df_balanced)} stars")
    
    return df_balanced


def plot_data_distribution(df, title="Gaia Data Distribution"):
    """
    Plot the data distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # R distribution
    ax = axes[0, 0]
    ax.hist(df['R_kpc'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Count')
    ax.set_title('Radial Distribution')
    ax.axvline(8.0, color='red', linestyle='--', label='Solar radius')
    ax.legend()
    
    # Rotation curve
    ax = axes[0, 1]
    ax.scatter(df['R_kpc'], df['v_obs'], alpha=0.3, s=10)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.set_title('Rotation Curve')
    ax.grid(True, alpha=0.3)
    
    # Sky distribution
    ax = axes[1, 0]
    ax.scatter(df['l'], df['b'], alpha=0.3, s=10, c=df['R_kpc'], cmap='viridis')
    ax.set_xlabel('l (deg)')
    ax.set_ylabel('b (deg)')
    ax.set_title('Sky Distribution (colored by R)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-30, 30)
    
    # R vs z
    ax = axes[1, 1]
    ax.scatter(df['R_kpc'], df['z_kpc'], alpha=0.3, s=10)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('z (kpc)')
    ax.set_title('Vertical Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def load_gaia_with_validation(n_stars=1000, cache_dir="gaia_cache", force_new=False):
    """
    Main function to load validated Gaia data.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    raw_cache = cache_dir / "gaia_disk_raw.parquet"
    proc_cache = cache_dir / "gaia_disk_processed.parquet"
    final_cache = cache_dir / "gaia_query_cache_DR3_processed_for_fit.parquet"
    
    if not force_new and final_cache.exists():
        print(f"💾 Loading final processed data from {final_cache}")
        df_final = pd.read_parquet(final_cache)
        if validate_radial_distribution(df_final):
            return df_final
        else:
            print("   Cached data failed validation, regenerating...")
    
    # Get raw data
    df_raw = query_gaia_disk_stars(
        n_stars_per_bin=max(200, n_stars // 5),
        cache_file=str(raw_cache)
    )
    
    # Process to galactocentric
    df_processed = process_to_galactocentric(df_raw, cache_file=str(proc_cache))
    
    # Create balanced sample
    n_per_bin = max(50, n_stars // 7)  # ~7 radial bins
    df_balanced = create_balanced_sample(df_processed, n_per_bin=n_per_bin)
    
    # Final sample to requested size
    if len(df_balanced) > n_stars:
        # Stratified sampling to maintain distribution
        df_final = df_balanced.groupby(pd.cut(df_balanced['R_kpc'], bins=7)).apply(
            lambda x: x.sample(n=max(1, int(len(x) * n_stars / len(df_balanced))), random_state=42)
        ).reset_index(drop=True)
    else:
        df_final = df_balanced
    
    print(f"\n✅ Final sample: {len(df_final)} stars")
    
    # Save in expected format
    df_final[['R_kpc', 'v_obs', 'sigma_v', 'z_kpc']].to_parquet(final_cache)
    
    # Plot distribution
    fig = plot_data_distribution(df_final, f"Final Gaia Sample (N={len(df_final)})")
    fig.savefig(cache_dir / "gaia_data_distribution.png", dpi=150)
    plt.close()
    
    return df_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stars', type=int, default=1000, help='Number of stars to load')
    parser.add_argument('--force_new', action='store_true', help='Force new query')
    args = parser.parse_args()
    
    print("="*60)
    print("GAIA PROPER DATA LOADER")
    print("="*60)
    
    df = load_gaia_with_validation(n_stars=args.n_stars, force_new=args.force_new)
    
    print("\n" + "="*60)
    print("Data ready for dynesty!")
    print(f"File: gaia_cache/gaia_query_cache_DR3_processed_for_fit.parquet")
    print("="*60)