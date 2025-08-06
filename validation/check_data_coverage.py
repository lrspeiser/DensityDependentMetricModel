#!/usr/bin/env python3
"""
check_data_coverage.py - A diagnostic script to analyze the spatial
distribution of stars in the cached Gaia dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

# --- Configuration ---
# This should point to the directory where your processed slice files are.
GAIA_SLICES_DIRECTORY = 'gaia_sky_slices'
# ---------------------

def load_gaia_slices_from_cache(cache_dir: str) -> Optional[pd.DataFrame]:
    """
    Loads and combines all processed .parquet files from the sky slices cache.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ Error: Cache directory '{cache_dir}' not found.")
        print("   Please make sure this script is in the same directory as 'gaia_sky_slices',")
        print("   or update the GAIA_SLICES_DIRECTORY variable.")
        return None

    slice_files = list(cache_path.glob("processed_*.parquet"))
    if not slice_files:
        print(f"❌ Error: No processed .parquet files found in '{cache_dir}'.")
        return None

    print(f"Found {len(slice_files)} data slices. Loading and combining...")
    df_list = [pd.read_parquet(f) for f in slice_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    print(f"✅ Successfully loaded a total of {len(full_df):,} stars from cache.")
    return full_df

def main():
    """
    Main function to run the data analysis.
    """
    print("--- Gaia Data Coverage Report ---")
    
    # Load the full dataset from your cached slices
    gaia_df = load_gaia_slices_from_cache(GAIA_SLICES_DIRECTORY)
    
    if gaia_df is None or gaia_df.empty:
        print("\nCould not load data. Aborting report.")
        return
        
    # --- Analysis ---
    
    # Ensure the correct column name is used
    if 'R_kpc' not in gaia_df.columns:
        print("❌ Error: The required column 'R_kpc' was not found in the DataFrame.")
        return
        
    r_kpc = gaia_df['R_kpc']
    
    # 1. Basic Statistics
    print("\n--- Overall Radial Distribution ---")
    print(f"  Total number of stars: {len(r_kpc):,}")
    print(f"  Minimum Radius (R_kpc):   {r_kpc.min():.2f} kpc")
    print(f"  Maximum Radius (R_kpc):   {r_kpc.max():.2f} kpc")
    print(f"  Median Radius (R_kpc):    {r_kpc.median():.2f} kpc")
    
    # 2. Binned Star Counts
    print("\n--- Star Counts in Radial Bins ---")
    bins = [0, 5, 10, 15, 20, 25, 30, 50]
    # Use pandas.cut to categorize the radii into bins
    binned_r = pd.cut(r_kpc, bins=bins, right=False)
    counts = binned_r.value_counts().sort_index()
    
    for bin_interval, count in counts.items():
        print(f"  Bin {str(bin_interval):<12}: {count:10,d} stars")
        
    # Specifically check the 20-30 kpc range
    stars_in_target_range = r_kpc[(r_kpc >= 20) & (r_kpc < 30)].count()
    print("\n--- Target Range Confirmation ---")
    if stars_in_target_range > 0:
        print(f"✅ CONFIRMED: Found {stars_in_target_range:,} stars between 20 and 30 kpc.")
    else:
        print(f"❌ NOT CONFIRMED: Found 0 stars between 20 and 30 kpc.")
        print(f"   The furthest star is at {r_kpc.max():.2f} kpc.")

    # 3. Visualization
    print("\nGenerating histogram of the radial distribution...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(r_kpc, bins=150, range=(0, 30), alpha=0.8)
    
    # Highlight the target region
    ax.axvspan(20, 30, color='red', alpha=0.2, label='Target Region (20-30 kpc)')
    
    ax.set_title('Radial Distribution of Stars in Gaia Dataset', fontsize=16)
    ax.set_xlabel('Galactocentric Radius (R_kpc)', fontsize=12)
    ax.set_ylabel('Number of Stars', fontsize=12)
    ax.set_xlim(0, 30)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_filename = 'data_radial_distribution.png'
    plt.savefig(output_filename, dpi=150)
    print(f"✅ Plot saved to '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()