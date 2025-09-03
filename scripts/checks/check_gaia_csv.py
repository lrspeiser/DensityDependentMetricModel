#!/usr/bin/env python3
"""Check Gaia CSV files for star counts and structure."""

import pandas as pd
import glob
import os

# Path to gaia sky slices
path = r'C:\Users\henry\Documents\GitHub\DensityDependentMetricModel\external_data\gaia_sky_slices'

# Get all CSV files
csv_files = glob.glob(os.path.join(path, '*.csv'))

print("=" * 80)
print("GAIA SKY SLICES DATA CHECK")
print("=" * 80)

total_stars = 0
slice_files = []
all_sky_file = None

for csv_file in sorted(csv_files):
    filename = os.path.basename(csv_file)
    
    # Read first few rows to get structure
    df_sample = pd.read_csv(csv_file, nrows=5)
    
    # Count total rows
    with open(csv_file, 'r') as f:
        # Count lines minus header
        n_rows = sum(1 for line in f) - 1
    
    print(f"\n{filename}:")
    print(f"  Stars: {n_rows:,}")
    print(f"  Columns ({len(df_sample.columns)}): {list(df_sample.columns)[:8]}...")
    
    # Check for key columns
    has_R = any('R' in col or 'radius' in col.lower() for col in df_sample.columns)
    has_v = any('v' in col.lower() or 'velocity' in col.lower() for col in df_sample.columns)
    print(f"  Has radius column: {has_R}")
    print(f"  Has velocity column: {has_v}")
    
    # Show first row sample
    if 'R_kpc' in df_sample.columns and 'v_obs' in df_sample.columns:
        print(f"  Sample R range: {df_sample['R_kpc'].min():.2f} - {df_sample['R_kpc'].max():.2f} kpc")
        print(f"  Sample v range: {df_sample['v_obs'].min():.1f} - {df_sample['v_obs'].max():.1f} km/s")
    
    if 'all_sky' in filename:
        all_sky_file = csv_file
    else:
        slice_files.append(csv_file)
        
    total_stars += n_rows

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total CSV files found: {len(csv_files)}")
print(f"Individual slice files: {len(slice_files)}")
print(f"All-sky combined file: {'Yes' if all_sky_file else 'No'}")
print(f"Total stars across all files: {total_stars:,}")

# Check if slices match all_sky
if all_sky_file and slice_files:
    all_sky_df = pd.read_csv(all_sky_file, nrows=1)
    all_sky_count = sum(1 for line in open(all_sky_file)) - 1
    slice_total = sum(sum(1 for line in open(f)) - 1 for f in slice_files)
    print(f"\nAll-sky file stars: {all_sky_count:,}")
    print(f"Sum of slice files: {slice_total:,}")
    print(f"Match: {all_sky_count == slice_total}")

# Check if we have the expected 144,000 stars
print(f"\nExpected stars: 144,000")
print(f"Actual total: {total_stars:,}")
if abs(total_stars - 144000) < 1000:
    print("✓ Close match to expected 144,000 stars!")
elif all_sky_file:
    all_sky_count = sum(1 for line in open(all_sky_file)) - 1
    if abs(all_sky_count - 144000) < 1000:
        print(f"✓ All-sky file has ~144,000 stars!")
