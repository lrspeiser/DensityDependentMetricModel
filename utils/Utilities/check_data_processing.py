#!/usr/bin/env python3
"""
check_data_processing.py - Debug Gaia data processing
"""
import numpy as np
import pandas as pd
from pathlib import Path

print("DEBUGGING GAIA DATA PROCESSING")
print("="*50)

# Check if processed cache exists and examine it
cache_file = "gaia_query_cache_DR3_processed_for_fit.parquet"
if Path(cache_file).exists():
    print(f"Found processed cache: {cache_file}")
    df = pd.read_parquet(cache_file)
    print(f"Cache shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'R_kpc' in df.columns:
        print(f"\nR_kpc statistics:")
        print(f"  Min: {df['R_kpc'].min():.6f}")
        print(f"  Max: {df['R_kpc'].max():.6f}")
        print(f"  Mean: {df['R_kpc'].mean():.6f}")
        print(f"  Std: {df['R_kpc'].std():.6f}")
        print(f"  Unique values: {df['R_kpc'].nunique()}")
        
        # Check if all values are exactly the same
        if df['R_kpc'].nunique() == 1:
            print(f"  ❌ ALL R VALUES ARE IDENTICAL: {df['R_kpc'].iloc[0]}")
            print(f"  This indicates a serious data processing error!")
        else:
            print(f"  ✅ R values span a range")
            
        # Show distribution
        print(f"\nR_kpc value distribution (first 20 unique values):")
        value_counts = df['R_kpc'].value_counts().head(20)
        for val, count in value_counts.items():
            print(f"  R = {val:.6f}: {count} stars")
            
    if 'v_obs' in df.columns:
        print(f"\nv_obs statistics:")
        print(f"  Min: {df['v_obs'].min():.1f} km/s")
        print(f"  Max: {df['v_obs'].max():.1f} km/s")
        print(f"  Mean: {df['v_obs'].mean():.1f} km/s")
        print(f"  Std: {df['v_obs'].std():.1f} km/s")
        
    if 'sigma_v' in df.columns:
        print(f"\nsigma_v statistics:")
        print(f"  Min: {df['sigma_v'].min():.1f} km/s")
        print(f"  Max: {df['sigma_v'].max():.1f} km/s")
        print(f"  Mean: {df['sigma_v'].mean():.1f} km/s")
        print(f"  Std: {df['sigma_v'].std():.1f} km/s")

# Check raw cache too
raw_cache_file = "gaia_query_cache_DR3_raw.csv"
if Path(raw_cache_file).exists():
    print(f"\n" + "="*50)
    print(f"Found raw cache: {raw_cache_file}")
    try:
        df_raw = pd.read_csv(raw_cache_file)
        print(f"Raw cache shape: {df_raw.shape}")
        
        if 'ra' in df_raw.columns and 'dec' in df_raw.columns:
            print(f"\nRaw coordinates:")
            print(f"  RA range: {df_raw['ra'].min():.3f} - {df_raw['ra'].max():.3f} deg")
            print(f"  Dec range: {df_raw['dec'].min():.3f} - {df_raw['dec'].max():.3f} deg")
            
        if 'parallax' in df_raw.columns:
            distances_pc = 1000 / df_raw['parallax']  # Convert mas to pc
            distances_kpc = distances_pc / 1000
            print(f"  Distance range: {distances_kpc.min():.2f} - {distances_kpc.max():.2f} kpc")
            print(f"  This should give R range of several kpc!")
            
    except Exception as e:
        print(f"Error reading raw cache: {e}")

print(f"\n" + "="*50)
print("DIAGNOSIS:")

# Check if the issue is in coordinate transformation
if Path(cache_file).exists():
    df = pd.read_parquet(cache_file)
    if 'R_kpc' in df.columns and df['R_kpc'].nunique() == 1:
        print("❌ COORDINATE TRANSFORMATION FAILED")
        print("All stars have identical galactocentric radius.")
        print("This suggests:")
        print("1. Bug in astropy coordinate transformation")
        print("2. All stars accidentally at same distance/position")
        print("3. Rounding error collapsing R values")
        print("\nFIX: Delete processed cache and regenerate with debugging")
    else:
        print("⚠️  Data processing seems OK, but check ranges")

print(f"\nRECOMMENDED ACTIONS:")
print(f"1. Delete corrupted cache: rm {cache_file}")
print(f"2. Run fresh data processing with debugging")
print(f"3. Check coordinate transformation in data_io.py")