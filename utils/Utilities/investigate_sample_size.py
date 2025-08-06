#!/usr/bin/env python3
"""
investigate_sample_size.py - Figure out why we only got 570 stars
"""
from data_io import load_gaia
import numpy as np

print("INVESTIGATING SAMPLE SIZE REDUCTION")
print("="*50)

# Test different sample_max values
sample_sizes = [1000, 5000, 10000, 20000, 50000, 80000]

for sample_max in sample_sizes:
    print(f"\nTesting sample_max = {sample_max:,}")
    try:
        data = load_gaia(sample_max=sample_max)
        if data:
            n_loaded = len(data['R_kpc'])
            r_min, r_max = np.min(data['R_kpc']), np.max(data['R_kpc'])
            v_min, v_max = np.min(data['v_obs']), np.max(data['v_obs'])
            print(f"  → Loaded: {n_loaded:,} stars")
            print(f"  → R range: {r_min:.2f} - {r_max:.2f} kpc") 
            print(f"  → v range: {v_min:.1f} - {v_max:.1f} km/s")
            
            if n_loaded < sample_max * 0.8:
                print(f"  ⚠️  Got {n_loaded:,} << {sample_max:,} requested")
        else:
            print("  ❌ Failed to load data")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Check the actual cache file
print(f"\n" + "="*50)
print("CHECKING CACHE FILE DIRECTLY")
try:
    import pandas as pd
    df = pd.read_parquet("gaia_query_cache_DR3_processed_for_fit.parquet")
    print(f"Cache contains: {len(df):,} total stars")
    print(f"R range in cache: {df['R_kpc'].min():.2f} - {df['R_kpc'].max():.2f} kpc")
    
    # Check for any filtering that might explain the reduction
    print(f"\nData quality checks:")
    n_finite_R = np.sum(np.isfinite(df['R_kpc']))
    n_finite_v = np.sum(np.isfinite(df['v_obs']))
    n_finite_sigma = np.sum(np.isfinite(df['sigma_v']))
    n_positive_R = np.sum(df['R_kpc'] > 0)
    n_reasonable_v = np.sum((df['v_obs'] > 0) & (df['v_obs'] < 1000))
    n_reasonable_sigma = np.sum((df['sigma_v'] > 0) & (df['sigma_v'] < 100))
    
    print(f"  Finite R: {n_finite_R:,} / {len(df):,}")
    print(f"  Finite v: {n_finite_v:,} / {len(df):,}")
    print(f"  Finite σ: {n_finite_sigma:,} / {len(df):,}")
    print(f"  R > 0: {n_positive_R:,} / {len(df):,}")
    print(f"  0 < v < 1000: {n_reasonable_v:,} / {len(df):,}")
    print(f"  0 < σ < 100: {n_reasonable_sigma:,} / {len(df):,}")
    
    # Combined quality mask
    quality_mask = (np.isfinite(df['R_kpc']) & 
                   np.isfinite(df['v_obs']) & 
                   np.isfinite(df['sigma_v']) &
                   (df['R_kpc'] > 0) &
                   (df['v_obs'] > 0) & (df['v_obs'] < 1000) &
                   (df['sigma_v'] > 0) & (df['sigma_v'] < 100))
    
    n_quality = np.sum(quality_mask)
    print(f"  All quality checks: {n_quality:,} / {len(df):,}")
    
    if n_quality < len(df) * 0.8:
        print(f"  ⚠️  Many stars fail quality checks!")
        
except Exception as e:
    print(f"Error reading cache: {e}")

print(f"\n" + "="*50)
print("POTENTIAL CAUSES:")
print("1. Data quality filtering in load_gaia() function")
print("2. Memory limitations during processing")
print("3. Hidden sampling in the coordinate transformation")
print("4. Bug in the sample_max parameter handling")
print("5. Very strict cuts removing most stars")