#!/usr/bin/env python3
"""
get_fresh_data.py - Robust script to fetch fresh Gaia DR3 kinematic data

- Uses enhanced spatial sampling and quality cuts
- Automatically retries on Gaia query failures
- Saves to cache: gaia_query_cache_DR3_processed_for_fit.parquet
- Prints summary statistics and radial coverage
"""

import time
from data_io import load_gaia
import numpy as np

MAX_ATTEMPTS = 5
SAMPLE_MAX = 300_000

print("📡 Requesting fresh Gaia DR3 data with enhanced quality filters...")
print(f"Target sample size: {SAMPLE_MAX:,} stars")

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n🔁 Attempt {attempt} of {MAX_ATTEMPTS}...")

    try:
        gaia_data = load_gaia(
            sample_max=SAMPLE_MAX,
            force_new_query_gaia=True,
            force_reprocess_raw=True,
            use_enhanced_query=True,
            validate_data=True
        )

        if gaia_data and len(gaia_data["R_kpc"]) >= 100_000:
            print(f"\n✅ Successfully loaded {len(gaia_data['R_kpc']):,} stars")
            print(f"   R range: {gaia_data['R_kpc'].min():.1f} – {gaia_data['R_kpc'].max():.1f} kpc")
            print(f"   ⟨v⟩ = {gaia_data['v_obs'].mean():.1f} ± {gaia_data['v_obs'].std():.1f} km/s")

            # Optional: Show radial distribution
            R_bins = [0, 5, 8, 10, 15, 20, 30]
            print("\n📊 Radial coverage:")
            for i in range(len(R_bins) - 1):
                mask = (gaia_data["R_kpc"] >= R_bins[i]) & (gaia_data["R_kpc"] < R_bins[i+1])
                print(f"   [{R_bins[i]:2d}, {R_bins[i+1]:2d}) kpc: {mask.sum():6d} stars")

            print("\n📁 Cached in: gaia_query_cache_DR3_processed_for_fit.parquet")
            break  # SUCCESS

        else:
            print("⚠️  Result invalid or too few stars. Retrying...")

    except Exception as e:
        print(f"❌ Error: {e}")
        if attempt < MAX_ATTEMPTS:
            print("   Waiting 20 seconds before retry...")
            time.sleep(20)
        else:
            print("❌ Final attempt failed. Check network or Gaia TAP availability.")

print("\n🏁 Done.")
