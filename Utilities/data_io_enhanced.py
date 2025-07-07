#!/usr/bin/env python3
"""
data_io_enhanced.py - Enhanced version with proper radial coverage validation
Add these functions to your existing data_io.py
"""

def validate_data_distribution(df_processed, min_std_r=2.0, min_bins_occupied=5):
    """
    Validate that processed data has sufficient radial distribution.
    
    Parameters:
    -----------
    df_processed : pandas.DataFrame
        Processed data with R_kpc column
    min_std_r : float
        Minimum standard deviation in R (kpc)
    min_bins_occupied : int
        Minimum number of radial bins that should have data
    
    Returns:
    --------
    bool : True if data passes validation
    """
    if 'R_kpc' not in df_processed.columns:
        print("❌ No R_kpc column found!")
        return False
    
    R_values = df_processed['R_kpc'].values
    
    # Check 1: Standard deviation
    r_std = np.std(R_values)
    print(f"\n📊 Data Distribution Validation:")
    print(f"   R range: {R_values.min():.1f} - {R_values.max():.1f} kpc")
    print(f"   R std dev: {r_std:.2f} kpc", end="")
    
    if r_std < min_std_r:
        print(f" ❌ Too concentrated! (need > {min_std_r} kpc)")
        return False
    else:
        print(" ✓")
    
    # Check 2: Bin occupancy
    r_bins = np.arange(3, 21, 2)  # 3, 5, 7, ..., 19 kpc
    counts, _ = np.histogram(R_values, bins=r_bins)
    occupied_bins = np.sum(counts > 10)  # At least 10 stars per bin
    
    print(f"   Radial bins with data: {occupied_bins}/{len(r_bins)-1}", end="")
    if occupied_bins < min_bins_occupied:
        print(f" ❌ Too few bins! (need >= {min_bins_occupied})")
        return False
    else:
        print(" ✓")
    
    # Show distribution
    print("   Distribution by radius:")
    for i in range(len(r_bins)-1):
        if counts[i] > 0:
            print(f"     R = {r_bins[i]:2.0f}-{r_bins[i+1]:2.0f} kpc: {counts[i]:4d} stars")
    
    return True


def perform_gaia_adql_query_multiregion(limit_val=100000, 
                                       cache_raw_filename="gaia_query_cache_DR3_raw.csv",
                                       force_live_query=False):
    """
    Enhanced query that samples multiple sky regions for better radial coverage.
    """
    if not HAS_ASTROPY_AND_QUERY:
        print("❌ Astroquery/Astropy not available.")
        return None
    
    raw_cache_path = Path(cache_raw_filename)
    
    # Check cache first
    if USE_LOCAL_CACHE and not force_live_query and raw_cache_path.exists():
        print(f"💾 Loading RAW Gaia query results from CSV cache: {raw_cache_path}")
        try:
            df_raw_cached = pd.read_csv(raw_cache_path)
            if not df_raw_cached.empty:
                print(f"   → Successfully loaded {len(df_raw_cached):,} raw star records from CSV cache.")
                return df_raw_cached
        except Exception as e:
            print(f"   ⚠️ Error loading cache: {e}")
    
    print(f"\n📡 Performing MULTI-REGION Gaia DR3 Query (target: {limit_val:,} stars total)...")
    
    # Query multiple regions to ensure radial coverage
    # Each region targets different distances based on sky position
    regions = [
        # (l_min, l_max, b_min, b_max, parallax_min, parallax_max, fraction)
        (30, 90, -5, 5, 0.05, 2.0, 0.3),      # Anti-center, distant stars
        (90, 150, -5, 5, 0.1, 5.0, 0.2),      # Perpendicular direction
        (210, 270, -5, 5, 0.2, 10.0, 0.3),    # Opposite perpendicular
        (270, 330, -5, 5, 0.1, 5.0, 0.2),     # Toward center (avoiding bulge)
    ]
    
    all_dfs = []
    
    for l_min, l_max, b_min, b_max, plx_min, plx_max, frac in regions:
        n_region = int(limit_val * frac)
        print(f"\n   Region l=[{l_min},{l_max}]°, parallax=[{plx_min},{plx_max}] mas...")
        
        query_adql = f"""
        SELECT TOP {n_region}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmra_error, pmdec, pmdec_error,
            radial_velocity, radial_velocity_error,
            ruwe, phot_g_mean_mag, b, l
        FROM gaiadr3.gaia_source
        WHERE l BETWEEN {l_min} AND {l_max}
          AND b BETWEEN {b_min} AND {b_max}
          AND parallax BETWEEN {plx_min} AND {plx_max}
          AND parallax_over_error > 5
          AND pmra IS NOT NULL AND pmdec IS NOT NULL
          AND pmra_error < 1.0 AND pmdec_error < 1.0
          AND radial_velocity IS NOT NULL
          AND radial_velocity_error < 20
          AND ruwe < 1.4
          AND phot_g_mean_mag < 17
        ORDER BY random_index
        """
        
        try:
            job = Gaia.launch_job_async(query_adql)
            tbl_results = job.get_results()
            df_region = tbl_results.to_pandas()
            print(f"      → Got {len(df_region)} stars")
            all_dfs.append(df_region)
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            continue
    
    if not all_dfs:
        print("❌ No successful queries!")
        return None
    
    # Combine all regions
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n   → Total stars from all regions: {len(df_combined):,}")
    
    # Save to cache
    if USE_LOCAL_CACHE:
        print(f"💾 Saving RAW Gaia query results to CSV cache: {raw_cache_path}")
        try:
            df_combined.to_csv(raw_cache_path, index=False)
        except Exception as e:
            print(f"   ⚠️ Error saving: {e}")
    
    return df_combined


def load_gaia_validated(sample_max=100_000,
                       force_new_query_gaia=False,
                       force_reprocess_raw=False,
                       raw_cache_filename="gaia_query_cache_DR3_raw.csv",
                       processed_cache_filename=None,
                       require_validation=True):
    """
    Enhanced load_gaia that validates radial distribution.
    """
    # Handle default value
    if processed_cache_filename is None:
        processed_cache_filename = PROCESSED_GAIA_CACHE_FILENAME_DEFAULT
    
    if not HAS_ASTROPY_AND_QUERY:
        print("❌ Astroquery/Astropy not available.")
        return None
    
    processed_cache_path = Path(processed_cache_filename)
    df_processed_output = None
    
    # Try to load processed data
    if USE_LOCAL_CACHE and not force_reprocess_raw and processed_cache_path.exists():
        print(f"💾 Loading PROCESSED Gaia data from Parquet cache: {processed_cache_path}")
        try:
            df_cached = pd.read_parquet(processed_cache_path)
            if not df_cached.empty:
                print(f"   → Successfully loaded {len(df_cached):,} processed stars from Parquet cache.")
                
                # Validate if required
                if require_validation:
                    if validate_data_distribution(df_cached):
                        df_processed_output = df_cached
                    else:
                        print("   ⚠️ Cached data failed validation! Will re-query with better coverage.")
                        force_new_query_gaia = True
                else:
                    df_processed_output = df_cached
        except Exception as e:
            print(f"   ⚠️ Error loading cache: {e}")
    
    # If no valid processed data, get raw data and process
    if df_processed_output is None:
        print("\n-- Attempting to obtain or generate processed data --")
        
        # Use multi-region query for better coverage
        df_raw_gaia = perform_gaia_adql_query_multiregion(
            limit_val=sample_max * 2,  # Query more to allow filtering
            cache_raw_filename=raw_cache_filename,
            force_live_query=force_new_query_gaia
        )
        
        if df_raw_gaia is not None and not df_raw_gaia.empty:
            df_processed_output = process_raw_gaia_df(df_raw_gaia)
            
            # Validate the processed data
            if require_validation and df_processed_output is not None:
                if not validate_data_distribution(df_processed_output):
                    print("\n⚠️  WARNING: Even after multi-region query, radial distribution is poor.")
                    print("   Consider using synthetic data for testing.")
            
            # Save if successful
            if df_processed_output is not None and not df_processed_output.empty and USE_LOCAL_CACHE:
                print(f"💾 Saving newly PROCESSED Gaia data to Parquet cache: {processed_cache_path}")
                try:
                    df_processed_output.to_parquet(processed_cache_path, index=False)
                except Exception as e:
                    print(f"   ⚠️ Error saving: {e}")
        else:
            print("❌ No raw Gaia data obtained")
            return None
    
    if df_processed_output is None or df_processed_output.empty:
        print("❌ No Gaia data was loaded or processed.")
        return None
    
    # Ensure z_kpc exists
    if 'z_kpc' not in df_processed_output.columns:
        df_processed_output['z_kpc'] = 0.0
    
    # Apply sampling if requested
    if sample_max and len(df_processed_output) > sample_max:
        # Stratified sampling to maintain radial distribution
        try:
            # Create radial bins
            r_bins = np.percentile(df_processed_output['R_kpc'], np.linspace(0, 100, 11))
            r_bins[0] -= 0.1  # Ensure all data included
            r_bins[-1] += 0.1
            
            # Sample from each bin proportionally
            df_sampled = df_processed_output.groupby(
                pd.cut(df_processed_output['R_kpc'], bins=r_bins)
            ).apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * sample_max / len(df_processed_output))),
                    random_state=42
                )
            ).reset_index(drop=True)
            
            df_processed_output = df_sampled
            print(f"   → Stratified sampling to {len(df_processed_output)} stars")
        except:
            # Fallback to simple sampling
            df_processed_output = df_processed_output.sample(n=sample_max, random_state=42)
            print(f"   → Random sampling to {sample_max} stars")
    
    return_dict = {
        "R_kpc": df_processed_output["R_kpc"].values,
        "v_obs": df_processed_output["v_obs"].values,
        "sigma_v": df_processed_output["sigma_v"].values,
        "z_kpc": df_processed_output["z_kpc"].values
    }
    if 'source_id' in df_processed_output.columns:
        return_dict["source_id"] = df_processed_output["source_id"].values
    
    return return_dict


# Make load_gaia an alias for the validated version
load_gaia = load_gaia_validated