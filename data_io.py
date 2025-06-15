#!/usr/bin/env python3
"""
data_io.py - Gaia DR3 data loading and caching utilities.
Implements two-stage caching:
1. Raw Gaia query results (CSV) - to avoid re-querying Gaia.
2. Processed kinematic data (Parquet) - to avoid re-processing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CylindricalDifferential
    from astroquery.gaia import Gaia
    HAS_ASTROPY_AND_QUERY = True
except ImportError:
    HAS_ASTROPY_AND_QUERY = False
    print("⚠️  Critical libraries (astropy/astroquery) not found in data_io.py. Please install them:")
    print("   pip install astropy astroquery numpy pandas")

# Cache Control Configuration
USE_LOCAL_CACHE = True
# RAW_GAIA_CACHE_FILENAME_DEFAULT = "gaia_query_cache_DR3_raw.csv" # Default for raw query
PROCESSED_GAIA_CACHE_FILENAME_DEFAULT = "gaia_query_cache_DR3_processed_for_fit.parquet" # Default for processed

# Galactocentric frame parameters for Milky Way
R0_KPC_ASTRO = 8.122 * u.kpc
ZSUN_KPC_ASTRO = 0.025 * u.kpc
VSUN_KMS_ASTRO = CartesianDifferential([11.1, 245.6, 7.25] * u.km/u.s)

def perform_gaia_adql_query(limit_val=100000, cache_raw_filename="gaia_query_cache_DR3_raw.csv", force_live_query=False):
    """
    Performs the Gaia ADQL query, saving/loading raw results from a CSV cache.
    Returns a pandas DataFrame with raw query results.
    """
    if not HAS_ASTROPY_AND_QUERY:
        print("❌ Astroquery/Astropy not available in perform_gaia_adql_query. Cannot fetch live Gaia data.")
        return None

    raw_cache_path = Path(cache_raw_filename)

    if USE_LOCAL_CACHE and not force_live_query and raw_cache_path.exists():
        print(f"💾 Loading RAW Gaia query results from CSV cache: {raw_cache_path}")
        try:
            df_raw_cached = pd.read_csv(raw_cache_path)
            if not df_raw_cached.empty:
                print(f"   → Successfully loaded {len(df_raw_cached):,} raw star records from CSV cache.")
                return df_raw_cached
            else:
                print(f"   ⚠️ Raw CSV cache file {raw_cache_path} is empty. Attempting live query.")
        except Exception as e_raw_cache:
            print(f"   ⚠️ Error loading raw CSV cache {raw_cache_path}: {e_raw_cache}. Attempting live query.")

    print(f"\n📡  Performing LIVE Gaia DR3 ADQL Query (limit {limit_val:,})...")
    # ... (ADQL query string remains the same) ...
    query_adql = f"""
    SELECT TOP {limit_val}
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        radial_velocity, radial_velocity_error,
        ruwe, phot_g_mean_mag, b, l
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
      AND parallax > 0.05
      AND parallax_over_error > 5
      AND pmra IS NOT NULL AND pmdec IS NOT NULL
      AND pmra_error IS NOT NULL AND pmdec_error IS NOT NULL
      AND pmra_error < 0.5 AND pmdec_error < 0.5
      AND radial_velocity IS NOT NULL
      AND radial_velocity_error IS NOT NULL AND radial_velocity_error < 20
      AND ruwe < 1.4
      AND phot_g_mean_mag < 18
      AND ABS(b) < 30
    ORDER BY random_index
    """
    try:
        job = Gaia.launch_job_async(query_adql)
        tbl_results = job.get_results()
        df_raw_live = tbl_results.to_pandas()
        print(f"   → Gaia ADQL query successful, {len(df_raw_live):,} raw stars returned.")
        if not df_raw_live.empty and USE_LOCAL_CACHE:
            print(f"💾 Saving RAW Gaia query results to CSV cache: {raw_cache_path}")
            try:
                df_raw_live.to_csv(raw_cache_path, index=False)
            except Exception as e_save_raw:
                print(f"   ⚠️ Error saving raw Gaia data to CSV cache {raw_cache_path}: {e_save_raw}")
        return df_raw_live
    except Exception as e_query:
        print(f"❌ Gaia ADQL query failed: {e_query}"); return None


def process_raw_gaia_df(df_raw):
    """
    Processes a raw Gaia DataFrame (from query) into kinematic quantities.
    Returns a processed pandas DataFrame.
    """
    if df_raw is None or df_raw.empty:
        print("   No raw Gaia data to process.")
        return pd.DataFrame()

    print("\n🌠 Processing raw Gaia data: Sky → 6‑D Galactocentric coordinates and velocities...")
    gc_frame = Galactocentric(galcen_distance=R0_KPC_ASTRO,
                              z_sun=ZSUN_KPC_ASTRO,
                              galcen_v_sun=VSUN_KMS_ASTRO)
    try:
        coords_icrs = SkyCoord(ra=df_raw['ra'].values*u.deg,
                               dec=df_raw['dec'].values*u.deg,
                               distance=(df_raw['parallax'].values*u.mas).to(u.pc, equivalencies=u.parallax()),
                               pm_ra_cosdec=df_raw['pmra'].values*u.mas/u.yr,
                               pm_dec=df_raw['pmdec'].values*u.mas/u.yr,
                               radial_velocity=df_raw['radial_velocity'].values*u.km/u.s,
                               frame='icrs')
    except Exception as e_skycoord:
        print(f"❌ Error creating SkyCoord object: {e_skycoord}")
        return pd.DataFrame()

    coords_gc = coords_icrs.transform_to(gc_frame)

    df_processed = pd.DataFrame()
    if 'source_id' in df_raw.columns:
        df_processed['source_id'] = df_raw['source_id']

    df_processed['R_kpc'] = coords_gc.cylindrical.rho.to(u.kpc).value
    df_processed['z_kpc'] = coords_gc.z.to(u.kpc).value

    # Corrected velocity calculation
    cyl_vel_diff = coords_gc.velocity.represent_as(CylindricalDifferential, coords_gc.data)
    v_phi_kms = (coords_gc.cylindrical.rho * cyl_vel_diff.d_phi).to(u.km/u.s, equivalencies=u.dimensionless_angles()).value
    df_processed['v_obs'] = np.abs(v_phi_kms)

    # Error propagation for v_obs
    distance_kpc = coords_icrs.distance.to(u.kpc).value
    rv_error_kms = df_raw['radial_velocity_error'].values
    pmra_error_masyr = df_raw['pmra_error'].values
    pmdec_error_masyr = df_raw['pmdec_error'].values

    rv_error_kms = np.nan_to_num(rv_error_kms)
    pmra_error_masyr = np.nan_to_num(pmra_error_masyr)
    pmdec_error_masyr = np.nan_to_num(pmdec_error_masyr)
    distance_kpc = np.nan_to_num(distance_kpc)

    pm_tot_error_masyr = np.hypot(pmra_error_masyr, pmdec_error_masyr)
    v_tan_error_kms = 4.74047 * distance_kpc * pm_tot_error_masyr

    v_err_combined = np.hypot(rv_error_kms, v_tan_error_kms)

    v_err_combined[v_err_combined < 5.0] = 5.0 # Floor
    v_err_combined[~np.isfinite(v_err_combined) | (v_err_combined > 500.0)] = 50.0 # Sensible fallback and cap

    df_processed['sigma_v'] = v_err_combined

    df_processed = df_processed[
        np.isfinite(df_processed['R_kpc']) & (df_processed['R_kpc'] > 0.01) &
        np.isfinite(df_processed['v_obs']) & (df_processed['v_obs'] < 700) &
        np.isfinite(df_processed['sigma_v']) & (df_processed['sigma_v'] < 100)
    ].copy()

    print(f"   → Successfully processed {len(df_processed)} stars with valid kinematics and errors.")
    return df_processed


def load_gaia(sample_max=100_000, force_new_query_gaia=False, force_reprocess_raw=False,
              raw_cache_filename="gaia_query_cache_DR3_raw.csv", # Default raw cache name
              processed_cache_filename=PROCESSED_GAIA_CACHE_FILENAME_DEFAULT): # Default processed cache name
    """
    Loads Gaia data, using a two-stage caching system.
    """
    if not HAS_ASTROPY_AND_QUERY:
        print("❌ data_io.py: Astroquery/Astropy not available. Cannot load Gaia data.")
        return None

    processed_cache_path = Path(processed_cache_filename)
    df_processed_output = None

    # Stage 1: Try to load already PROCESSED data from Parquet cache
    if USE_LOCAL_CACHE and not force_reprocess_raw and processed_cache_path.exists():
        print(f"💾 Loading PROCESSED Gaia data from Parquet cache: {processed_cache_path}")
        try:
            df_cached_processed = pd.read_parquet(processed_cache_path)
            if not df_cached_processed.empty and all(col in df_cached_processed.columns for col in ['R_kpc', 'v_obs', 'sigma_v', 'z_kpc']):
                print(f"   → Successfully loaded {len(df_cached_processed):,} processed stars from Parquet cache.")
                df_processed_output = df_cached_processed
            else:
                print(f"   ⚠️ Processed Parquet cache {processed_cache_path} seems invalid. Will attempt to regenerate from raw.")
        except Exception as e_processed_cache:
            print(f"   ⚠️ Error loading processed Parquet cache {processed_cache_path}: {e_processed_cache}. Will attempt to regenerate from raw.")

    # Stage 2: If processed data not loaded, get RAW data (from its cache or live query) and then process it
    if df_processed_output is None:
        print("\n-- Attempting to obtain or generate processed data --")
        # Get raw Gaia data (this function handles its own CSV caching or live query)
        df_raw_gaia = perform_gaia_adql_query(limit_val=sample_max,
                                              cache_raw_filename=raw_cache_filename,
                                              force_live_query=force_new_query_gaia)

        if df_raw_gaia is not None and not df_raw_gaia.empty:
            # Process the (now available) raw data
            df_processed_output = process_raw_gaia_df(df_raw_gaia)

            # Save the newly processed data to Parquet cache if successful and caching is enabled
            if df_processed_output is not None and not df_processed_output.empty and USE_LOCAL_CACHE:
                print(f"💾 Saving newly PROCESSED Gaia data to Parquet cache: {processed_cache_path}")
                try:
                    df_processed_output.to_parquet(processed_cache_path, index=False)
                except Exception as e_save_processed:
                    print(f"   ⚠️ Error saving processed data to Parquet cache {processed_cache_path}: {e_save_processed}")
        else:
            print("❌ No raw Gaia data obtained (from cache or live query), so cannot produce processed data.")
            return None

    # Final check and return
    if df_processed_output is None or df_processed_output.empty:
        print("❌ Ultimately, no Gaia data was loaded or processed.")
        return None

    if 'z_kpc' not in df_processed_output.columns: # Should be there from process_raw_gaia_df
        df_processed_output['z_kpc'] = 0.0

    return_dict = {
        "R_kpc": df_processed_output["R_kpc"].values,
        "v_obs": df_processed_output["v_obs"].values,
        "sigma_v": df_processed_output["sigma_v"].values,
        "z_kpc": df_processed_output["z_kpc"].values
    }
    if 'source_id' in df_processed_output.columns:
         return_dict["source_id"] = df_processed_output["source_id"].values

    return return_dict

if __name__ == '__main__':
    print("Testing data_io.py with two-stage caching...")
    raw_csv_name = "test_gaia_raw.csv"
    processed_parquet_name = "test_gaia_processed.parquet"

    print("\n--- Test Run 1: Force new query & reprocessing ---")
    gaia_data = load_gaia(sample_max=500, force_new_query_gaia=True, force_reprocess_raw=True,
                          raw_cache_filename=raw_csv_name, processed_cache_filename=processed_parquet_name)
    if gaia_data and gaia_data["R_kpc"] is not None :
        print(f"Run 1: Loaded data for {len(gaia_data['R_kpc'])} stars.")
    else: print("Run 1: Failed to load data.")

    print("\n--- Test Run 2: Use caches if available ---")
    gaia_data_cached = load_gaia(sample_max=500, force_new_query_gaia=False, force_reprocess_raw=False,
                                 raw_cache_filename=raw_csv_name, processed_cache_filename=processed_parquet_name)
    if gaia_data_cached and gaia_data_cached["R_kpc"] is not None :
        print(f"Run 2: Loaded data for {len(gaia_data_cached['R_kpc'])} stars.")
        if gaia_data and len(gaia_data["R_kpc"]) == len(gaia_data_cached["R_kpc"]):
            print("   Data length matches Run 1, caches likely used.")
    else: print("Run 2: Failed to load data from cache.")

    # Clean up test cache files
    # if Path(raw_csv_name).exists(): Path(raw_csv_name).unlink()
    # if Path(processed_parquet_name).exists(): Path(processed_parquet_name).unlink()
    # print("\nCleaned up test cache files (if they existed).")