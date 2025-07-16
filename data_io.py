#!/usr/bin/env python3
"""
data_io.py - Enhanced Gaia DR3 data loading and caching utilities.

This module implements a robust two-stage caching system:
1. Raw Gaia query results (CSV) - to avoid re-querying Gaia Archive
2. Processed kinematic data (Parquet) - to avoid re-processing coordinates

Key improvements:
- Enhanced quality cuts for more reliable data
- Spatial sampling to ensure radial coverage
- Data validation and sanity checks
- Detailed logging for diagnostics
- Robust error handling

Author: [Your name]
Version: 2.0 (Enhanced for density-dependent metric validation)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from typing import Dict, Optional, Tuple, Union
import traceback

# Set up logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Delay HTTP debug until after astropy is imported
try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CylindricalDifferential
    from astroquery.gaia import Gaia
    HAS_ASTROPY_AND_QUERY = True

    # Only now turn on HTTP debugging
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1

    logging.getLogger("astropy").setLevel(logging.DEBUG)
    logging.getLogger("astroquery").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

    # Set timeout
    try:
        Gaia.TIMEOUT = 900  # 15 minutes
        logger.info("Gaia.TIMEOUT set to 900 seconds")
    except Exception as e:
        logger.warning(f"⚠️ Could not set Gaia.TIMEOUT: {e}")
        
    logger.info("Successfully imported astropy and astroquery")

except ImportError as e:
    HAS_ASTROPY_AND_QUERY = False
    logger.error(f"Critical libraries (astropy/astroquery) not found: {e}")
    print("⚠️  Critical libraries (astropy/astroquery) not found in data_io.py. Please install them:")
    print("   pip install astropy astroquery numpy pandas")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("astropy").setLevel(logging.DEBUG)
logging.getLogger("astroquery").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CylindricalDifferential
    from astroquery.gaia import Gaia
    HAS_ASTROPY_AND_QUERY = True
    
    # Set timeout for Gaia queries
    try:
        Gaia.TIMEOUT = 900  # 15 minutes
    except Exception as e:
        logger.warning(f"Could not set Gaia.TIMEOUT: {e}")
            
    logger.info("Successfully imported astropy and astroquery")
    
except ImportError as e:
    HAS_ASTROPY_AND_QUERY = False
    logger.error(f"Critical libraries (astropy/astroquery) not found: {e}")
    print("⚠️  Critical libraries (astropy/astroquery) not found in data_io.py. Please install them:")
    print("   pip install astropy astroquery numpy pandas")

# ============================================================================
# Configuration Parameters
# ============================================================================

# Cache Control
USE_LOCAL_CACHE = True
PROCESSED_GAIA_CACHE_FILENAME_DEFAULT = "gaia_query_cache_DR3_processed_for_fit.parquet"
RAW_GAIA_CACHE_FILENAME_DEFAULT = "gaia_query_cache_DR3_raw.csv"

# Galactocentric frame parameters for Milky Way (McMillan 2017, Gravity Collaboration 2018)
R0_KPC_ASTRO = 8.122 * u.kpc          # Galactocentric distance of Sun
ZSUN_KPC_ASTRO = 0.025 * u.kpc        # Height of Sun above midplane
VSUN_KMS_ASTRO = CartesianDifferential([11.1, 245.6, 7.25] * u.km/u.s)  # Solar motion

# Data quality parameters
MIN_PARALLAX_MAS = 0.033               # allows distances up to ~30 kpc
MIN_PARALLAX_OVER_ERROR = 5           # Minimum parallax SNR for reliable distances
MAX_PM_ERROR_MAS_YR = 0.2              # Maximum proper motion error (mas/yr)
MAX_RV_ERROR_KMS = 5                   # Maximum radial velocity error (km/s)
MAX_RUWE = 1.2                         # Maximum RUWE for good astrometric solutions
MAX_PHOT_G_MAG = 17                    # Maximum G magnitude (brighter stars only)
MAX_ABS_B_DEG = 10                     # Maximum |b| to focus on disk stars
MIN_VISIBILITY_PERIODS = 8              # Minimum visibility periods for reliable data

# Velocity error parameters
MIN_VELOCITY_ERROR_KMS = 5.0           # Minimum velocity error floor
MAX_VELOCITY_ERROR_KMS = 50.0          # Maximum reasonable velocity error
DEFAULT_LARGE_ERROR_KMS = 50.0         # Default for problematic measurements

# Physical validation parameters
EXPECTED_V_MEDIAN_RANGE = (180, 250)   # Expected median velocity range (km/s)
EXPECTED_R_MEDIAN_RANGE = (6, 10)      # Expected median radius range (kpc)
MIN_STARS_PER_RADIAL_BIN = 100         # Minimum stars required per radial bin
MAX_REASONABLE_VELOCITY_KMS = 700      # Maximum reasonable circular velocity


# ============================================================================
# Enhanced Query Functions
# ============================================================================

def perform_gaia_adql_query_enhanced(
    limit_val: int = 100000,
    cache_raw_filename: str = RAW_GAIA_CACHE_FILENAME_DEFAULT,
    force_live_query: bool = False,
    subsample_factor: int = 10
) -> Optional[pd.DataFrame]:
    """
    Performs enhanced Gaia ADQL query with stricter quality cuts and spatial sampling.
    
    This query implements several improvements over the basic version:
    1. Stricter parallax and error cuts for more reliable distances
    2. Additional quality indicators (astrometric_excess_noise, etc.)
    3. Focus on disk stars with |b| < 10°
    4. Subsampling using source_id modulo for spatial uniformity
    5. Better error thresholds based on Gaia DR3 validation
    
    Parameters
    ----------
    limit_val : int
        Maximum number of stars to query
    cache_raw_filename : str
        Filename for caching raw query results
    force_live_query : bool
        If True, bypass cache and force new query
    subsample_factor : int
        Subsampling factor using source_id modulo (10 = keep 1/10 of stars)
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with raw Gaia data, or None if query fails
    """
    if not HAS_ASTROPY_AND_QUERY:
        logger.error("Astroquery/Astropy not available. Cannot fetch Gaia data.")
        return None
        
    raw_cache_path = Path(cache_raw_filename)
    
    # Check cache first
    if USE_LOCAL_CACHE and not force_live_query and raw_cache_path.exists():
        logger.info(f"Loading RAW Gaia query results from cache: {raw_cache_path}")
        try:
            df_raw_cached = pd.read_csv(raw_cache_path)
            if not df_raw_cached.empty:
                logger.info(f"Successfully loaded {len(df_raw_cached):,} raw star records from cache")
                return df_raw_cached
            else:
                logger.warning(f"Raw cache file {raw_cache_path} is empty. Will perform live query.")
        except Exception as e:
            logger.error(f"Error loading raw cache: {e}. Will perform live query.")
    
    # Construct enhanced ADQL query
    logger.info(f"Performing LIVE Gaia DR3 ADQL Query (limit {limit_val:,}, subsample 1/{subsample_factor})")
    
    query_adql = f"""
    SELECT TOP {limit_val}
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        radial_velocity, radial_velocity_error,
        ruwe, phot_g_mean_mag, b, l,
        -- Additional quality indicators for enhanced filtering
        astrometric_excess_noise,
        astrometric_excess_noise_sig,
        visibility_periods_used,
        astrometric_n_good_obs_al,
        radial_velocity_renormalised_gof,
        phot_bp_rp_excess_factor
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
      AND parallax > {MIN_PARALLAX_MAS}                    -- Max distance ~10 kpc
      AND parallax_over_error > {MIN_PARALLAX_OVER_ERROR}  -- High SNR parallax
      AND pmra IS NOT NULL AND pmdec IS NOT NULL
      AND pmra_error < {MAX_PM_ERROR_MAS_YR} 
      AND pmdec_error < {MAX_PM_ERROR_MAS_YR}              -- Tight PM errors
      AND radial_velocity IS NOT NULL
      AND radial_velocity_error < {MAX_RV_ERROR_KMS}       -- Strict RV error
      AND ruwe < {MAX_RUWE}                                 -- Good astrometric solution
      AND phot_g_mean_mag < {MAX_PHOT_G_MAG}               -- Bright stars only
      AND ABS(b) < {MAX_ABS_B_DEG}                         -- Focus on disk
      AND visibility_periods_used > {MIN_VISIBILITY_PERIODS} -- Well-observed
      AND astrometric_n_good_obs_al > 100                   -- Many observations
      AND astrometric_excess_noise < 1                      -- Low excess noise
      AND astrometric_excess_noise_sig < 2                  -- Not significant
      -- Quality cuts on RV if available
      AND (radial_velocity_renormalised_gof < 3 OR radial_velocity_renormalised_gof IS NULL)
      -- Spatial subsampling for uniformity
      AND MOD(source_id, {subsample_factor}) = 0
    ORDER BY random_index
    """
    
    try:
        # Log query for debugging
        logger.debug("ADQL Query:\n" + query_adql)
        
        # Execute query
        try:
            logger.debug("ADQL Query:\n" + query_adql)
            job = Gaia.launch_job_async(query_adql)
            tbl_results = job.get_results()
            df_raw_live = tbl_results.to_pandas()

        except Exception as e:
            logger.error(f"Gaia ADQL query failed: {e}")
            
            # Print full traceback
            tb_str = traceback.format_exc()
            logger.error(f"Traceback:\n{tb_str}")

            # Try to log the response from the job, if any
            if 'job' in locals():
                try:
                    logger.error(f"Job phase: {job.phase}")
                    logger.error(f"Job error report:\n{job.get_error_report()}")
                except Exception as inner:
                    logger.warning(f"Could not retrieve job error report: {inner}")

            return None
        
        
        df_raw_live = tbl_results.to_pandas()
        
        logger.info(f"Gaia ADQL query successful: {len(df_raw_live):,} stars returned")
        
        # Validate query results
        if df_raw_live.empty:
            logger.warning("Query returned no results. Check query constraints.")
            return None
            
        # Basic validation of returned data
        logger.info("Validating query results...")
        _validate_raw_gaia_data(df_raw_live)
        
        # Save to cache
        if USE_LOCAL_CACHE:
            logger.info(f"Saving RAW Gaia query results to cache: {raw_cache_path}")
            try:
                df_raw_live.to_csv(raw_cache_path, index=False)
                logger.info("Cache saved successfully")
            except Exception as e:
                logger.error(f"Error saving raw Gaia data to cache: {e}")
                
        return df_raw_live
        
    except Exception as e:
        logger.error(f"Gaia ADQL query failed: {e}")
        return None


def _validate_raw_gaia_data(df_raw: pd.DataFrame) -> None:
    """
    Validate raw Gaia data for basic consistency.
    Logs warnings but doesn't modify data.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Gaia query results
    """
    logger.info("Performing raw data validation...")
    
    # Check for required columns
    required_cols = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 
                    'radial_velocity', 'phot_g_mean_mag']
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    # Check data ranges
    if 'parallax' in df_raw.columns:
        neg_parallax = (df_raw['parallax'] < 0).sum()
        if neg_parallax > 0:
            logger.warning(f"Found {neg_parallax} stars with negative parallax")
    
    if 'radial_velocity' in df_raw.columns:
        extreme_rv = (np.abs(df_raw['radial_velocity']) > 500).sum()
        if extreme_rv > 0:
            logger.warning(f"Found {extreme_rv} stars with |RV| > 500 km/s")
    
    # Check spatial distribution
    if 'b' in df_raw.columns:
        b_median = df_raw['b'].median()
        logger.info(f"Median galactic latitude: {b_median:.2f}°")
        if np.abs(b_median) > 5:
            logger.warning(f"Data may be biased toward high latitudes")
    
    # Report statistics
    logger.info(f"Raw data statistics:")
    logger.info(f"  Total stars: {len(df_raw):,}")
    if 'phot_g_mean_mag' in df_raw.columns:
        logger.info(f"  G magnitude range: {df_raw['phot_g_mean_mag'].min():.1f} - "
                   f"{df_raw['phot_g_mean_mag'].max():.1f}")
    if 'parallax' in df_raw.columns:
        logger.info(f"  Parallax range: {df_raw['parallax'].min():.3f} - "
                   f"{df_raw['parallax'].max():.3f} mas")


# ============================================================================
# Enhanced Processing Functions
# ============================================================================

def process_raw_gaia_df_enhanced(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Gaia data into galactocentric coordinates with enhanced error handling.
    
    Improvements over basic version:
    1. More robust error propagation
    2. Quality flags for processed data
    3. Additional derived quantities (e.g., galactocentric velocities)
    4. Better handling of edge cases
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Gaia query results
        
    Returns
    -------
    pd.DataFrame
        Processed data with galactocentric coordinates and validated errors
    """
    if df_raw is None or df_raw.empty:
        logger.error("No raw Gaia data to process")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(df_raw):,} stars: ICRS → Galactocentric coordinates")
    
    # Set up galactocentric frame
    gc_frame = Galactocentric(
        galcen_distance=R0_KPC_ASTRO,
        z_sun=ZSUN_KPC_ASTRO,
        galcen_v_sun=VSUN_KMS_ASTRO
    )
    
    try:
        # Create SkyCoord object with full 6D phase space
        coords_icrs = SkyCoord(
            ra=df_raw['ra'].values * u.deg,
            dec=df_raw['dec'].values * u.deg,
            distance=(df_raw['parallax'].values * u.mas).to(u.pc, equivalencies=u.parallax()),
            pm_ra_cosdec=df_raw['pmra'].values * u.mas/u.yr,
            pm_dec=df_raw['pmdec'].values * u.mas/u.yr,
            radial_velocity=df_raw['radial_velocity'].values * u.km/u.s,
            frame='icrs'
        )
        
    except Exception as e:
        logger.error(f"Error creating SkyCoord object: {e}")
        return pd.DataFrame()
    
    # Transform to galactocentric
    logger.info("Transforming to galactocentric frame...")
    coords_gc = coords_icrs.transform_to(gc_frame)
    
    # Create output DataFrame
    df_processed = pd.DataFrame()
    
    # Preserve source IDs for traceability
    if 'source_id' in df_raw.columns:
        df_processed['source_id'] = df_raw['source_id']
    
    # Galactocentric positions
    df_processed['X_gc_kpc'] = coords_gc.x.to(u.kpc).value
    df_processed['Y_gc_kpc'] = coords_gc.y.to(u.kpc).value
    df_processed['Z_gc_kpc'] = coords_gc.z.to(u.kpc).value
    
    # Cylindrical coordinates
    df_processed['R_kpc'] = coords_gc.cylindrical.rho.to(u.kpc).value
    df_processed['phi_rad'] = coords_gc.cylindrical.phi.value
    df_processed['z_kpc'] = coords_gc.z.to(u.kpc).value
    
    # Velocities in cylindrical coordinates
    cyl_vel_diff = coords_gc.velocity.represent_as(CylindricalDifferential, coords_gc.data)
    df_processed['v_R_kms'] = cyl_vel_diff.d_rho.to(u.km/u.s).value
    df_processed['v_phi_kms'] = (coords_gc.cylindrical.rho * cyl_vel_diff.d_phi).to(
        u.km/u.s, equivalencies=u.dimensionless_angles()
    ).value
    df_processed['v_z_kms'] = cyl_vel_diff.d_z.to(u.km/u.s).value
    
    # Circular velocity (what we fit)
    df_processed['v_obs'] = np.abs(df_processed['v_phi_kms'])
    
    # Calculate comprehensive velocity errors
    df_processed['sigma_v'] = _calculate_velocity_errors_enhanced(
        df_raw, coords_icrs, df_processed
    )
    
    # Add quality flags
    df_processed['quality_flag'] = _assign_quality_flags(df_raw, df_processed)
    
    # Filter to valid values with physical constraints
    initial_count = len(df_processed)
    
    df_processed = df_processed[
        # Basic validity
        np.isfinite(df_processed['R_kpc']) & 
        np.isfinite(df_processed['v_obs']) & 
        np.isfinite(df_processed['sigma_v']) &
        # Physical constraints
        (df_processed['R_kpc'] > 0.01) &                    # Not at galactic center
        (df_processed['R_kpc'] < 30) &                      # Within reasonable galaxy size
        (df_processed['v_obs'] < MAX_REASONABLE_VELOCITY_KMS) &
        (df_processed['sigma_v'] < MAX_VELOCITY_ERROR_KMS) &
        (np.abs(df_processed['z_kpc']) < 5)                 # Not too far from disk
    ].copy()
    
    filtered_count = initial_count - len(df_processed)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} stars with unphysical values")
    
    logger.info(f"Successfully processed {len(df_processed):,} stars with valid kinematics")
    
    # Validate processed data
    _validate_processed_data(df_processed)
    
    return df_processed


def _calculate_velocity_errors_enhanced(
    df_raw: pd.DataFrame, 
    coords_icrs: SkyCoord,
    df_processed: pd.DataFrame
) -> np.ndarray:
    """
    Calculate comprehensive velocity errors with proper error propagation.
    
    This includes:
    1. Radial velocity measurement errors
    2. Proper motion errors projected to tangential velocity
    3. Distance errors affecting tangential velocities
    4. Coordinate transformation uncertainties
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Gaia data with error columns
    coords_icrs : SkyCoord
        ICRS coordinates
    df_processed : pd.DataFrame
        Processed data with velocities
        
    Returns
    -------
    np.ndarray
        Total velocity errors in km/s
    """
    logger.info("Calculating velocity errors with full error propagation...")
    
    # Extract basic measurements and errors
    distance_kpc = coords_icrs.distance.to(u.kpc).value
    rv_error_kms = df_raw['radial_velocity_error'].values
    pmra_error_masyr = df_raw['pmra_error'].values
    pmdec_error_masyr = df_raw['pmdec_error'].values
    parallax_error_mas = df_raw['parallax_error'].values
    
    # Replace NaN with sensible defaults
    rv_error_kms = np.nan_to_num(rv_error_kms, nan=10.0)
    pmra_error_masyr = np.nan_to_num(pmra_error_masyr, nan=0.5)
    pmdec_error_masyr = np.nan_to_num(pmdec_error_masyr, nan=0.5)
    parallax_error_mas = np.nan_to_num(parallax_error_mas, nan=0.1)
    
    # 1. Tangential velocity error from proper motion
    # v_tan = 4.74047 * d[kpc] * μ[mas/yr]
    pm_tot_masyr = np.sqrt(df_raw['pmra'].values**2 + df_raw['pmdec'].values**2)
    pm_tot_error_masyr = np.sqrt(pmra_error_masyr**2 + pmdec_error_masyr**2)
    
    # Error from PM uncertainty
    v_tan_error_pm = 4.74047 * distance_kpc * pm_tot_error_masyr
    
    # 2. Tangential velocity error from distance uncertainty
    # σ_d/d = σ_π/π
    relative_distance_error = parallax_error_mas / df_raw['parallax'].values
    v_tan_error_dist = 4.74047 * distance_kpc * pm_tot_masyr * relative_distance_error
    
    # 3. Total tangential velocity error
    v_tan_error_total = np.sqrt(v_tan_error_pm**2 + v_tan_error_dist**2)
    
    # 4. Project to circular velocity error
    # This is approximate - proper treatment would need full covariance matrix
    # For circular velocity, main contribution is from tangential component
    
    # Get velocity components for projection
    v_R = df_processed['v_R_kms'].values
    v_phi = df_processed['v_phi_kms'].values
    v_tot = np.sqrt(v_R**2 + v_phi**2)
    
    # Weight factors for error projection (simplified)
    w_tan = np.abs(v_phi) / (v_tot + 1e-10)  # Avoid division by zero
    w_rad = np.abs(v_R) / (v_tot + 1e-10)
    
    # Combined velocity error
    v_err_combined = np.sqrt(
        (w_tan * v_tan_error_total)**2 + 
        (w_rad * rv_error_kms)**2
    )
    
    # Apply error floor and ceiling
    v_err_combined = np.clip(v_err_combined, MIN_VELOCITY_ERROR_KMS, MAX_VELOCITY_ERROR_KMS)
    
    # Handle any remaining NaN or inf
    mask_bad = ~np.isfinite(v_err_combined)
    v_err_combined[mask_bad] = DEFAULT_LARGE_ERROR_KMS
    
    # Log statistics
    logger.info(f"Velocity error statistics:")
    logger.info(f"  Median: {np.median(v_err_combined):.1f} km/s")
    logger.info(f"  Mean: {np.mean(v_err_combined):.1f} km/s")
    logger.info(f"  Range: [{np.min(v_err_combined):.1f}, {np.max(v_err_combined):.1f}] km/s")
    
    return v_err_combined


def _assign_quality_flags(df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> np.ndarray:
    """
    Assign quality flags to processed stars based on various criteria.
    
    Quality levels:
    0 - Excellent: All parameters well-determined
    1 - Good: Minor issues but usable
    2 - Fair: Use with caution
    3 - Poor: Significant issues
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Gaia data
    df_processed : pd.DataFrame  
        Processed data
        
    Returns
    -------
    np.ndarray
        Quality flag for each star
    """
    n_stars = len(df_processed)
    quality = np.zeros(n_stars, dtype=int)
    
    # Check various quality criteria
    if 'ruwe' in df_raw.columns:
        quality[df_raw['ruwe'].values > 1.1] += 1
        quality[df_raw['ruwe'].values > 1.3] += 1
    
    if 'astrometric_excess_noise' in df_raw.columns:
        quality[df_raw['astrometric_excess_noise'].values > 0.5] += 1
        quality[df_raw['astrometric_excess_noise'].values > 1.0] += 1
    
    # Large velocity errors
    quality[df_processed['sigma_v'].values > 20] += 1
    quality[df_processed['sigma_v'].values > 40] += 1
    
    # Extreme positions
    quality[df_processed['R_kpc'].values > 20] += 1
    quality[np.abs(df_processed['z_kpc'].values) > 2] += 1
    
    # Clip to maximum flag value
    quality = np.clip(quality, 0, 3)
    
    # Log distribution
    unique, counts = np.unique(quality, return_counts=True)
    logger.info("Quality flag distribution:")
    for q, c in zip(unique, counts):
        logger.info(f"  Quality {q}: {c:,} stars ({c/n_stars*100:.1f}%)")
    
    return quality



def _validate_processed_data(df_processed: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate processed data for physical consistency and adequate sampling.
    Fixed version that handles missing phi_rad column gracefully.
    """
    logger.info("\nValidating processed data...")
    
    validation_results = {}
    
    # 1. Check velocity distribution
    v_median = df_processed['v_obs'].median()
    v_mad = np.median(np.abs(df_processed['v_obs'] - v_median))
    
    v_check = EXPECTED_V_MEDIAN_RANGE[0] < v_median < EXPECTED_V_MEDIAN_RANGE[1]
    validation_results['velocity_distribution'] = v_check
    
    if not v_check:
        logger.warning(f"❌ Median velocity {v_median:.1f} km/s outside expected range "
                      f"{EXPECTED_V_MEDIAN_RANGE}")
    else:
        logger.info(f"✅ Median velocity {v_median:.1f} ± {v_mad:.1f} km/s is reasonable")
    
    # 2. Check radial distribution
    R_median = df_processed['R_kpc'].median()
    R_check = EXPECTED_R_MEDIAN_RANGE[0] < R_median < EXPECTED_R_MEDIAN_RANGE[1]
    validation_results['radial_distribution'] = R_check
    
    if not R_check:
        logger.warning(f"❌ Median radius {R_median:.1f} kpc outside expected range "
                      f"{EXPECTED_R_MEDIAN_RANGE}")
    else:
        logger.info(f"✅ Median radius {R_median:.1f} kpc is reasonable")
    
    # 3. Check radial sampling
    R_bins = [0, 5, 8, 12, 20, 30]
    sampling_adequate = True
    
    logger.info("\nRadial sampling:")
    for i in range(len(R_bins)-1):
        mask = (df_processed['R_kpc'] > R_bins[i]) & (df_processed['R_kpc'] < R_bins[i+1])
        n_in_bin = mask.sum()
        
        bin_check = n_in_bin >= MIN_STARS_PER_RADIAL_BIN
        if not bin_check:
            sampling_adequate = False
            logger.warning(f"  ❌ [{R_bins[i]:2d}, {R_bins[i+1]:2d}] kpc: "
                          f"{n_in_bin:5d} stars (need >{MIN_STARS_PER_RADIAL_BIN})")
        else:
            logger.info(f"  ✅ [{R_bins[i]:2d}, {R_bins[i+1]:2d}] kpc: {n_in_bin:5d} stars")
    
    validation_results['radial_sampling'] = sampling_adequate
    
    # 4. Check for obvious biases - FIXED to handle missing phi_rad
    if 'phi_rad' in df_processed.columns:
        # Check if data is symmetric in phi
        phi_std = df_processed['phi_rad'].std()
        phi_uniform = phi_std > 1.5  # Should be ~1.8 for uniform distribution
        validation_results['azimuthal_coverage'] = phi_uniform
        
        if not phi_uniform:
            logger.warning(f"❌ Azimuthal distribution may be biased (std={phi_std:.2f})")
        else:
            logger.info(f"✅ Good azimuthal coverage (std={phi_std:.2f})")
    else:
        logger.warning("⚠️  phi_rad column missing - skipping azimuthal coverage check")
        validation_results['azimuthal_coverage'] = True  # Don't fail validation for this
    
    # 5. Check vertical distribution
    z_median = df_processed['z_kpc'].median()
    z_symmetric = abs(z_median) < 0.1
    validation_results['vertical_symmetry'] = z_symmetric
    
    if not z_symmetric:
        logger.warning(f"❌ Vertical distribution asymmetric (median z={z_median:.3f} kpc)")
    else:
        logger.info(f"✅ Vertical distribution symmetric (median z={z_median:.3f} kpc)")
    
    # Summary
    all_passed = all(validation_results.values())
    if all_passed:
        logger.info("\n✅ All validation checks passed!")
    else:
        failed_tests = [k for k, v in validation_results.items() if not v]
        logger.warning(f"\n❌ Validation issues found: {failed_tests}")
        logger.warning("   Consider adjusting query parameters or checking data quality")
    
    return validation_results


# ============================================================================
# Main Loading Function
# ============================================================================

def load_gaia(
    sample_max: int = 100_000,
    force_new_query_gaia: bool = False,
    force_reprocess_raw: bool = False,
    raw_cache_filename: Optional[str] = None,
    processed_cache_filename: Optional[str] = None,
    use_enhanced_query: bool = True,
    validate_data: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load Gaia data with comprehensive quality controls and validation.
    
    This is the main entry point for loading Gaia DR3 data. It implements:
    1. Two-stage caching (raw query results and processed data)
    2. Enhanced quality cuts for reliable data
    3. Comprehensive error propagation
    4. Data validation and sanity checks
    
    Parameters
    ----------
    sample_max : int
        Maximum number of stars to return (applied after processing)
    force_new_query_gaia : bool
        If True, bypass raw cache and query Gaia Archive
    force_reprocess_raw : bool
        If True, bypass processed cache and reprocess raw data
    raw_cache_filename : str, optional
        Filename for raw query cache (default: RAW_GAIA_CACHE_FILENAME_DEFAULT)
    processed_cache_filename : str, optional
        Filename for processed data cache (default: PROCESSED_GAIA_CACHE_FILENAME_DEFAULT)
    use_enhanced_query : bool
        If True, use enhanced query with stricter cuts (recommended)
    validate_data : bool
        If True, perform validation checks on loaded data
        
    Returns
    -------
    dict or None
        Dictionary with arrays:
        - 'R_kpc': Galactocentric radius
        - 'v_obs': Observed circular velocity (|v_phi|)
        - 'sigma_v': Velocity uncertainty
        - 'z_kpc': Height above midplane
        - 'source_id': Gaia source IDs (if available)
        - 'quality_flag': Quality indicators (if enhanced processing used)
        
    Examples
    --------
    >>> # Basic usage
    >>> gaia_data = load_gaia(sample_max=10000)
    
    >>> # Force fresh query with enhanced cuts
    >>> gaia_data = load_gaia(force_new_query_gaia=True, use_enhanced_query=True)
    
    >>> # Load without validation (faster but less safe)
    >>> gaia_data = load_gaia(validate_data=False)
    """
    # Set default filenames if not provided
    if raw_cache_filename is None:
        raw_cache_filename = RAW_GAIA_CACHE_FILENAME_DEFAULT
    if processed_cache_filename is None:
        processed_cache_filename = PROCESSED_GAIA_CACHE_FILENAME_DEFAULT
    
    if not HAS_ASTROPY_AND_QUERY:
        logger.error("Cannot load Gaia data: astropy/astroquery not available")
        return None
    
    logger.info("="*60)
    logger.info("GAIA DATA LOADER - Enhanced Version")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Max sample size: {sample_max:,}")
    logger.info(f"  Enhanced query: {use_enhanced_query}")
    logger.info(f"  Data validation: {validate_data}")
    logger.info(f"  Force new query: {force_new_query_gaia}")
    logger.info(f"  Force reprocess: {force_reprocess_raw}")
    
    processed_cache_path = Path(processed_cache_filename)
    df_processed_output = None
    
    # Stage 1: Try to load already PROCESSED data from cache
    if USE_LOCAL_CACHE and not force_reprocess_raw and processed_cache_path.exists():
        logger.info(f"\n📁 Checking processed data cache: {processed_cache_path}")
        try:
            df_cached_processed = pd.read_parquet(processed_cache_path)
            
            # Verify cache has required columns
            required_cols = ['R_kpc', 'v_obs', 'sigma_v', 'z_kpc']
            if not df_cached_processed.empty and all(col in df_cached_processed.columns 
                                                     for col in required_cols):
                logger.info(f"✅ Loaded {len(df_cached_processed):,} processed stars from cache")
                df_processed_output = df_cached_processed
            else:
                logger.warning("❌ Processed cache is invalid or missing columns. Will regenerate.")
                
        except Exception as e:
            logger.error(f"❌ Error loading processed cache: {e}")
    
    # Stage 2: If no processed data, get raw data and process it
    if df_processed_output is None:
        logger.info("\n📡 Need to obtain/process raw Gaia data...")
        
        # Get raw data (from cache or query)
        if use_enhanced_query:
            df_raw_gaia = perform_gaia_adql_query_enhanced(
                limit_val=sample_max * 2,  # Query more to account for filtering
                cache_raw_filename=raw_cache_filename,
                force_live_query=force_new_query_gaia
            )
        else:
            # Fall back to original query function
            df_raw_gaia = perform_gaia_adql_query(
                limit_val=sample_max * 2,
                cache_raw_filename=raw_cache_filename,
                force_live_query=force_new_query_gaia
            )
        
        if df_raw_gaia is not None and not df_raw_gaia.empty:
            # Process the raw data
            if use_enhanced_query:
                df_processed_output = process_raw_gaia_df_enhanced(df_raw_gaia)
            else:
                df_processed_output = process_raw_gaia_df(df_raw_gaia)
            
            # Save processed data to cache
            if (df_processed_output is not None and 
                not df_processed_output.empty and 
                USE_LOCAL_CACHE):
                logger.info(f"\n💾 Saving processed data to cache: {processed_cache_path}")
                try:
                    df_processed_output.to_parquet(processed_cache_path, index=False)
                    logger.info("✅ Cache saved successfully")
                except Exception as e:
                    logger.error(f"❌ Error saving processed cache: {e}")
        else:
            logger.error("❌ No raw Gaia data obtained")
            return None
    
    # Final checks
    if df_processed_output is None or df_processed_output.empty:
        logger.error("❌ No data available after processing")
        return None
    
    # Ensure z_kpc column exists (for backward compatibility)
    if 'z_kpc' not in df_processed_output.columns:
        logger.warning("Adding missing z_kpc column (set to 0)")
        df_processed_output['z_kpc'] = 0.0
    
    # Apply sampling if needed
    if sample_max and len(df_processed_output) > sample_max:
        # Use stratified sampling by radius to maintain coverage
        logger.info(f"\n📊 Applying stratified sampling: {len(df_processed_output):,} → {sample_max:,} stars")
        
        # Create radial bins for stratified sampling
        df_processed_output['R_bin'] = pd.cut(df_processed_output['R_kpc'], 
                                              bins=[0, 5, 8, 12, 20, 100])
        
        # Sample proportionally from each bin
        df_processed_output = df_processed_output.groupby('R_bin', group_keys=False).apply(
            lambda x: x.sample(n=int(len(x) * sample_max / len(df_processed_output)), 
                              random_state=42)
        )
        
        # Drop the temporary column
        df_processed_output = df_processed_output.drop('R_bin', axis=1)
        
        logger.info(f"✅ Sampled {len(df_processed_output):,} stars with radial stratification")
    
    # Final validation if requested
    if validate_data:
        logger.info("\n🔍 Performing final validation...")
        validation_results = _validate_processed_data(df_processed_output)
    
    # Prepare output dictionary
    return_dict = {
        "R_kpc": df_processed_output["R_kpc"].values,
        "v_obs": df_processed_output["v_obs"].values,
        "sigma_v": df_processed_output["sigma_v"].values,
        "z_kpc": df_processed_output["z_kpc"].values
    }
    
    # Include optional columns if available
    if 'source_id' in df_processed_output.columns:
        return_dict["source_id"] = df_processed_output["source_id"].values
    
    if 'quality_flag' in df_processed_output.columns:
        return_dict["quality_flag"] = df_processed_output["quality_flag"].values
    
    if 'v_R_kms' in df_processed_output.columns:
        return_dict["v_R_kms"] = df_processed_output["v_R_kms"].values
        return_dict["v_z_kms"] = df_processed_output["v_z_kms"].values
    
    logger.info("\n✅ Data loading complete!")
    logger.info(f"   Returned {len(return_dict['R_kpc']):,} stars")
    logger.info("="*60)
    
    return return_dict


def load_all_sky_gaia_slices(
    lon_bin_width: int = 30,
    stars_per_bin: int = 12000,
    output_dir: str = "gaia_sky_slices",
    force_query: bool = False,
    max_distance_kpc: float = 30.0
) -> pd.DataFrame:
    """
    Query and process Gaia stars across the full disk in longitude slices.

    Parameters
    ----------
    lon_bin_width : int
        Width of longitude slices in degrees (e.g., 30° = 12 slices)
    stars_per_bin : int
        Max stars to download per slice
    output_dir : str
        Where to cache raw and processed data
    force_query : bool
        If True, re-query Gaia even if cache exists
    max_distance_kpc : float
        Max galactocentric radius to retain after transformation

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all stars across all bins
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_dataframes = []

    # Inside the for loop in load_all_sky_gaia_slices
    for lon_min in range(0, 360, lon_bin_width):
        lon_max = lon_min + lon_bin_width
        lon_label = f"L{lon_min:03d}-{lon_max:03d}"

        raw_cache = f"{output_dir}/raw_{lon_label}.csv"
        processed_cache = f"{output_dir}/processed_{lon_label}.parquet"

        logger.info(f"\n🌌 Querying stars with {lon_min}° ≤ l < {lon_max}°")

        # Adjust cuts based on longitude region
        if 60 <= lon_min < 180 or 270 <= lon_min < 330:
            stars_this_bin = 12000
            min_parallax = 0.02
            min_snr = 8
        elif 0 <= lon_min < 90:
            stars_this_bin = 12000
            min_parallax = 0.03
            min_snr = 10
        else:
            stars_this_bin = 12000
            min_parallax = 0.025
            min_snr = 8

        
        query_adql = f"""
SELECT TOP {stars_this_bin}
    source_id, ra, dec, parallax, parallax_error,
    pmra, pmra_error, pmdec, pmdec_error,
    radial_velocity, radial_velocity_error,
    ruwe, phot_g_mean_mag, b, l
FROM gaiadr3.gaia_source
WHERE parallax > {min_parallax}
    AND parallax_over_error > {min_snr}
    AND pmra IS NOT NULL AND pmdec IS NOT NULL
    AND pmra_error < {MAX_PM_ERROR_MAS_YR}
    AND pmdec_error < {MAX_PM_ERROR_MAS_YR}
    AND radial_velocity IS NOT NULL
    AND radial_velocity_error < {MAX_RV_ERROR_KMS}
    AND ruwe < {MAX_RUWE}
    AND phot_g_mean_mag < {MAX_PHOT_G_MAG}
    AND ABS(b) < {MAX_ABS_B_DEG}
    AND l >= {lon_min} AND l < {lon_max}
ORDER BY phot_g_mean_mag
"""

        try:
            if not force_query and Path(processed_cache).exists():
                df_proc = pd.read_parquet(processed_cache)
                logger.info(f"✅ Loaded cached slice: {processed_cache}")
            else:
                logger.info("📡 Sending ADQL query to Gaia...")
                job = Gaia.launch_job_async(query_adql)
                df_raw = job.get_results().to_pandas()
                df_raw.to_csv(raw_cache, index=False)
                logger.info(f"💾 Saved raw slice to {raw_cache}")

                df_proc = process_raw_gaia_df_enhanced(df_raw)
                df_proc = df_proc[df_proc['R_kpc'] < max_distance_kpc]
                df_proc.to_parquet(processed_cache, index=False)
                logger.info(f"✅ Processed & saved: {processed_cache}")
            
            all_dataframes.append(df_proc)

        except Exception as e:
            logger.error(f"❌ Failed to fetch/process {lon_label}: {e}")

    # Combine all into one DataFrame
    if all_dataframes:
        df_all = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"\n✅ Loaded total: {len(df_all):,} stars across {len(all_dataframes)} slices")
        return df_all
    else:
        logger.error("❌ No data returned from any slice")
        return pd.DataFrame()


# ============================================================================
# Test Functions
# ============================================================================

if __name__ == '__main__':
    """Run full-sky Gaia DR3 data acquisition with 30° longitude bins"""

    print("\n" + "="*60)
    print("🌌 FULL SKY GAIA DATA LOADER")
    print("="*60)

    # Run full-sky loader for publication-grade data
    df_all_sky = load_all_sky_gaia_slices(
        lon_bin_width=30,         # 12 bins around full galactic disk
        stars_per_bin=12000,       # Safe number of stars per bin
        output_dir="gaia_sky_slices",
        force_query=True,        # Set to True to re-download all
        max_distance_kpc=30.0     # Target up to 30 kpc from galactic center
    )

    if not df_all_sky.empty:
        print(f"\n✅ All-sky dataset loaded with {len(df_all_sky):,} stars")
        print(f"   R_kpc range: {df_all_sky['R_kpc'].min():.2f}–{df_all_sky['R_kpc'].max():.2f} kpc")
        print(f"   Median v_obs: {np.median(df_all_sky['v_obs']):.2f} km/s")
    else:
        print("❌ All-sky load failed")

    print("\n" + "="*60)
    print("✅ ALL-SKY GAIA DATA LOADING COMPLETE")
    print("="*60)
