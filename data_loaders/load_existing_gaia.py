#!/usr/bin/env python3
"""
Load existing Gaia data from parquet files for split-region analysis.

This script loads the processed Gaia data that's already available in the
gaia_sky_slices folder instead of querying the database.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_gaia_data(sample_max=None):
    """
    Load existing Gaia data from processed parquet files.
    
    Parameters:
    -----------
    sample_max : int, optional
        Maximum number of stars to return (for testing)
        
    Returns:
    --------
    dict : Dictionary with Gaia data arrays
    """
    logger.info("Loading existing Gaia data from processed parquet files...")
    
    # Path to processed data
    gaia_dir = Path("gaia_sky_slices")
    
    # Find all processed parquet files
    parquet_files = list(gaia_dir.glob("processed_L*.parquet"))
    
    if not parquet_files:
        logger.error("No processed parquet files found in gaia_sky_slices/")
        return None
    
    logger.info(f"Found {len(parquet_files)} processed parquet files")
    
    # Load and combine all data
    all_data = []
    
    for file_path in parquet_files:
        logger.info(f"Loading {file_path.name}...")
        try:
            df = pd.read_parquet(file_path)
            all_data.append(df)
            logger.info(f"  Loaded {len(df)} stars from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            continue
    
    if not all_data:
        logger.error("No data could be loaded from parquet files")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} total stars")
    
    # Check required columns
    required_columns = ['R_kpc', 'v_obs', 'sigma_v']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(combined_df.columns)}")
        return None
    
    # Apply sample limit if requested
    if sample_max and len(combined_df) > sample_max:
        logger.info(f"Subsampling to {sample_max} stars")
        combined_df = combined_df.sample(n=sample_max, random_state=42)
    
    # Convert to arrays
    gaia_data = {
        'R_kpc': combined_df['R_kpc'].values,
        'v_obs': combined_df['v_obs'].values,
        'sigma_v': combined_df['sigma_v'].values
    }
    
    # Add optional columns if available
    if 'z_kpc' in combined_df.columns:
        gaia_data['z_kpc'] = combined_df['z_kpc'].values
    
    if 'source_id' in combined_df.columns:
        gaia_data['source_id'] = combined_df['source_id'].values
    
    logger.info("Gaia data loaded successfully!")
    logger.info(f"  Radius range: {gaia_data['R_kpc'].min():.2f} - {gaia_data['R_kpc'].max():.2f} kpc")
    logger.info(f"  Velocity range: {gaia_data['v_obs'].min():.1f} - {gaia_data['v_obs'].max():.1f} km/s")
    logger.info(f"  Mean velocity: {gaia_data['v_obs'].mean():.1f} ± {gaia_data['v_obs'].std():.1f} km/s")
    
    return gaia_data

if __name__ == "__main__":
    # Test loading
    data = load_existing_gaia_data(sample_max=1000)
    if data:
        print("✅ Successfully loaded Gaia data!")
        print(f"   {len(data['R_kpc'])} stars")
    else:
        print("❌ Failed to load Gaia data") 