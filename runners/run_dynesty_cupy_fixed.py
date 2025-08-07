#!/usr/bin/env python3
"""
run_dynesty_cupy_fixed.py - Fixed version that handles CuPy arrays properly

This version stores data as numpy arrays and converts to CuPy only inside
the likelihood function to avoid multiprocessing issues.
"""

# Copy the imports from run_dynesty_cupy.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import argparse
from pathlib import Path
import pickle
import json
import threading
import time
from multiprocessing import Pool, freeze_support
from datetime import datetime
import pandas as pd

# CuPy imports for GPU acceleration
import cupy as cp

# Import the physics functions directly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import density_metric_cupy

# Global data storage (as numpy arrays to avoid multiprocessing issues)
GLOBAL_R_DATA = None
GLOBAL_V_DATA = None
GLOBAL_SIGMA_DATA = None

def log_likelihood_dynesty_fixed(theta, param_names, args):
    """
    Fixed likelihood that uses global numpy data and converts to CuPy internally.
    """
    global GLOBAL_R_DATA, GLOBAL_V_DATA, GLOBAL_SIGMA_DATA
    
    try:
        # Ensure CuPy is properly initialized for this process
        cp.cuda.Device(0).use()
        # Check physical plausibility
        for i, (name, value) in enumerate(zip(param_names, theta)):
            if 'M_' in name and 'solar' in name and value <= 0:
                return -np.inf
            if any(x in name for x in ['R_', 'r_', 'a_']) and value <= 0:
                return -np.inf
        
        # Convert numpy data to CuPy for GPU computation
        # Use cp.array() instead of cp.asarray() to force copy
        R_data_cupy = cp.array(GLOBAL_R_DATA, dtype=cp.float32)
        v_data_cupy = cp.array(GLOBAL_V_DATA, dtype=cp.float32)
        sigma_data_cupy = cp.array(GLOBAL_SIGMA_DATA, dtype=cp.float32)
        
        # Create parameter dictionary
        params = dict(zip(param_names, theta))
        
        # Compute model velocities - call directly without extra conversions
        v_model = density_metric_cupy.v_total_kms_cupy(R_data_cupy, params, xi_type=args.xi)
        
        # Check for NaN/Inf
        if not cp.all(cp.isfinite(v_model)):
            return -np.inf
        
        # Compute chi-squared
        chi2 = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
        
        # Convert to float and return log-likelihood
        logl = -0.5 * float(chi2.get())  # Use .get() to transfer from GPU
        
        return logl
        
    except Exception as e:
        print(f"Likelihood error: {e}")
        return -np.inf

def prior_transform_dynesty_fixed(u, param_names, bounds_low, bounds_high, use_log_prior):
    """Prior transform for dynesty."""
    theta = np.zeros(len(u))
    for i in range(len(u)):
        if use_log_prior[i]:
            # Log-uniform prior
            log_low = np.log10(bounds_low[i])
            log_high = np.log10(bounds_high[i])
            theta[i] = 10**(log_low + u[i] * (log_high - log_low))
        else:
            # Uniform prior
            theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
    return theta

def main():
    """Main function with fixed CuPy handling."""
    global GLOBAL_R_DATA, GLOBAL_V_DATA, GLOBAL_SIGMA_DATA
    
    parser = argparse.ArgumentParser(description="Fixed CuPy dynesty runner")
    parser.add_argument('--xi', type=str, required=True,
                       choices=['grav_color_void_safe', 'enhanced', 'power', 'grav_color'],
                       help='Xi function type')
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--maxcall', type=int, default=10000000)
    parser.add_argument('--dlogz_target', type=float, default=0.01)
    parser.add_argument('--max_sample_gaia', type=int, default=144000)
    parser.add_argument('--num_threads', type=int, default=1)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"runs/{args.xi}_fixed_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load Gaia data
    logger.info("Loading Gaia data...")
    try:
        # Use the existing merged cache file directly
        cache_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
        
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            gaia_df = pd.read_csv(cache_file)
            
            # Process the data if needed
            from core.data_io import process_gaia_data
            if 'R_kpc' not in gaia_df.columns:
                gaia_df = process_gaia_data(gaia_df)
        else:
            logger.error(f"Cache file not found: {cache_file}")
            return
        
        # Convert to numpy arrays (NOT CuPy yet)
        GLOBAL_R_DATA = gaia_df['R_kpc'].values.astype(np.float32)
        GLOBAL_V_DATA = gaia_df['v_obs'].values.astype(np.float32)
        GLOBAL_SIGMA_DATA = gaia_df['sigma_v'].values.astype(np.float32)
        
        # Limit sample size if requested
        if args.max_sample_gaia and len(GLOBAL_R_DATA) > args.max_sample_gaia:
            indices = np.random.choice(len(GLOBAL_R_DATA), args.max_sample_gaia, replace=False)
            GLOBAL_R_DATA = GLOBAL_R_DATA[indices]
            GLOBAL_V_DATA = GLOBAL_V_DATA[indices]
            GLOBAL_SIGMA_DATA = GLOBAL_SIGMA_DATA[indices]
        
        logger.info(f"Loaded {len(GLOBAL_R_DATA)} stars")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Setup parameters based on xi_type
    from run_dynesty_cupy import setup_parameter_bounds
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    
    logger.info(f"Fitting {len(param_names)} parameters: {param_names}")
    
    # Import and run dynesty
    import dynesty
    
    # Create sampler with fixed likelihood
    if args.num_threads > 1:
        with Pool(processes=args.num_threads) as pool:
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood_dynesty_fixed,
                prior_transform_dynesty_fixed,
                ndim=len(param_names),
                logl_args=(param_names, args),
                ptform_args=(param_names, bounds_low, bounds_high, use_log_prior),
                pool=pool,
                queue_size=args.num_threads,
                nlive=args.nlive,
                sample='rslice',
                bound='multi'
            )
            
            # Run sampling
            sampler.run_nested(
                maxcall=args.maxcall,
                dlogz_init=args.dlogz_target,
                print_progress=True
            )
    else:
        sampler = dynesty.DynamicNestedSampler(
            log_likelihood_dynesty_fixed,
            prior_transform_dynesty_fixed,
            ndim=len(param_names),
            logl_args=(param_names, args),
            ptform_args=(param_names, bounds_low, bounds_high, use_log_prior),
            nlive=args.nlive,
            sample='rslice',
            bound='multi'
        )
        
        # Run sampling
        sampler.run_nested(
            maxcall=args.maxcall,
            dlogz_init=args.dlogz_target,
            print_progress=True
        )
    
    # Save results
    results = sampler.results
    
    # Save pickle
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save NPZ
    np.savez(
        output_dir / "posterior_samples.npz",
        samples=results.samples,
        logl=results.logl,
        logz=results.logz[-1],
        param_names=param_names,
        xi_type=args.xi
    )
    
    logger.info(f"Sampling complete! LogZ = {results.logz[-1]:.2f}")
    
    # Model comparison
    BASELINE_LOGZ_GR = -1490897.53
    delta_logz = results.logz[-1] - BASELINE_LOGZ_GR
    logger.info(f"Delta LogZ vs GR: {delta_logz:+.2f}")
    
    if delta_logz > 0:
        logger.info("DDMM model preferred over GR!")
    else:
        logger.info("GR model preferred over DDMM")

if __name__ == "__main__":
    freeze_support()
    main()