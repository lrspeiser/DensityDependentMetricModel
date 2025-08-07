#!/usr/bin/env python3
"""
run_dynesty_nocupy.py - Version that bypasses CuPy multiprocessing issues
by keeping everything in the main process.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import cupy as cp
import argparse
from pathlib import Path
import pickle
import json
from datetime import datetime
import pandas as pd

# Import the physics functions
from core.density_metric_cupy import v_total_kms_cupy

class SingleProcessSampler:
    """Wrapper to run dynesty in single process to avoid CuPy issues."""
    
    def __init__(self, R_data, v_data, sigma_data, xi_type):
        # Store data as CuPy arrays from the start
        self.R_data = cp.array(R_data, dtype=cp.float32)
        self.v_data = cp.array(v_data, dtype=cp.float32)
        self.sigma_data = cp.array(sigma_data, dtype=cp.float32)
        self.xi_type = xi_type
        self.call_count = 0
        
    def log_likelihood(self, theta, param_names):
        """Likelihood function that uses pre-loaded CuPy arrays."""
        self.call_count += 1
        
        try:
            # Check physical plausibility
            for name, value in zip(param_names, theta):
                if 'M_' in name and 'solar' in name and value <= 0:
                    return -np.inf
                if any(x in name for x in ['R_', 'r_', 'a_']) and value <= 0:
                    return -np.inf
            
            # Create parameter dictionary
            params = dict(zip(param_names, theta))
            
            # Compute model velocities - data is already CuPy
            v_model = v_total_kms_cupy(self.R_data, params, xi_type=self.xi_type)
            
            # Check for NaN/Inf
            if not cp.all(cp.isfinite(v_model)):
                return -np.inf
            
            # Compute chi-squared
            chi2 = cp.sum(((self.v_data - v_model) / self.sigma_data)**2)
            
            # Convert to float and return log-likelihood
            logl = -0.5 * float(chi2.get())
            
            # Print progress occasionally
            if self.call_count <= 5 or self.call_count % 1000 == 0:
                print(f"Call {self.call_count}: logL = {logl:.2f}")
            
            return logl
            
        except Exception as e:
            print(f"Likelihood error: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf

def prior_transform(u, param_names, bounds_low, bounds_high, use_log_prior):
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
    """Main function."""
    
    parser = argparse.ArgumentParser(description="Single-process dynesty runner")
    parser.add_argument('--xi', type=str, required=True,
                       choices=['grav_color_void_safe', 'enhanced', 'power', 'grav_color'],
                       help='Xi function type')
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--maxcall', type=int, default=10000000)
    parser.add_argument('--dlogz_target', type=float, default=0.01)
    parser.add_argument('--max_sample_gaia', type=int, default=144000)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"runs/{args.xi}_single_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load Gaia data
    logger.info("Loading Gaia data...")
    cache_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
    
    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        return
    
    logger.info(f"Loading from cache: {cache_file}")
    gaia_df = pd.read_csv(cache_file)
    
    # Process the data if needed
    from core.data_io import process_gaia_data
    if 'R_kpc' not in gaia_df.columns:
        gaia_df = process_gaia_data(gaia_df)
    
    # Convert to numpy arrays first
    R_data = gaia_df['R_kpc'].values.astype(np.float32)
    v_data = gaia_df['v_obs'].values.astype(np.float32)
    sigma_data = gaia_df['sigma_v'].values.astype(np.float32)
    
    # Limit sample size if requested
    if args.max_sample_gaia and len(R_data) > args.max_sample_gaia:
        indices = np.random.choice(len(R_data), args.max_sample_gaia, replace=False)
        R_data = R_data[indices]
        v_data = v_data[indices]
        sigma_data = sigma_data[indices]
    
    logger.info(f"Loaded {len(R_data)} stars")
    
    # Setup parameters
    from run_dynesty_cupy import setup_parameter_bounds
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    
    logger.info(f"Fitting {len(param_names)} parameters: {param_names}")
    
    # Create sampler wrapper
    sampler_wrapper = SingleProcessSampler(R_data, v_data, sigma_data, args.xi)
    
    # Import and run dynesty - NO MULTIPROCESSING
    import dynesty
    
    sampler = dynesty.DynamicNestedSampler(
        lambda theta: sampler_wrapper.log_likelihood(theta, param_names),
        lambda u: prior_transform(u, param_names, bounds_low, bounds_high, use_log_prior),
        ndim=len(param_names),
        nlive=args.nlive,
        sample='rslice',
        bound='multi'
    )
    
    logger.info("Starting sampling (single process)...")
    
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
        logz=results.logz[-1] if hasattr(results, 'logz') and len(results.logz) > 0 else np.nan,
        param_names=param_names,
        xi_type=args.xi
    )
    
    final_logz = results.logz[-1] if hasattr(results, 'logz') and len(results.logz) > 0 else np.nan
    logger.info(f"Sampling complete! LogZ = {final_logz:.2f}")
    
    # Model comparison
    BASELINE_LOGZ_GR = -1490897.53
    delta_logz = final_logz - BASELINE_LOGZ_GR
    logger.info(f"Delta LogZ vs GR: {delta_logz:+.2f}")
    
    if delta_logz > 0:
        logger.info("DDMM model preferred over GR!")
    else:
        logger.info("GR model preferred over DDMM")

if __name__ == "__main__":
    main()