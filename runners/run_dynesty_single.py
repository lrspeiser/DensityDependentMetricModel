#!/usr/bin/env python3
"""
run_dynesty_single.py - Robust single-process runner for CuPy models
Completely avoids multiprocessing to prevent '_ArrayProxy' serialization errors.
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
import dynesty
from functools import partial

# Import the physics functions
from core.density_metric_cupy import v_total_kms_cupy
from runners.run_dynesty_cupy import setup_parameter_bounds
from enhanced_summary import DynestyRunSummary

# Set CuPy memory pool for better performance
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

class CuPySampler:
    """Single-process sampler that keeps CuPy arrays in memory."""
    
    def __init__(self, R_data, v_data, sigma_data, xi_type, output_dir):
        """Initialize with data already converted to CuPy arrays."""
        # Ensure we're using GPU 0
        cp.cuda.Device(0).use()
        
        # Store data as CuPy arrays
        self.R_data = cp.asarray(R_data, dtype=cp.float32)
        self.v_data = cp.asarray(v_data, dtype=cp.float32)
        self.sigma_data = cp.asarray(sigma_data, dtype=cp.float32)
        self.xi_type = xi_type
        self.output_dir = output_dir
        
        # Tracking
        self.call_count = 0
        self.best_logl = -np.inf
        self.best_params = None
        self.start_time = datetime.now()
        
        # Summary generator
        self.summary_gen = DynestyRunSummary(output_dir)
        
        logging.info(f"Initialized sampler with {len(R_data)} stars")
        logging.info(f"Data on GPU: R_data shape={self.R_data.shape}, dtype={self.R_data.dtype}")
        
    def log_likelihood(self, theta, param_names):
        """Compute log-likelihood using pre-loaded CuPy arrays."""
        self.call_count += 1
        
        try:
            # Parameter validation
            for name, value in zip(param_names, theta):
                if 'M_' in name and value <= 0:
                    return -np.inf
                if any(x in name for x in ['R_', 'r_', 'a_', 'hz_']) and value <= 0:
                    return -np.inf
            
            # Create parameter dictionary
            params = dict(zip(param_names, theta))
            
            # Compute model velocities using CuPy arrays
            v_model = v_total_kms_cupy(self.R_data, params, xi_type=self.xi_type)
            
            # Check for NaN/Inf
            if not cp.all(cp.isfinite(v_model)):
                return -np.inf
            
            # Compute chi-squared on GPU
            residuals = (self.v_data - v_model) / self.sigma_data
            chi2 = cp.sum(residuals**2)
            
            # Convert to Python float
            logl = -0.5 * float(chi2.get())
            
            # Track best parameters
            if logl > self.best_logl:
                self.best_logl = logl
                self.best_params = theta.copy()
            
            # Progress reporting
            if self.call_count <= 10 or self.call_count % 1000 == 0:
                logging.info(f"Call {self.call_count}: logL = {logl:.2f} (best = {self.best_logl:.2f})")
                
                # Memory usage check every 10000 calls
                if self.call_count % 10000 == 0:
                    used_bytes = mempool.used_bytes()
                    total_bytes = mempool.total_bytes()
                    logging.info(f"GPU memory: {used_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB used")
            
            return logl
            
        except Exception as e:
            logging.error(f"Likelihood error at call {self.call_count}: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf
    
    def save_checkpoint(self, sampler):
        """Save checkpoint for resume capability."""
        try:
            checkpoint_file = self.output_dir / "checkpoint.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'results': sampler.results,
                    'call_count': self.call_count,
                    'best_logl': self.best_logl,
                    'best_params': self.best_params,
                    'xi_type': self.xi_type
                }, f)
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")

def prior_transform(u, bounds_low, bounds_high, use_log_prior):
    """Transform unit cube to prior space."""
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
    parser = argparse.ArgumentParser(description="Single-process CuPy dynesty runner")
    parser.add_argument('--xi', type=str, required=True,
                       choices=['balanced_screening', 'grav_color_void_safe', 'enhanced', 'power', 'grav_color',
                               'elastic_strain', 'hookean', 'tension_field'],
                       help='Xi function type')
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--maxcall', type=int, default=10000000)
    parser.add_argument('--dlogz_target', type=float, default=0.01)
    parser.add_argument('--max_sample_gaia', type=int, default=144000)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--run_dir', type=str, help='Directory to resume from')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dynesty_run.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # GPU info
    logger.info(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        logger.info(f"Using GPU: {props['name'].decode()} (compute capability: {props['major']}.{props['minor']})")
        mem_info = cp.cuda.runtime.memGetInfo()
        logger.info(f"GPU memory: {mem_info[1]/1e9:.2f} GB total, {mem_info[0]/1e9:.2f} GB free")
    
    # Create or identify output directory
    if args.resume and args.run_dir:
        output_dir = Path(args.run_dir)
        if not output_dir.exists():
            logger.error(f"Resume directory not found: {output_dir}")
            return
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"runs/{args.xi}_single_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save run configuration
    config = {
        'xi_type': args.xi,
        'nlive': args.nlive,
        'maxcall': args.maxcall,
        'dlogz_target': args.dlogz_target,
        'max_sample_gaia': args.max_sample_gaia,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load Gaia data
    logger.info("Loading Gaia data...")
    cache_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
    
    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        logger.info("Please run data preparation first or check the data path")
        return
    
    logger.info(f"Loading from cache: {cache_file}")
    gaia_df = pd.read_csv(cache_file)
    
    # Process the data if needed
    from core.data_io import process_gaia_data
    if 'R_kpc' not in gaia_df.columns:
        logger.info("Processing Gaia data...")
        gaia_df = process_gaia_data(gaia_df)
    
    # Extract arrays as numpy first
    R_data = gaia_df['R_kpc'].values.astype(np.float32)
    v_data = gaia_df['v_obs'].values.astype(np.float32)
    sigma_data = gaia_df['sigma_v'].values.astype(np.float32)
    
    # Sample if requested
    if args.max_sample_gaia and len(R_data) > args.max_sample_gaia:
        logger.info(f"Sampling {args.max_sample_gaia} stars from {len(R_data)}")
        indices = np.random.choice(len(R_data), args.max_sample_gaia, replace=False)
        R_data = R_data[indices]
        v_data = v_data[indices]
        sigma_data = sigma_data[indices]
    
    logger.info(f"Using {len(R_data)} stars for fitting")
    
    # Setup parameters
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    ndim = len(param_names)
    
    logger.info(f"Fitting {ndim} parameters:")
    for i, name in enumerate(param_names):
        if use_log_prior[i]:
            logger.info(f"  {name}: [{bounds_low[i]:.2e}, {bounds_high[i]:.2e}] (log-uniform)")
        else:
            logger.info(f"  {name}: [{bounds_low[i]:.4f}, {bounds_high[i]:.4f}] (uniform)")
    
    # Create sampler wrapper with CuPy arrays
    sampler_wrapper = CuPySampler(R_data, v_data, sigma_data, args.xi, output_dir)
    
    # Setup likelihood and prior functions
    logl_func = partial(sampler_wrapper.log_likelihood, param_names=param_names)
    prior_func = partial(prior_transform, bounds_low=bounds_low, 
                        bounds_high=bounds_high, use_log_prior=use_log_prior)
    
    # Create or resume sampler
    if args.resume and args.run_dir:
        checkpoint_file = output_dir / "checkpoint.pkl"
        if checkpoint_file.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                sampler_wrapper.call_count = checkpoint['call_count']
                sampler_wrapper.best_logl = checkpoint['best_logl']
                sampler_wrapper.best_params = checkpoint['best_params']
                logger.info(f"Resumed at call {sampler_wrapper.call_count}")
    
    # Create Dynamic Nested Sampler - NO MULTIPROCESSING
    logger.info("Creating DynamicNestedSampler (single process)...")
    sampler = dynesty.DynamicNestedSampler(
        logl_func,
        prior_func,
        ndim=ndim,
        nlive=args.nlive,
        sample='rslice',  # Robust slice sampling
        bound='multi',    # Multiple bounding ellipsoids
        update_interval=100  # Update bounds every 100 iterations
    )
    
    logger.info("=" * 60)
    logger.info(f"Starting sampling with {args.nlive} live points")
    logger.info(f"Target: dlogz < {args.dlogz_target}, maxcall = {args.maxcall}")
    logger.info("=" * 60)
    
    # Run sampling with periodic checkpoints
    try:
        start_time = datetime.now()
        
        # Initial batch
        sampler.run_nested(
            maxcall=min(10000, args.maxcall),
            dlogz_init=args.dlogz_target * 10,  # Start with looser tolerance
            print_progress=True
        )
        
        # Save initial checkpoint
        sampler_wrapper.save_checkpoint(sampler)
        
        # Continue with tighter tolerance
        remaining_calls = args.maxcall - sampler.ncall
        if remaining_calls > 0:
            sampler.run_nested(
                maxcall=args.maxcall,
                dlogz_init=args.dlogz_target,
                print_progress=True
            )
        
        # Final results
        results = sampler.results
        
        # Generate and save summary
        logger.info("\nGenerating final summary...")
        summary = sampler_wrapper.summary_gen.generate_summary(
            sampler, param_names, args, start_time, status="completed"
        )
        
        with open(output_dir / "final_summary.txt", 'w') as f:
            f.write(summary)
        
        logger.info("\n" + summary)
        
        # Save results
        logger.info("Saving results...")
        
        # Pickle format
        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        # NPZ format for easy loading
        np.savez(
            output_dir / "posterior_samples.npz",
            samples=results.samples,
            logl=results.logl,
            logwt=results.logwt,
            logz=results.logz,
            logzerr=results.logzerr,
            param_names=param_names,
            xi_type=args.xi,
            best_params=sampler_wrapper.best_params,
            best_logl=sampler_wrapper.best_logl
        )
        
        # Model comparison
        final_logz = results.logz[-1] if hasattr(results, 'logz') and len(results.logz) > 0 else np.nan
        BASELINE_LOGZ_GR = -1490897.53  # From power-law baseline
        
        delta_logz = final_logz - BASELINE_LOGZ_GR
        
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"LogZ = {final_logz:.2f} ± {results.logzerr[-1]:.2f}")
        logger.info(f"Delta LogZ vs GR: {delta_logz:+.2f}")
        
        if delta_logz > 0:
            logger.info(">>> DDMM model PREFERRED over GR! <<<")
        else:
            logger.info(">>> GR model preferred over DDMM <<<")
        
        logger.info(f"\nBest-fit logL: {sampler_wrapper.best_logl:.2f}")
        if sampler_wrapper.best_params is not None:
            logger.info("Best-fit parameters:")
            for name, value in zip(param_names, sampler_wrapper.best_params):
                if 'M_' in name:
                    logger.info(f"  {name}: {value:.3e}")
                else:
                    logger.info(f"  {name}: {value:.6f}")
        
        # Clean up GPU memory
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
    except KeyboardInterrupt:
        logger.info("\nSampling interrupted by user")
        sampler_wrapper.save_checkpoint(sampler)
        logger.info("Checkpoint saved")
    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        sampler_wrapper.save_checkpoint(sampler)

if __name__ == "__main__":
    main()