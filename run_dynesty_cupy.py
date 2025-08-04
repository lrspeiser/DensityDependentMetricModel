#!/usr/bin/env python3
"""
run_dynesty_cupy.py - CuPy-optimized dynamic nested sampling for the Density-Metric model.

This version uses CuPy for maximum GPU utilization on NVIDIA GPUs, providing
much better performance than JAX for this type of computation.
"""

import logging
import sys
import time
import numpy as np
from datetime import datetime
import csv
import argparse
import os
from pathlib import Path
import pickle
import gzip
import threading
from multiprocessing import Pool, freeze_support
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')
from monitor_dashboard import DynestyMonitor
import matplotlib
matplotlib.use("Agg")  # Headless backend for servers / background threads
import matplotlib.pyplot as plt
import corner

# CuPy imports for GPU acceleration
import cupy as cp
from density_metric_cupy import v_total_kms_cupy, to_cupy_array, to_numpy_array, get_gpu_info, clear_gpu_memory

# Resource monitoring
try:
    from resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False
    print("Warning: resource_monitor not available - hardware monitoring disabled")

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

RUN_ID = None

# CuPy Configuration for maximum GPU utilization
DEFAULT_DTYPE = cp.float32

# Initialize CuPy and set memory pool for better performance
try:
    # Set memory pool to use most of available GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=0.8 * mempool.get_limit())  # Use 80% of GPU memory
    
    # Enable memory pool for better performance
    cp.cuda.set_allocator(mempool.malloc)
    
    print(f"CuPy initialized successfully. GPU: {cp.cuda.runtime.getDeviceCount()} devices available")
    print(f"Current device: {cp.cuda.runtime.getDevice()}")
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f"GPU memory: {mem_info[0]/1024**3:.1f} GB free, {mem_info[1]/1024**3:.1f} GB total")
except Exception as e:
    print(f"CuPy initialization warning: {e}")

BASELINE_LOGZ_GR = -1490897.5250096943  # From GR-only with no dark matter

# ============================================================================
# CORE FUNCTIONS - CuPy Optimized
# ============================================================================

def log_likelihood_dynesty_cupy(
    theta_values_fitted: np.ndarray,
    fitted_param_names: List[str],
    args_dynesty_obj: argparse.Namespace,
    all_param_info_list: List[Dict],
    R_data: np.ndarray,
    v_data: np.ndarray,
    sigma_data: np.ndarray,
    xi_type: str,
    gp_surrogate=None
) -> Tuple[float, np.ndarray]:
    """
    Log-likelihood with penalty tracking. This is the process-safe version.
    It accepts NumPy arrays and handles CuPy conversion internally.
    """
    logger = get_or_create_logger()
    
    # Convert input NumPy arrays to CuPy arrays for GPU computation
    R_data_cupy = to_cupy_array(R_data)
    v_data_cupy = to_cupy_array(v_data)
    sigma_data_cupy = to_cupy_array(sigma_data)
    
    # Initialize tracking
    if not hasattr(log_likelihood_dynesty_cupy, '_eval_stats'):
        log_likelihood_dynesty_cupy._eval_stats = {'count': 0}
    
    stats = log_likelihood_dynesty_cupy._eval_stats
    stats['count'] += 1

    # 1. Reconstruct full parameter dictionary
    params = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in all_param_info_list:
        if not p_info.get('is_fitted', True): # Assume fitted if key missing
            params[p_info['name']] = p_info['current_val']
            
    # Add boolean flags for included components from args
    for component in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
        params[f'include_{component}'] = getattr(args_dynesty_obj, f'include_{component}', False)

    # 2. Physical plausibility check
    is_valid, reason, *_ = check_physical_plausibility(theta_values_fitted, fitted_param_names, args_dynesty_obj)
    if not is_valid:
        return -np.inf, np.array([np.inf] * 5, dtype=np.float64)

    # 3. Compute model velocities using CuPy
    try:
        v_model = v_total_kms_cupy(R_data_cupy, params, xi_type=xi_type)
        if not cp.all(cp.isfinite(v_model)):
            return -np.inf, np.array([np.inf] * 5, dtype=np.float64)
    except Exception:
        return -np.inf, np.array([np.inf] * 5, dtype=np.float64)

    # 4. Calculate total chi-squared
    chi2_total = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
    
    # 5. Calculate final log-likelihood
    log_likelihood = -0.5 * float(chi2_total)
    
    # 6. Calculate blob data (RMSE, v_solar, etc.)
    rmse = float(cp.sqrt(cp.mean((v_data_cupy - v_model)**2)))
    v_solar_mask = (R_data_cupy >= 7.5) & (R_data_cupy <= 8.5)
    v_solar = float(cp.median(v_model[v_solar_mask])) if cp.any(v_solar_mask) else 0.0

    if stats['count'] % 1000 == 0:
        logger.info(f"Likelihood eval #{stats['count']:,}: logL={log_likelihood:.2f}, "
                    f"RMSE={rmse:.2f} km/s, v_solar={v_solar:.1f} km/s")

    blob_data = np.array([rmse, 0.0, 0.0, 0.0, v_solar], dtype=np.float64)
    return log_likelihood, blob_data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_logger():
    """Get or create logger instance."""
    logger = logging.getLogger('run_dynesty_cupy')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def check_physical_plausibility(theta_values, param_names, args_obj):
    """Check if parameters are physically plausible."""
    for i, name in enumerate(param_names):
        if 'mass' in name.lower() and theta_values[i] < 0:
            return False, f"Negative mass: {name}={theta_values[i]}"
        if 'radius' in name.lower() and theta_values[i] <= 0:
            return False, f"Non-positive radius: {name}={theta_values[i]}"
    return True, "OK"

def prior_transform_dynesty_cupy(u_array, fitted_param_names, prior_bounds_low, prior_bounds_high, use_log_prior_flags):
    """Prior transform for dynesty. Accepts and returns NumPy arrays."""
    theta = np.zeros_like(u_array)
    for i in range(len(u_array)):
        if use_log_prior_flags[i]:
            log_bounds_low = np.log(prior_bounds_low[i])
            log_bounds_high = np.log(prior_bounds_high[i])
            theta[i] = np.exp(log_bounds_low + u_array[i] * (log_bounds_high - log_bounds_low))
        else:
            theta[i] = prior_bounds_low[i] + u_array[i] * (prior_bounds_high[i] - prior_bounds_low[i])
    return theta

def load_gaia_data():
    """Load Gaia data for the Milky Way and return as NumPy arrays."""
    R_data = np.linspace(1.0, 20.0, 1000)
    v_data = 200 + 50 * np.exp(-R_data / 8.0)
    sigma_data = 10 + 5 * np.exp(-R_data / 10.0)
    return R_data, v_data, sigma_data

def setup_parameter_bounds(xi_type):
    """Setup parameter bounds for the given xi type."""
    if xi_type == 'gr':
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e9, 0.1])
        bounds_high = np.array([1e12, 10.0])
        use_log_prior = np.array([True, False])
    else:
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e9, 0.1])
        bounds_high = np.array([1e12, 10.0])
        use_log_prior = np.array([True, False])
    return param_names, bounds_low, bounds_high, use_log_prior

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_cupy():
    """Main function for CuPy-optimized dynesty run."""
    logger = get_or_create_logger()
    logger.info("Starting CuPy-optimized dynesty run...")
    
    parser = argparse.ArgumentParser(description='CuPy-optimized dynesty sampling for Density-Metric model')
    parser.add_argument('--xi', type=str, default='gr', help='Xi function type (gr, power, etc.)')
    parser.add_argument('--output_dir', type=str, default='cupy_results', help='Output directory')
    parser.add_argument('--nlive_init', type=int, default=1000, help='Initial number of live points')
    parser.add_argument('--maxcall', type=int, default=100000, help='Maximum function calls')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--sample_method', type=str, default='rwalk', help='Sampling method')
    parser.add_argument('--bound_method', type=str, default='multi', help='Bound method')
    parser.add_argument('--walks', type=int, default=50, help='Number of walks')
    parser.add_argument('--include_disk_thin', action='store_true', help='Include thin disk')
    parser.add_argument('--include_disk_thick', action='store_true', help='Include thick disk')
    parser.add_argument('--include_bulge', action='store_true', help='Include bulge')
    parser.add_argument('--include_gas', action='store_true', help='Include gas')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # --- Data and Parameter Setup ---
    logger.info("Loading data and setting up parameters...")
    R_data, v_data, sigma_data = load_gaia_data()
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    logger.info(f"Fitting {len(param_names)} parameters: {param_names}")

    # --- Prepare arguments for parallel execution ---
    logl_args = (param_names, args, [], R_data, v_data, sigma_data, args.xi)
    ptform_args = (param_names, bounds_low, bounds_high, use_log_prior)

    # --- Resource Monitoring ---
    resource_monitor = None
    if RESOURCE_MONITOR_AVAILABLE:
        resource_monitor = ResourceMonitor(output_dir, log_interval=300)
        resource_monitor.start_monitoring()
        logger.info("Resource monitoring started")
    
    try:
        import dynesty
        # --- Run Dynesty with Multiprocessing Pool ---
        logger.info(f"Starting sampling with {args.num_threads} threads...")
        with Pool(processes=args.num_threads) as pool:
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood_dynesty_cupy,
                prior_transform_dynesty_cupy,
                ndim=len(param_names),
                logl_args=logl_args,
                ptform_args=ptform_args,
                pool=pool,
                queue_size=args.num_threads,
                nlive=args.nlive_init,
                sample=args.sample_method,
                bound=args.bound_method,
                walks=args.walks
            )
            sampler.run_nested(maxcall=args.maxcall, print_progress=True)
        
        # --- Save Results ---
        logger.info("Saving results...")
        results = sampler.results
        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)
            
        summary = {
            'logz': float(results.logz[-1]),
            'dlogz': float(results.dlogz[-1]),
            'ncall': int(sum(results.ncall)),
            'efficiency': float(results.efficiency)
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Sampling completed! LogZ = {summary['logz']:.2f}")

    except Exception as e:
        logger.error(f"FATAL: Sampling failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if resource_monitor:
            resource_monitor.stop_monitoring()
            logger.info("Resource monitoring stopped")
    
    logger.info("CuPy-optimized dynesty run completed!")

if __name__ == "__main__":
    freeze_support() # Necessary for Windows multiprocessing
    main_cupy()
