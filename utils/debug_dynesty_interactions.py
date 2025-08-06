#!/usr/bin/env python3
"""
debug_dynesty_interactions.py - A script to diagnose the interaction between dynesty and the likelihood function.
Each worker process will write to its own log file to trace the returned values.
"""

import logging
import sys
import numpy as np
import argparse
from pathlib import Path
import pickle
import json
from multiprocessing import Pool, freeze_support, current_process
import os

import cupy as cp
from density_metric_cupy import v_total_kms_cupy, to_cupy_array

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_worker_logger(output_dir):
    """Creates a logger for a specific worker process."""
    worker_name = current_process().name
    logger = logging.getLogger(worker_name)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    # Ensure the output directory for logs exists
    log_path = output_dir / "worker_logs"
    log_path.mkdir(exist_ok=True)
    
    # Create a unique log file for this process
    handler = logging.FileHandler(log_path / f"{worker_name}.log", mode='w')
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def check_physical_plausibility(theta_values, param_names):
    for i, name in enumerate(param_names):
        if 'mass' in name.lower() and theta_values[i] < 0: return False
        if 'radius' in name.lower() and theta_values[i] <= 0: return False
    return True

# ============================================================================
# CORE LIKELIHOOD AND PRIOR FUNCTIONS (WITH HEAVY DEBUGGING)
# ============================================================================

def log_likelihood_for_debugging(theta, param_names, args, R_data, v_data, sigma_data, output_dir):
    """
    This version of the likelihood function writes its every move to a process-specific log file.
    """
    logger = get_worker_logger(output_dir)
    log_prefix = f"theta={theta}:"

    try:
        if not check_physical_plausibility(theta, param_names):
            logl = -np.inf
            blob = np.array([np.inf] * 5, dtype=np.float64)
            logger.warning(f"{log_prefix} PHYSICALITY FAIL -> Returning logl={logl}, blob={blob}")
            return logl, blob

        R_data_cupy = to_cupy_array(R_data)
        v_data_cupy = to_cupy_array(v_data)
        sigma_data_cupy = to_cupy_array(sigma_data)
        
        params = dict(zip(param_names, theta))

        v_model = v_total_kms_cupy(R_data_cupy, params, xi_type=args.xi)
        if not cp.all(cp.isfinite(v_model)):
            logl = -np.inf
            blob = np.array([np.inf] * 5, dtype=np.float64)
            logger.warning(f"{log_prefix} V_MODEL NON-FINITE -> Returning logl={logl}, blob={blob}")
            return logl, blob

        chi2 = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
        logl = -0.5 * float(chi2)

        rmse = float(cp.sqrt(cp.mean((v_data_cupy - v_model)**2)))
        blob = np.array([rmse, 0., 0., 0., 0.], dtype=np.float64)
        
        logger.info(f"{log_prefix} SUCCESS -> Returning logl={logl} (type {type(logl)}), blob={blob} (shape {blob.shape}, type {type(blob)})")
        return logl, blob

    except Exception as e:
        logger.error(f"{log_prefix} EXCEPTION -> Returning -inf. Error: {e}", exc_info=True)
        logl = -np.inf
        blob = np.array([np.inf] * 5, dtype=np.float64)
        return logl, blob

def prior_transform(u, param_names, bounds_low, bounds_high, use_log_prior):
    theta = np.zeros_like(u)
    for i in range(len(u)):
        if use_log_prior[i]:
            theta[i] = 10**(np.log10(bounds_low[i]) + u[i] * (np.log10(bounds_high[i]) - np.log10(bounds_low[i])))
        else:
            theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
    return theta

# ============================================================================
# DATA AND PARAMETER SETUP
# ============================================================================

def load_data():
    R_data = np.linspace(1.0, 20.0, 1000)
    v_data = 200 + 50 * np.exp(-R_data / 8.0)
    sigma_data = 10.0 * np.ones_like(R_data)
    return R_data, v_data, sigma_data

def setup_params(xi_type):
    param_names = ['M_disk_solar', 'R_d_kpc']
    bounds_low = np.array([1e9, 0.1])
    bounds_high = np.array([1e12, 10.0])
    use_log_prior = np.array([True, False])
    return param_names, bounds_low, bounds_high, use_log_prior

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for the dynesty run."""
    # Use a main logger for the controlling process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | [MainProcess] | %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--xi', type=str, default='gr')
    parser.add_argument('--output_dir', type=str, default='debug_dynesty_run')
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--num_threads', type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "worker_logs").mkdir(exist_ok=True)
    logging.info(f"Output and worker logs will be in: {output_dir}")

    R_data, v_data, sigma_data = load_data()
    param_names, bounds_low, bounds_high, use_log_prior = setup_params(args.xi)
    logging.info(f"Fitting {len(param_names)} parameters: {param_names}")

    # Pass the output_dir to the likelihood function for logging
    logl_args = (param_names, args, R_data, v_data, sigma_data, output_dir)
    ptform_args = (param_names, bounds_low, bounds_high, use_log_prior)

    try:
        import dynesty
        logging.info(f"Starting dynesty with {args.num_threads} threads...")
        with Pool(processes=args.num_threads) as pool:
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood_for_debugging,
                prior_transform,
                ndim=len(param_names),
                logl_args=logl_args,
                ptform_args=ptform_args,
                pool=pool,
                queue_size=args.num_threads,
                nlive=args.nlive
            )
            sampler.run_nested(print_progress=True)
        
        logging.info("Sampling completed successfully!")
        results = sampler.results
        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

    except Exception as e:
        logging.error(f"FATAL: Sampling failed during run_nested call.")
        logging.error(e, exc_info=True)
        logging.error("Please check the log files in the 'worker_logs' subdirectory for clues.")

if __name__ == "__main__":
    freeze_support()
    main()
