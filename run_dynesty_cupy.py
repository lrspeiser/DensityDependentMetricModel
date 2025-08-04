#!/usr/bin/env python3
"""
run_dynesty_cupy.py - CuPy-optimized dynamic nested sampling for the Density-Metric model.
"""

import logging
import sys
import numpy as np
import argparse
from pathlib import Path
import pickle
import json
import threading, time
from multiprocessing import Pool, freeze_support

# CuPy imports for GPU acceleration
import cupy as cp
from density_metric_cupy import v_total_kms_cupy, to_cupy_array, to_numpy_array

# Resource monitoring
try:
    from resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_logger():
    """Get or create logger instance."""
    logger = logging.getLogger('run_dynesty_cupy')
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def check_physical_plausibility(theta_values, param_names):
    """Check if parameters are physically plausible."""
    for i, name in enumerate(param_names):
        if 'mass' in name.lower() and theta_values[i] < 0: return False
        if 'radius' in name.lower() and theta_values[i] <= 0: return False
    return True

# ============================================================================
# CORE LIKELIHOOD AND PRIOR FUNCTIONS (PROCESS-SAFE)
# ============================================================================

def log_likelihood_dynesty_cupy(theta, param_names, args, R_data, v_data, sigma_data):
    """
    Process-safe log-likelihood. Accepts NumPy arrays, uses CuPy internally.
    Based on the successful debug_cupy_parallel.py script.
    """
    try:
        # Log first few calls to verify function is working
        if hasattr(log_likelihood_dynesty_cupy, 'call_count'):
            log_likelihood_dynesty_cupy.call_count += 1
        else:
            log_likelihood_dynesty_cupy.call_count = 1
            
        if log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD CALL #{log_likelihood_dynesty_cupy.call_count}: theta={theta}, params={param_names}")
        
        if not check_physical_plausibility(theta, param_names):
            return -np.inf

        # Convert to CuPy arrays for GPU computation
        R_data_cupy = to_cupy_array(R_data)
        v_data_cupy = to_cupy_array(v_data)
        sigma_data_cupy = to_cupy_array(sigma_data)
        
        params = dict(zip(param_names, theta))

        v_model = v_total_kms_cupy(R_data_cupy, params, xi_type=args.xi)
        if not cp.all(cp.isfinite(v_model)):
            return -np.inf

        chi2 = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
        logl = -0.5 * float(chi2)

        rmse = float(cp.sqrt(cp.mean((v_data_cupy - v_model)**2)))
        
        if log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD RESULT #{log_likelihood_dynesty_cupy.call_count}: logl={logl:.2f}, rmse={rmse:.2f}")
        
        return logl
    except Exception as e:
        if log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD ERROR #{log_likelihood_dynesty_cupy.call_count}: {e}")
        return -np.inf

def prior_transform_dynesty_cupy(u, param_names, bounds_low, bounds_high, use_log_prior):
    """Prior transform. Accepts and returns NumPy arrays."""
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

def load_gaia_data():
    """Load or generate data, returning NumPy arrays."""
    R_data = np.linspace(1.0, 20.0, 1000)
    v_data = 200 + 50 * np.exp(-R_data / 8.0)
    sigma_data = 10.0 * np.ones_like(R_data)
    return R_data, v_data, sigma_data

def setup_parameter_bounds(xi_type):
    """Setup parameter bounds, returning NumPy arrays."""
    if xi_type == 'gr':
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e9, 0.1])
        bounds_high = np.array([1e12, 10.0])
        use_log_prior = np.array([True, False])
    else: # Fallback
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e9, 0.1])
        bounds_high = np.array([1e12, 10.0])
        use_log_prior = np.array([True, False])
    return param_names, bounds_low, bounds_high, use_log_prior

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_cupy():
    """Main function for the dynesty run."""
    logger = get_or_create_logger()
    logger.info("Starting CuPy-optimized dynesty run...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--xi', type=str, default='gr')
    parser.add_argument('--output_dir', type=str, default='cupy_results')
    parser.add_argument('--nlive', type=int, default=500)
    parser.add_argument('--maxcall', type=int, default=200000)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--checkpoint_every', type=int, default=900, help='Seconds between automatic checkpoints')
    # Post-processing flags
    parser.add_argument('--run_analysis', action='store_true', help='Run analyze_results.py on the final .npz')
    parser.add_argument('--run_validation', action='store_true', help='Run validate_ddmm.py on the final .npz')
    parser.add_argument('--run_plots', action='store_true', help='Run generate_paper_figures.py for quick plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Setup Data and Parameters (as NumPy) ---
    R_data, v_data, sigma_data = load_gaia_data()
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    logger.info(f"Fitting {len(param_names)} parameters: {param_names}")

    # --- Prepare arguments for parallel execution ---
    logl_args = (param_names, args, R_data, v_data, sigma_data)
    ptform_args = (param_names, bounds_low, bounds_high, use_log_prior)

    # --- Resource Monitoring ---
    resource_monitor = None
    if RESOURCE_MONITOR_AVAILABLE:
        resource_monitor = ResourceMonitor(output_dir)
        resource_monitor.start_monitoring()

    try:
        logger.info("STEP 1: Importing dynesty...")
        import dynesty
        logger.info(f"✓ Dynesty imported successfully (version: {dynesty.__version__})")
        
        logger.info(f"STEP 2: Creating multiprocessing pool with {args.num_threads} threads...")
        with Pool(processes=args.num_threads) as pool:
            logger.info("STEP 3: Creating DynamicNestedSampler...")
            logger.info(f"  - ndim: {len(param_names)}")
            logger.info(f"  - nlive: {args.nlive}")
            logger.info(f"  - queue_size: {args.num_threads}")
            logger.info(f"  - logl_args length: {len(logl_args)}")
            logger.info(f"  - ptform_args length: {len(ptform_args)}")
            
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood_dynesty_cupy,
                prior_transform_dynesty_cupy,
                ndim=len(param_names),
                logl_args=logl_args,
                ptform_args=ptform_args,
                pool=pool,
                queue_size=args.num_threads,
                nlive=args.nlive
            )
            logger.info("✓ DynamicNestedSampler created successfully")
            logger.info("STEP 4: Setting up checkpoint thread...")
            stop_event = threading.Event()
            def _checkpoint_worker():
                while not stop_event.wait(args.checkpoint_every):
                    try:
                        with open(output_dir / "dynesty_checkpoint.pkl", "wb") as cf:
                            pickle.dump(sampler.results, cf)
                        logger.info("✓ Checkpoint saved.")
                    except Exception as e:
                        logger.warning(f"✗ Checkpoint save failed: {e}")
            chk_thread = threading.Thread(target=_checkpoint_worker, daemon=True)
            chk_thread.start()
            logger.info("✓ Checkpoint thread started")

            logger.info("STEP 5: Starting nested sampling...")
            logger.info(f"  - maxcall: {args.maxcall}")
            logger.info(f"  - print_progress: True")
            logger.info("  - Calling sampler.run_nested()...")
            
            try:
                sampler.run_nested(maxcall=args.maxcall, print_progress=True)
                logger.info("✓ sampler.run_nested() completed successfully")
            except Exception as e:
                logger.error(f"✗ sampler.run_nested() failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error args: {e.args}")
                raise
            
            logger.info("STEP 6: Saving progress tracking...")
            try:
                # Helper function to convert numpy types to JSON-serializable
                def to_json_serializable(obj):
                    if hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    else:
                        return obj
                
                progress_data = {
                    "iterations": len(sampler.results.logz),
                    "logz": to_json_serializable(sampler.results.logz),
                    "logzerr": to_json_serializable(sampler.results.logzerr) if hasattr(sampler.results, 'logzerr') else [],
                    "ncall": to_json_serializable(sampler.results.ncall) if hasattr(sampler.results, 'ncall') else None,
                    "efficiency": to_json_serializable(getattr(sampler.results, 'efficiency', None)),
                    "final_logz": to_json_serializable(sampler.results.logz[-1]),
                    "final_logzerr": to_json_serializable(sampler.results.logzerr[-1]) if hasattr(sampler.results, 'logzerr') else None
                }
                with open(output_dir / "dynesty_progress.json", "w") as f:
                    json.dump(progress_data, f, indent=2)
                logger.info("✓ Saved dynesty_progress.json")
            except Exception as e:
                logger.error(f"✗ Failed to save progress: {e}")
                raise

            # Stop checkpoint thread after sampling completes
            stop_event.set()
            chk_thread.join(timeout=2)
        
        logger.info("STEP 7: Saving final results...")
        results = sampler.results
        logger.info(f"  - Results type: {type(results)}")
        logger.info(f"  - Available attributes: {dir(results)}")
        
        try:
            with open(output_dir / "results.pkl", "wb") as f:
                pickle.dump(results, f)
            logger.info("✓ Saved results.pkl")
        except Exception as e:
            logger.error(f"✗ Failed to save results.pkl: {e}")
            raise
        logger.info("STEP 8: Saving posterior samples...")
        try:
            # Extract weights (not always present as an attribute)
            if 'weights' in results:
                weights_arr = results['weights']
                logger.info("  - Using results['weights']")
            elif hasattr(results, 'weights'):
                weights_arr = results.weights
                logger.info("  - Using results.weights")
            else:
                # Compute importance weights from logwt and final evidence
                weights_arr = np.exp(results['logwt'] - results['logz'][-1])
                logger.info("  - Computed weights from logwt")
            
            logger.info(f"  - Weights shape: {weights_arr.shape}")
            logger.info(f"  - Samples shape: {results['samples'].shape}")
            logger.info(f"  - LogL shape: {results['logl'].shape}")
            
            np.savez(
                output_dir / "posterior_samples.npz",
                samples=results['samples'],
                logl=results['logl'],
                weights=weights_arr,
                logz=results['logz'][-1],
                dlogz=(results['dlogz'][-1] if 'dlogz' in results else (
                    results.logzerr[-1] if hasattr(results, 'logzerr') else np.nan
                ))
            )
            logger.info("✓ Saved posterior_samples.npz")
        except Exception as e:
            logger.error(f"✗ Failed to save posterior_samples.npz: {e}")
            raise
        logger.info(f"Sampling completed! LogZ = {results.logz[-1]:.2f}")

        # Optional post-processing ------------------------------------------------
        posterior_npz = output_dir / "posterior_samples.npz"
        if args.run_analysis:
            try:
                import analyze_results as ar
                # Fix: analyze_results.main() doesn't take arguments
                import sys
                original_argv = sys.argv
                sys.argv = ['analyze_results.py', str(posterior_npz)]
                ar.main()
                sys.argv = original_argv
                logger.info("analyze_results.py finished.")
            except Exception as e:
                logger.error(f"analyze_results.py failed: {e}")
        if args.run_validation:
            try:
                import validate_ddmm as vd
                vd.main([str(posterior_npz)])
                logger.info("validate_ddmm.py finished.")
            except Exception as e:
                logger.error(f"validate_ddmm.py failed: {e}")
        if args.run_plots:
            try:
                import sys
                sys.path.append("Older Files")
                import generate_paper_figures as gpf
                # Execute the main block directly
                exec(open("Older Files/generate_paper_figures.py").read())
                logger.info("generate_paper_figures.py finished.")
            except Exception as e:
                logger.error(f"generate_paper_figures.py failed: {e}")

    except Exception as e:
        logger.error(f"FATAL: Sampling failed: {e}", exc_info=True)
    finally:
        if resource_monitor:
            resource_monitor.stop_monitoring()

if __name__ == "__main__":
    freeze_support()
    main_cupy()
