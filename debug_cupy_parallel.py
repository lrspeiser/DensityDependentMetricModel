
import numpy as np
import cupy as cp
import argparse
import logging
from multiprocessing import Pool, freeze_support
from typing import List, Dict, Tuple

# --- Functions copied from run_dynesty_cupy.py for isolated testing ---

def to_cupy_array(arr):
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr)

def v_total_kms_cupy(R_kpc, params, xi_type='gr'):
    # A simplified placeholder for v_total_kms_cupy
    M_disk = params.get('M_disk_solar', 5e10)
    R_d = params.get('R_d_kpc', 3.0)
    # Simplified Newtonian velocity calculation for testing
    G = 4.30091e-3 # (km/s)^2 pc/M_sun
    R_pc = R_kpc * 1000
    v_sq = G * M_disk / R_pc
    return cp.sqrt(v_sq)

def get_or_create_logger():
    """Get or create logger instance."""
    logger = logging.getLogger('debug_cupy_parallel')
    if not logger.handlers:
        # Use a specific handler for this logger to avoid conflicts
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(processName)s] %(levelname)-8s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def check_physical_plausibility(theta_values, param_names, args_obj):
    for i, name in enumerate(param_names):
        if 'mass' in name.lower() and theta_values[i] < 0: return False, "Negative mass"
        if 'radius' in name.lower() and theta_values[i] <= 0: return False, "Non-positive radius"
    return True, "OK"

def log_likelihood_for_debug(
    theta_values_fitted: np.ndarray,
    fitted_param_names: List[str],
    args_dynesty_obj: argparse.Namespace,
    R_data: np.ndarray,
    v_data: np.ndarray,
    sigma_data: np.ndarray,
    xi_type: str
) -> Tuple[float, np.ndarray]:
    """
    A version of the likelihood function with heavy internal logging for debugging.
    """
    logger = get_or_create_logger()
    log_prefix = f"theta={theta_values_fitted}:"

    try:
        # --- Internal CuPy Conversion ---
        R_data_cupy = to_cupy_array(R_data)
        v_data_cupy = to_cupy_array(v_data)
        sigma_data_cupy = to_cupy_array(sigma_data)

        params = dict(zip(fitted_param_names, theta_values_fitted))
        
        is_valid, reason, *_ = check_physical_plausibility(theta_values_fitted, fitted_param_names, args_dynesty_obj)
        if not is_valid:
            logger.warning(f"{log_prefix} Physical plausibility failed: {reason}")
            return -np.inf, np.array([np.inf] * 5, dtype=np.float64)

        v_model = v_total_kms_cupy(R_data_cupy, params, xi_type=xi_type)
        if not cp.all(cp.isfinite(v_model)):
            logger.warning(f"{log_prefix} v_model calculation returned non-finite values.")
            return -np.inf, np.array([np.inf] * 5, dtype=np.float64)

        chi2_total = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
        log_likelihood = -0.5 * float(chi2_total)
        rmse = float(cp.sqrt(cp.mean((v_data_cupy - v_model)**2)))
        
        # --- Type and Shape Logging ---
        logl_type = type(log_likelihood)
        blob = np.array([rmse, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        blob_type = type(blob)
        blob_shape = blob.shape
        
        logger.info(f"{log_prefix} SUCCESS -> logL={log_likelihood:.2f} (type: {logl_type}), blob_shape={blob_shape} (type: {blob_type})")

        return log_likelihood, blob

    except Exception as e:
        logger.error(f"{log_prefix} EXCEPTION CAUGHT: {e}", exc_info=True)
        return -np.inf, np.array([np.inf] * 5, dtype=np.float64)

def worker_task(args_tuple):
    """Wrapper function for pool.map to pass multiple arguments."""
    return log_likelihood_for_debug(*args_tuple)

def main():
    """Main diagnostic function."""
    logger = get_or_create_logger()
    logger.info("--- Starting Parallel Likelihood Debugger ---")

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to test')
    parser.add_argument('--num_calls', type=int, default=100, help='Number of likelihood calls to simulate')
    args = parser.parse_args()

    # --- 1. Set up data and parameters (as NumPy arrays) ---
    logger.info("Setting up test data and parameters...")
    R_data = np.linspace(1.0, 20.0, 1000)
    v_data = 200 + 50 * np.exp(-R_data / 8.0)
    sigma_data = 10 + 5 * np.exp(-R_data / 10.0)
    
    param_names = ['M_disk_solar', 'R_d_kpc']
    bounds_low = np.array([1e9, 0.1])
    bounds_high = np.array([1e12, 10.0])
    
    # --- 2. Generate random parameter vectors to test ---
    logger.info(f"Generating {args.num_calls} random parameter vectors...")
    np.random.seed(42)
    thetas_to_test = np.random.rand(args.num_calls, len(param_names))
    # Scale them to be within bounds for this test
    thetas_to_test[:, 0] = bounds_low[0] + thetas_to_test[:, 0] * (bounds_high[0] - bounds_low[0])
    thetas_to_test[:, 1] = bounds_low[1] + thetas_to_test[:, 1] * (bounds_high[1] - bounds_low[1])

    # --- 3. Prepare arguments for parallel mapping ---
    # Create a list of tuples, where each tuple contains all args for one call
    tasks = [(
        theta,
        param_names,
        args,  # The argparse namespace object
        R_data,
        v_data,
        sigma_data,
        'gr' # xi_type
    ) for theta in thetas_to_test]

    # --- 4. Run the parallel test ---
    logger.info(f"Starting parallel test with {args.num_threads} workers...")
    results = []
    with Pool(processes=args.num_threads) as pool:
        try:
            results = pool.map(worker_task, tasks)
        except Exception as e:
            logger.error(f"A fatal error occurred in the multiprocessing pool: {e}", exc_info=True)
            return

    # --- 5. Analyze the results ---
    logger.info("--- Analysis of Results ---")
    if not results:
        logger.error("No results were returned from the pool. A major error likely occurred.")
        return

    num_success = 0
    num_fail = 0
    inconsistent_logl_types = []
    inconsistent_blob_types = []
    inconsistent_blob_shapes = []

    # Expected types/shapes from a successful run
    expected_logl_type = float
    expected_blob_type = np.ndarray
    expected_blob_shape = (5,)

    for i, (logl, blob) in enumerate(results):
        is_consistent = True
        if logl > -np.inf:
            num_success += 1
            if not isinstance(logl, expected_logl_type):
                inconsistent_logl_types.append((i, type(logl)))
                is_consistent = False
            if not isinstance(blob, expected_blob_type):
                inconsistent_blob_types.append((i, type(blob)))
                is_consistent = False
            if blob.shape != expected_blob_shape:
                inconsistent_blob_shapes.append((i, blob.shape))
                is_consistent = False
        else:
            num_fail += 1

    logger.info(f"Total calls: {len(results)}")
    logger.info(f"Successful calls (finite logL): {num_success}")
    logger.info(f"Failed calls (-inf logL): {num_fail}")
    
    if inconsistent_logl_types or inconsistent_blob_types or inconsistent_blob_shapes:
        logger.error("!!!!!! INCONSISTENCY DETECTED !!!!!!")
        if inconsistent_logl_types:
            logger.error(f"Found {len(inconsistent_logl_types)} logL values with wrong type (expected {expected_logl_type}): {inconsistent_logl_types}")
        if inconsistent_blob_types:
            logger.error(f"Found {len(inconsistent_blob_types)} blobs with wrong type (expected {expected_blob_type}): {inconsistent_blob_types}")
        if inconsistent_blob_shapes:
            logger.error(f"Found {len(inconsistent_blob_shapes)} blobs with wrong shape (expected {expected_blob_shape}): {inconsistent_blob_shapes}")
    else:
        logger.info("✓✓✓ SUCCESS: All returned types and shapes are consistent.")

if __name__ == '__main__':
    freeze_support() # For Windows compatibility
    main()
