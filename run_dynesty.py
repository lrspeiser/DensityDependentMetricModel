#!/usr/bin/env python3
"""
run_dynesty.py - Run dynesty dynamic nested sampling on the Density-Metric model for the Milky Way.
Saves posterior samples to specified output. Includes self-tests and advanced progress logging.
"""
import logging
import sys
import time
import numpy as np
import argparse
import os
from pathlib import Path
import pickle
import gzip
from multiprocessing import Pool, freeze_support
from datetime import timedelta

import matplotlib.pyplot as plt
import corner

try:
    from dynesty import DynamicNestedSampler, utils as dyfunc
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    print("CRITICAL: Dynesty library not found. Please install it: pip install dynesty")
    sys.exit(1)

try:
    from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP, run_physics_self_tests
    from data_io import load_gaia
    from main2 import get_param_labels_and_bounds as get_param_config_main_module
except ImportError as e:
    print(f"CRITICAL: Could not import local modules: {e}")
    sys.exit(1)

logger = None

# --- Parameter Configuration (Moved here to be self-contained) ---
MW_MULTI_COMP_PARAM_CONFIG = {
    'rho_c_solar_kpc3': {'label': r"$\rho_c$ ($M_\odot/kpc^3$)", 'fixed_val_from_arg': 'rho_c_fixed', 'default_fixed': 1e7, 'low': 1e5, 'high': 2e9, 'fit_flag_arg': 'fit_xi_params'},
    'n_exp': {'label': r"$n$", 'fixed_val_from_arg': 'n_exp_fixed', 'default_fixed': 1.5, 'low': 0.1, 'high': 4.0, 'fit_flag_arg': 'fit_xi_params'},
    'M_disk_thin_solar': {'label': r"$M_{d,thin}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thin_fixed', 'default_fixed': 4.0e10, 'low': 1e10, 'high': 8e10, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin'},
    'R_d_thin_kpc': {'label': r"$R_{d,thin}$ (kpc)", 'fixed_val_from_arg': 'R_d_thin_fixed', 'default_fixed': 2.5, 'low': 1.5, 'high': 5.0, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin'},
    'h_z_thin_kpc': {'label': r"$h_{z,thin}$ (kpc)", 'fixed_val_from_arg': 'h_z_thin_fixed', 'default_fixed': 0.3, 'low': 0.15, 'high': 0.45, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin'},
    'M_disk_thick_solar': {'label': r"$M_{d,thick}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thick_fixed', 'default_fixed': 1.0e10, 'low': 0.1e10, 'high': 5e10, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick'},
    'R_d_thick_kpc': {'label': r"$R_{d,thick}$ (kpc)", 'fixed_val_from_arg': 'R_d_thick_fixed', 'default_fixed': 3.5, 'low': 2.0, 'high': 6.0, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick'},
    'h_z_thick_kpc': {'label': r"$h_{z,thick}$ (kpc)", 'fixed_val_from_arg': 'h_z_thick_fixed', 'default_fixed': 0.9, 'low': 0.5, 'high': 1.5, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick'},
    'M_bulge_solar': {'label': r"$M_{bulge}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_bulge_fixed', 'default_fixed': 0.9e10, 'low': 0.1e10, 'high': 3.0e10, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge'},
    'a_bulge_kpc': {'label': r"$a_{bulge}$ (kpc)", 'fixed_val_from_arg': 'a_bulge_fixed', 'default_fixed': 0.5, 'low': 0.1, 'high': 2.0, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge'},
    'M_gas_solar': {'label': r"$M_{gas}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_gas_fixed', 'default_fixed': 1.0e10, 'low': 0.2e10, 'high': 2.0e10, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas'},
    'R_d_gas_kpc': {'label': r"$R_{d,gas}$ (kpc)", 'fixed_val_from_arg': 'R_d_gas_fixed', 'default_fixed': 7.0, 'low': 3.0, 'high': 15.0, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas'},
    'h_z_gas_kpc': {'label': r"$h_{z,gas}$ (kpc)", 'fixed_val_from_arg': 'h_z_gas_fixed', 'default_fixed': 0.15, 'low': 0.05, 'high': 0.5, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas'},
}


# --- Likelihood and Prior Transform Functions ---
def prior_transform_dynesty(u_array, fitted_param_names, prior_bounds_low, prior_bounds_high):
    if fitted_param_names is None or prior_bounds_low is None or prior_bounds_high is None:
        raise ValueError("prior_transform_dynesty received None for essential arguments.")
    params = np.empty_like(u_array)
    for i in range(len(fitted_param_names)):
        params[i] = prior_bounds_low[i] + u_array[i] * (prior_bounds_high[i] - prior_bounds_low[i])
    return params

def v_model_for_dynesty(R_kpc_array, p_all_params_dict, xi_type_str, ARGS_obj_dynesty):
    rho_c_solar_kpc3 = p_all_params_dict['rho_c_solar_kpc3']
    n_exp = p_all_params_dict['n_exp']
    if ARGS_obj_dynesty.fit_target == 'milkyway':
        v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict)
        rho_midplane_for_xi = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict)
    else:
        raise NotImplementedError("SPARC fitting not yet fully configured.")
    v_n_kms = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)
    rho_midplane_for_xi = np.nan_to_num(rho_midplane_for_xi, nan=0.0, posinf=1e10, neginf=0.0)
    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    xi_values = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp)
    xi_values = np.nan_to_num(xi_values, nan=1.0, posinf=1e10, neginf=0.0)
    xi_values_safe = np.maximum(xi_values, 0.0)
    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
    return v_mod_kms

def log_likelihood_dynesty(theta_values_fitted, fitted_param_names, args_dynesty_obj,
                           all_param_info_list, R_data, v_data, sigma_data, xi_type):
    if any(arg is None for arg in locals().values()): return -np.inf, [np.inf]
    current_params_full_dict = dict(zip(fitted_param_names, theta_values_fitted))
    if all_param_info_list:
        for p_info in all_param_info_list:
            if not p_info['is_fitted']: current_params_full_dict[p_info['name']] = p_info['current_val']
        if args_dynesty_obj.fit_target == 'milkyway':
            for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
                if 'include_flag_arg' in p_details_cfg:
                    current_params_full_dict[p_details_cfg['include_flag_arg']] = getattr(args_dynesty_obj, p_details_cfg['include_flag_arg'])
            current_params_full_dict['include_bulge_density'] = args_dynesty_obj.include_bulge
    v_predicted = v_model_for_dynesty(R_data, current_params_full_dict, xi_type, args_dynesty_obj)
    if not np.all(np.isfinite(v_predicted)): return -np.inf, [np.inf]
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals = v_data - v_predicted
    rmse = np.sqrt(np.mean(residuals**2))
    chi_squared_terms = (residuals / sigma_data_safe)**2
    log_L_terms = chi_squared_terms + np.log(2 * np.pi * sigma_data_safe**2)
    if not np.all(np.isfinite(log_L_terms)): return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    log_L = -0.5 * np.sum(log_L_terms)
    if not np.isfinite(log_L): return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    return log_L, [rmse]

def _check_flag_consistency(args, logger_obj):
    logger_obj.info("Performing CLI flag consistency check...")
    components_to_check = {
        "xi_params": ["rho_c_fixed", "n_exp_fixed"],
        "disk_thin": ["M_disk_thin_fixed", "R_d_thin_fixed", "h_z_thin_fixed"],
        "disk_thick": ["M_disk_thick_fixed", "R_d_thick_fixed", "h_z_thick_fixed"],
        "bulge": ["M_bulge_fixed", "a_bulge_fixed"],
        "gas": ["M_gas_fixed", "R_d_gas_fixed", "h_z_gas_fixed"]
    }
    inconsistent_flags_found = False
    for comp_name, fixed_arg_names in components_to_check.items():
        fit_flag_name = f"fit_{comp_name}"
        if not hasattr(args, fit_flag_name): continue
        fit_flag_is_set = getattr(args, fit_flag_name, False)
        if fit_flag_is_set:
            for fixed_name in fixed_arg_names:
                if f"--{fixed_name}" in sys.argv:
                    logger_obj.error(f"Inconsistent CLI: --{fit_flag_name} and --{fixed_name} provided.")
                    inconsistent_flags_found = True
    if inconsistent_flags_found:
        raise ValueError("Inconsistent command-line arguments detected. Aborting.")
    logger_obj.info("CLI flag consistency OK.")

def get_param_labels_and_bounds(ARGS):
    param_info_list = []
    config_to_use = MW_MULTI_COMP_PARAM_CONFIG
    logger.info("Configuring parameters for NEW multi-component Milky Way model.")
    for p_name, p_details in config_to_use.items():
        is_included = 'include_flag_arg' not in p_details or getattr(ARGS, p_details['include_flag_arg'], False)
        if not is_included: continue
        is_fitted = 'fit_flag_arg' in p_details and getattr(ARGS, p_details['fit_flag_arg'], False)
        current_val = getattr(ARGS, p_details['fixed_val_from_arg'])
        param_info_list.append({'name': p_name, 'label': p_details['label'], 'current_val': current_val,
                                'low': p_details['low'], 'high': p_details['high'], 'is_fitted': is_fitted})
    ARGS.all_param_info_list = param_info_list
    fitted_params_info = [p for p in param_info_list if p['is_fitted']]
    if not fitted_params_info: 
        logger.error("No parameters configured to be fitted! You must use at least one --fit_* flag.")
        sys.exit(1)
    return ([p['name'] for p in fitted_params_info], [p['label'] for p in fitted_params_info],
            np.array([p['current_val'] for p in fitted_params_info]),
            np.array([p['low'] for p in fitted_params_info]), np.array([p['high'] for p in fitted_params_info]))

def main_dynesty():
    global logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger = logging.getLogger("run_dynesty")
    logger.info("Starting main_dynesty function.")

    if not DYNESTY_AVAILABLE: logger.error("Dynesty library not found."); sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Run Dynesty for Density-Metric model.")
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'])
    parser.add_argument('--max_sample_gaia', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default="chains_dynesty")
    parser.add_argument('--nlive_init', type=int, default=800)
    parser.add_argument('--nlive_batch', type=int, default=200, help="Number of live points to add per batch.")
    parser.add_argument('--dlogz_target', type=float, default=0.01, help="Target dlogz for stopping.")
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--update_interval', type=float, default=0.6, help="Dynesty update_interval.")
    parser.add_argument('--maxcall', type=int, default=2000000, help="Hard limit on likelihood calls.")
    parser.add_argument('--progress_update_interval_s', type=int, default=60, help="Interval in seconds for printing custom progress.")
    parser.add_argument('--debug_likelihood_params', type=str, default=None, help="Comma-separated physical parameters to test likelihood function.")

    mw_model_g = parser.add_argument_group('Milky Way Model Configuration')
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True)
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)
    
    fit_g = parser.add_argument_group('Fitting Flags (specify what to fit)')
    fit_g.add_argument('--fit_xi_params', action='store_true', help="Fit the xi function parameters (rho_c, n_exp).")
    fit_g.add_argument('--fit_disk_thin', action='store_true', help="Fit the thin disk parameters.")
    fit_g.add_argument('--fit_disk_thick', action='store_true', help="Fit the thick disk parameters.")
    fit_g.add_argument('--fit_bulge', action='store_true', help="Fit the bulge parameters.")
    fit_g.add_argument('--fit_gas', action='store_true', help="Fit the gas disk parameters.")

    fixed_g = parser.add_argument_group('Fixed Value Arguments (used if not fitting)')
    for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
        fixed_g.add_argument(f"--{p_details_cfg['fixed_val_from_arg']}", type=float, default=p_details_cfg['default_fixed'],
                                help=f"Fixed/initial value for {p_name_cfg}.")
    
    args = parser.parse_args()
    
    _check_flag_consistency(args, logger)
    run_physics_self_tests()

    args.fit_target = 'milkyway'
    gaia_data_dict = load_gaia(sample_max=args.max_sample_gaia)
    if gaia_data_dict is None: logger.error("Failed to load Gaia data."); sys.exit(1)
    R_data_for_run, v_data_for_run, sigma_data_for_run = gaia_data_dict["R_kpc"], gaia_data_dict["v_obs"], gaia_data_dict["sigma_v"]
    logger.info(f"Loaded {len(R_data_for_run)} Gaia data points.")

    fitted_p_names, fitted_p_labels, _, p_low, p_high = get_param_labels_and_bounds(args)
    ndim_dynesty = len(fitted_p_names)
    logger.info(f"Dynesty fitting {ndim_dynesty} parameters: {fitted_p_names}")
    
    ptform_args_tuple = (fitted_p_names, np.array(p_low), np.array(p_high))
    logl_args_tuple = (fitted_p_names, args, args.all_param_info_list, R_data_for_run, v_data_for_run, sigma_data_for_run, args.xi)

    pool_obj, queue_size_for_sampler = None, None
    if args.num_threads > 1:
        try:
            pool_obj = Pool(args.num_threads)
            queue_size_for_sampler = args.num_threads
            logger.info(f"Dynesty will run with {args.num_threads} threads.")
        except Exception as e:
            logger.warning(f"Failed to create Pool: {e}. Running serially.")
    
    sampler = DynamicNestedSampler(log_likelihood_dynesty, prior_transform_dynesty, ndim_dynesty,
                                   pool=pool_obj, queue_size=queue_size_for_sampler,
                                   ptform_args=ptform_args_tuple, logl_args=logl_args_tuple,
                                   blob=True)
    
    run_start_time = time.time()
    last_progress_log_time = time.time()
    best_rmse_so_far = np.inf
    
    try:
        logger.info(f"Running initial sampling with nlive_init = {args.nlive_init}...")
        for _ in sampler.sample_initial(nlive=args.nlive_init, maxcall=args.maxcall, save_samples=True):
            if time.time() - last_progress_log_time > args.progress_update_interval_s:
                last_progress_log_time = time.time()
                logger.info(f"Initial Sampling | Calls: {sampler.results.ncall}/{args.maxcall} | Live: {len(sampler.live_logl)}")

        logger.info("Initial sampling complete. Starting batch processing...")
        
        while sampler.results.ncall < args.maxcall:
            stop_val, _ = sampler.stopping_function(sampler.results)
            if stop_val < args.dlogz_target:
                logger.info(f"Stopping criterion met: dlogz ({stop_val:.4f}) < target ({args.dlogz_target:.4f}).")
                break
            
            sampler.add_batch(nlive=args.nlive_batch, maxcall=args.maxcall, save_samples=True)
            
            if time.time() - last_progress_log_time > args.progress_update_interval_s:
                last_progress_log_time = time.time()
                res = sampler.results
                
                if res.blob is not None and len(res.blob) > 0:
                    all_rmses = np.array([b[0] for b in res.blob if b])
                    finite_rmses = all_rmses[np.isfinite(all_rmses)]
                    if len(finite_rmses) > 0:
                         best_rmse_so_far = np.nanmin(finite_rmses)
                
                eta_str = "N/A"
                elapsed_time = time.time() - run_start_time
                if res.ncall > args.nlive_init and elapsed_time > 1:
                    rate = res.ncall / elapsed_time
                    remaining_calls = args.maxcall - res.ncall
                    if rate > 0 and remaining_calls > 0: eta_str = str(timedelta(seconds=int(remaining_calls/rate)))

                logz_str = f"{res.logz[-1]:.2f}"
                if not np.isfinite(res.logz[-1]): logz_str = f"WARNING: {res.logz[-1]}"

                logger.info(
                    f"Progress | Calls: {res.ncall}/{args.maxcall} | dlogz: {stop_val:.4f} "
                    f"| logZ: {logz_str} | Best RMSE so far: {best_rmse_so_far:.2f} km/s | ETA: {eta_str}"
                )
        
        logger.info("Sampling loop finished.")

    finally:
        if pool_obj:
            logger.info("Closing and joining multiprocessing Pool.")
            pool_obj.close(); pool_obj.join()

    res = sampler.results
    tdelta = time.time() - run_start_time
    logger.info("Dynesty run complete in %.1f min (%.2f hr).", tdelta / 60, tdelta / 3600)
    
    logZ_final, logZerr_final = res.logz[-1], res.logzerr[-1]
    
    try: ess = res.effective_sample_size
    except AttributeError:
        weights = np.exp(res.logwt - res.logz[-1]); ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0
    
    logger.info(f"Effective Sample Size (ESS): {ess:.0f}")

    if not np.isfinite(logZ_final):
        logger.error(f"FINAL LOGZ IS PROBLEMATIC: {logZ_final:.2f}. Fit likely failed.")
    else:
        logger.info("log(Z) = %.2f +/- %.2f (evidence)", logZ_final, logZerr_final)

    output_fname_parts = ["dynesty_mw", args.xi]
    if args.include_bulge: output_fname_parts.append("B" + ("f" if args.fit_bulge else "x"))
    if args.include_disk_thin: output_fname_parts.append("DT" + ("f" if args.fit_disk_thin else "x"))
    if args.include_disk_thick: output_fname_parts.append("DK" + ("f" if args.fit_disk_thick else "x"))
    if args.include_gas: output_fname_parts.append("G" + ("f" if args.fit_gas else "x"))
    output_basename = "_".join(output_fname_parts)
    
    output_npz_file = Path(args.output_dir) / f"{output_basename}_samples.npz"
    np.savez(output_npz_file, samples=res.samples, weights=np.exp(res.logwt - res.logz[-1]),
             logl=res.logl, logz=res.logz, logzerr=res.logzerr, ess=ess, blob=res.blob)
    logger.info(f"Results saved to {output_npz_file}")

    output_pickle_file = Path(args.output_dir) / f"{output_basename}_results.pkl.gz"
    try:
        with gzip.open(output_pickle_file, "wb") as fh: pickle.dump(res, fh)
        logger.info(f"Full Dynesty results object saved to {output_pickle_file}")
    except Exception as e: logger.error(f"Failed to save full results object: {e}")

    if len(res.samples) > 0 and ndim_dynesty > 0 and np.isfinite(ess) and ess > ndim_dynesty * 10:
        logger.info("Generating corner plot.")
        samples_eq = dyfunc.resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
        fig = corner.corner(samples_eq, labels=fitted_p_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 10})
        corner_plot_file = Path(args.output_dir) / f"{output_basename}_corner.png"
        fig.savefig(corner_plot_file); plt.close(fig)
        logger.info(f"Corner plot saved to {corner_plot_file}")
    else:
        logger.warning(f"Skipping corner plot. Insufficient ESS ({ess:.0f} vs {ndim_dynesty*10} needed).")
    
    logger.info("main_dynesty function finished.")


if __name__ == "__main__":
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL (main entry): Dynesty library not found.")
        sys.exit(1)
    
    main_dynesty()