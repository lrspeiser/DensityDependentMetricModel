#!/usr/bin/env python3
"""
main.py - Main script to orchestrate MCMC fitting of density-dependent
             metric models to Gaia rotation curve data (Milky Way) or SPARC data (external galaxies).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import argparse
from tqdm import tqdm
import time
import os
import signal
import subprocess
import shlex
from multiprocessing import Pool, cpu_count, freeze_support
import platform
import logging

# Import from local modules
try:
    from data_io import load_gaia
    from density_metric import (
        v_newton_kms, volume_density_total_midplane_solar_kpc3, XI_FUNCTION_MAP,
        check_vertical_kinematics_Kz, calculate_microlensing_tau_baade, # Updated names
        R_SUN_KPC, SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2, TAU_MICRO_TARGET_BAADE_MAX, # Updated constants
        G_ASTRO_UNITS # Using G in astro units for v_N calculations
    )
    SPARC_AVAILABLE = False
    try:
        from sparc_io import load_single_sparc_galaxy, load_sparc_metadata, BASE_M_L_3_6_MICRON_DISK, BASE_M_L_3_6_MICRON_BULGE
        SPARC_AVAILABLE = True
        logging.info("sparc_io module loaded successfully.")
    except ImportError as e_sparc:
        logging.warning(f"sparc_io.py not found or has issues ({e_sparc}). SPARC galaxy fitting will not be available.")

except ImportError as e:
    logging.error(f"Error importing local modules: {e}")
    logging.error("Ensure data_io.py, density_metric.py (and optionally sparc_io.py) are in the same directory or Python path.")
    import sys
    sys.exit(1)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def kill_existing_instances(script_name_to_kill="main.py"):
    # (Content unchanged from previous version)
    current_pid = os.getpid()
    logger.info(f"🌬️  Attempting to terminate other instances of '{script_name_to_kill}' (current PID: {current_pid})...")
    try:
        ps_cmd = "ps aux"
        grep_python_cmd = "grep python"
        grep_script_cmd = f"grep {shlex.quote(script_name_to_kill)}"
        grep_exclude_cmd = "grep -v grep"
        p1 = subprocess.Popen(shlex.split(ps_cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(shlex.split(grep_python_cmd), stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p1.stdout: p1.stdout.close()
        p3 = subprocess.Popen(shlex.split(grep_script_cmd), stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p2.stdout: p2.stdout.close()
        p4 = subprocess.Popen(shlex.split(grep_exclude_cmd), stdin=p3.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p3.stdout: p3.stdout.close()
        result_stdout, result_stderr = p4.communicate()
        killed_count = 0
        if result_stdout:
            lines = result_stdout.decode().strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        if pid != current_pid:
                            logger.info(f"   Killing process {pid}: {' '.join(parts[10:])}")
                            os.kill(pid, signal.SIGKILL)
                            killed_count += 1
                            time.sleep(0.1)
                    except ValueError: pass
                    except ProcessLookupError: pass
                    except Exception as e_kill: logger.error(f"   Error killing process {pid} from line '{line}': {e_kill}")
        if killed_count > 0: logger.info(f"   ✅ Terminated {killed_count} other instance(s).")
        else: logger.info(f"   No other running instances of '{script_name_to_kill}' found to terminate.")
    except Exception as e: logger.warning(f"   ⚠️ Error during attempt to kill existing instances: {e}")
    logger.info("-" * 30)


def v_model_for_emcee(R_kpc_array, theta_params_dict, xi_type_str="power", ARGS_obj=None,
                      # For SPARC mode, pass the SPARC data dictionary:
                      sparc_galaxy_data_dict_full=None):
    """
    Calculates the model velocity.
    theta_params_dict contains *only the fitted parameters*.
    ARGS_obj provides fixed parameters and flags.
    sparc_galaxy_data_dict_full provides all pre-loaded SPARC data for the galaxy.
    """
    rho_c_solar_kpc3 = theta_params_dict['rho_c_solar_kpc3']
    n_exp = theta_params_dict['n_exp']

    if ARGS_obj.fit_target == 'sparc':
        # --- SPARC Mode ---
        if sparc_galaxy_data_dict_full is None:
            raise ValueError("SPARC mode requires sparc_galaxy_data_dict_full.")

        stellar_ML_factor = theta_params_dict.get('stellar_ML_factor', 1.0) # Default to 1 if not fitted

        # Scale SPARC V_disk and V_bulge components by sqrt(ML_factor)
        # V_gas is independent of stellar M/L
        v_disk_scaled_sq = (sparc_galaxy_data_dict_full['V_disk_comp_kms']**2) * stellar_ML_factor
        v_bulge_scaled_sq = (sparc_galaxy_data_dict_full['V_bulge_comp_kms']**2) * stellar_ML_factor
        v_gas_sq = sparc_galaxy_data_dict_full['V_gas_comp_kms']**2
        
        v_n_kms_sq = v_disk_scaled_sq + v_bulge_scaled_sq + v_gas_sq
        v_n_kms = np.sqrt(np.maximum(0, v_n_kms_sq))

        # Scale stellar part of rho_midplane by ML_factor
        rho_star_mid_scaled = sparc_galaxy_data_dict_full['rho_star_mid_Msun_kpc3_baseML'] * stellar_ML_factor
        rho_midplane_solar_kpc3 = rho_star_mid_scaled + sparc_galaxy_data_dict_full['rho_gas_mid_Msun_kpc3']

    elif ARGS_obj.fit_target == 'milkyway':
        # --- Milky Way Parametric Mode ---
        # Get parameters that are being fitted from theta_params_dict
        # Get fixed parameters from ARGS_obj
        M_disk_solar = theta_params_dict.get('M_disk_solar', ARGS_obj.M_disk_fixed)
        R_d_kpc = theta_params_dict.get('R_d_kpc', ARGS_obj.R_d_fixed)
        h_z_disk_kpc = theta_params_dict.get('h_z_disk_kpc', ARGS_obj.h_z_disk_fixed)

        M_bulge_solar, R_b_kpc = 0.0, 0.5 # Defaults if not included or fixed
        if ARGS_obj.include_bulge:
            M_bulge_solar = theta_params_dict.get('M_bulge_solar', ARGS_obj.M_bulge_fixed)
            R_b_kpc = theta_params_dict.get('R_b_kpc', ARGS_obj.R_b_fixed)

        M_gas_solar, R_gas_kpc, h_z_gas_kpc = 0.0, 7.0, 0.15 # Defaults
        if ARGS_obj.include_gas:
            M_gas_solar = theta_params_dict.get('M_gas_solar', ARGS_obj.M_gas_fixed)
            R_gas_kpc = theta_params_dict.get('R_gas_kpc', ARGS_obj.R_gas_fixed)
            h_z_gas_kpc = theta_params_dict.get('h_z_gas_kpc', ARGS_obj.h_z_gas_fixed)
        
        if R_d_kpc <= 1e-9: # Or M_disk_solar
            return np.zeros_like(R_kpc_array, dtype=float)

        v_n_kms = v_newton_kms(R_kpc_array,
                               M_disk_solar, R_d_kpc,
                               M_bulge_solar, R_b_kpc, ARGS_obj.include_bulge,
                               M_gas_solar, R_gas_kpc, ARGS_obj.include_gas)
        
        rho_midplane_solar_kpc3 = volume_density_total_midplane_solar_kpc3(
            R_kpc_array,
            M_disk_solar, R_d_kpc, h_z_disk_kpc,
            M_bulge_solar, R_b_kpc, ARGS_obj.include_bulge,
            M_gas_solar, R_gas_kpc, h_z_gas_kpc, ARGS_obj.include_gas
        )
    else:
        raise ValueError(f"Unknown fit_target: {ARGS_obj.fit_target}")

    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    if xi_func is None:
        raise ValueError(f"Unknown xi_type: {xi_type_str}. Available: {list(XI_FUNCTION_MAP.keys())}")

    xi_values = xi_func(rho_midplane_solar_kpc3, rho_c_solar_kpc3, n_exp)
    xi_values_safe = np.maximum(xi_values, 0.0)
    xi_values_safe = np.nan_to_num(xi_values_safe, nan=0.0)

    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
    return v_mod_kms


def get_param_labels_and_bounds(ARGS):
    param_info_list = [] # List of dicts: {name, label, guess, low, high, is_fitted}

    # Core density metric parameters (always fitted)
    param_info_list.append({'name': 'rho_c_solar_kpc3', 'label': r"$\rho_c$ ($M_\odot/kpc^3$)", 'guess': 1e7, 'low': 1e5, 'high': 1e9, 'is_fitted': True})
    param_info_list.append({'name': 'n_exp', 'label': r"$n$", 'guess': 1.0, 'low': 0.1, 'high': 4.0, 'is_fitted': True})

    if ARGS.fit_target == 'milkyway':
        param_info_list.append({'name': 'M_disk_solar', 'label': r"$M_\mathrm{disk}$ ($M_\odot$)", 'guess': 6e10, 'low': 1e10, 'high': 2.5e11, 'is_fitted': True}) # Increased upper M_disk bound
        param_info_list.append({'name': 'R_d_kpc', 'label': r"$R_d$ (kpc)", 'guess': 3.0, 'low': 1.5, 'high': 5.0, 'is_fitted': True})
        param_info_list.append({'name': 'h_z_disk_kpc', 'label': r"$h_z^\mathrm{disk}$ (kpc)", 'guess': 0.3, 'low': 0.1, 'high': 0.7, 'is_fitted': True})
        
        if ARGS.include_bulge:
            param_info_list.append({'name': 'M_bulge_solar', 'label': r"$M_\mathrm{bulge}$ ($M_\odot$)", 'guess': ARGS.M_bulge_fixed, 'low': 0.1e10, 'high': 2.0e10, 'is_fitted': ARGS.fit_bulge})
            param_info_list.append({'name': 'R_b_kpc', 'label': r"$R_b$ (kpc)", 'guess': ARGS.R_b_fixed, 'low': 0.1, 'high': 2.0, 'is_fitted': ARGS.fit_bulge})
        
        if ARGS.include_gas:
            param_info_list.append({'name': 'M_gas_solar', 'label': r"$M_\mathrm{gas}$ ($M_\odot$)", 'guess': ARGS.M_gas_fixed, 'low': 0.1e10, 'high': 3e10, 'is_fitted': ARGS.fit_gas})
            param_info_list.append({'name': 'R_gas_kpc', 'label': r"$R_g$ (kpc)", 'guess': ARGS.R_gas_fixed, 'low': 3.0, 'high': 15.0, 'is_fitted': ARGS.fit_gas})
            param_info_list.append({'name': 'h_z_gas_kpc', 'label': r"$h_z^\mathrm{gas}$ (kpc)", 'guess': ARGS.h_z_gas_fixed, 'low': 0.05, 'high': 0.5, 'is_fitted': ARGS.fit_gas})

    elif ARGS.fit_target == 'sparc':
        if ARGS.fit_sparc_ML: # Only add if fitting M/L
             param_info_list.append({'name': 'stellar_ML_factor', 'label': r"$(M/L)_* factor$", 'guess': 1.0, 'low': 0.2, 'high': 2.5, 'is_fitted': True})

    # Filter for fitted parameters
    fitted_params_info = [p for p in param_info_list if p['is_fitted']]
    
    param_names = [p['name'] for p in fitted_params_info]
    param_labels = [p['label'] for p in fitted_params_info]
    p0_guess_means = np.array([p['guess'] for p in fitted_params_info])
    prior_bounds_low = np.array([p['low'] for p in fitted_params_info])
    prior_bounds_high = np.array([p['high'] for p in fitted_params_info])
    
    # Store all param info (fitted and fixed) in ARGS for later reconstruction
    ARGS.all_param_info_list = param_info_list

    return param_names, param_labels, p0_guess_means, prior_bounds_low, prior_bounds_high


def reconstruct_full_theta_dict(theta_values_fitted, fitted_param_names, ARGS):
    """Reconstructs the full parameter dictionary including fixed values."""
    full_theta_dict = dict(zip(fitted_param_names, theta_values_fitted))
    
    for p_info in ARGS.all_param_info_list:
        if not p_info['is_fitted'] and p_info['name'] not in full_theta_dict:
            # Get fixed value from ARGS (e.g., ARGS.M_bulge_fixed)
            fixed_val_attr = f"{p_info['name']}_fixed" # Assumes fixed params are stored like M_bulge_fixed
            # A bit hacky; better to have a more structured way to store fixed vals if this becomes complex
            if hasattr(ARGS, p_info['name']): # If simple name like M_disk_solar
                 full_theta_dict[p_info['name']] = getattr(ARGS, p_info['name'], p_info['guess'])
            elif hasattr(ARGS, fixed_val_attr):
                 full_theta_dict[p_info['name']] = getattr(ARGS, fixed_val_attr, p_info['guess'])
            else: # Fallback to guess if not found in ARGS (should be set in ARGS)
                 full_theta_dict[p_info['name']] = p_info['guess']
                 logger.warning(f"Fixed parameter {p_info['name']} not found directly in ARGS, using guess value. Ensure it's set if intended to be fixed.")
    
    # Ensure component flags are present
    if ARGS.fit_target == 'milkyway':
        full_theta_dict['include_bulge'] = ARGS.include_bulge
        full_theta_dict['include_gas'] = ARGS.include_gas
        # Ensure all necessary h_z values are present if components are included
        if ARGS.include_bulge and 'h_z_bulge_eff_kpc' not in full_theta_dict : # Example for a potentially fixed param for density
            full_theta_dict['h_z_bulge_eff_kpc'] = ARGS.h_z_bulge_fixed if hasattr(ARGS, 'h_z_bulge_fixed') else 0.3

    return full_theta_dict


def log_prior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted, ARGS):
    for i, val in enumerate(theta_values_fitted):
        if not (prior_bounds_low_fitted[i] <= val <= prior_bounds_high_fitted[i]): # Inclusive bounds
            return -np.inf
    
    if ARGS.fit_target == 'milkyway':
        # Reconstruct full parameter dict for consistency checks
        params_for_checks = reconstruct_full_theta_dict(theta_values_fitted, fitted_param_names, ARGS)
        
        if ARGS.check_kz:
            if not check_vertical_kinematics_Kz(params_for_checks, R_solar_val=R_SUN_KPC,
                                                target_sigma_z_max_msun_kpc2=SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2):
                return -np.inf
        if ARGS.check_microlensing:
            if not calculate_microlensing_tau_baade(params_for_checks,
                                                   target_tau_max=TAU_MICRO_TARGET_BAADE_MAX):
                return -np.inf
    return 0.0

def log_likelihood(theta_values_fitted, fitted_param_names, R_data, v_data, sigma_data, xi_type_selected, ARGS,
                   sparc_galaxy_data_dict_full=None): # Pass full SPARC dict
    
    current_params_full_dict = reconstruct_full_theta_dict(theta_values_fitted, fitted_param_names, ARGS)
    
    v_predicted = v_model_for_emcee(R_data, current_params_full_dict, xi_type_str=xi_type_selected, ARGS_obj=ARGS,
                                    sparc_galaxy_data_dict_full=sparc_galaxy_data_dict_full)
    
    if not np.all(np.isfinite(v_predicted)): return -np.inf
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals_sq = ((v_data - v_predicted) / sigma_data_safe)**2
    log_L = -0.5 * np.sum(residuals_sq + np.log(2 * np.pi * sigma_data_safe**2))
    return log_L if np.isfinite(log_L) else -np.inf

def log_posterior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted,
                  R_data, v_data, sigma_data, xi_type_selected, ARGS,
                  sparc_galaxy_data_dict_full=None):

    lp = log_prior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted, ARGS)
    if not np.isfinite(lp): return -np.inf
    
    ll = log_likelihood(theta_values_fitted, fitted_param_names, R_data, v_data, sigma_data, xi_type_selected, ARGS,
                        sparc_galaxy_data_dict_full)
    if not np.isfinite(ll): return -np.inf
    return lp + ll


def run_mcmc_analysis(ARGS_in):
    logger.info(f"--- Starting MCMC Analysis for target: {ARGS_in.fit_target} ---")
    
    # Get only the parameters that will be fitted
    fitted_param_names, param_labels_fitted, p0_guess_means_fitted, prior_bounds_low_fitted, prior_bounds_high_fitted = get_param_labels_and_bounds(ARGS_in)
    ndim = len(fitted_param_names)
    logger.info(f"Fitting {ndim} parameters: {fitted_param_names}")
    if ndim == 0:
        logger.error("No parameters selected for fitting. Check --fit_... flags. Exiting.")
        sys.exit(1)

    R_obs_kpc, v_obs_kms, sigma_v_kms = None, None, None
    sparc_galaxy_data_dict_for_fit = None

    if ARGS_in.fit_target == 'milkyway':
        # ... (Gaia data loading, same as before) ...
        logger.info("--- Loading Milky Way (Gaia) Data ---")
        gaia_data_dict = load_gaia(sample_max=ARGS_in.max_sample_gaia,
                                   force_new_query_gaia=ARGS_in.force_live_gaia,
                                   force_reprocess_raw=ARGS_in.force_reprocess)
        if gaia_data_dict is None or gaia_data_dict.get("R_kpc") is None or len(gaia_data_dict["R_kpc"]) == 0:
            logger.error("❌ Failed to load Gaia data or data is empty. Exiting.")
            sys.exit(1)
        R_obs_kpc = gaia_data_dict["R_kpc"]
        v_obs_kms = gaia_data_dict["v_obs"]
        sigma_v_kms = gaia_data_dict["sigma_v"]

    elif ARGS_in.fit_target == 'sparc':
        if not SPARC_AVAILABLE: # ... (SPARC availability check) ...
            logger.error("❌ SPARC fitting selected, but sparc_io module is not available. Exiting.")
            sys.exit(1)
        if not ARGS_in.galaxy_id: # ... (galaxy_id check) ...
            logger.error("❌ SPARC fitting selected, but no --galaxy_id provided. Exiting.")
            sys.exit(1)
        logger.info(f"--- Loading SPARC Galaxy Data for: {ARGS_in.galaxy_id} ---")
        sparc_galaxy_data_dict_for_fit = load_single_sparc_galaxy(
            ARGS_in.galaxy_id,
            sparc_dir=ARGS_in.sparc_data_dir,
            assume_stellar_hz_kpc=ARGS_in.sparc_hz_star,
            assume_gas_hz_kpc=ARGS_in.sparc_hz_gas
        ) # ... (SPARC data loading, error check) ...
        if sparc_galaxy_data_dict_for_fit is None:
            logger.error(f"❌ Failed to load SPARC data for {ARGS_in.galaxy_id}. Exiting.")
            sys.exit(1)
        R_obs_kpc = sparc_galaxy_data_dict_for_fit['R_kpc']
        v_obs_kms = sparc_galaxy_data_dict_for_fit['V_obs']
        sigma_v_kms = sparc_galaxy_data_dict_for_fit['e_V_obs']
        # Note: V_newton components and rho components from sparc_galaxy_data_dict_for_fit
        # will be scaled by stellar_ML_factor inside the likelihood/v_model if that's fitted.

    logger.info(f"Loaded {len(R_obs_kpc)} data points for fitting.") # ... (empty data check) ...
    if len(R_obs_kpc) == 0:
        logger.error("❌ No data points to fit after loading. Exiting.")
        sys.exit("❌ No data points to fit after loading. Exiting.")


    # ... (MCMC setup: n_cores, pos0, sampler_moves, backend - largely same) ...
    logger.info(f"\n--- Setting up Likelihood & Priors for xi_type = '{ARGS_in.xi}' ---")

    n_cores_to_use = ARGS_in.ncores if ARGS_in.ncores >= 1 else 1
    if ARGS_in.ncores > 1:
        try:
            available_cpus = cpu_count()
            if n_cores_to_use > available_cpus:
                logger.warning(f"Requested {n_cores_to_use} cores, but only {available_cpus} seem available. Using {available_cpus}.")
                n_cores_to_use = available_cpus
        except NotImplementedError:
            logger.warning("cpu_count() not available. Defaulting to serial for MCMC if --ncores > 1 was used.")
            if ARGS_in.ncores > 1: n_cores_to_use = 1

    logger.info(f"\n--- Running MCMC ({ARGS_in.nwalkers} walkers, {ARGS_in.nsteps} steps, on {n_cores_to_use} core(s)) ---")
    pos0 = np.zeros((ARGS_in.nwalkers, ndim))
    for i in range(ndim): 
        pos0[:, i] = np.random.uniform(prior_bounds_low_fitted[i], prior_bounds_high_fitted[i], ARGS_in.nwalkers)
    
    sampler_moves_obj = None
    if ARGS_in.sampler_move == 'kdemove':
        if hasattr(emcee, 'moves') and hasattr(emcee.moves, 'KDEMove'):
            try:
                sampler_moves_obj = emcee.moves.KDEMove()
                logger.info("Using emcee.moves.KDEMove for sampling.")
            except Exception as e_kdemove:
                logger.warning(f"Failed to initialize KDEMove ({e_kdemove}). Using default moves.")
        else:
            logger.warning("emcee.moves.KDEMove not available in this emcee version. Using default moves.")

    backend_filename_parts = [
        ARGS_in.fit_target,
        ARGS_in.galaxy_id if ARGS_in.galaxy_id else 'MW',
        ARGS_in.xi
    ]
    if ARGS_in.fit_target == 'milkyway':
        if ARGS_in.include_bulge: backend_filename_parts.append("bulge" + ("fit" if ARGS_in.fit_bulge else "fix"))
        if ARGS_in.include_gas: backend_filename_parts.append("gas" + ("fit" if ARGS_in.fit_gas else "fix"))

    backend_filename = f"backend_{'_'.join(backend_filename_parts)}.h5"
    backend = emcee.backends.HDFBackend(backend_filename)

    if not ARGS_in.resume_mcmc or not os.path.exists(backend_filename): 
        backend.reset(ARGS_in.nwalkers, ndim)
        logger.info(f"Initialized new MCMC run. Backend: {backend_filename}")
    else:
        logger.info(f"Attempting to resume MCMC from backend: {backend_filename}")
        try:
            # Check if backend is compatible
            if backend.iteration > 0 and backend.shape[1] == ndim: # Check ndim consistency
                logger.info(f"Resuming from step {backend.iteration}.")
                latest_sample = backend.get_last_sample()
                if latest_sample.shape == (ARGS_in.nwalkers, ndim):
                    pos0 = latest_sample
                else:
                    logger.warning(f"Backend last sample shape {latest_sample.shape} mismatch with current ({ARGS_in.nwalkers}, {ndim}). Resetting walkers.")
                    # pos0 remains random from prior
            else:
                if backend.iteration > 0 and backend.shape[1] != ndim:
                    logger.warning(f"Backend ndim ({backend.shape[1]}) mismatch with current ({ndim}). Resetting backend.")
                else: # iteration == 0
                    logger.info("Backend is empty. Starting new run.")
                backend.reset(ARGS_in.nwalkers, ndim)
        except Exception as e_backend:
            logger.error(f"Error initializing/resuming backend: {e_backend}. Resetting backend.")
            backend.reset(ARGS_in.nwalkers, ndim)


    sampler_pool_arg_for_emcee = None
    # ... (pool setup, same as before) ...
    if n_cores_to_use > 1:
        logger.info(f"   MCMC will attempt parallel run on {n_cores_to_use} cores.")
        sampler_pool_arg_for_emcee = Pool(processes=n_cores_to_use)
    else:
        logger.info("   MCMC running in serial mode.")


    sampler = emcee.EnsembleSampler(ARGS_in.nwalkers, ndim, log_posterior,
                                    args=(fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted,
                                          R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi, ARGS_in,
                                          sparc_galaxy_data_dict_for_fit),
                                    pool=sampler_pool_arg_for_emcee,
                                    moves=sampler_moves_obj,
                                    backend=backend)
    
    # ... (MCMC sampling loop with tqdm, backend.iteration, same as before) ...
    start_time_mcmc = time.time()
    initial_step = sampler.iteration
    steps_to_run = ARGS_in.nsteps - initial_step
    
    if steps_to_run > 0 :
        logger.info(f"Running MCMC for {steps_to_run} new steps (total target {ARGS_in.nsteps}).")
        with tqdm(total=steps_to_run, desc="MCMC Sampling", unit="step", initial=0) as pbar:
            for _ in sampler.sample(pos0, iterations=steps_to_run, progress=False, store=True):
                pbar.update(1)
    else:
        logger.info(f"MCMC already run for {initial_step} steps. Target {ARGS_in.nsteps} reached or exceeded.")
    end_time_mcmc = time.time()

    if sampler_pool_arg_for_emcee is not None:
        sampler_pool_arg_for_emcee.close()
        sampler_pool_arg_for_emcee.join()

    logger.info(f"MCMC finished in {(end_time_mcmc - start_time_mcmc)/60:.2f} minutes for this session.")
    logger.info(f"Total steps in backend: {sampler.iteration}")


    # ... (Autocorrelation, burn-in, thinning, chain saving - largely same, ensure using fitted_param_labels) ...
    actual_burnin = ARGS_in.burnin
    actual_thin = ARGS_in.thin
    try:
        current_chain_length = sampler.iteration
        discard_for_tau = min(ARGS_in.burnin, current_chain_length - 1 if current_chain_length > 0 else 0)
        
        if current_chain_length - discard_for_tau > 100 * ndim : # Heuristic: need enough samples post-discard
            autocorr_time = sampler.get_autocorr_time(discard=discard_for_tau, tol=0, quiet=True) # Use discard here
            logger.info(f"Autocorrelation time estimates: {autocorr_time}")
            finite_autocorr = autocorr_time[np.isfinite(autocorr_time)]
            if len(finite_autocorr) > 0:
                max_autocorr = np.max(finite_autocorr)
                recommended_burnin = int(np.ceil(max_autocorr * 5))
                recommended_thin = int(np.ceil(max_autocorr / 2)); recommended_thin = max(1, recommended_thin)
                logger.info(f"Recommended burn-in for analysis: ~{recommended_burnin}")
                logger.info(f"Recommended thinning for analysis: ~{recommended_thin}")
                if ARGS_in.burnin_for_analysis < recommended_burnin: # Using separate burnin for analysis
                    logger.warning(f"User analysis burn-in ({ARGS_in.burnin_for_analysis}) is less than recommended ({recommended_burnin}).")
                    actual_burnin = recommended_burnin 
                else:
                    actual_burnin = ARGS_in.burnin_for_analysis

                if ARGS_in.thin_for_analysis < recommended_thin:
                    logger.warning(f"User analysis thinning ({ARGS_in.thin_for_analysis}) is less than recommended ({recommended_thin}).")
                    actual_thin = recommended_thin
                else:
                    actual_thin = ARGS_in.thin_for_analysis
            else: 
                logger.warning("All autocorrelation times are non-finite. Using user-specified burn-in/thin for analysis.")
                actual_burnin = ARGS_in.burnin_for_analysis
                actual_thin = ARGS_in.thin_for_analysis
        else: 
            logger.warning("Chain too short after initial discard to reliably estimate autocorr time. Using user-specified burn-in/thin for analysis.")
            actual_burnin = ARGS_in.burnin_for_analysis
            actual_thin = ARGS_in.thin_for_analysis
    except emcee.autocorr.AutocorrError as e_acorr:
        logger.warning(f"Could not estimate autocorrelation time: {e_acorr}. Using user-specified values for analysis.")
        actual_burnin = ARGS_in.burnin_for_analysis
        actual_thin = ARGS_in.thin_for_analysis
    except Exception as e_autocorr_generic:
        logger.warning(f"An unexpected error during autocorrelation time estimation: {e_autocorr_generic}. Using user-specified values for analysis.")
        actual_burnin = ARGS_in.burnin_for_analysis
        actual_thin = ARGS_in.thin_for_analysis

    actual_burnin = min(actual_burnin, sampler.iteration -1 if sampler.iteration > 0 else 0); actual_burnin = max(0, actual_burnin)
    actual_thin = max(1, actual_thin)
    logger.info(f"Using actual burn-in for analysis: {actual_burnin}")
    logger.info(f"Using actual thinning for analysis: {actual_thin}")

    chain_flat = np.array([])
    if sampler.iteration > actual_burnin and (sampler.iteration - actual_burnin) >= actual_thin:
        chain_flat = sampler.get_chain(discard=actual_burnin, thin=actual_thin, flat=True)
        if (sampler.iteration - actual_burnin) / actual_thin < 50 * ndim : 
             logger.warning(f"Consider increasing total steps or adjusting burn/thin. Current effective samples for analysis: {len(chain_flat)}")
    else:
        logger.warning(f"nsteps in backend ({sampler.iteration}) is too small for the chosen analysis burn-in ({actual_burnin}) and thinning ({actual_thin}). Chain for analysis will be empty or very small.")

    output_prefix_parts = [
        ARGS_in.fit_target,
        ARGS_in.galaxy_id if ARGS_in.galaxy_id else 'MW',
        ARGS_in.xi
    ]
    if ARGS_in.fit_target == 'milkyway': # Add MW model config to output name
        if ARGS_in.include_bulge: output_prefix_parts.append("b" + ("F" if ARGS_in.fit_bulge else "X")) # Fit/Fixed
        if ARGS_in.include_gas: output_prefix_parts.append("g" + ("F" if ARGS_in.fit_gas else "X"))

    output_prefix = "_".join(output_prefix_parts)
    chain_filename = f"chain_{output_prefix}.npy"
    np.save(chain_filename, chain_flat)
    logger.info(f"Saved flattened chain to {chain_filename} ({len(chain_flat)} samples)")


    # --- Posterior Diagnostics & Plot ---
    # (Plotting and summary generation logic needs param_labels_fitted)
    param_summary_text = ""
    median_params_fitted_values = p0_guess_means_fitted # Fallback
    
    logger.info("\n--- Generating Posterior Diagnostics & Plot ---")
    if len(chain_flat) < ndim * 2 :
        logger.warning(f"Chain has very few samples ({len(chain_flat)}) for robust analysis. Plots may be misleading or fail.")
        if len(chain_flat) > 0 : median_params_fitted_values = np.median(chain_flat, axis=0)
    else:
        # Corner plot with fitted parameters
        figure = corner.corner(chain_flat, labels=param_labels_fitted, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        figure.savefig(f"corner_{output_prefix}.png"); plt.close(figure)
        logger.info(f"Saved corner plot to corner_{output_prefix}.png")
        
        median_params_fitted_values = np.median(chain_flat, axis=0)
        params_16th_fitted_values, params_84th_fitted_values = np.percentile(chain_flat, [16, 84], axis=0)
        
        param_summary_text = "Fitted Parameters (Median & 68% CI):\n"
        for i, label_text in enumerate(param_labels_fitted):
            summary_line = (f"  {label_text:<30}: {median_params_fitted_values[i]:.3e} "
                            f" (+{params_84th_fitted_values[i]-median_params_fitted_values[i]:.2e} / -{median_params_fitted_values[i]-params_16th_fitted_values[i]:.2e})")
            logger.info(summary_line); param_summary_text += summary_line + "\n"

    # Reconstruct full parameter dictionary for plotting the median model
    median_params_full_dict_for_plot = reconstruct_full_theta_dict(median_params_fitted_values, fitted_param_names, ARGS_in)

    R_plot_curve_min = np.min(R_obs_kpc) if len(R_obs_kpc) > 0 and np.any(np.isfinite(R_obs_kpc)) else 0.1
    R_plot_curve_max = np.max(R_obs_kpc) if len(R_obs_kpc) > 0 and np.any(np.isfinite(R_obs_kpc)) else 25.0
    R_plot_curve_min = max(0.01, R_plot_curve_min)
    R_plot_curve = np.linspace(R_plot_curve_min, R_plot_curve_max, 300)
    
    v_median_curve_plot = v_model_for_emcee(R_plot_curve, median_params_full_dict_for_plot, ARGS_in.xi, ARGS_obj=ARGS_in,
                                            sparc_galaxy_data_dict_full=sparc_galaxy_data_dict_for_fit)

    # Envelope calculation
    v_16th_curve, v_84th_curve = np.zeros_like(R_plot_curve), np.zeros_like(R_plot_curve)
    if len(chain_flat) > ndim: # Need some samples for envelope
        n_samples_for_envelope = min(500, len(chain_flat)) # Reduced for speed
        chain_subset_indices = np.random.choice(len(chain_flat), n_samples_for_envelope, replace=False)
        chain_subset_for_envelope = chain_flat[chain_subset_indices]
        
        v_model_samples_list = []
        for fitted_pars_array in chain_subset_for_envelope:
            current_full_pars_dict = reconstruct_full_theta_dict(fitted_pars_array, fitted_param_names, ARGS_in)
            v_sample = v_model_for_emcee(R_plot_curve, current_full_pars_dict, ARGS_in.xi, ARGS_obj=ARGS_in,
                                         sparc_galaxy_data_dict_full=sparc_galaxy_data_dict_for_fit)
            v_model_samples_list.append(v_sample)
        
        if v_model_samples_list:
            v_model_samples_arr = np.array(v_model_samples_list)
            v_16th_curve, v_84th_curve = np.nanpercentile(v_model_samples_arr, [16, 84], axis=0)
        else: # Fallback if envelope calculation fails
            v_16th_curve, v_84th_curve = v_median_curve_plot * 0.9, v_median_curve_plot * 1.1
    else:
        v_16th_curve, v_84th_curve = v_median_curve_plot * 0.9, v_median_curve_plot * 1.1


    plt.figure(figsize=(10,6))
    plt.errorbar(R_obs_kpc, v_obs_kms, yerr=sigma_v_kms, fmt=".k", alpha=0.02, label=f"{ARGS_in.fit_target.capitalize()} Data", zorder=1)
    plt.plot(R_plot_curve, v_median_curve_plot, color="red", lw=2.5, label=f"Density-Metric Median ({ARGS_in.xi} $\\xi$)", zorder=3)
    plt.fill_between(R_plot_curve, v_16th_curve, v_84th_curve, color="red", alpha=0.3, zorder=2, label="68% Credible Interval")

    # Newtonian curve for plot
    v_newton_plot_median_params = np.zeros_like(R_plot_curve)
    if ARGS_in.fit_target == 'milkyway':
        v_newton_plot_median_params = v_newton_kms(R_plot_curve,
                                                   median_params_full_dict_for_plot.get('M_disk_solar',0), median_params_full_dict_for_plot.get('R_d_kpc',1),
                                                   median_params_full_dict_for_plot.get('M_bulge_solar',0), median_params_full_dict_for_plot.get('R_b_kpc',0.5), median_params_full_dict_for_plot.get('include_bulge',False),
                                                   median_params_full_dict_for_plot.get('M_gas_solar',0), median_params_full_dict_for_plot.get('R_gas_kpc',7.0), median_params_full_dict_for_plot.get('include_gas',False))
    elif ARGS_in.fit_target == 'sparc' and sparc_galaxy_data_dict_for_fit is not None:
        # Reconstruct V_N from SPARC components using the median fitted M/L factor
        stellar_ML_factor_median = median_params_full_dict_for_plot.get('stellar_ML_factor', 1.0)
        v_disk_sq_plot = (sparc_galaxy_data_dict_for_fit['V_disk_comp_kms']**2) * stellar_ML_factor_median
        v_bulge_sq_plot = (sparc_galaxy_data_dict_for_fit['V_bulge_comp_kms']**2) * stellar_ML_factor_median
        v_gas_sq_plot = sparc_galaxy_data_dict_for_fit['V_gas_comp_kms']**2
        v_n_bary_obs_R_sq = v_disk_sq_plot + v_bulge_sq_plot + v_gas_sq_plot
        v_n_bary_obs_R = np.sqrt(np.maximum(0, v_n_bary_obs_R_sq))
        # Interpolate this onto R_plot_curve
        interp_func_vn = interp1d(sparc_galaxy_data_dict_for_fit['R_kpc'], v_n_bary_obs_R,
                                  kind='linear', bounds_error=False, fill_value=(np.nan, np.nan))
        v_newton_plot_median_params = interp_func_vn(R_plot_curve)
        
    plt.plot(R_plot_curve, v_newton_plot_median_params, color="green", ls="--", lw=2, label="Newtonian (Baryons, Median Fit)", zorder=2.5)
    
    # ... (Plot labels, title, legend, savefig - largely same, use output_prefix) ...
    plt.xlabel("Galactocentric Radius R (kpc)", fontsize=12)
    plt.ylabel("Tangential Velocity v (km s$^{-1}$)", fontsize=12)
    title_text = f"{ARGS_in.galaxy_id if ARGS_in.galaxy_id else 'Milky Way'} Fit: Density-Metric ({ARGS_in.xi} $\\xi$)"
    if ARGS_in.fit_target == 'milkyway':
        title_text += f" (Disk"
        if ARGS_in.include_bulge: title_text += "+Bulge" + ("(fit)" if ARGS_in.fit_bulge else "(fix)")
        if ARGS_in.include_gas: title_text += "+Gas" + ("(fit)" if ARGS_in.fit_gas else "(fix)")
        title_text += ")"
    elif ARGS_in.fit_target == 'sparc' and ARGS_in.fit_sparc_ML:
        title_text += " (fit M/L)"


    plt.title(title_text, fontsize=14)
    plt.legend(fontsize=9); plt.grid(True, ls=':', alpha=0.7)
    valid_y_for_lim = v_84th_curve[np.isfinite(v_84th_curve)] if isinstance(v_84th_curve, np.ndarray) and len(v_84th_curve)>0 else v_median_curve_plot[np.isfinite(v_median_curve_plot)]
    ylim_max_val = np.max(valid_y_for_lim) if len(valid_y_for_lim) > 0 and np.any(np.isfinite(valid_y_for_lim)) else 350
    ylim_max_plot = max(350, ylim_max_val * 1.1 if pd.notna(ylim_max_val) else 350)

    # Ensure v_obs_kms is finite for sensible y limits
    finite_v_obs = v_obs_kms[np.isfinite(v_obs_kms)]
    if len(finite_v_obs) > 0:
        ylim_max_plot = max(ylim_max_plot, np.percentile(finite_v_obs, 99) * 1.1)


    plt.ylim(0, ylim_max_plot)
    plt.xlim(0, np.max(R_plot_curve) if len(R_plot_curve) > 0 and np.any(np.isfinite(R_plot_curve)) else 25.0)

    plt.tight_layout(); plt.savefig(f"rotation_curve_fit_{output_prefix}.png", dpi=150)
    logger.info(f"Saved rotation curve plot to rotation_curve_fit_{output_prefix}.png"); plt.close()


    # ... (AIC/BIC, RMS - largely same, ensure using median_params_full_dict_for_plot for v_model calls) ...
    # ... (Gelman-Rubin, Dynesty placeholders) ...
    # ... (Save summary text) ...
    aic_bic_text = "AIC/BIC: Not calculable (chain too small or error).\n"
    if len(chain_flat) > ndim : 
        logL_samples_all = sampler.get_log_prob(discard=actual_burnin, thin=actual_thin, flat=True)
        # Ensure logL_samples_all correspond to chain_flat (posterior, not just likelihood)
        valid_log_probs = logL_samples_all[np.isfinite(logL_samples_all)] # Use posterior samples
        if len(valid_log_probs) > 0:
            # For AIC/BIC, typically use max likelihood from the MCMC chain
            # Need to recompute likelihood for best-fit theta if log_prob stored posterior
            max_log_prob_idx = np.argmax(logL_samples_all) # Index in chain_flat
            best_fit_theta_mcmc_fitted = chain_flat[max_log_prob_idx]
            best_fit_theta_mcmc_full = reconstruct_full_theta_dict(best_fit_theta_mcmc_fitted, fitted_param_names, ARGS_in)

            max_log_likelihood_val = log_likelihood(best_fit_theta_mcmc_fitted, fitted_param_names, 
                                                    R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi, ARGS_in,
                                                    sparc_galaxy_data_dict_for_fit)

            if np.isfinite(max_log_likelihood_val):
                k_params = ndim; N_data = len(R_obs_kpc)
                AIC = 2 * k_params - 2 * max_log_likelihood_val
                BIC = k_params * np.log(N_data) - 2 * max_log_likelihood_val
                aic_bic_text = (f"Model: {ARGS_in.xi} on {ARGS_in.fit_target} ({ARGS_in.galaxy_id if ARGS_in.galaxy_id else 'MW'})\n"
                                f"Max Log-Likelihood (from MCMC best-fit): {max_log_likelihood_val:.2f}\n"
                                f"k (fitted params): {k_params}, N (data points): {N_data}\nAIC: {AIC:.2f}\nBIC: {BIC:.2f}\n\n")
                logger.info(f"\n--- Model Comparison Metrics ---\n{aic_bic_text.strip()}")
            else: logger.warning("⚠️ Max log-likelihood is not finite. Cannot calculate AIC/BIC.")
        else: logger.warning("⚠️ No finite log probabilities in chain for AIC/BIC.")
    else: logger.warning("⚠️ Empty or too small chain, skipping AIC/BIC calculation.")

    # RMS Residuals
    rms_text = ""
    if len(R_obs_kpc) > 0 :
        mask_inner = (R_obs_kpc < 5.0)
        mask_outer = (R_obs_kpc > 10.0) & (R_obs_kpc < 20.0)
        
        def calculate_rms_residuals(data_mask, params_for_rms_calc_full_dict):
            if np.sum(data_mask) == 0: return np.nan
            v_pred_masked = v_model_for_emcee(R_obs_kpc[data_mask], params_for_rms_calc_full_dict, ARGS_in.xi, ARGS_obj=ARGS_in,
                                            sparc_galaxy_data_dict_full=sparc_galaxy_data_dict_for_fit)
            if not np.all(np.isfinite(v_pred_masked)): return np.nan # Check if all are finite
            return np.sqrt(np.nanmean((v_obs_kms[data_mask] - v_pred_masked)**2))

        rms_inner = calculate_rms_residuals(mask_inner, median_params_full_dict_for_plot)
        rms_outer = calculate_rms_residuals(mask_outer, median_params_full_dict_for_plot)
        logger.info(f"RMS Δv (Median Params) for R < 5 kpc  : {rms_inner:.2f} km/s (N_stars={np.sum(mask_inner)})")
        logger.info(f"RMS Δv (Median Params) for 10 < R < 20 kpc : {rms_outer:.2f} km/s (N_stars={np.sum(mask_outer)})")
        rms_text = (f"RMS inner (R < 5 kpc): {rms_inner:.2f} km/s\n"
                    f"RMS outer (10 < R < 20 kpc): {rms_outer:.2f} km/s\n")
    else:
        rms_text = "RMS: No data to calculate.\n"


    info_filename = f"info_summary_{output_prefix}.txt"
    with open(info_filename, "w") as f:
        f.write(f"--- Fit Summary for {output_prefix} ---\n")
        f.write(param_summary_text) # Already includes header
        # Add fixed parameter summary
        f.write("\nFixed Parameters (if any):\n")
        any_fixed = False
        for p_info in ARGS_in.all_param_info_list:
            if not p_info['is_fitted']:
                fixed_val = median_params_full_dict_for_plot.get(p_info['name'], "N/A_fixed")
                f.write(f"  {p_info['label']:<30}: {fixed_val} (fixed)\n")
                any_fixed = True
        if not any_fixed: f.write("  None\n")

        f.write("\n--- Model Comparison ---\n"); f.write(aic_bic_text)
        f.write("\n--- RMS Residuals ---\n"); f.write(rms_text)
    logger.info(f"\nSaved summary information to {info_filename}")


if __name__ == '__main__':
    if platform.system() in ["Windows", "Darwin"]: freeze_support()

    parser = argparse.ArgumentParser(description="Fit density-dependent metric model.")
    
    # General settings
    parser.add_argument('--fit_target', type=str, default='milkyway', choices=['milkyway', 'sparc'], help='Target for fitting')
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'], help='Type of xi(rho) function')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    # MCMC settings
    parser.add_argument('--nwalkers', type=int, default=64, help='MCMC walkers')
    parser.add_argument('--nsteps', type=int, default=2000, help='Total MCMC steps (reduced for quick test)') # Reduced default
    parser.add_argument('--burnin_for_analysis', type=int, default=500, help='Burn-in steps for analysis (not for sampler)') # New name
    parser.add_argument('--thin_for_analysis', type=int, default=50, help='Thinning factor for analysis') # New name
    parser.add_argument('--ncores', type=int, default=1, help='CPU cores for MCMC (0 or 1 serial)')
    parser.add_argument('--sampler_move', type=str, default='default', choices=['default', 'kdemove'], help="emcee sampler move type")
    parser.add_argument('--resume_mcmc', action='store_true', help='Resume MCMC from backend file')

    # Milky Way specific settings
    mw_group = parser.add_argument_group('Milky Way Specific Settings')
    mw_group.add_argument('--max_sample_gaia', type=int, default=10000, help='Max Gaia stars (reduced for quick test)') # Reduced
    mw_group.add_argument('--force_live_gaia', action='store_true', help='Force new Gaia query')
    mw_group.add_argument('--force_reprocess', action='store_true', help='Force reprocessing raw Gaia data')
    
    mw_group.add_argument('--include_bulge', action='store_true', help='[MW] Include bulge component.')
    mw_group.add_argument('--fit_bulge', action='store_true', help='[MW] Fit bulge parameters (if --include_bulge). Else use fixed.')
    mw_group.add_argument('--M_bulge_fixed', type=float, default=0.9e10, help='[MW] Fixed bulge mass (Msun).')
    mw_group.add_argument('--R_b_fixed', type=float, default=0.5, help='[MW] Fixed bulge scale radius (kpc).')
    # Add h_z_bulge_fixed if needed for some density models
    
    mw_group.add_argument('--include_gas', action='store_true', help='[MW] Include gas component.')
    mw_group.add_argument('--fit_gas', action='store_true', help='[MW] Fit gas parameters (if --include_gas). Else use fixed.')
    mw_group.add_argument('--M_gas_fixed', type=float, default=1.0e10, help='[MW] Fixed gas mass (Msun).')
    mw_group.add_argument('--R_gas_fixed', type=float, default=7.0, help='[MW] Fixed gas scale radius (kpc).')
    mw_group.add_argument('--h_z_gas_fixed', type=float, default=0.15, help='[MW] Fixed gas scale height (kpc).')

    # MW Consistency checks (fixed values for these are in density_metric.py)
    mw_group.add_argument('--check_kz', action='store_true', help='[MW] Enable Kz vertical kinematics check.')
    mw_group.add_argument('--check_microlensing', action='store_true', help='[MW] Enable microlensing optical depth check.')
    # Fixed parameters for MW model if not fitted (these are defaults if not specified and component not fitted)
    mw_group.add_argument('--M_disk_fixed', type=float, default=6e10)
    mw_group.add_argument('--R_d_fixed', type=float, default=3.0)
    mw_group.add_argument('--h_z_disk_fixed', type=float, default=0.3)


    # SPARC specific settings
    sparc_group = parser.add_argument_group('SPARC Specific Settings')
    sparc_group.add_argument('--galaxy_id', type=str, default=None, help='[SPARC] ID of SPARC galaxy')
    sparc_group.add_argument('--sparc_data_dir', type=str, default="data/sparc_data", help='[SPARC] SPARC data directory.')
    sparc_group.add_argument('--sparc_hz_star', type=float, default=0.3, help='[SPARC] Assumed stellar h_z (kpc)')
    sparc_group.add_argument('--sparc_hz_gas', type=float, default=0.1, help='[SPARC] Assumed gas h_z (kpc)')
    sparc_group.add_argument('--fit_sparc_ML', action='store_true', help='[SPARC] Fit stellar M/L factor. If false, uses base M/L.')


    # Advanced sampling / convergence (placeholders)
    adv_group = parser.add_argument_group('Advanced Sampling/Convergence (Placeholders)')
    adv_group.add_argument('--use_dynesty', action='store_true', help='Use Dynesty nested sampler.')
    adv_group.add_argument('--run_gelman_rubin', action='store_true', help='Run Gelman-Rubin diagnostic.')
    adv_group.add_argument('--nchains_gr', type=int, default=4, help='Num chains for Gelman-Rubin.')
    adv_group.add_argument('--gr_threshold', type=float, default=1.05, help='Gelman-Rubin R_hat threshold.')

    parser.add_argument('--kill_existing', action='store_true', help='Kill other running instances of this script.')

    ARGS_global = parser.parse_args()
    
    logger_level_numeric = getattr(logging, ARGS_global.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(logger_level_numeric)
    # Ensure individual loggers also respect this, or set them specifically
    logging.getLogger("sparc_loader").setLevel(logger_level_numeric)
    logging.getLogger("density_metric").setLevel(logger_level_numeric)


    if ARGS_global.kill_existing:
        current_script_name = os.path.basename(__file__)
        kill_existing_instances(script_name_to_kill=current_script_name)
        logger.info("Continuing with the current script run...")

    if ARGS_global.ncores < 1: # Auto-set ncores if invalid
        try:
            num_cpus = cpu_count()
            ARGS_global.ncores = max(1, num_cpus - 2 if num_cpus and num_cpus > 2 else 1)
            logger.info(f"Auto-set ncores: {ARGS_global.ncores}")
        except NotImplementedError:
            ARGS_global.ncores = 1
    
    # Store all_param_info_list in ARGS_global once after parsing
    _, _, _, _, _ = get_param_labels_and_bounds(ARGS_global) # This call populates ARGS_global.all_param_info_list

    # Backwards compatibility for burnin/thin if user only provides old flags
    if not hasattr(ARGS_global, 'burnin_for_analysis') : ARGS_global.burnin_for_analysis = ARGS_global.burnin
    if not hasattr(ARGS_global, 'thin_for_analysis') : ARGS_global.thin_for_analysis = ARGS_global.thin


    run_mcmc_analysis(ARGS_global)

    logger.info("\n--- main.py Script Finished ---")