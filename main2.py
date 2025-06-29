#!/usr/bin/env python3
"""
main2.py - Main script to orchestrate MCMC fitting of density-dependent
             metric models to Gaia rotation curve data (Milky Way) or SPARC data (external galaxies).
             Integrates multi-component baryonic model capabilities. (Emcee version)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
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
from scipy.interpolate import interp1d 
import sys 
import emcee
from pathlib import Path

# Import from local modules
try:
    from data_io import load_gaia
    # Import the self-test function along with the physics models
    from density_metric2 import ( 
        v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3,
        XI_FUNCTION_MAP,
        check_vertical_kinematics_Kz, calculate_microlensing_tau_baade,
        R_SUN_KPC, SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2, TAU_MICRO_TARGET_BAADE_MAX,
        MSUN_PC2_TO_MSUN_KPC2, G_ASTRO_UNITS,
        v_newton_kms as v_newton_kms_single_disk, 
        volume_density_total_midplane_solar_kpc3 as volume_density_midplane_single_disk,
        run_physics_self_tests # <-- IMPORT THE SELF-TEST FUNCTION
    )
    SPARC_AVAILABLE = False
    try:
        from sparc_io import load_single_sparc_galaxy, load_sparc_metadata
        SPARC_AVAILABLE = True
    except ImportError:
        # This warning is deferred until after the logger is fully configured.
        pass

except ImportError as e:
    # Use print here as logger is not yet configured.
    print(f"CRITICAL ERROR importing local modules: {e}", file=sys.stderr)
    sys.exit(1) 

# Configure logger at the top level, but allow main to set the level.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# --- HELPER FUNCTIONS ---

def _check_flag_consistency(args, logger_obj):
    """Checks for contradictory CLI flags for fitting vs. fixing parameters."""
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
                    logger_obj.error(
                        f"Inconsistent CLI: --{fit_flag_name} is set AND --{fixed_name} is provided. "
                        "You cannot both fit a component's parameters and fix them. Please choose one."
                    )
                    inconsistent_flags_found = True
    
    if inconsistent_flags_found:
        raise ValueError("Inconsistent command-line arguments detected. Aborting.")
    logger_obj.info("CLI flag consistency OK.")

def kill_existing_instances(script_name_to_kill="main2.py"):
    current_pid = os.getpid()
    logger.info(f"🌬️  Attempting to terminate other instances of '{script_name_to_kill}' (current PID: {current_pid})...")
    try:
        ps_cmd = "ps aux"; grep_python_cmd = "grep python"; grep_script_cmd = f"grep {shlex.quote(script_name_to_kill)}"; grep_exclude_cmd = "grep -v grep"
        p1 = subprocess.Popen(shlex.split(ps_cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(shlex.split(grep_python_cmd), stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); p1.stdout.close() if p1.stdout else None
        p3 = subprocess.Popen(shlex.split(grep_script_cmd), stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); p2.stdout.close() if p2.stdout else None
        p4 = subprocess.Popen(shlex.split(grep_exclude_cmd), stdin=p3.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); p3.stdout.close() if p3.stdout else None
        result_stdout, _ = p4.communicate(); killed_count = 0
        if result_stdout:
            for line in result_stdout.decode().strip().split('\n'):
                if not line.strip(): continue
                parts = line.split(); pid_str = parts[1] if len(parts) > 1 else None
                if pid_str and pid_str.isdigit():
                    pid = int(pid_str)
                    if pid != current_pid: 
                        try:
                            logger.info(f"   Killing process {pid}"); os.kill(pid, signal.SIGKILL); killed_count += 1; time.sleep(0.1)
                        except ProcessLookupError: pass 
                        except Exception as e_kill_inner: logger.error(f"   Error killing process {pid}: {e_kill_inner}")
        logger.info(f"   Terminated {killed_count} other instance(s)." if killed_count > 0 else f"   No other instances of '{script_name_to_kill}' found.")
    except Exception as e: logger.warning(f"   ⚠️ Error during attempt to kill existing instances: {e}")
    logger.info("-" * 30)

def v_model_for_emcee(R_kpc_array, p_all_params_dict_for_model, xi_type_str="power", ARGS_obj_for_target=None,
                      sparc_galaxy_data_dict_full_for_model=None):
    if ARGS_obj_for_target is None:
        raise ValueError("ARGS_obj_for_target must be provided to v_model_for_emcee")

    rho_c_solar_kpc3 = p_all_params_dict_for_model['rho_c_solar_kpc3']
    n_exp = p_all_params_dict_for_model['n_exp']
    
    v_n_kms = np.zeros_like(np.atleast_1d(R_kpc_array), dtype=np.float64)
    rho_midplane_for_xi = np.zeros_like(np.atleast_1d(R_kpc_array), dtype=np.float64)

    is_multi_component_mw = ARGS_obj_for_target.fit_target == 'milkyway' and \
                            (ARGS_obj_for_target.include_bulge or 
                             ARGS_obj_for_target.include_disk_thick or 
                             ARGS_obj_for_target.include_gas or
                             not ARGS_obj_for_target.use_old_single_disk_params)

    if ARGS_obj_for_target.fit_target == 'sparc':
        if sparc_galaxy_data_dict_full_for_model is None:
            raise ValueError("SPARC mode requires sparc_galaxy_data_dict_full_for_model.")
        stellar_ML_factor = p_all_params_dict_for_model.get('stellar_ML_factor', 1.0)
        v_disk_comp_sq = (sparc_galaxy_data_dict_full_for_model['V_disk_comp_kms']**2) * stellar_ML_factor
        v_bulge_comp_sq = (sparc_galaxy_data_dict_full_for_model['V_bulge_comp_kms']**2) * stellar_ML_factor
        v_gas_comp_sq = sparc_galaxy_data_dict_full_for_model['V_gas_comp_kms']**2
        v_n_kms_sq_total = v_disk_comp_sq + v_bulge_comp_sq + v_gas_comp_sq
        v_n_kms = np.sqrt(np.maximum(0, v_n_kms_sq_total))
        rho_star_mid_scaled = sparc_galaxy_data_dict_full_for_model['rho_star_mid_Msun_kpc3_baseML'] * stellar_ML_factor
        rho_midplane_for_xi = rho_star_mid_scaled + sparc_galaxy_data_dict_full_for_model['rho_gas_mid_Msun_kpc3']
    elif is_multi_component_mw:
        v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict_for_model)
        rho_midplane_for_xi = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict_for_model)
    elif ARGS_obj_for_target.fit_target == 'milkyway': 
        logger.debug("Using OLD single-disk v_newton and volume_density functions for MW.")
        M_d = p_all_params_dict_for_model['M_disk_solar']
        R_d = p_all_params_dict_for_model['R_d_kpc']
        h_z_d = p_all_params_dict_for_model['h_z_disk_kpc']
        v_n_kms = v_newton_kms_single_disk(R_kpc_array, M_d, R_d, 0.0, 0.5, False, 0.0, 7.0, False)
        rho_midplane_for_xi = volume_density_midplane_single_disk( R_kpc_array, M_d, R_d, h_z_d, 0.0, 0.5, 0.3, False, 0.0, 7.0, 0.15, False )
    else:
        raise ValueError(f"Unknown fit_target or MW configuration: {ARGS_obj_for_target.fit_target}")

    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    if xi_func is None: raise ValueError(f"Unknown xi_type: {xi_type_str}.")
    rho_midplane_for_xi_safe = np.maximum(rho_midplane_for_xi, 0.0)
    xi_values = xi_func(rho_midplane_for_xi_safe, rho_c_solar_kpc3, n_exp)
    xi_values = np.nan_to_num(xi_values, nan=1.0) 
    xi_values_safe = np.maximum(xi_values, 0.0)
    v_n_kms_safe = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)
    v_mod_kms = v_n_kms_safe * np.sqrt(xi_values_safe)
    return v_mod_kms

MW_SINGLE_DISK_OLD_PARAMS = {
    'M_disk_solar': {'label': r"$M_{disk}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_fixed', 'default_fixed': 6e10, 'low': 1e10, 'high': 2.5e11},
    'R_d_kpc': {'label': r"$R_d$ (kpc)", 'fixed_val_from_arg': 'R_d_fixed', 'default_fixed': 3.0, 'low': 1.5, 'high': 5.0},
    'h_z_disk_kpc': {'label': r"$h_z$ (kpc)", 'fixed_val_from_arg': 'h_z_disk_fixed', 'default_fixed': 0.3, 'low': 0.1, 'high': 0.7},
}
MW_MULTI_COMP_PARAM_CONFIG = {
    'rho_c_solar_kpc3': {'label': r"$\rho_c$ ($M_\odot/kpc^3$)", 'fixed_val_from_arg': 'rho_c_fixed', 'default_fixed': 1e7, 'low': 1e5, 'high': 2e9, 'fit_flag_arg': 'fit_xi_params', 'default_fit_flag': True},
    'n_exp': {'label': r"$n$", 'fixed_val_from_arg': 'n_exp_fixed', 'default_fixed': 1.5, 'low': 0.1, 'high': 4.0, 'fit_flag_arg': 'fit_xi_params', 'default_fit_flag': True},
    'M_bulge_solar': {'label': r"$M_{bulge}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_bulge_fixed', 'default_fixed': 0.9e10, 'low': 0.1e10, 'high': 3.0e10, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge', 'default_fit_flag': False},
    'a_bulge_kpc': {'label': r"$a_{bulge}$ (kpc)", 'fixed_val_from_arg': 'a_bulge_fixed', 'default_fixed': 0.5, 'low': 0.1, 'high': 2.0, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge', 'default_fit_flag': False},
    'M_disk_thin_solar': {'label': r"$M_{d,thin}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thin_fixed', 'default_fixed': 4.0e10, 'low': 1e10, 'high': 8e10, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'default_fit_flag': True},
    'R_d_thin_kpc': {'label': r"$R_{d,thin}$ (kpc)", 'fixed_val_from_arg': 'R_d_thin_fixed', 'default_fixed': 2.5, 'low': 1.5, 'high': 5.0, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'default_fit_flag': True},
    'h_z_thin_kpc': {'label': r"$h_{z,thin}$ (kpc)", 'fixed_val_from_arg': 'h_z_thin_fixed', 'default_fixed': 0.3, 'low': 0.15, 'high': 0.45, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'default_fit_flag': True},
    'M_disk_thick_solar': {'label': r"$M_{d,thick}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thick_fixed', 'default_fixed': 1.0e10, 'low': 0.1e10, 'high': 5e10, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'default_fit_flag': False},
    'R_d_thick_kpc': {'label': r"$R_{d,thick}$ (kpc)", 'fixed_val_from_arg': 'R_d_thick_fixed', 'default_fixed': 3.5, 'low': 2.0, 'high': 6.0, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'default_fit_flag': False},
    'h_z_thick_kpc': {'label': r"$h_{z,thick}$ (kpc)", 'fixed_val_from_arg': 'h_z_thick_fixed', 'default_fixed': 0.9, 'low': 0.5, 'high': 1.5, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'default_fit_flag': False},
    'M_gas_solar': {'label': r"$M_{gas}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_gas_fixed', 'default_fixed': 1.0e10, 'low': 0.2e10, 'high': 2.0e10, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'default_fit_flag': False},
    'R_d_gas_kpc': {'label': r"$R_{d,gas}$ (kpc)", 'fixed_val_from_arg': 'R_d_gas_fixed', 'default_fixed': 7.0, 'low': 3.0, 'high': 15.0, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'default_fit_flag': False},
    'h_z_gas_kpc': {'label': r"$h_{z,gas}$ (kpc)", 'fixed_val_from_arg': 'h_z_gas_fixed', 'default_fixed': 0.15, 'low': 0.05, 'high': 0.5, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'default_fit_flag': False},
}
SPARC_PARAM_CONFIG = {
    'rho_c_solar_kpc3': {'label': r"$\rho_c$ ($M_\odot/kpc^3$)", 'fixed_val_from_arg': 'rho_c_fixed', 'default_fixed': 1e7, 'low': 1e5, 'high': 2e9, 'default_fit_flag': True},
    'n_exp': {'label': r"$n$", 'fixed_val_from_arg': 'n_exp_fixed', 'default_fixed': 1.5, 'low': 0.1, 'high': 4.0, 'default_fit_flag': True},
    'stellar_ML_factor': {'label': r"$(M/L)_* factor$", 'fixed_val_from_arg': 'stellar_ML_factor_fixed', 'default_fixed': 1.0, 'low': 0.2, 'high': 2.5, 'fit_flag_arg': 'fit_sparc_ML', 'default_fit_flag': False},
}

def get_param_labels_and_bounds(ARGS):
    param_info_list = []
    config_to_use = MW_MULTI_COMP_PARAM_CONFIG
    logger.info("Configuring parameters for NEW multi-component Milky Way model.")
    
    for p_name, p_details in config_to_use.items():
        is_included = True 
        if 'include_flag_arg' in p_details and not getattr(ARGS, p_details.get('include_flag_arg', ''), False):
            is_included = False
        if not is_included:
            continue

        # *** THE CRITICAL FIX IS HERE ***
        # A parameter is fitted ONLY if its component's `--fit_*` flag is explicitly True.
        # We ignore the 'default_fit_flag' from the config dict, as it's confusing.
        # The user MUST specify what they want to fit.
        is_fitted = False
        if 'fit_flag_arg' in p_details:
            # getattr will find the value of --fit_disk_thin, --fit_bulge, etc. from the parsed args.
            # If the flag was given, it will be True. If not, it will be False.
            is_fitted = getattr(ARGS, p_details['fit_flag_arg'], False)
        
        current_val = getattr(ARGS, p_details['fixed_val_from_arg'])
        param_info_list.append({'name': p_name, 'label': p_details['label'], 'current_val': current_val,
                                'low': p_details['low'], 'high': p_details['high'], 'is_fitted': is_fitted})

    ARGS.all_param_info_list = param_info_list
    fitted_params_info = [p for p in param_info_list if p['is_fitted']]
    if not fitted_params_info: 
        logger.error("No parameters configured to be fitted! You must use at least one --fit_* flag (e.g., --fit_xi_params).")
        sys.exit(1)
        
    return ([p['name'] for p in fitted_params_info], [p['label'] for p in fitted_params_info],
            np.array([p['current_val'] for p in fitted_params_info]),
            np.array([p['low'] for p in fitted_params_info]), np.array([p['high'] for p in fitted_params_info]))
def reconstruct_full_theta_dict(theta_values_fitted, fitted_param_names, ARGS):
    full_theta_dict = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in ARGS.all_param_info_list: 
        if p_info['name'] not in full_theta_dict: 
            full_theta_dict[p_info['name']] = p_info['current_val'] 
    if ARGS.fit_target == 'milkyway':
        is_multi_comp_mode = ARGS.include_bulge or ARGS.include_disk_thick or ARGS.include_gas or not ARGS.use_old_single_disk_params
        if is_multi_comp_mode:
            full_theta_dict['include_bulge'] = ARGS.include_bulge
            full_theta_dict['include_disk_thin'] = ARGS.include_disk_thin
            full_theta_dict['include_disk_thick'] = ARGS.include_disk_thick
            full_theta_dict['include_gas'] = ARGS.include_gas
            full_theta_dict['include_bulge_density'] = ARGS.include_bulge 
    return full_theta_dict

def log_prior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted, ARGS):
    for i, val in enumerate(theta_values_fitted):
        if not (prior_bounds_low_fitted[i] <= val <= prior_bounds_high_fitted[i]): return -np.inf
    return 0.0

def log_likelihood(theta_values_fitted, fitted_param_names, R_data, v_data, sigma_data, xi_type_selected, ARGS,
                   sparc_galaxy_data_dict_full=None):
    current_params_full_dict = reconstruct_full_theta_dict(theta_values_fitted, fitted_param_names, ARGS)
    v_predicted = v_model_for_emcee(R_data, current_params_full_dict, xi_type_selected, ARGS, sparc_galaxy_data_dict_full)
    if not np.all(np.isfinite(v_predicted)): return -np.inf
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals_sq = ((v_data - v_predicted) / sigma_data_safe)**2
    log_L_val = -0.5 * np.sum(residuals_sq + np.log(2 * np.pi * sigma_data_safe**2))
    return log_L_val if np.isfinite(log_L_val) else -np.inf

def log_posterior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted,
                  R_data, v_data, sigma_data, xi_type_selected, ARGS,
                  sparc_galaxy_data_dict_full=None):
    lp = log_prior(theta_values_fitted, fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted, ARGS)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(theta_values_fitted, fitted_param_names, R_data, v_data, sigma_data, xi_type_selected, ARGS, sparc_galaxy_data_dict_full)
    if not np.isfinite(ll): return -np.inf
    return lp + ll

def run_mcmc_analysis(ARGS_in):
    logger.info(f"--- MCMC Analysis Setup for target: {ARGS_in.fit_target} ---")
    
    fitted_param_names, param_labels_fitted, p0_guess_means_fitted, prior_bounds_low_fitted, prior_bounds_high_fitted = get_param_labels_and_bounds(ARGS_in)
    ndim = len(fitted_param_names)
    
    logger.info(f"Fitting {ndim} parameters:")
    for p_info in ARGS_in.all_param_info_list:
        if p_info['is_fitted']:
            logger.info(f"  - (FIT) {p_info['name']:<25} | Prior Range: [{p_info['low']:.2e}, {p_info['high']:.2e}]")

    logger.info("The following parameters are FIXED:")
    any_fixed = False
    for p_info in ARGS_in.all_param_info_list:
        if not p_info['is_fitted']:
            logger.info(f"  - (FIX) {p_info['name']:<25} | Value: {p_info['current_val']:.2e}")
            any_fixed = True
    if not any_fixed:
        logger.info("  - None")

    R_obs_kpc, v_obs_kms, sigma_v_kms, sparc_galaxy_data_dict_for_fit = None, None, None, None
    if ARGS_in.fit_target == 'milkyway':
        logger.info("--- Loading Milky Way (Gaia) Data ---")
        gaia_data_dict = load_gaia(sample_max=ARGS_in.max_sample_gaia, force_new_query_gaia=ARGS_in.force_live_gaia, force_reprocess_raw=ARGS_in.force_reprocess)
        if not gaia_data_dict: logger.error("Failed to load Gaia data."); sys.exit(1)
        R_obs_kpc, v_obs_kms, sigma_v_kms = gaia_data_dict["R_kpc"], gaia_data_dict["v_obs"], gaia_data_dict["sigma_v"]
    elif ARGS_in.fit_target == 'sparc': 
        if not SPARC_AVAILABLE: logger.error("SPARC io not available."); sys.exit(1)
        if not ARGS_in.galaxy_id: logger.error("No SPARC --galaxy_id."); sys.exit(1)
        logger.info(f"--- Loading SPARC Galaxy: {ARGS_in.galaxy_id} ---")
        sparc_galaxy_data_dict_for_fit = load_single_sparc_galaxy(ARGS_in.galaxy_id, sparc_dir=ARGS_in.sparc_data_dir, assume_stellar_hz_kpc=ARGS_in.sparc_hz_star, assume_gas_hz_kpc=ARGS_in.sparc_hz_gas)
        if not sparc_galaxy_data_dict_for_fit: logger.error(f"Failed to load SPARC data."); sys.exit(1)
        R_obs_kpc, v_obs_kms, sigma_v_kms = sparc_galaxy_data_dict_for_fit['R_kpc'], sparc_galaxy_data_dict_for_fit['V_obs'], sparc_galaxy_data_dict_for_fit['e_V_obs']
    if R_obs_kpc is None or len(R_obs_kpc) == 0: logger.error("No data points to fit."); sys.exit(1)
    logger.info(f"Loaded {len(R_obs_kpc)} data points.")

    n_cores_to_use = ARGS_in.ncores
    
    logger.info(f"--- Running MCMC ({ARGS_in.nwalkers} walkers, {ARGS_in.nsteps} steps, on {n_cores_to_use} core(s)) using Emcee ---")
    pos0 = np.zeros((ARGS_in.nwalkers, ndim))
    for i in range(ndim):
        low, high, guess = prior_bounds_low_fitted[i], prior_bounds_high_fitted[i], p0_guess_means_fitted[i]
        spread = (high - low) * 0.05 
        walker_low, walker_high = np.clip(guess - spread/2, low, high), np.clip(guess + spread/2, low, high)
        if walker_low >= walker_high: walker_low, walker_high = low, high
        pos0[:, i] = np.random.uniform(walker_low, walker_high, ARGS_in.nwalkers)

    sampler_moves_obj = None
    if ARGS_in.sampler_move == 'kdemove' and hasattr(emcee, 'moves') and hasattr(emcee.moves, 'KDEMove'):
        try: sampler_moves_obj = emcee.moves.KDEMove(); logger.info("Using KDEMove.")
        except Exception as e_kdemove: logger.warning(f"KDEMove init failed: {e_kdemove}. Using default moves.")

    output_dir_path = Path(ARGS_in.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir_path.resolve()}")

    output_prefix_parts = [ARGS_in.fit_target, ARGS_in.galaxy_id if ARGS_in.galaxy_id else 'MW', ARGS_in.xi]
    if ARGS_in.fit_target == 'milkyway':
        model_desc_list_fn = []
        if ARGS_in.use_old_single_disk_params and not (ARGS_in.include_bulge or ARGS_in.include_disk_thick or ARGS_in.include_gas):
            model_desc_list_fn.append("SingleDiskOld")
        else: 
            if ARGS_in.include_bulge: model_desc_list_fn.append("B" + ("f" if ARGS_in.fit_bulge else "x"))
            if ARGS_in.include_disk_thin: model_desc_list_fn.append("DT" + ("f" if ARGS_in.fit_disk_thin else "x"))
            if ARGS_in.include_disk_thick: model_desc_list_fn.append("DK" + ("f" if ARGS_in.fit_disk_thick else "x"))
            if ARGS_in.include_gas: model_desc_list_fn.append("G" + ("f" if ARGS_in.fit_gas else "x"))
        if model_desc_list_fn : output_prefix_parts.extend(model_desc_list_fn)
        elif ARGS_in.fit_xi_params and len(fitted_param_names) == 2 : output_prefix_parts.append("XiOnly")
        elif not any([ARGS_in.include_bulge, ARGS_in.include_disk_thick, ARGS_in.include_gas]) and ARGS_in.include_disk_thin :
             output_prefix_parts.append("DT" + ("f" if ARGS_in.fit_disk_thin else "x"))
             
    output_prefix_final = "_".join(filter(None, output_prefix_parts))
    backend_filename = output_dir_path / f"backend_{output_prefix_final}.h5"
    
    backend = emcee.backends.HDFBackend(backend_filename)
    if not ARGS_in.resume_mcmc or not os.path.exists(backend_filename): 
        backend.reset(ARGS_in.nwalkers, ndim); logger.info(f"Initialized new MCMC. Backend: {backend_filename}")
    else:
        logger.info(f"Attempting resume from: {backend_filename}")
        try:
            if backend.iteration > 0 and backend.get_chain().shape[-1] == ndim:
                logger.info(f"Resuming from step {backend.iteration}.")
                pos0 = backend.get_last_sample().coords
            else: 
                logger.warning(f"Backend incompatible or empty. Resetting."); backend.reset(ARGS_in.nwalkers, ndim)
        except Exception as e_bk: logger.error(f"Backend error: {e_bk}. Resetting."); backend.reset(ARGS_in.nwalkers, ndim)

    sampler_pool_arg_for_emcee = None
    if n_cores_to_use > 1: sampler_pool_arg_for_emcee = Pool(processes=n_cores_to_use)
    
    sampler = emcee.EnsembleSampler(ARGS_in.nwalkers, ndim, log_posterior,
                                    args=(fitted_param_names, prior_bounds_low_fitted, prior_bounds_high_fitted,
                                          R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi, ARGS_in,
                                          sparc_galaxy_data_dict_for_fit),
                                    pool=sampler_pool_arg_for_emcee, moves=sampler_moves_obj, backend=backend)
    
    start_time_mcmc = time.time(); initial_step = sampler.iteration; steps_to_run = ARGS_in.nsteps - initial_step
    if steps_to_run > 0 :
        logger.info(f"Running MCMC for {steps_to_run} new steps (total target {ARGS_in.nsteps}).")
        # Use sampler's internal progress bar
        sampler.run_mcmc(pos0, steps_to_run, progress=True)
    else: 
        logger.info(f"MCMC target {ARGS_in.nsteps} reached/exceeded (current: {initial_step}).")
        
    if sampler_pool_arg_for_emcee: sampler_pool_arg_for_emcee.close(); sampler_pool_arg_for_emcee.join()
    logger.info(f"MCMC finished in {(time.time() - start_time_mcmc)/60:.2f} min. Total steps in backend: {sampler.iteration}")

    actual_burnin, actual_thin = ARGS_in.burnin_for_analysis, ARGS_in.thin_for_analysis
    try:
        current_chain_len = sampler.iteration
        discard_tau = min(ARGS_in.burnin_for_analysis, current_chain_len - 1 if current_chain_len > 0 else 0)
        if current_chain_len > discard_tau: 
            tau_est = sampler.get_autocorr_time(discard=discard_tau, tol=0, quiet=True)
            logger.info(f"Autocorr time est: {tau_est}")
            valid_tau = tau_est[np.isfinite(tau_est)]
            if len(valid_tau) > 0:
                max_tau = np.max(valid_tau)
                rec_burn, rec_thin = int(np.ceil(max_tau * 5)), max(1, int(np.ceil(max_tau / 2)))
                logger.info(f"Recommended analysis burn: ~{rec_burn}, thin: ~{rec_thin}")
                if ARGS_in.burnin_for_analysis < rec_burn: actual_burnin = rec_burn
                if ARGS_in.thin_for_analysis < rec_thin: actual_thin = rec_thin
    except Exception as e_ac: logger.warning(f"Autocorr est error: {e_ac}")
    actual_burnin = min(max(0, actual_burnin), sampler.iteration - 1 if sampler.iteration > 0 else 0)
    actual_thin = max(1, actual_thin)
    logger.info(f"Using analysis burn-in: {actual_burnin}, thinning: {actual_thin}")
    
    chain_flat = np.array([])
    if sampler.iteration > actual_burnin :
        try: chain_flat = sampler.get_chain(discard=actual_burnin, thin=actual_thin, flat=True)
        except Exception as e_gc: logger.error(f"Error getting chain: {e_gc}")
    logger.info(f"Effective samples for analysis plot: {len(chain_flat)}")
    if 0 < len(chain_flat) < 50 * ndim: logger.warning("Low effective samples for robust posteriors!")
    
    chain_filename = output_dir_path / f"chain_{output_prefix_final}.npy"
    np.save(chain_filename, chain_flat); logger.info(f"Saved chain: {chain_filename}")

    logger.info("\n--- Generating Posterior Diagnostics & Plot ---")
    param_summary_text = "Fitted Parameters (Median & 68% CI):\n"
    median_params_fitted_values = p0_guess_means_fitted 
    if len(chain_flat) > max(ndim * 5, 50):
        try:
            corner_plot_filename = output_dir_path / f"corner_{output_prefix_final}.png"
            figure = corner.corner(chain_flat, labels=param_labels_fitted, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 10})
            figure.savefig(corner_plot_filename); plt.close(figure)
            logger.info(f"Saved corner plot: {corner_plot_filename}")
        except Exception as e_corn: logger.error(f"Corner plot failed: {e_corn}")
        
        median_params_fitted_values = np.median(chain_flat, axis=0)
        p16, p84 = np.percentile(chain_flat, [16, 84], axis=0)
        for i, label in enumerate(param_labels_fitted):
            line = f"  {label:<35}: {median_params_fitted_values[i]:.4e} (+{p84[i]-median_params_fitted_values[i]:.2e} / -{median_params_fitted_values[i]-p16[i]:.2e})"
            logger.info(line); param_summary_text += line + "\n"
    else: 
        logger.warning("Chain too short for robust posteriors/corner plot. Using initial guesses for summary.")
        for i, label in enumerate(param_labels_fitted):
            line = f"  {label:<35}: {p0_guess_means_fitted[i]:.4e} (initial guess - chain too short)"
            logger.info(line); param_summary_text += line + "\n"
    
    median_params_full_dict = reconstruct_full_theta_dict(median_params_fitted_values, fitted_param_names, ARGS_in)
    
    R_plot = np.linspace(max(0.01, np.min(R_obs_kpc) if len(R_obs_kpc)>0 else 0.1), np.max(R_obs_kpc) if len(R_obs_kpc)>0 else 25.0, 200)
    v_median_plot = v_model_for_emcee(R_plot, median_params_full_dict, ARGS_in.xi, ARGS_in, sparc_galaxy_data_dict_for_fit)
    v16_plot, v84_plot = v_median_plot*0.9, v_median_plot*1.1
    if len(chain_flat) > ndim:
        n_env = min(200, len(chain_flat))
        if n_env > 0:
            v_samples_list = [v_model_for_emcee(R_plot, reconstruct_full_theta_dict(p_fit, fitted_param_names, ARGS_in), ARGS_in.xi, ARGS_in, sparc_galaxy_data_dict_for_fit) for p_fit in chain_flat[np.random.choice(len(chain_flat), n_env, replace=False)]]
            if v_samples_list: 
                v_model_samples_arr = np.array(v_samples_list)
                finite_v_samples = v_model_samples_arr[np.all(np.isfinite(v_model_samples_arr), axis=1)]
                if len(finite_v_samples) > 0: v16_plot, v84_plot = np.nanpercentile(finite_v_samples, [16,84], axis=0)
    
    plt.figure(figsize=(10,6)); plt.errorbar(R_obs_kpc, v_obs_kms, yerr=sigma_v_kms, fmt=".k", alpha=0.02, label=f"{ARGS_in.fit_target.capitalize()} Data", zorder=1)
    plt.plot(R_plot, v_median_plot, "r-", lw=2.5, label=f"Density-Metric Median ({ARGS_in.xi})", zorder=3)
    plt.fill_between(R_plot, v16_plot, v84_plot, color="red", alpha=0.3, zorder=2, label="68% CI")
    v_newton_plot = np.zeros_like(R_plot)
    if ARGS_in.fit_target == 'milkyway': 
        if ARGS_in.use_old_single_disk_params and not (ARGS_in.include_bulge or ARGS_in.include_disk_thick or ARGS_in.include_gas):
            M_d_plot = median_params_full_dict['M_disk_solar']; R_d_plot = median_params_full_dict['R_d_kpc']; h_z_d_plot = median_params_full_dict['h_z_disk_kpc']
            v_newton_plot = v_newton_kms_single_disk(R_plot, M_d_plot, R_d_plot, 0,0.5,False,0,7,False)
        else: v_newton_plot = v_baryon_total_newtonian_kms(R_plot, median_params_full_dict)
    elif ARGS_in.fit_target == 'sparc' and sparc_galaxy_data_dict_for_fit:
        ml = median_params_full_dict.get('stellar_ML_factor', 1.0)
        vds,vbs,vgs = sparc_galaxy_data_dict_for_fit['V_disk_comp_kms']**2*ml, sparc_galaxy_data_dict_for_fit['V_bulge_comp_kms']**2*ml, sparc_galaxy_data_dict_for_fit['V_gas_comp_kms']**2
        vn_sparc = np.sqrt(np.maximum(0, vds+vbs+vgs))
        if len(sparc_galaxy_data_dict_for_fit['R_kpc']) > 1 :
            if_vn = interp1d(sparc_galaxy_data_dict_for_fit['R_kpc'], vn_sparc, kind='linear', bounds_error=False, fill_value=(np.nan, np.nan))
            v_newton_plot = if_vn(R_plot)
    plt.plot(R_plot, v_newton_plot, "g--", lw=2, label="Newtonian (Baryons, Median)", zorder=2.5)
    
    title_parts_list = [(ARGS_in.galaxy_id or 'Milky Way'), f"Fit ({ARGS_in.xi})"]
    if ARGS_in.fit_target == 'milkyway':
        model_desc_list_title = []
        if ARGS_in.use_old_single_disk_params and not (ARGS_in.include_bulge or ARGS_in.include_disk_thick or ARGS_in.include_gas):
            model_desc_list_title.append("SingleDisk(Old)")
        else:
            if ARGS_in.include_bulge: model_desc_list_title.append("B" + ("f" if ARGS_in.fit_bulge else "x"))
            if ARGS_in.include_disk_thin: model_desc_list_title.append("DT" + ("f" if ARGS_in.fit_disk_thin else "x"))
            if ARGS_in.include_disk_thick: model_desc_list_title.append("DK" + ("f" if ARGS_in.fit_disk_thick else "x"))
            if ARGS_in.include_gas: model_desc_list_title.append("G" + ("f" if ARGS_in.fit_gas else "x"))
        if model_desc_list_title : title_parts_list.append(f"({'+'.join(model_desc_list_title)})")
        elif ARGS_in.fit_xi_params : title_parts_list.append("(Xi-Params Only)") 
    plt.title(' '.join(title_parts_list), fontsize=12); plt.xlabel("R (kpc)"); plt.ylabel("v (km/s)")
    plt.legend(fontsize=8); plt.grid(True, ls=':', alpha=0.7); plt.ylim(bottom=0); plt.xlim(left=0)
    rc_plot_filename = output_dir_path / f"rotation_curve_fit_{output_prefix_final}.png"
    plt.tight_layout(); plt.savefig(rc_plot_filename, dpi=150); plt.close()
    logger.info(f"Saved RC plot: {rc_plot_filename}")
    
    aic_bic_text = "AIC/BIC: Not calculable.\n"; rms_text = "RMS: No data.\n"
    if len(chain_flat) > ndim:
        max_logL_val = log_likelihood(median_params_fitted_values, fitted_param_names, R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi, ARGS_in, sparc_galaxy_data_dict_for_fit)
        if np.isfinite(max_logL_val): 
            k,N=ndim,len(R_obs_kpc)
            AIC,BIC=2*k-2*max_logL_val,k*np.log(N)-2*max_logL_val
            aic_bic_text=f"MaxLogL(median): {max_logL_val:.2f}\nk:{k}, N:{N}\nAIC:{AIC:.2f}\nBIC:{BIC:.2f}\n"
    if len(R_obs_kpc)>0:
        m_in = R_obs_kpc < 5; m_out = (R_obs_kpc > 10) & (R_obs_kpc < 20)
        def calc_rms(m,p_full):
            if np.sum(m)==0: return np.nan
            vp = v_model_for_emcee(R_obs_kpc[m],p_full,ARGS_in.xi,ARGS_in,sparc_galaxy_data_dict_for_fit)
            return np.sqrt(np.nanmean((v_obs_kms[m]-vp)**2)) if np.all(np.isfinite(vp)) else np.nan
        rms_i,rms_o=calc_rms(m_in,median_params_full_dict),calc_rms(m_out,median_params_full_dict)
        rms_text=f"RMS R<5kpc: {rms_i:.2f} (N={np.sum(m_in)})\nRMS 10<R<20kpc: {rms_o:.2f} (N={np.sum(m_out)})\n"
    
    summary_filename = output_dir_path / f"info_summary_{output_prefix_final}.txt"
    with open(summary_filename, "w") as f:
        f.write(f"--- Fit Summary: {output_prefix_final} ---\n"); f.write(param_summary_text)
        f.write("\nFixed Parameters (values are from ARGS defaults/user input):\n"); any_fixed=False
        if hasattr(ARGS_in, 'all_param_info_list'):
            for p_info in ARGS_in.all_param_info_list:
                if not p_info['is_fitted']: f.write(f"  {p_info['label']:<35}: {p_info['current_val']:.4e} (fixed)\n"); any_fixed=True
        if not any_fixed: f.write("  None\n")
        f.write("\n--- Model Comparison ---\n"); f.write(aic_bic_text); f.write("\n--- RMS Residuals ---\n"); f.write(rms_text)
    logger.info(f"Saved summary: {summary_filename}")


# --- Script Entry Point ---
if __name__ == '__main__':
    if platform.system() in ["Windows", "Darwin"]:
        freeze_support()

    parser = argparse.ArgumentParser(description="Fit density-dependent metric model (Emcee runner).")
    parser.add_argument('--output_dir', type=str, default="chains_emcee", 
                        help="Directory to save output chains, plots, and summaries.")
    parser.add_argument('--fit_target', type=str, default='milkyway', choices=['milkyway', 'sparc'])
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'])
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    mcmc_g = parser.add_argument_group('MCMC Settings')
    mcmc_g.add_argument('--nwalkers', type=int, default=64)
    mcmc_g.add_argument('--nsteps', type=int, default=5000) 
    mcmc_g.add_argument('--burnin_for_analysis', type=int, default=1000)
    mcmc_g.add_argument('--thin_for_analysis', type=int, default=50)
    mcmc_g.add_argument('--ncores', type=int, default=-1, help="Number of cores. -1 to auto-detect and use all but one.")
    mcmc_g.add_argument('--sampler_move', type=str, default='default', choices=['default', 'kdemove'])
    mcmc_g.add_argument('--resume_mcmc', action='store_true')
    mw_data_g = parser.add_argument_group('Milky Way Data Settings')
    mw_data_g.add_argument('--max_sample_gaia', type=int, default=80000)
    mw_data_g.add_argument('--force_live_gaia', action='store_true')
    mw_data_g.add_argument('--force_reprocess', action='store_true')
    mw_model_g = parser.add_argument_group('Milky Way Model Components')
    mw_model_g.add_argument('--use_old_single_disk_params', action='store_true', default=False, 
                           help="Use old single disk param names (M_disk_solar etc.) & physics functions.")
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True) 
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)
    added_cli_flags = set()
    for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
        if 'fit_flag_arg' in p_details_cfg:
            fit_flag_cli_name = f"--{p_details_cfg['fit_flag_arg']}"
            if fit_flag_cli_name not in added_cli_flags:
                 mw_model_g.add_argument(fit_flag_cli_name, action='store_true', default=False,
                                         help=f"Fit parameters for {p_details_cfg['fit_flag_arg'].replace('fit_', '')} component(s).")
                 added_cli_flags.add(fit_flag_cli_name)
        fixed_val_cli_name = f"--{p_details_cfg['fixed_val_from_arg']}"
        if fixed_val_cli_name not in added_cli_flags:
            mw_model_g.add_argument(fixed_val_cli_name, type=float, default=p_details_cfg['default_fixed'], 
                                    help=f"Fixed/initial for multi-comp {p_name_cfg}. Default: {p_details_cfg['default_fixed']:.2e}")
            added_cli_flags.add(fixed_val_cli_name)
    for p_name_cfg, p_details_cfg in MW_SINGLE_DISK_OLD_PARAMS.items():
        fixed_val_cli_name = f"--{p_details_cfg['fixed_val_from_arg']}"
        if fixed_val_cli_name not in added_cli_flags:
            mw_model_g.add_argument(fixed_val_cli_name, type=float, default=p_details_cfg['default_fixed'],
                                    help=f"Fixed/initial for OLD single-disk {p_name_cfg}. Default: {p_details_cfg['default_fixed']:.2e}")
            added_cli_flags.add(fixed_val_cli_name)
    mw_model_g.add_argument('--check_kz', action='store_true', default=False)
    mw_model_g.add_argument('--check_microlensing', action='store_true', default=False)
    sparc_g = parser.add_argument_group('SPARC Specific Settings')
    sparc_g.add_argument('--galaxy_id', type=str, default=None)
    sparc_g.add_argument('--sparc_data_dir', type=str, default="data/sparc_data")
    sparc_g.add_argument('--sparc_hz_star', type=float, default=0.3)
    sparc_g.add_argument('--sparc_hz_gas', type=float, default=0.1)
    for p_name_cfg, p_details_cfg in SPARC_PARAM_CONFIG.items():
        if 'fit_flag_arg' in p_details_cfg:
            fit_flag_cli_name = f"--{p_details_cfg['fit_flag_arg']}"
            if fit_flag_cli_name not in added_cli_flags:
                 sparc_g.add_argument(fit_flag_cli_name, action='store_true', default=False, 
                                      help=f"Fit {p_details_cfg['fit_flag_arg'].replace('fit_', '')} for SPARC.")
                 added_cli_flags.add(fit_flag_cli_name)
        fixed_val_cli_name = f"--{p_details_cfg['fixed_val_from_arg']}"
        if fixed_val_cli_name not in added_cli_flags:
            sparc_g.add_argument(fixed_val_cli_name, type=float, default=p_details_cfg['default_fixed'],
                                 help=f"Fixed/initial value for SPARC {p_name_cfg}. Default: {p_details_cfg['default_fixed']:.2e}")
            added_cli_flags.add(fixed_val_cli_name)
    parser.add_argument('--kill_existing', action='store_true')
    
    ARGS_global = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, ARGS_global.log_level.upper(), logging.INFO))
    if SPARC_AVAILABLE: logger.info("sparc_io module loaded successfully.")
    else: logger.warning("sparc_io module not available. SPARC galaxy fitting disabled.")

    _check_flag_consistency(ARGS_global, logger)
    run_physics_self_tests() 

    for config_dict in [MW_MULTI_COMP_PARAM_CONFIG, SPARC_PARAM_CONFIG]:
        for p_details_cfg in config_dict.values():
            if 'fit_flag_arg' in p_details_cfg:
                fit_flag_name = p_details_cfg['fit_flag_arg']
                if not f"--{fit_flag_name}" in sys.argv:
                    setattr(ARGS_global, fit_flag_name, p_details_cfg['default_fit_flag'])
    
    if ARGS_global.ncores < 1:
        try: 
            n_auto_cores = max(1, cpu_count() - 1)
            ARGS_global.ncores = n_auto_cores
            logger.info(f"Auto-set ncores to: {ARGS_global.ncores}")
        except NotImplementedError: 
            ARGS_global.ncores = 1
            logger.warning("cpu_count() failed, defaulting ncores to 1.")
            
    if ARGS_global.kill_existing: kill_existing_instances(os.path.basename(__file__))
    
    run_mcmc_analysis(ARGS_global)
    logger.info("\n--- main2.py Script Finished ---")