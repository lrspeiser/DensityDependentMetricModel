#!/usr/bin/env python3
"""
run_dynesty.py - Enhanced dynamic nested sampling for the Density-Metric model.

This module implements Bayesian parameter estimation for the density-dependent
gravitational modification model using the dynesty nested sampling package.

Key features:
- Dynamic nested sampling with adaptive live points
- Physical plausibility checks to prevent unphysical parameter exploration
- Curriculum learning approach for complex parameter spaces
- Gaussian Process surrogate modeling for computational efficiency
- Comprehensive monitoring and diagnostic output
- Checkpoint support for long runs
- Multi-threading support

Major improvements in v2.0:
- Parameter sanity checks during sampling
- Tighter, physically motivated prior bounds
- Enhanced monitoring with parameter health checks
- Automatic detection of pathological parameter regions
- Better initialization strategies
- Validation integration

Author: [Your name]
Version: 2.0 (Enhanced with physical constraints)
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
from density_metric2 import v_total_kms 

# Control debug output
DEBUG_COUNTER_MAX = 100  # Maximum debug messages to prevent log spam
debug_counter = 0

prior_data = np.load("chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz")
samples = prior_data['samples']
weights = prior_data['weights']

param_names = [
    'rho_c_solar_kpc3', 'n_exp',
    'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
    'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
    'M_bulge_solar', 'a_bulge_kpc',
    'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
]

median_vals = np.average(samples, weights=weights, axis=0)
previous_best = dict(zip(param_names, median_vals))


# ============================================================================
# Optional imports for advanced features
# ============================================================================

# Gaussian Process surrogate modeling
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import qmc  # For Latin Hypercube sampling
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("WARNING: scikit-learn not found. Gaussian Process surrogate modeling disabled.")

# Dynesty nested sampling
try:
    from dynesty import DynamicNestedSampler, utils as dyfunc
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    print("CRITICAL: Dynesty library not found. Please install it: pip install dynesty")
    sys.exit(1)

# Local physics modules
try:
    from density_metric2 import (
        v_baryon_total_newtonian_kms, 
        rho_baryon_total_midplane_solar_kpc3, 
        XI_FUNCTION_MAP, 
        run_physics_self_tests,
        G_ASTRO_UNITS,
        R_SUN_KPC
    )
    from data_io import load_gaia
    from main2 import get_param_labels_and_bounds as get_param_config_main_module
except ImportError as e:
    print(f"CRITICAL: Could not import local modules: {e}")
    sys.exit(1)


# ============================================================================
# Set Up Multi-Thread Safe Logging
# ============================================================================


# Set up module logger
logger = None

import run_history
from run_history import finalize_record

def get_or_create_logger():
    """Get or create the module logger. Safe for multiprocessing."""
    import logging
    logger = logging.getLogger("run_dynesty")
    if not logger.handlers:  # Only add handler if none exist
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


# ============================================================================
# Physical Constraints and Validation
# ============================================================================

# Physical bounds for parameters (based on MW observations and theory)
PHYSICAL_BOUNDS = {
    # Mass parameters (M☉)
    'M_disk_thin_solar':   {'min': 2.4e10, 'max': 9.0e10, 'typical': 4.0e10},   # Lower min, slightly higher max
    'M_disk_thick_solar':  {'min': 5e9,    'max': 3.5e10, 'typical': 1.5e10},   # Allow more mass if needed
    'M_bulge_solar':       {'min': 0.5e10, 'max': 2.5e10, 'typical': 1.2e10},   # Centered on MW bulge fits
    'M_gas_solar':         {'min': 5e9,    'max': 6e10,   'typical': 3.0e10},

    # Scale lengths (kpc)
    'R_d_thin_kpc':        {'min': 2.0,  'max': 4.5,    'typical': 2.6},     # Gaia + Bovy 2017 range
    'R_d_thick_kpc':       {'min': 3.5,  'max': 9.5,    'typical': 4.5},     # Extended if bimodality demands it
    'R_d_gas_kpc':         {'min': 4.0,  'max': 15.0,   'typical': 7.0},
    'a_bulge_kpc':         {'min': 0.2,  'max': 2.0,    'typical': 0.7},

    # Scale heights (kpc)
    'h_z_thin_kpc':        {'min': 0.15, 'max': 0.5,    'typical': 0.3},     # Based on star counts and vertical profile
    'h_z_thick_kpc':       {'min': 0.7,  'max': 1.5,    'typical': 0.9},     # Raised min for stability; ~2–3× thin
    'h_z_gas_kpc':         {'min': 0.05, 'max': 0.4,    'typical': 0.15},

    # Other parameters
    'M_total':             {'min': 5e10, 'max': 2e11,   'typical': 1e11},
    'rho_c_solar_kpc3':    {'min': 1e12,  'max': 1e15,   'typical': 1e13},
    'n_exp':               {'min': 0.5,  'max': 4.0,    'typical': 2.7},
}


# Expected ranges for validation
EXPECTED_XI_AT_SOLAR = (0.5, 3.0)  # Xi should not suppress gravity too much at R_sun
EXPECTED_V_AT_SOLAR = (100, 300)   # TEMPORARILY RELAXED for initial exploration

def load_previous_best_params():
    """Load previous best parameters if available."""
    try:
        prior_data = np.load("chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz")
        samples = prior_data['samples']
        weights = prior_data['weights']
        
        param_names = [
            'rho_c_solar_kpc3', 'n_exp',
            'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
            'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
            'M_bulge_solar', 'a_bulge_kpc',
            'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
        ]
        
        median_vals = np.average(samples, weights=weights, axis=0)
        return dict(zip(param_names, median_vals))
    except Exception as e:
        logger.warning(f"Could not load previous results: {e}")
        return None


# ============================================================================
# Parameter Configuration with Enhanced Bounds
# ============================================================================
MW_MULTI_COMP_PARAM_CONFIG = {
    # --- Gravity Parameters with MORE LENIENT PRIORS ---
    'rho_c_solar_kpc3': {
        'label': "rho_c (M_sun/kpc^3)", 
        'fixed_val_from_arg': 'rho_c_fixed', 
        'default_fixed': 1e13,    # Cassini-safe value
        'low': 1e12,              # Must be >> 1e8 (galaxy)
        'high': 1e15,             # Must be << 1e29 (Saturn)
        'fit_flag_arg': 'fit_xi_params', 
        'log_prior': True
    },
    'A': {
        'label': "A (enhancement factor)",
        'fixed_val_from_arg': 'A_fixed',
        'default_fixed': 1.0,  # This gives 2x total enhancement
        'low': 0.5,
        'high': 3.0,
        'fit_flag_arg': 'fit_xi_params',
        'log_prior': False
    },
    'n_exp': {
        'label': "n", 'fixed_val_from_arg': 'n_exp_fixed', 'default_fixed': 1.0, # Lower default
        'low': 0.1, 'high': 2.5, 'fit_flag_arg': 'fit_xi_params', 'log_prior': False
    },
    
    # --- Baryonic Components with Correct, Hardcoded Defaults ---
    'M_disk_thin_solar': {
        'label': "M_disk_thin (M_sun)", 'fixed_val_from_arg': 'M_disk_thin_fixed', 'default_fixed': 4.0e10,
        'low': 2.4e10, 'high': 9.0e10, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'log_prior': True
    },
    'R_d_thin_kpc': {
        'label': "R_d_thin (kpc)", 'fixed_val_from_arg': 'R_d_thin_fixed', 'default_fixed': 2.6,
        'low': 2.0, 'high': 4.5, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'log_prior': False
    },
    'h_z_thin_kpc': {
        'label': "h_z_thin (kpc)", 'fixed_val_from_arg': 'h_z_thin_fixed', 'default_fixed': 0.3,
        'low': 0.15, 'high': 0.5, 'fit_flag_arg': 'fit_disk_thin', 'include_flag_arg': 'include_disk_thin', 'log_prior': False
    },
    'M_disk_thick_solar': {
        'label': "M_disk_thick (M_sun)", 'fixed_val_from_arg': 'M_disk_thick_fixed', 'default_fixed': 1.5e10,
        'low': 5e9, 'high': 3.5e10, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'log_prior': True
    },
    'R_d_thick_kpc': {
        'label': "R_d_thick (kpc)", 'fixed_val_from_arg': 'R_d_thick_fixed', 'default_fixed': 4.5,
        'low': 3.5, 'high': 9.5, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'log_prior': False
    },
    'h_z_thick_kpc': {
        'label': "h_z_thick (kpc)", 'fixed_val_from_arg': 'h_z_thick_fixed', 'default_fixed': 0.9,
        'low': 0.7, 'high': 1.5, 'fit_flag_arg': 'fit_disk_thick', 'include_flag_arg': 'include_disk_thick', 'log_prior': False
    },
    'M_bulge_solar': {
        'label': "M_bulge (M_sun)", 'fixed_val_from_arg': 'M_bulge_fixed', 'default_fixed': 1.2e10,
        'low': 0.5e10, 'high': 2.5e10, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge', 'log_prior': True
    },
    'a_bulge_kpc': {
        'label': "a_bulge (kpc)", 'fixed_val_from_arg': 'a_bulge_fixed', 'default_fixed': 0.7,
        'low': 0.2, 'high': 2.0, 'fit_flag_arg': 'fit_bulge', 'include_flag_arg': 'include_bulge', 'log_prior': False
    },
    'M_gas_solar': {
        'label': "M_gas (M_sun)", 'fixed_val_from_arg': 'M_gas_fixed', 'default_fixed': 3.0e10,
        'low': 5e9, 'high': 6e10, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'log_prior': True
    },
    'R_d_gas_kpc': {
        'label': "R_d_gas (kpc)", 'fixed_val_from_arg': 'R_d_gas_fixed', 'default_fixed': 7.0,
        'low': 4.0, 'high': 15.0, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'log_prior': False
    },
    'h_z_gas_kpc': {
        'label': "h_z_gas (kpc)", 'fixed_val_from_arg': 'h_z_gas_fixed', 'default_fixed': 0.15,
        'low': 0.05, 'high': 0.4, 'fit_flag_arg': 'fit_gas', 'include_flag_arg': 'include_gas', 'log_prior': False
    },
    
    # --- MASS THRESHOLD PARAMETERS (FIXED FOR GALAXY SCALES) ---
    'M_crit_msun': {
        'label': "M_crit (M_sun)",  # Fixed: removed "log10" from label
        'fixed_val_from_arg': 'M_crit_fixed', 
        'default_fixed': 5e10,  # Fixed: galaxy-scale default (was 0.01)
        'low': 1e9,             # Fixed: dwarf galaxy scale (was 1e-4)
        'high': 1e12,           # Fixed: massive galaxy scale (was 1.0)
        'fit_flag_arg': 'fit_xi_params', 
        'log_prior': True
    },
    'xi_boost': {
        'label': "xi_boost", 
        'fixed_val_from_arg': 'xi_boost_fixed', 
        'default_fixed': 2.0,  # Changed: more moderate default (was 3.0)
        'low': 1.1,            # Changed: allow smaller enhancements (was 2.0)
        'high': 4.0,           # Keep upper bound
        'fit_flag_arg': 'fit_xi_params', 
        'log_prior': False
    },
    'width': {
        'label': "width (M_crit fraction)",  # Clarified: it's a fraction of M_crit
        'fixed_val_from_arg': 'width_fixed', 
        'default_fixed': 0.3,  # Changed: wider transition (was 0.1)
        'low': 0.01,           # Keep lower bound
        'high': 1.0,           # Keep upper bound
        'fit_flag_arg': 'fit_xi_params', 
        'log_prior': False
    },
}

# ============================================================================
# Physical Plausibility Checks
# ============================================================================

def check_physical_plausibility(
    theta_values: np.ndarray, 
    param_names: List[str],
    args_obj: argparse.Namespace
) -> Tuple[bool, str]:
    """
    Check if parameters are physically reasonable. Includes extensive logging for diagnostics.
    """
    logger = get_or_create_logger()
    logger.debug("\n--- Running check_physical_plausibility ---")

    try:
        params = dict(zip(param_names, theta_values))
        logger.debug(f"Parameter vector length: {len(theta_values)}")
        logger.debug(f"Parameter names: {param_names}")
        logger.debug(f"Parameter values: {theta_values}")
    except Exception as e:
        logger.error(f"❌ Failed to unpack parameters: {e}")
        return False, "Parameter unpacking failure"

    # 1. Check total baryonic mass
    try:
        mass_components = ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']
        total_mass = sum(
            params.get(comp, 0.0)
            for comp in mass_components
            if getattr(args_obj, f"include_{comp.split('_solar')[0]}", False)
        )
        logger.debug(f"Total baryonic mass (included components only): {total_mass:.2e} M☉")

        if total_mass > 1e6 and (
            total_mass < PHYSICAL_BOUNDS['M_total']['min'] or
            total_mass > PHYSICAL_BOUNDS['M_total']['max']
        ):
            logger.warning("Total mass outside physical bounds")
            return False, f"Total mass {total_mass:.2e} outside physical bounds."
    except Exception as e:
        logger.error(f"Error during mass check: {e}")
        return False, "Failed during total mass check"

    # 2. Check scale length ordering
    if 'R_d_thick_kpc' in params and 'R_d_thin_kpc' in params:
        if params['R_d_thick_kpc'] < params['R_d_thin_kpc']:
            logger.warning("Thick disk scale length smaller than thin disk")
            return False, "Thick disk scale length cannot be smaller than thin disk."

    # 3. Check scale height ordering
    if 'h_z_thick_kpc' in params and 'h_z_thin_kpc' in params:
        if params['h_z_thick_kpc'] < 2 * params['h_z_thin_kpc']:
            logger.warning("Thick disk scale height less than 2x thin disk")
            return False, "Thick disk scale height must be at least 2x thin disk."

    # 4. Check xi and velocity at solar radius
    try:
        logger.debug("Beginning velocity/xi plausibility check at R_sun")

        # Reconstruct full param set with includes and fixed values
        params_for_calc = params.copy()

        if getattr(args_obj, "all_param_info_list", None):
            for p_info in args_obj.all_param_info_list:
                if p_info['name'] not in params_for_calc:
                    params_for_calc[p_info['name']] = p_info['current_val']
        else:
            logger.warning("⚠️ args_obj.all_param_info_list is None — skipping fixed-value injection")

        for component in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
            include_flag = getattr(args_obj, f'include_{component}', False)
            params_for_calc[f'include_{component}'] = include_flag
            logger.debug(f"Component include_{component} = {include_flag}")

        r_solar = np.array([R_SUN_KPC])
        logger.debug(f"R_SUN_KPC = {R_SUN_KPC:.2f} kpc")

        # Calculate velocities using v_total_kms which handles all xi types correctly
        try:
            v_model_solar = v_total_kms(r_solar, params_for_calc, xi_type=args_obj.xi)[0]
            v_newton_solar = v_baryon_total_newtonian_kms(r_solar, params_for_calc)[0]
            logger.debug(f"v_model(R_sun) = {v_model_solar:.2f} km/s")
            logger.debug(f"v_newton(R_sun) = {v_newton_solar:.2f} km/s")
        except Exception as e:
            logger.warning(f"Failed to calculate velocities: {e}")
            return False, f"Velocity calculation failed: {e}"

        # Compute effective xi from velocity ratio
        if v_newton_solar > 1e-6:
            xi_solar = (v_model_solar / v_newton_solar) ** 2
        else:
            xi_solar = 1.0
        logger.debug(f"xi(R_sun) = {xi_solar:.3f}")

        # 4a. Xi range check - more lenient for mass_threshold
        if args_obj.xi == 'mass_threshold':
            # Mass threshold can have different behavior, be more lenient
            if xi_solar < 0.1 or xi_solar > 5.0:
                return False, f"xi at R_sun = {xi_solar:.3f} is extreme for mass_threshold model"
        else:
            # Standard xi range check for other models
            if not (EXPECTED_XI_AT_SOLAR[0] <= xi_solar <= EXPECTED_XI_AT_SOLAR[1]):
                return False, (
                    f"xi at R_sun = {xi_solar:.3f} is outside the expected range "
                    f"[{EXPECTED_XI_AT_SOLAR[0]}, {EXPECTED_XI_AT_SOLAR[1]}]"
                )

        # 4b. Check velocity magnitude
        if not (EXPECTED_V_AT_SOLAR[0] <= v_model_solar <= EXPECTED_V_AT_SOLAR[1]):
            return False, (
                f"Predicted v(R_sun) = {v_model_solar:.0f} km/s is outside the expected range "
                f"[{EXPECTED_V_AT_SOLAR[0]}, {EXPECTED_V_AT_SOLAR[1]}]"
            )

        # 5. Cassini test
        passes_cassini, cassini_msg = check_cassini_compatibility(params, args_obj.xi)
        if not passes_cassini:
            logger.warning(f"Cassini test failed: {cassini_msg}")
            return False, cassini_msg

    except Exception as e:
        import traceback
        logger.warning(f"⚠️ Plausibility check for xi/velocity failed with error: {e}")
        logger.debug(traceback.format_exc())
        return False, "Velocity calculation failed during plausibility check"


    # If all checks pass
    return True, "OK"


def check_parameter_evolution(
    recent_samples: np.ndarray,
    param_names: List[str],
    logger: logging.Logger = None  # Make logger optional
) -> Dict[str, Any]:
    """
    Analyze parameter evolution to detect pathological behavior.
    
    Checks for:
    - Parameters stuck at bounds
    - Runaway mass accumulation
    - Extreme parameter correlations
    - Bimodal distributions
    
    Parameters
    ----------
    recent_samples : np.ndarray
        Recent samples from the sampler (N_samples x N_params)
    param_names : list
        Parameter names
    logger : logging.Logger, optional
        Logger for warnings (will create one if None)
        
    Returns
    -------
    dict
        Analysis results with warnings and statistics
    """
    # Get logger safely for multiprocessing
    if logger is None:
        logger = get_or_create_logger()
    
    if len(recent_samples) < 100:
        return {'status': 'insufficient_samples'}
    
    results = {
        'status': 'ok',
        'warnings': [],
        'stats': {}
    }
    
    # 1. Check for parameters stuck at bounds
    for i, param in enumerate(param_names):
        if param not in MW_MULTI_COMP_PARAM_CONFIG:
            continue
            
        config = MW_MULTI_COMP_PARAM_CONFIG[param]
        values = recent_samples[:, i]
        
        # Check if >90% of samples are within 5% of bounds
        near_lower = np.sum(values < config['low'] * 1.05) / len(values)
        near_upper = np.sum(values > config['high'] * 0.95) / len(values)
        
        if near_lower > 0.9:
            results['warnings'].append(f"{param} stuck at lower bound")
            results['status'] = 'boundary_issue'
        elif near_upper > 0.9:
            results['warnings'].append(f"{param} stuck at upper bound")
            results['status'] = 'boundary_issue'
    
    # 2. Check total mass evolution
    mass_indices = [i for i, name in enumerate(param_names) 
                   if 'M_' in name and 'solar' in name]
    
    if mass_indices:
        total_masses = np.sum(recent_samples[:, mass_indices], axis=1)
        mass_trend = np.polyfit(range(len(total_masses)), total_masses, 1)[0]
        
        # Check if mass is growing rapidly
        if mass_trend > 1e9:  # Growing by >1e9 M_sun per sample
            results['warnings'].append(f"Runaway mass accumulation: {mass_trend:.2e} M_sun/sample")
            results['status'] = 'mass_runaway'
        
        results['stats']['total_mass_median'] = np.median(total_masses)
        results['stats']['total_mass_std'] = np.std(total_masses)
    
    # 3. Check parameter correlations
    if len(recent_samples) > 200:
        corr_matrix = np.corrcoef(recent_samples.T)
        np.fill_diagonal(corr_matrix, 0)
        
        # Find problematic correlations
        high_corr_threshold = 0.95
        high_corr_indices = np.where(np.abs(corr_matrix) > high_corr_threshold)
        
        for idx1, idx2 in zip(high_corr_indices[0], high_corr_indices[1]):
            if idx1 < idx2:  # Avoid duplicates
                corr_val = corr_matrix[idx1, idx2]
                results['warnings'].append(
                    f"High correlation ({corr_val:.2f}): {param_names[idx1]} ↔ {param_names[idx2]}"
                )
                if results['status'] == 'ok':
                    results['status'] = 'high_correlation'
    
    # 4. Check for bimodality (might indicate multiple solutions)
    for i, param in enumerate(param_names):
        values = recent_samples[:, i]
        
        # Simple bimodality check using Hartigan's dip test approximation
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        if n > 50:
            # Check for gap in middle 50% of distribution
            mid_range = sorted_vals[int(n*0.25):int(n*0.75)]
            if len(mid_range) > 2:
                gaps = np.diff(mid_range)
                max_gap = np.max(gaps)
                median_gap = np.median(gaps)
                
                if max_gap > 5 * median_gap:
                    results['warnings'].append(f"{param} shows bimodal distribution")
                    if results['status'] == 'ok':
                        results['status'] = 'bimodal'
    
    return results

def setup_xi_parameters_for_mode(args):
    """
    Setup parameter configuration based on the xi mode.
    This ensures all necessary parameters are included.
    """
    logger = logging.getLogger("run_dynesty")
    
    if args.xi == 'grav_color':
        logger.info("📊 Setting up parameters for gravitational color confinement")
        
        # CRITICAL: Ensure rho_c is in the args with a default value
        if not hasattr(args, 'rho_c_fixed') or args.rho_c_fixed is None:
            args.rho_c_fixed = 5e8  # Default value
            logger.info(f"   Setting default rho_c_fixed = {args.rho_c_fixed:.1e}")
        
        # CRITICAL: Ensure n_exp has a value (even if not used)
        if not hasattr(args, 'n_exp_fixed') or args.n_exp_fixed is None:
            args.n_exp_fixed = 2.0  # Not used but needed
            logger.info(f"   Setting default n_exp_fixed = {args.n_exp_fixed}")

        # Add/update rho_c (still needed!)
        if 'rho_c_solar_kpc3' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'] = {}
        
        MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'].update({
            'label': "rho_c (M_sun/kpc^3)",
            'fixed_val_from_arg': 'rho_c_fixed',
            'default_fixed': 1e13,  # Galaxy-appropriate
            'low': 1e12,
            'high': 1e15,
            'fit_flag_arg': 'fit_rho_c',
            'log_prior': True,
            'physical_check': True
        })
        
        # Ensure gamma_exp is configured
        if 'gamma_exp' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['gamma_exp'] = {}
            
        MW_MULTI_COMP_PARAM_CONFIG['gamma_exp'].update({
            'label': "γ (grav-color)",
            'fixed_val_from_arg': 'gamma_fixed',
            'default_fixed': 2.0,  # Galaxy-appropriate
            'low': 1.0,
            'high': 3.0,
            'fit_flag_arg': 'fit_gamma',
            'log_prior': False,
            'physical_check': False
        })
        
        # Ensure lambda_g is configured
        if 'lambda_g' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['lambda_g'] = {}
            
        MW_MULTI_COMP_PARAM_CONFIG['lambda_g'].update({
            'label': "λ_g",
            'fixed_val_from_arg': 'lambda_g_fixed',
            'default_fixed': 1.5,  # Galaxy-appropriate (NOT 8!)
            'low': 0.5,
            'high': 4.0,
            'fit_flag_arg': 'fit_lambda_g',
            'log_prior': False,
            'physical_check': False
        })
        
        # n_exp is not used for grav_color but might be needed for compatibility
        if 'n_exp' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['n_exp'] = {}
            
        MW_MULTI_COMP_PARAM_CONFIG['n_exp'].update({
            'label': "n",
            'fixed_val_from_arg': 'n_exp_fixed',
            'default_fixed': 2.0,
            'low': 0.5,
            'high': 2.0,
            'fit_flag_arg': 'fit_n_exp',  # Different flag
            'log_prior': False,
            'physical_check': True
        })
        
        # Set appropriate defaults if not specified
        if not hasattr(args, 'rho_c_fixed') or args.rho_c_fixed is None:
            args.rho_c_fixed = 5e7
        if not hasattr(args, 'gamma_fixed') or args.gamma_fixed is None:
            args.gamma_fixed = 2.0
        if not hasattr(args, 'lambda_g_fixed') or args.lambda_g_fixed is None:
            args.lambda_g_fixed = 1.5
        if not hasattr(args, 'n_exp_fixed') or args.n_exp_fixed is None:
            args.n_exp_fixed = 2.0  # Not used but needed for compatibility
        
        # Add fit_rho_c flag if using fit_xi_params
        if hasattr(args, 'fit_xi_params') and args.fit_xi_params:
            args.fit_rho_c = True
            logger.info("   fit_xi_params enabled → fitting rho_c")
        
    else:
        # Standard xi functions need rho_c and n_exp
        logger.info(f"📊 Setting up parameters for {args.xi} xi function")
        
        # Ensure standard parameters are present
        if 'rho_c_solar_kpc3' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'] = {}
            
        MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'].update({
            'label': "rho_c (M_sun/kpc^3)",
            'fixed_val_from_arg': 'rho_c_fixed',
            'default_fixed': 1e13,
            'low': 1e12,
            'high': 1e15,
            'fit_flag_arg': 'fit_xi_params',
            'log_prior': True,
            'physical_check': True
        })
        
        if 'n_exp' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['n_exp'] = {}
            
        MW_MULTI_COMP_PARAM_CONFIG['n_exp'].update({
            'label': "n",
            'fixed_val_from_arg': 'n_exp_fixed',
            'default_fixed': 1.5,
            'low': 0.5,
            'high': 2.0,
            'fit_flag_arg': 'fit_xi_params',
            'log_prior': False,
            'physical_check': True
        })


# ============================================================================
# Enhanced Monitoring Functions
# ============================================================================

# Initialize global tracking
convergence_tracker = None

class ConvergenceTracker:
    """Enhanced convergence tracking with parameter health monitoring."""
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.history = []
        self.param_history = {name: [] for name in param_names}
        self.last_logz = None
        self.stuck_counter = 0
        self.efficiency_history = []
        self.logz_history = []
        self.health_warnings = []
        
    def update(self, logz: float, params: np.ndarray, efficiency: float, samples: np.ndarray = None):
        """Update tracking with new information."""
        current_time = time.time()
        
        self.history.append({
            'time': current_time,
            'logz': logz,
            'params': params.copy() if params is not None else None,
            'efficiency': efficiency
        })
        
        # Update parameter histories
        if params is not None:
            for i, name in enumerate(self.param_names):
                if i < len(params):
                    self.param_history[name].append(params[i])
        
        # Keep only last hour
        cutoff_time = current_time - 3600
        self.history = [h for h in self.history if h['time'] > cutoff_time]
        
        # Update rolling statistics
        self.efficiency_history.append(efficiency)
        if len(self.efficiency_history) > 20:
            self.efficiency_history.pop(0)
            
        self.logz_history.append(logz)
        if len(self.logz_history) > 20:
            self.logz_history.pop(0)
        
        # Check for problems if we have samples
        if samples is not None and len(samples) > 100:
            evolution_check = check_parameter_evolution(samples, self.param_names, logger)
            if evolution_check['status'] != 'ok':
                self.health_warnings = evolution_check['warnings']
    
    def get_progress_report(self) -> str:
        """Generate comprehensive progress report."""
        if len(self.history) < 2:
            return "Not enough data for progress analysis"
        
        lines = []
        current_time = time.time()
        current_logz = self.history[-1]['logz']
        
        # LogZ progress
        logz_10min_ago = None
        logz_30min_ago = None
        
        for h in reversed(self.history):
            if logz_10min_ago is None and current_time - h['time'] > 600:
                logz_10min_ago = h['logz']
            if logz_30min_ago is None and current_time - h['time'] > 1800:
                logz_30min_ago = h['logz']
                break
        
        if logz_10min_ago is not None:
            logz_change_10min = current_logz - logz_10min_ago
            if abs(logz_change_10min) < 0.01:
                lines.append("⚠️  Log(Z) barely changed in last 10 min")
                self.stuck_counter += 1
            else:
                lines.append(f"✓ Log(Z) changed by {logz_change_10min:+.3f} in last 10 min")
                self.stuck_counter = max(0, self.stuck_counter - 1)
        
        if logz_30min_ago is not None:
            logz_change_30min = current_logz - logz_30min_ago
            lines.append(f"  30-min Δlog(Z): {logz_change_30min:+.3f}")
        
        # Efficiency trend
        if len(self.efficiency_history) > 5:
            recent_eff = np.mean(self.efficiency_history[-5:])
            older_eff = np.mean(self.efficiency_history[:5])
            eff_trend = recent_eff - older_eff
            
            if eff_trend < -0.5:
                lines.append(f"⚠️  Efficiency declining: {older_eff:.2f}% → {recent_eff:.2f}%")
            elif recent_eff < 2.0:
                lines.append(f"⚠️  Low efficiency: {recent_eff:.2f}%")
            else:
                lines.append(f"✓ Efficiency stable: {recent_eff:.2f}%")
        
        # Health warnings
        if self.health_warnings:
            lines.append("\n⚠️  PARAMETER HEALTH WARNINGS:")
            for warning in self.health_warnings:
                lines.append(f"   - {warning}")
        
        # Stuck detection
        if self.stuck_counter > 3:
            lines.append(f"\n❗ SAMPLING APPEARS STUCK (count: {self.stuck_counter})")
            lines.append("   Consider: wider priors, different sampler settings, or curriculum learning")
        
        return "\n".join(lines)

class AdaptiveModeMonitor:
    """Monitor sampling and adapt strategy in real-time."""
    
    def __init__(self, param_names, switch_threshold=0.7):
        self.param_names = param_names
        self.switch_threshold = switch_threshold
        self.mode_history = []
        self.current_mode = None
        self.mode_lifetimes = {}
        
    def update(self, samples, weights):
        """Check current mode and recommend actions."""
        if len(samples) < 500:
            return None
            
        # Identify current dominant mode
        analyzer = BimodalAnalyzer(None)  # Modify to accept arrays
        analyzer.samples = samples
        analyzer.weights = weights
        analyzer.param_names = self.param_names
        
        physical_mode, unphysical_mode = analyzer.separate_physical_modes()
        
        # Track mode evolution
        mode_info = {
            'iteration': len(self.mode_history),
            'physical_weight': physical_mode['weight_fraction'],
            'n_physical': len(physical_mode['samples']),
            'n_unphysical': len(unphysical_mode['samples'])
        }
        self.mode_history.append(mode_info)
        
        # Decide on action
        if physical_mode['weight_fraction'] > self.switch_threshold:
            if self.current_mode != 'physical':
                logger.info("🎯 Switching focus to PHYSICAL mode")
                self.current_mode = 'physical'
                return {
                    'action': 'tighten_bounds',
                    'mode_params': analyzer.get_mode_parameters(
                        physical_mode['samples'], 
                        physical_mode['weights']
                    )[0]
                }
        elif physical_mode['weight_fraction'] < 0.3:
            logger.warning("⚠️ Sampling dominated by unphysical mode!")
            return {
                'action': 'add_constraints',
                'constraint_strength': 200
            }
            
        return None

def enhanced_monitor_sampler_progress(
    sampler,
    fitted_param_names: List[str],
    fitted_param_labels: List[str],
    start_time: float,
    logger: logging.Logger,
    gp_surrogate=None,
    args_obj=None,
    dashboard_monitor=None
):
    """
    Enhanced monitoring with parameter health checks, convergence diagnostics,
    and a direct DDMM vs. Newtonian comparison.
    """
    global convergence_tracker
    
    # This 'try' block wraps the entire function to catch any unexpected errors during monitoring.
    try:
        res = sampler.results

        logger.info("=" * 80)
        logger.info(f"🔍 DYNESTY PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        if not hasattr(res, 'samples') or len(res.samples) == 0:
            logger.info("❌ No samples available yet")
            return

        samples = res.samples
        n_samples, n_params = samples.shape
        ncall_total = np.sum(res.ncall) if isinstance(res.ncall, np.ndarray) else res.ncall
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        eff = 100.0 * n_samples / ncall_total if ncall_total > 0 else 0.0
        
        logger.info(f"⏱️  Elapsed: {elapsed_str} | 📊 Samples: {n_samples:,} | 🎲 Calls: {ncall_total:,} | 📈 Eff: {eff:.2f}%")

        if gp_surrogate is not None and GP_AVAILABLE:
            gp_stats = gp_surrogate.get_statistics()
            logger.info(f"🤖 GP Surrogate: {gp_stats['n_real_calls']:,} real, "
                        f"{gp_stats['n_surrogate_calls']:,} surrogate (speedup: {gp_stats['speedup_factor']:.1f}x)")

        if n_samples < 50:
            logger.info("⚠️  Too few samples for detailed analysis")
            return

        # LogZ stats and dlogz
        current_logz = -np.inf
        dlogz = np.nan  # Default

        if hasattr(res, 'logz') and len(res.logz) > 0:
            current_logz = res.logz[-1]
            
            if not np.isfinite(current_logz):
                logger.error("❌ log(Z) = -inf. All live points have likelihood = -inf.")
                return
            else:
                logger.info(f"📊 Log(Z): {current_logz:.3f}")

                # Compute dlogz from last two logz values
                if len(res.logz) >= 2:
                    prev_logz = res.logz[-2]
                    dlogz = current_logz - prev_logz
                    logger.info(f"   Δlog(Z): {dlogz:.6f}")
                else:
                    logger.info("   Δlog(Z): (not enough samples yet)")

                # Show error estimate
                if hasattr(res, 'logzerr') and len(res.logzerr) > 0:
                    logger.info(f"   Error: ±{res.logzerr[-1]:.3f}")

        # Store dlogz for later JSON/dash export
        args_obj._current_dlogz = dlogz  # Or however you pass it to dashboard_monitor

       # Convergence check
        recent_samples = samples[-min(1000, len(samples)):]
        current_params = np.median(recent_samples, axis=0)
        if convergence_tracker is None:
            convergence_tracker = ConvergenceTracker(fitted_param_names)
        convergence_tracker.update(current_logz, current_params, eff, recent_samples)
        logger.info("\n🎯 CONVERGENCE PROGRESS:")
        logger.info("─" * 60)
        logger.info(convergence_tracker.get_progress_report())

        # dlogz stopping
        dlogz = np.inf
        if hasattr(res, 'logz') and len(res.logz) > 2:
            logz = float(res.logz[-1])
            sampler.saved_logz.append(logz)
            if len(sampler.saved_logz) > 2:
                dlogz = float(sampler.saved_logz[-1] - sampler.saved_logz[-2])
            else:
                dlogz = float("nan")
            logger.info(f"\n📏 Stopping criterion: dlogz = {dlogz:.4f}")
            if args_obj and hasattr(args_obj, 'dlogz_target'):
                if dlogz < args_obj.dlogz_target:
                    logger.info(f"   → Close to convergence target ({args_obj.dlogz_target})!")

        # Parameter stats
        logger.info(f"\n📊 PARAMETER ESTIMATES:")
        logger.info("─" * 80)
        logger.info(f"{'Parameter':<25} {'Median':<15} {'MAD':<15} {'Status':<15}")
        logger.info("─" * 80)
        for i, (param_name, param_label) in enumerate(zip(fitted_param_names, fitted_param_labels)):
            values = recent_samples[:, i]
            median_val = np.median(values)
            mad = np.median(np.abs(values - median_val))
            formatted_val = format_parameter_value_enhanced(median_val, param_name)
            formatted_mad = format_parameter_value_enhanced(mad, param_name)
            status = "✓"
            if param_name in MW_MULTI_COMP_PARAM_CONFIG:
                config = MW_MULTI_COMP_PARAM_CONFIG[param_name]
                if median_val < config['low'] * 1.1:
                    status = "⚠️ Near lower bound"
                elif median_val > config['high'] * 0.9:
                    status = "⚠️ Near upper bound"
            param_display = param_name.replace('_solar', '').replace('_kpc3', '').replace('_kpc', '')
            logger.info(f"{param_display:<25} {formatted_val:<15} {formatted_mad:<15} {status:<15}")

        # Physical plausibility check
        logger.info(f"\n🔍 PHYSICAL PLAUSIBILITY CHECK:")
        logger.info("─" * 60)
        is_valid, reason, *_ = check_physical_plausibility(current_params, fitted_param_names, args_obj)
        if is_valid:
            logger.info("✅ Median parameters pass all physical checks")
        else:
            logger.error(f"❌ Median parameters FAIL physical checks: {reason}")
        
        # --- NEW: LIVE MODEL COMPARISON ---
        logger.info(f"\n🌟 MODEL PREDICTIONS AT SOLAR RADIUS (R = {R_SUN_KPC:.2f} kpc):")
        logger.info("─" * 60)
        
        full_params = dict(zip(fitted_param_names, current_params))
        for p_info in args_obj.all_param_info_list:
            if not p_info['is_fitted']:
                full_params[p_info['name']] = p_info['current_val']
        full_params.update({
            'include_disk_thin': args_obj.include_disk_thin,
            'include_disk_thick': args_obj.include_disk_thick,
            'include_bulge': args_obj.include_bulge,
            'include_gas': args_obj.include_gas,
        })

        try:
            r_solar = np.array([R_SUN_KPC])
            
            v_newton_solar = v_baryon_total_newtonian_kms(r_solar, full_params)[0]
            rho_solar = rho_baryon_total_midplane_solar_kpc3(r_solar, full_params)[0]
            
            xi_func = XI_FUNCTION_MAP.get(args_obj.xi, XI_FUNCTION_MAP['power'])
            n_key = 'gamma_exp' if 'gamma_exp' in full_params else 'n_exp'
            A_key = 'lambda_g' if 'lambda_g' in full_params else 'A'
            
            xi_solar_val = xi_func(rho_solar, full_params['rho_c_solar_kpc3'], full_params[n_key], full_params.get(A_key, 1.0))
            xi_solar = np.minimum(xi_solar_val, 5.0)[0]
            v_model_solar = v_newton_solar * np.sqrt(xi_solar)

            logger.info(f"   Newtonian Velocity (Baryons Only): {v_newton_solar:.1f} km/s")
            logger.info(f"   DDMM Predicted Velocity:             {v_model_solar:.1f} km/s")
            logger.info(f"   Enhancement Factor (ξ):              {xi_solar:.3f}")
            logger.info(f"   Difference (DDMM - Newtonian):       {v_model_solar - v_newton_solar:+.1f} km/s")

        except Exception as e:
            logger.error(f"❌ Error calculating model predictions: {e}")
        # --- END OF NEW SECTION ---
        
        # Dashboard update
        if dashboard_monitor is not None:
            try:
                dashboard_state = {
                    "elapsed_time": float(elapsed_time / 3600),
                    "n_samples": int(n_samples),
                    "n_calls": int(ncall_total),
                    "efficiency": float(eff),
                    "logz": float(current_logz),
                    "logz_err": float(res.logzerr[-1]) if hasattr(res, 'logzerr') and len(res.logzerr) > 0 else 0.0,
                    "dlogz": float(dlogz) if dlogz is not None and np.isfinite(dlogz) else None,
                    "current_nlive": int(len(res.live_points)) if hasattr(res, 'live_points') else 0,
                    "parameter_estimates": {},
                    "parameter_uncertainties": {},
                    "health_warnings": convergence_tracker.health_warnings if convergence_tracker else []
                }
                
                for i, name in enumerate(fitted_param_names):
                    dashboard_state["parameter_estimates"][name] = float(current_params[i])
                    dashboard_state["parameter_uncertainties"][name] = float(np.std(recent_samples[:, i]))

                dashboard_state = make_json_serializable(dashboard_state)
                dashboard_monitor.update_progress(dashboard_state)

            except Exception as e:
                logger.warning(f"⚠️ Dashboard update failed: {e}")

        logger.info("=" * 80)
    
    # This is the 'except' block that matches the 'try' at the top and fixes the SyntaxError.
    except Exception as e:
        logger.error(f"An unexpected error occurred in the monitoring function: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def format_parameter_value_enhanced(value: float, param_name: str) -> str:
    """
    Format parameter values with appropriate units and precision.
    
    Parameters
    ----------
    value : float
        Parameter value
    param_name : str
        Parameter name to determine formatting
        
    Returns
    -------
    str
        Formatted value string
    """
    if 'M_' in param_name and 'solar' in param_name:
        # Mass parameters
        if value > 1e11:
            return f"{value/1e11:.2f}×10¹¹ M☉"
        elif value > 1e10:
            return f"{value/1e10:.2f}e10 M☉"
        else:
            return f"{value:.2e} M☉"
    elif 'rho_c' in param_name:
        # Density parameters
        return f"{value:.2e} M☉/kpc³"
    elif 'R_d' in param_name or 'a_' in param_name:
        # Scale lengths
        return f"{value:.3f} kpc"
    elif 'h_z' in param_name:
        # Scale heights
        return f"{value:.3f} kpc"
    elif 'n_exp' in param_name:
        # Power law exponent
        return f"{value:.3f}"
    elif param_name in ('gamma_exp', 'lambda_g'):
        return f"{value:.3f}"      # dimensionless
    else:
        # Default scientific notation
        return f"{value:.3e}"


# ============================================================================
# Prior Transform and Likelihood Functions
# ============================================================================

def prior_transform_dynesty(
    u_array: np.ndarray,
    fitted_param_names: List[str],
    prior_bounds_low: np.ndarray,
    prior_bounds_high: np.ndarray,
    use_log_prior_flags: List[bool]
) -> np.ndarray:
    """
    PRIOR TRANSFORM WITH SOLAR SYSTEM CONSTRAINTS.
    This version uses extremely tight, physically motivated priors on the gravity
    parameters to FORCE the solution to be compatible with Solar System tests.
    The MCMC will then determine if such a solution can also fit the galaxy data.
    """
    params = np.zeros_like(u_array)
    u_dict = dict(zip(fitted_param_names, u_array))

    # --- Gravity / DDMM Parameters (SOLAR SYSTEM COMPATIBLE PRIORS) ---
    if 'rho_c_solar_kpc3' in u_dict:
        # Log-uniform prior between 10^12 and 10^16 M☉/kpc³.
        # This is a much higher range, guaranteed to screen gravity in the Solar System.
        log_low, log_high = 12.0, 15.0
        params[fitted_param_names.index('rho_c_solar_kpc3')] = 10**(log_low + u_dict['rho_c_solar_kpc3'] * (log_high - log_low))

    if 'n_exp' in u_dict or 'gamma_exp' in u_dict:
        # A higher 'n' creates a sharper, more step-like transition, which also helps.
        key = 'gamma_exp' if 'gamma_exp' in u_dict else 'n_exp'
        low, high = 2.0, 6.0
        params[fitted_param_names.index(key)] = low + u_dict[key] * (high - low)
        
    if 'gamma_exp' in u_dict:
        # Use the same wide prior for 'gamma_exp'
        low, high = 0.1, 2.5
        params[fitted_param_names.index('gamma_exp')] = low + u_dict['gamma_exp'] * (high - low)

    if 'lambda_g' in u_dict:
        # Log-Uniform prior between 0.1 and 2.0.
        log_low, log_high = np.log10(0.1), np.log10(2.0)
        params[fitted_param_names.index('lambda_g')] = 10**(log_low + u_dict['lambda_g'] * (log_high - log_low))

    # --- Baryonic Component Parameters (STANDARD PRIORS) ---
    if 'M_disk_thin_solar' in u_dict:
        log_low, log_high = 10.5, 11.1
        params[fitted_param_names.index('M_disk_thin_solar')] = 10**(log_low + u_dict['M_disk_thin_solar'] * (log_high - log_low))
    if 'R_d_thin_kpc' in u_dict:
        low, high = 2.0, 4.5
        params[fitted_param_names.index('R_d_thin_kpc')] = low + u_dict['R_d_thin_kpc'] * (high - low)
    if 'h_z_thin_kpc' in u_dict:
        low, high = 0.15, 0.5
        params[fitted_param_names.index('h_z_thin_kpc')] = low + u_dict['h_z_thin_kpc'] * (high - low)
    if 'M_disk_thick_solar' in u_dict:
        log_low, log_high = 9.5, 10.7
        params[fitted_param_names.index('M_disk_thick_solar')] = 10**(log_low + u_dict['M_disk_thick_solar'] * (log_high - log_low))
    if 'R_d_thick_kpc' in u_dict:
        low, high = 3.5, 9.5
        params[fitted_param_names.index('R_d_thick_kpc')] = low + u_dict['R_d_thick_kpc'] * (high - low)
    if 'h_z_thick_kpc' in u_dict:
        low, high = 0.7, 1.5
        params[fitted_param_names.index('h_z_thick_kpc')] = low + u_dict['h_z_thick_kpc'] * (high - low)
    if 'M_bulge_solar' in u_dict:
        log_low, log_high = 9.7, 10.4
        params[fitted_param_names.index('M_bulge_solar')] = 10**(log_low + u_dict['M_bulge_solar'] * (log_high - log_low))
    if 'a_bulge_kpc' in u_dict:
        low, high = 0.2, 2.0
        params[fitted_param_names.index('a_bulge_kpc')] = low + u_dict['a_bulge_kpc'] * (high - low)
    if 'M_gas_solar' in u_dict:
        log_low, log_high = 9.7, 10.8
        params[fitted_param_names.index('M_gas_solar')] = 10**(log_low + u_dict['M_gas_solar'] * (log_high - log_low))
    if 'R_d_gas_kpc' in u_dict:
        low, high = 4.0, 15.0
        params[fitted_param_names.index('R_d_gas_kpc')] = low + u_dict['R_d_gas_kpc'] * (high - low)
    if 'h_z_gas_kpc' in u_dict:
        low, high = 0.05, 0.4
        params[fitted_param_names.index('h_z_gas_kpc')] = low + u_dict['h_z_gas_kpc'] * (high - low)
        
    return params

def log_likelihood_dynesty(
    theta_values_fitted: np.ndarray,
    fitted_param_names: List[str],
    args_dynesty_obj: argparse.Namespace,
    all_param_info_list: List[Dict],
    R_data: np.ndarray,
    v_data: np.ndarray,
    sigma_data: np.ndarray,
    xi_type: str,  # <-- This parameter exists here
    gp_surrogate=None
) -> Tuple[float, List[float]]:

    """
    MASTER log-likelihood function that now correctly passes the FULL parameter set
    to the physical plausibility check and uses the v_total_kms master function.
    """
    # 1. Reconstruct the full parameter dictionary, including fixed values
    params = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in all_param_info_list:
        if not p_info['is_fitted']:
            params[p_info['name']] = p_info['current_val']
    
    # Add boolean flags for which components to include in calculations
    for component in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
        params[f'include_{component}'] = getattr(args_dynesty_obj, f'include_{component}', False)
    
    # 2. Perform plausibility checks on the full set of parameters
    # The check function needs the full list of names and values
    all_param_names_for_check = [p['name'] for p in all_param_info_list]
    all_param_values_for_check = np.array([params[name] for name in all_param_names_for_check if name in params])
    
    # Ensure the arrays have the same length before calling the check
    if len(all_param_names_for_check) != len(all_param_values_for_check):
         # This can happen if a parameter is in the config but not in the current run's `params` dict
         # We will filter `all_param_names_for_check` to only include keys present in `params`
         all_param_names_for_check = [name for name in all_param_names_for_check if name in params]

    is_valid, reason, *_ = check_physical_plausibility(all_param_values_for_check, all_param_names_for_check, args_dynesty_obj)
    if not is_valid:
        return -np.inf, [np.inf]
    
    # 2b. Enforce Cassini constraints
    try:
        rho_saturn = 2.3e21  # Cassini orbit density
        
        # Import at function level to avoid circular imports
        from density_metric2 import XI_FUNCTION_MAP
        
        xi_func = XI_FUNCTION_MAP.get(xi_type, XI_FUNCTION_MAP['power'])
        
        if xi_type == 'grav_color':
            gamma = params.get('gamma_exp', 2.7)
            lambda_g = params.get('lambda_g', 1.5)
            rho_c = params.get('rho_c_solar_kpc3', 1e13)
            xi_saturn = xi_func(rho_saturn, rho_c, gamma, lambda_g)[0]
        elif xi_type in ('power', 'logistic', 'enhanced'):
            n = params.get('n_exp', 1.5)
            A = params.get('A', 1.0) if 'enhanced' in xi_type else None
            rho_c = params.get('rho_c_solar_kpc3', 1e13)
            if A is not None:
                xi_saturn = xi_func(rho_saturn, rho_c, n, A)[0]
            else:
                xi_saturn = xi_func(rho_saturn, rho_c, n)[0]
        else:
            xi_saturn = 1.0  # Skip check for mass_threshold etc.
        
        if abs(xi_saturn - 1.0) > 2.3e-5:  # Cassini tolerance
            return -np.inf, [np.inf]  # Reject sample
            
    except Exception as e:
        logger = get_or_create_logger()
        logger.warning(f"⚠️ Cassini check failed: {e}")
        return -np.inf, [np.inf]

    # 3. Compute the model velocity using the master v_total_kms function
    try:
        v_model = v_total_kms(R_data, params, xi_type=xi_type)
        if not np.all(np.isfinite(v_model)):
            return -np.inf, [np.inf]
    except Exception:
        return -np.inf, [np.inf]
    
    # 4. Plausibility penalty for overshooting
    v_model_solar_mask = (R_data > 7.5) & (R_data < 8.5)
    if np.any(v_model_solar_mask):
        v_model_solar = np.median(v_model[v_model_solar_mask])
        if v_model_solar > 300.0:
            return -np.inf, [np.inf]

    # 5. Calculate chi-squared and final log-likelihood
    chi2 = np.sum(((v_data - v_model) / sigma_data)**2)
    log_L = -0.5 * chi2
    
    if not np.isfinite(log_L):
        return -np.inf, [np.inf]

    rmse = np.sqrt(np.mean((v_data - v_model)**2))
    return log_L, [rmse]


def v_model_for_dynesty(
    R_kpc_array: np.ndarray,
    p_all_params_dict: Dict[str, float],
    xi_type_str: str,
    ARGS_obj_dynesty: argparse.Namespace
) -> np.ndarray:
    """
    Calculate model velocities with density-dependent modification.
    Fixed for multiprocessing compatibility.
    """
    # Get logger safely for multiprocessing
    logger = get_or_create_logger()
    
    # Import needed functions
    from density_metric2 import XI_FUNCTION_MAP, xi_gravitational_color

    # DEBUG: Print what parameters we have (only once per process)
    if not hasattr(v_model_for_dynesty, "_params_logged"):
        try:
            logger.info(f"\n[PARAMS DEBUG] xi_type: {xi_type_str}")
            logger.info(f"[PARAMS DEBUG] Available parameters: {list(p_all_params_dict.keys())}")
            logger.info(f"[PARAMS DEBUG] Parameter values:")
            for k, v in p_all_params_dict.items():
                if isinstance(v, (int, float)):
                    logger.info(f"   {k}: {v:.3e}")
            v_model_for_dynesty._params_logged = True
        except:
            # If logging fails in multiprocessing, just continue
            pass
    
    # Extract parameters with proper error handling
    try:
        if xi_type_str == 'grav_color':
            # For gravitational color, we need rho_c, gamma, and lambda_g
            rho_c_solar_kpc3 = p_all_params_dict.get('rho_c_solar_kpc3', 5e7)
            gamma = p_all_params_dict.get('gamma_exp', 
                                         getattr(ARGS_obj_dynesty, 'fix_gamma', 2.0))
            lambda_g = p_all_params_dict.get('lambda_g', 
                                           getattr(ARGS_obj_dynesty, 'fix_lambda_g', 1.5))
            n_exp = p_all_params_dict.get('n_exp', 2.0)  # Not used but might be needed
        else:
            # Standard xi functions need rho_c and n_exp
            if 'rho_c_solar_kpc3' not in p_all_params_dict:
                logger.error(f"ERROR: rho_c_solar_kpc3 missing!")
                logger.error(f"Available: {list(p_all_params_dict.keys())}")
                return np.zeros_like(R_kpc_array)
            if 'n_exp' not in p_all_params_dict:
                logger.error(f"ERROR: n_exp missing!")
                return np.zeros_like(R_kpc_array)
                
            rho_c_solar_kpc3 = p_all_params_dict['rho_c_solar_kpc3']
            n_exp = p_all_params_dict['n_exp']
            gamma = None
            lambda_g = None
            
    except Exception as e:
        logger.error(f"Error extracting parameters: {e}")
        return np.zeros_like(R_kpc_array)
    
    # Validate xi parameters
    if not np.isfinite(rho_c_solar_kpc3):
        logger.warning(f"Non-finite rho_c: {rho_c_solar_kpc3}")
        return np.zeros_like(R_kpc_array)
    
    # Calculate Newtonian velocities and densities
    if ARGS_obj_dynesty.fit_target == 'milkyway':
        v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict)
        rho_midplane_for_xi = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict)
    else:
        raise NotImplementedError("Only Milky Way fitting currently supported")
    
    # Validate intermediate results
    if not np.all(np.isfinite(v_n_kms)):
        logger.warning("⚠️ Non-finite Newtonian velocities detected")
        v_n_kms = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(np.isfinite(rho_midplane_for_xi)):
        logger.warning("⚠️ Non-finite densities detected")
        rho_midplane_for_xi = np.nan_to_num(rho_midplane_for_xi, nan=0.0, posinf=1e10, neginf=0.0)

    # Calculate xi based on the selected type
    try:
        if xi_type_str == 'grav_color':
            # Use gravitational color function directly
            xi_raw = xi_gravitational_color(rho_midplane_for_xi, rho_c_solar_kpc3, gamma, lambda_g)
        elif xi_type_str == 'enhanced':
            xi_func = XI_FUNCTION_MAP['enhanced']
            # For enhanced, use A=8.0 for theory test
            xi_raw = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp, 8.0)
        else:
            # Standard xi functions (power, logistic, etc.)
            xi_func = XI_FUNCTION_MAP.get(xi_type_str, XI_FUNCTION_MAP['power'])
            xi_raw = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp)
            
    except Exception as e:
        logger.error(f"Error calculating xi with {xi_type_str}: {e}")
        xi_raw = np.ones_like(rho_midplane_for_xi)

    # Log xi verification (only once per process)
    if not hasattr(v_model_for_dynesty, "_has_logged_xi"):
        try:
            logger.info(f"[XI VERIFICATION] Using xi_type: '{xi_type_str}'")
            if xi_type_str == 'grav_color':
                logger.info(f"[XI VERIFICATION] Parameters: ρ_c={rho_c_solar_kpc3:.2e}, γ={gamma:.2f}, λ_g={lambda_g:.2f}")
            else:
                logger.info(f"[XI VERIFICATION] Parameters: ρ_c={rho_c_solar_kpc3:.2e}, n={n_exp:.2f}")

            # Test xi at different densities
            test_densities = [1e6, 1e8, 1e10]
            summary = []
            for test_rho in test_densities:
                try:
                    if xi_type_str == 'grav_color':
                        test_xi_raw = xi_gravitational_color(test_rho, rho_c_solar_kpc3, gamma, lambda_g)
                    else:
                        xi_func = XI_FUNCTION_MAP.get(xi_type_str, XI_FUNCTION_MAP['power'])
                        test_xi_raw = xi_func(test_rho, rho_c_solar_kpc3, n_exp)
                    test_xi = test_xi_raw[0] if hasattr(test_xi_raw, '__getitem__') else float(test_xi_raw)
                    summary.append(f"ρ={test_rho:.0e} → ξ={test_xi:.3f}")
                except Exception as e:
                    summary.append(f"ρ={test_rho:.0e} → error: {e}")
            logger.info("[XI VERIFICATION SUMMARY] " + "; ".join(summary))
            
            v_model_for_dynesty._has_logged_xi = True
        except:
            # If logging fails in multiprocessing, just continue
            pass

    # Convert xi_raw to array safely
    if not hasattr(xi_raw, "__getitem__"):
        xi_values = np.full_like(v_n_kms, float(xi_raw))
    else:
        xi_values = np.asarray(xi_raw, dtype=np.float64)

    # Sanitize xi values
    xi_values = np.nan_to_num(xi_values, nan=1.0, posinf=1.0, neginf=0.0)
    xi_values_safe = np.maximum(xi_values, 0.0)

    # Apply modified gravity
    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)

    # Final velocity validation
    if not np.all(np.isfinite(v_mod_kms)):
        logger.warning("⚠️ Non-finite final velocities detected")
        v_mod_kms = np.nan_to_num(v_mod_kms, nan=0.0, posinf=0.0, neginf=0.0)

    return v_mod_kms


# Also need to ensure the parameter configuration includes rho_c for grav_color
def ensure_grav_color_params_in_config(args):
    """
    Ensure that when using grav_color xi, we still include rho_c_solar_kpc3
    in the parameter configuration.
    """
    if args.xi == 'grav_color':
        # Make sure rho_c is available even if not fitted
        if not hasattr(args, 'rho_c_fixed') or args.rho_c_fixed is None:
            args.rho_c_fixed = 5e7  # Galaxy-appropriate default
            logger.info(f"Setting default rho_c_fixed = {args.rho_c_fixed:.1e} for grav_color")
        
        # Ensure it's in the parameter config
        if 'rho_c_solar_kpc3' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'] = {
                'label': "rho_c (M_sun/kpc^3)", 
                'fixed_val_from_arg': 'rho_c_fixed', 
                'default_fixed': 1e13,  # Galaxy-appropriate
                'low': 1e12,
                'high': 1e15,
                'fit_flag_arg': 'fit_rho_c',  # New flag
                'log_prior': True,
                'physical_check': True
            }


def log_likelihood_dynesty_debug(
    theta_values_fitted: np.ndarray,
    fitted_param_names: List[str],
    args_dynesty_obj: argparse.Namespace,
    all_param_info_list: List[Dict],
    R_data: np.ndarray,
    v_data: np.ndarray,
    sigma_data: np.ndarray,
    xi_type: str,
    gp_surrogate=None
) -> Tuple[float, List[float]]:
    """Debug version with extensive logging - FIXED to handle full parameter set"""
    
    # Get logger safely for multiprocessing
    logger = get_or_create_logger()
    
    # Initialize warning counts
    if not hasattr(log_likelihood_dynesty_debug, 'warning_counts'):
        log_likelihood_dynesty_debug.warning_counts = {}
    
    # Handle debug counter for multiprocessing
    if not hasattr(log_likelihood_dynesty_debug, 'debug_counter'):
        log_likelihood_dynesty_debug.debug_counter = 0
    
    # First check if we have valid data
    if R_data is None or len(R_data) == 0:
        logger.error("ERROR: R_data is None or empty!")
        return -np.inf, [np.inf]
    
    if v_data is None or len(v_data) == 0:
        logger.error("ERROR: v_data is None or empty!")
        return -np.inf, [np.inf]
    
    if sigma_data is None or len(sigma_data) == 0:
        logger.error("ERROR: sigma_data is None or empty!")
        return -np.inf, [np.inf]
    
    # FIXED: Reconstruct full parameter dictionary FIRST
    params = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in all_param_info_list:
        if not p_info['is_fitted']:
            params[p_info['name']] = p_info['current_val']
    
    # Hard constraint: thick disk scale length must be > thin disk scale length
    if ('R_d_thick_kpc' in params and 'R_d_thin_kpc' in params):
        R_d_thick = params['R_d_thick_kpc']
        R_d_thin = params['R_d_thin_kpc']
        
        if R_d_thick < 1.05 * R_d_thin:
            # Use warning counter for this specific constraint
            warning_key = "R_d_thick_thin"
            if warning_key not in log_likelihood_dynesty_debug.warning_counts:
                log_likelihood_dynesty_debug.warning_counts[warning_key] = 0
            
            if log_likelihood_dynesty_debug.warning_counts[warning_key] < 10:
                logger.debug(f"Rejecting: R_d_thick ({R_d_thick:.2f}) < 1.05 * R_d_thin ({R_d_thin:.2f})")
                log_likelihood_dynesty_debug.warning_counts[warning_key] += 1
            
            return -np.inf, [np.inf]

    # Hard constraint: thick disk scale height must be > thin disk scale height
    if ('h_z_thick_kpc' in params and 'h_z_thin_kpc' in params):
        h_z_thick = params['h_z_thick_kpc']
        h_z_thin = params['h_z_thin_kpc']
        
        if h_z_thick < 2.0 * h_z_thin:
            # Use warning counter for this specific constraint
            warning_key = "h_z_thick_thin"
            if warning_key not in log_likelihood_dynesty_debug.warning_counts:
                log_likelihood_dynesty_debug.warning_counts[warning_key] = 0
            
            if log_likelihood_dynesty_debug.warning_counts[warning_key] < 10:
                logger.debug(f"Rejecting: h_z_thick ({h_z_thick:.3f}) < 2 * h_z_thin ({h_z_thin:.3f})")
                log_likelihood_dynesty_debug.warning_counts[warning_key] += 1
            
            return -np.inf, [np.inf]
        
    # Log first few data points (only once)
    if not hasattr(log_likelihood_dynesty_debug, '_logged_data'):
        logger.info(f"DEBUG: Data shapes - R: {R_data.shape}, v: {v_data.shape}, sigma: {sigma_data.shape}")
        logger.info(f"DEBUG: First 5 R values: {R_data[:5]}")
        logger.info(f"DEBUG: First 5 v values: {v_data[:5]}")
        logger.info(f"DEBUG: First 5 sigma values: {sigma_data[:5]}")
        logger.info(f"DEBUG: R range: [{np.min(R_data):.2f}, {np.max(R_data):.2f}]")
        logger.info(f"DEBUG: v range: [{np.min(v_data):.2f}, {np.max(v_data):.2f}]")
        
        # Log the full parameter set being used
        logger.info("DEBUG: Full parameter set:")
        for name, value in params.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {name}: {value:.3e}")
        
        log_likelihood_dynesty_debug._logged_data = True
    
    # Now call the original function with the fix
    return log_likelihood_dynesty(
        theta_values_fitted, fitted_param_names, args_dynesty_obj,
        all_param_info_list, R_data, v_data, sigma_data, xi_type, gp_surrogate
    )
    
def check_cassini_compatibility(params, xi_type):
    """
    Check if parameters pass Cassini test.
    """
    from density_metric2 import XI_FUNCTION_MAP
    
    rho_saturn = 2.3e21  # Saturn orbit density
    cassini_precision = 2.3e-5
    
    xi_func = XI_FUNCTION_MAP.get(xi_type)
    if xi_func is None:
        return False, f"Unknown xi_type: {xi_type}"
    
    try:
        if xi_type == 'mass_threshold':
            return False, "mass_threshold model cannot pass Cassini test"
        elif xi_type == 'grav_color':
            rho_c = params.get('rho_c_solar_kpc3', 1e13)
            gamma = params.get('gamma_exp', 2.7)
            lambda_g = params.get('lambda_g', 8.0)
            xi_saturn = xi_func(rho_saturn, rho_c, gamma, lambda_g)[0]
        else:
            rho_c = params.get('rho_c_solar_kpc3', 1e13)
            n_exp = params.get('n_exp', 1.5)
            A = params.get('A', 1.0)
            xi_saturn = xi_func(rho_saturn, rho_c, n_exp, A)[0]
        
        if abs(xi_saturn - 1.0) > cassini_precision:
            return False, f"Fails Cassini: |ξ({rho_saturn:.1e}) - 1| = {abs(xi_saturn - 1.0):.2e} > {cassini_precision}"
        
        return True, "Passes Cassini test"
        
    except Exception as e:
        return False, f"Cassini check error: {e}"

def check_prior_bounds_compatibility(args):
    """Check if prior bounds are compatible with constraints"""
    
    if args.fit_disk_thin and args.fit_disk_thick:
        # Check R_d constraint compatibility
        R_d_thin_max = MW_MULTI_COMP_PARAM_CONFIG['R_d_thin_kpc']['high']
        R_d_thick_min = MW_MULTI_COMP_PARAM_CONFIG['R_d_thick_kpc']['low']
        
        if R_d_thin_max * 1.1 > R_d_thick_min:
            logger.warning(f"WARNING: Prior bounds may be incompatible!")
            logger.warning(f"  R_d_thin can go up to {R_d_thin_max} kpc")
            logger.warning(f"  R_d_thick must be > {R_d_thin_max * 1.1:.2f} kpc")
            logger.warning(f"  But R_d_thick minimum is {R_d_thick_min} kpc")
            logger.warning("  Consider adjusting bounds or constraint")
            
            # Suggest fix
            new_R_d_thin_max = R_d_thick_min / 1.1
            logger.warning(f"  Suggestion: Set R_d_thin_kpc max to {new_R_d_thin_max:.2f}")

    

# ============================================================================
# Parameter Configuration Functions
# ============================================================================

def get_param_labels_and_bounds(ARGS):
    """
    Enhanced parameter configuration with log-prior flags, optional prior tightening,
    and validation. Supports starting from previous best-fit and narrowing bounds.
    """
    param_info_list = []
    config_to_use = MW_MULTI_COMP_PARAM_CONFIG
    logger.info("Configuring parameters for multi-component Milky Way model")

    # === OPTIONAL: Load previous best-fit medians ===
    previous_best = None
    bounds_modified = {}

    # CRITICAL: For grav_color, ensure xi parameters are included
    if ARGS.xi == 'grav_color':
        # Make sure rho_c_solar_kpc3 is in the config
        if 'rho_c_solar_kpc3' not in config_to_use:
            logger.error("ERROR: rho_c_solar_kpc3 missing from config!")
        
        # Force include rho_c even if not fitting
        if not hasattr(ARGS, 'include_rho_c'):
            ARGS.include_rho_c = True

    if getattr(ARGS, 'use_previous_best', False):
        try:

            logger.info("✅ Loaded previous best-fit medians successfully.")

            # Apply tightened bounds if requested
            if getattr(ARGS, 'tighten_bounds_factor', 0) > 0:
                factor = ARGS.tighten_bounds_factor
                for param, val in previous_best.items():
                    if param in PHYSICAL_BOUNDS:
                        delta = factor * abs(val)
                        bounds_modified[param] = {
                            'low': max(PHYSICAL_BOUNDS[param]['min'], val - delta),
                            'high': min(PHYSICAL_BOUNDS[param]['max'], val + delta)
                        }
                        logger.info(f"🔒 Tightened bounds for {param}: [{bounds_modified[param]['low']:.2e}, {bounds_modified[param]['high']:.2e}]")

        except Exception as e:
            logger.warning(f"⚠️ Could not load previous_best: {e}")
            previous_best = None

    # === Main parameter loop ===
    for p_name, p_details in config_to_use.items():
        # Special handling for xi parameters when using grav_color
        if ARGS.xi == 'grav_color' and p_name in ['rho_c_solar_kpc3', 'n_exp', 'gamma_exp', 'lambda_g']:
            # Always include these for grav_color mode
            is_included = True
        else:
            # Standard component check
            is_included = 'include_flag_arg' not in p_details or \
                          getattr(ARGS, p_details['include_flag_arg'], False)
        
        if not is_included:
            continue

        # Define which parameters belong to which xi model
        xi_model_params = {
            'power': ['rho_c_solar_kpc3', 'n_exp'],
            'mass_threshold': ['M_crit_msun', 'xi_boost', 'width'],
            'grav_color': ['rho_c_solar_kpc3', 'gamma_exp', 'lambda_g']
            # Add other models as you define them
        }

        # Should we fit this parameter?
        is_fitted = False
        if p_details.get('fit_flag_arg') == 'fit_xi_params':
            # This is a gravity parameter. Only fit it if it belongs to the
            # currently selected model AND the --fit_xi_params flag is active.
            if ARGS.fit_xi_params and p_name in xi_model_params.get(ARGS.xi, []):
                is_fitted = True
        elif 'fit_flag_arg' in p_details and getattr(ARGS, p_details['fit_flag_arg'], False):
            # This is a baryonic parameter. Fit it if its specific flag is active.
            is_fitted = True

        # Allow a fixed value on the command line to override fitting
        fixed_arg_name = p_details['fixed_val_from_arg']
        default_val = p_details['default_fixed']
        # Check if the user provided a value different from the default
        if getattr(ARGS, fixed_arg_name, default_val) != default_val:
            is_fitted = False

        # Get current value
        if p_details.get('log_prior', False):
            current_val = 10 ** (0.5 * (np.log10(p_details['low']) +
                                        np.log10(p_details['high'])))
        else:
            current_val = 0.5 * (p_details['low'] + p_details['high'])
        # Allow CLI --<param>_fixed to override the auto start
        cli_override = getattr(ARGS, p_details['fixed_val_from_arg'])
        if not is_fitted and cli_override is not None:
            current_val = cli_override
            
        # Bounds (tightened if available)
        if p_name in bounds_modified:
            low = bounds_modified[p_name]['low']
            high = bounds_modified[p_name]['high']
        else:
            low = p_details['low']
            high = p_details['high']

        # Validate initial value against PHYSICAL_BOUNDS
        if p_name in PHYSICAL_BOUNDS:
            pb = PHYSICAL_BOUNDS[p_name]
            if current_val < pb['min'] or current_val > pb['max']:
                logger.warning(f"  {p_name}: Initial value {current_val:.2e} outside "
                               f"physical bounds [{pb['min']:.2e}, {pb['max']:.2e}]")
                current_val = np.clip(current_val, pb['min'], pb['max'])
                logger.warning(f"  Clipped to: {current_val:.2e}")

        param_info_list.append({
            'name': p_name,
            'label': p_details['label'],
            'current_val': current_val,
            'low': low,
            'high': high,
            'is_fitted': is_fitted,
            'log_prior': p_details.get('log_prior', False),
            'physical_check': p_details.get('physical_check', True)
        })

    ARGS.all_param_info_list = param_info_list
    fitted_params_info = [p for p in param_info_list if p['is_fitted']]

    if not fitted_params_info:
        logger.error("No parameters configured to be fitted!")
        logger.error("You must use at least one --fit_* flag (e.g., --fit_xi_params)")
        sys.exit(1)

    # Logging fitted params
    logger.info(f"\nFitting {len(fitted_params_info)} parameters:")
    for p in fitted_params_info:
        prior_type = "Log-uniform" if p['log_prior'] else "Uniform"
        logger.info(f"  {p['name']:<25} | Prior: {prior_type} | Range: [{p['low']:.2e}, {p['high']:.2e}]")

    # Return as required
    use_log_flags = [p['log_prior'] for p in fitted_params_info]
    return (
        [p['name'] for p in fitted_params_info],
        [p['label'] for p in fitted_params_info],
        np.array([p['current_val'] for p in fitted_params_info]),
        np.array([p['low'] for p in fitted_params_info]),
        np.array([p['high'] for p in fitted_params_info]),
        use_log_flags
    )


# ---------------------------------------------------------------------------
# Helper: return the list of fitted‑parameter names for any Namespace
# ---------------------------------------------------------------------------
def fitted_names_for(args_obj):
    names, *_ = get_param_labels_and_bounds(args_obj)
    return names

# ============================================================================
# Convert numpy types to Python native types for JSON serialization
# ============================================================================

def make_json_serializable(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    Recursively handles dicts, lists, tuples, arrays, and NumPy scalars.
    """
    import numpy as np

    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif isinstance(obj, (np.bytes_, bytes)):
        return obj.decode('utf-8')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj



# ============================================================================
# Gaussian Process Surrogate Model (unchanged but included for completeness)
# ============================================================================

class GPSurrogateModel:
    """
    Gaussian Process surrogate model for fast likelihood evaluation.
    Uses active learning to intelligently call the expensive physics model.
    """
    def __init__(self, param_names: List[str], param_bounds: np.ndarray, 
                 uncertainty_threshold: float = 0.1, n_initial: int = 500):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.ndim = len(param_names)
        self.uncertainty_threshold = uncertainty_threshold
        self.n_initial = n_initial
        
        # Training data
        self.X_train = []
        self.y_train = []
        
        # GP model with Matern kernel (good for smooth functions)
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        
        # Statistics
        self.n_real_calls = 0
        self.n_surrogate_calls = 0
        self.gp_trained = False
        
        logger.info(f"🤖 GP Surrogate initialized for {self.ndim}D parameter space")
    
    def generate_initial_training_data(self, physics_function, args_obj):
        """Generate initial training data using Latin Hypercube sampling"""
        logger.info(f"🎲 Generating {self.n_initial} initial training points...")
        
        # Latin Hypercube sampling for better coverage
        sampler = qmc.LatinHypercube(d=self.ndim)
        samples_unit = sampler.random(n=self.n_initial)
        
        # Transform to parameter bounds
        samples = qmc.scale(samples_unit, self.param_bounds[:, 0], self.param_bounds[:, 1])
        
        # Evaluate physics model
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"   Training point {i}/{self.n_initial}")
            
            # Create parameter dictionary
            param_dict = dict(zip(self.param_names, sample))
            
            # Call physics model
            try:
                v_pred = physics_function(param_dict, args_obj)
                self.X_train.append(sample)
                self.y_train.append(v_pred)
                self.n_real_calls += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to evaluate training point: {e}")
        
        # Train initial GP
        self._train_gp()
        logger.info(f"✅ Initial GP training complete with {len(self.X_train)} points")
    
    def _train_gp(self):
        """Train or retrain the GP model"""
        if len(self.X_train) < 10:
            if logger:
                logger.warning("Too few training points for GP")
            return
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        # Normalize features for better GP performance
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        # Flatten y if needed (for multi-output)
        if y.ndim > 1:
            y = y.flatten()
        
        # Train GP
        try:
            self.gp.fit(X_norm, y)
            self.gp_trained = True
        except Exception as e:
            logger.error(f"GP training failed: {e}")
            self.gp_trained = False
    
    def predict(self, params: np.ndarray, physics_function=None, args_obj=None):
        """
        Predict using GP with uncertainty quantification.
        Falls back to physics model if uncertainty is high.
        """
        self.n_surrogate_calls += 1
        
        if not self.gp_trained or physics_function is None:
            # Fallback to physics model
            if physics_function is not None:
                self.n_real_calls += 1
                param_dict = dict(zip(self.param_names, params))
                return physics_function(param_dict, args_obj), None
            else:
                raise ValueError("No trained GP and no physics function provided")
        
        # Normalize input
        X_test = (params.reshape(1, -1) - self.X_mean) / self.X_std
        
        # GP prediction with uncertainty
        y_pred, y_std = self.gp.predict(X_test, return_std=True)
        
        # Check uncertainty threshold
        relative_uncertainty = y_std[0] / (np.abs(y_pred[0]) + 1e-8)
        
        if relative_uncertainty > self.uncertainty_threshold:
            # High uncertainty - call real model and update training set
            logger.debug(f"🎯 High uncertainty ({relative_uncertainty:.3f}) - calling real model")
            
            self.n_real_calls += 1
            param_dict = dict(zip(self.param_names, params))
            y_real = physics_function(param_dict, args_obj)
            
            # Add to training set
            self.X_train.append(params)
            self.y_train.append(y_real)
            
            # Retrain periodically
            if len(self.X_train) % 50 == 0:
                logger.info(f"🔄 Retraining GP with {len(self.X_train)} points")
                self._train_gp()
            
            return y_real, y_std[0]
        
        return y_pred[0], y_std[0]
    
    def get_statistics(self):
        """Return usage statistics"""
        total_calls = self.n_real_calls + self.n_surrogate_calls
        speedup = self.n_surrogate_calls / self.n_real_calls if self.n_real_calls > 0 else 0
        
        return {
            'n_real_calls': self.n_real_calls,
            'n_surrogate_calls': self.n_surrogate_calls,
            'total_calls': total_calls,
            'surrogate_fraction': self.n_surrogate_calls / total_calls if total_calls > 0 else 0,
            'speedup_factor': speedup,
            'n_training_points': len(self.X_train)
        }

# ============================================================================
# Data Validation
# ============================================================================


def validate_gaia_data_for_fitting(gaia_data_dict):
    """Validate Gaia data before fitting"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATING GAIA DATA FOR FITTING")
    logger.info("="*60)
    
    R = gaia_data_dict['R_kpc']
    v = gaia_data_dict['v_obs']
    sigma = gaia_data_dict['sigma_v']
    
    # Basic checks
    n_stars = len(R)
    logger.info(f"Number of stars: {n_stars}")
    
    if n_stars < 100:
        logger.error(f"ERROR: Only {n_stars} stars! Need at least 100 for reliable fit.")
        return False
    
    # Check for NaN/inf
    n_bad_R = np.sum(~np.isfinite(R))
    n_bad_v = np.sum(~np.isfinite(v))
    n_bad_sigma = np.sum(~np.isfinite(sigma))
    
    if n_bad_R + n_bad_v + n_bad_sigma > 0:
        logger.error(f"ERROR: Found non-finite values - R: {n_bad_R}, v: {n_bad_v}, sigma: {n_bad_sigma}")
        return False
    
    # Check ranges
    logger.info(f"R range: [{R.min():.2f}, {R.max():.2f}] kpc")
    logger.info(f"v range: [{v.min():.1f}, {v.max():.1f}] km/s")
    logger.info(f"sigma range: [{sigma.min():.1f}, {sigma.max():.1f}] km/s")
    
    # Check for outliers
    v_median = np.median(v)
    v_mad = np.median(np.abs(v - v_median))
    outliers = np.abs(v - v_median) > 5 * v_mad
    n_outliers = np.sum(outliers)
    
    if n_outliers > 0.1 * n_stars:
        logger.warning(f"WARNING: {n_outliers} velocity outliers ({n_outliers/n_stars*100:.1f}%)")
    
    # Check error distribution
    sigma_median = np.median(sigma)
    if sigma_median < 1.0:
        logger.warning(f"WARNING: Median error {sigma_median:.1f} km/s seems too small")
    elif sigma_median > 50.0:
        logger.warning(f"WARNING: Median error {sigma_median:.1f} km/s seems too large")
    
    # Radial coverage
    R_bins = np.histogram(R, bins=[0, 5, 8, 12, 20, 30])[0]
    logger.info("Radial distribution:")
    bin_labels = ["0-5", "5-8", "8-12", "12-20", "20-30"]
    for i, (label, count) in enumerate(zip(bin_labels, R_bins)):
        logger.info(f"  {label} kpc: {count} stars")
        if count < 10:
            logger.warning(f"  WARNING: Very few stars in {label} kpc bin!")
    
    return True


def test_likelihood_at_typical_params(args, gaia_data):
    """
    Test likelihood at the initial starting point of the MCMC run.
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING LIKELIHOOD AT INITIAL PARAMETERS")
    logger.info("="*60)
    
    # Get parameter configuration, which includes the correct starting guess (p0_guess)
    fitted_p_names, _, p0_guess, _, _, _ = \
        get_param_labels_and_bounds(args)
    
    # --- THIS IS THE CRITICAL FIX ---
    # We now use p0_guess directly. It is the true starting point of the run.
    test_params = p0_guess
    # --- END OF FIX ---
    
    logger.info("Testing with initial parameters (center of prior ranges):")
    for name, val in zip(fitted_p_names, test_params):
        logger.info(f"  {name}: {val:.3e}")
    
    logger.info("\nUsing fixed parameters:")
    if hasattr(args, 'all_param_info_list'):
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                logger.info(f"  {p_info['name']}: {p_info['current_val']:.3e}")
    
    # Temporarily make the plausibility check more lenient for this one test
    args._is_preflight_check = True
    
    logl_args_tuple = (fitted_p_names, args, args.all_param_info_list,
                       gaia_data['R_kpc'], gaia_data['v_obs'], gaia_data['sigma_v'],
                       args.xi, None)
    
    try:
        log_L, blob = log_likelihood_dynesty(test_params, *logl_args_tuple)
    except Exception as e:
        logger.error(f"Exception during likelihood evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Always remove the temporary flag after the test is done
        args._is_preflight_check = False

    logger.info(f"\nLog-likelihood at initial params: {log_L:.1f}")
    logger.info(f"RMSE at initial params: {blob[0]:.1f} km/s")
    
    if not np.isfinite(log_L) or log_L == -np.inf:
        logger.error("ERROR: Initial parameters give -inf likelihood!")
        logger.error("This suggests a fundamental problem. Suggestions:")
        logger.error("1. Check the prior ranges in `prior_transform_dynesty` are reasonable.")
        logger.error("2. Check the fixed baryonic parameter values in `MW_MULTI_COMP_PARAM_CONFIG`.")
        logger.error("3. Ensure the `log_likelihood_dynesty` plausibility checks are not too strict.")
        return False
    
    logger.info("✅ Likelihood test passed. Starting sampler.")
    return True



# ============================================================================
# Curriculum Learning Implementation
# ============================================================================

def run_curriculum_learning(args, gaia_data_dict, logger):
    """
    Implement curriculum learning approach with physical constraints.
    
    Enhanced with:
    - Better stage design based on parameter coupling
    - Physical validation between stages
    - Adaptive resource allocation
    """
    logger.info("🎓 Starting CURRICULUM LEARNING approach")
    
    all_results = {}
    cumulative_params = {}
    
    # Load previous best if requested
    previous_best = None
    if args.use_previous_best:
        previous_best = load_previous_best_params()
    
    # Define curriculum stages with physically motivated progression
    curriculum = [
        {
            'name': 'Stage 1: Initialize from previous best' if previous_best else 'Stage 1: Xi parameters only',
            'fit_flags': {
                'fit_xi_params': True,
                'fit_disk_thin': True,
                'fit_disk_thick': True,
                'fit_bulge': True,
                'fit_gas': True
            },
            'fixed_values': previous_best if previous_best else {},
            'nlive': 1000,
            'dlogz': 0.05,
            'maxcall': int(args.maxcall * 0.5)
        },
        {
            'name': 'Stage 2: Final Refinement',
            'fit_flags': {
                'fit_xi_params': True,
                'fit_disk_thin': True,
                'fit_disk_thick': True,
                'fit_bulge': True,
                'fit_gas': True
            },
            'use_previous': 'all',
            'nlive': args.nlive_init,
            'dlogz': args.dlogz_target,
            'maxcall': int(args.maxcall * 0.5)
        }
    ]
    
    stage_args_per_stage = {}                       # NEW – keep every stage’s Namespace
    
    for i, stage in enumerate(curriculum):
        logger.info(f"\n{'='*80}")
        logger.info(f"📚 {stage['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"Settings: nlive={stage.get('nlive')}, dlogz={stage.get('dlogz')}, "
                   f"maxcall={stage.get('maxcall'):,} ({stage.get('maxcall')/args.maxcall*100:.0f}% of total)")
        
        # Create stage-specific configuration
        stage_args = argparse.Namespace(**vars(args))  # Copy args
        
        # Set fit flags
        for flag, value in stage['fit_flags'].items():
            setattr(stage_args, flag, value)
        
        # Set fixed values
        if 'fixed_values' in stage:
            for param, value in stage['fixed_values'].items():
                fixed_name = param.replace('_solar', '_fixed').replace('_kpc', '_fixed')
                setattr(stage_args, fixed_name, value)
        
        # Use results from previous stages
        if i > 0 and 'use_previous' in stage:
            if stage['use_previous'] == 'all':
                for param, stats in cumulative_params.items():
                    fixed_name = param.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
                    setattr(stage_args, fixed_name, stats['median'])
                    logger.info(f"  Using previous {param}: {stats['median']:.3e}")
            else:
                for param in stage['use_previous']:
                    if param in cumulative_params:
                        fixed_name = param.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
                        setattr(stage_args, fixed_name, cumulative_params[param]['median'])
                        logger.info(f"  Using previous {param}: {cumulative_params[param]['median']:.3e}")
                        
        
        # Update sampler settings
        stage_args_per_stage[f'stage_{i+1}'] = stage_args    
        stage_args.nlive_init = stage.get('nlive', 500)
        stage_args.maxcall = stage.get('maxcall', 200000)
        stage_args.dlogz_target = stage.get('dlogz', args.dlogz_target)
        stage_args.output_dir = Path(args.output_dir) / f"stage_{i+1}"
        
        # Run this stage
        results = run_single_dynesty(stage_args, gaia_data_dict)
        
        if results is None:
            logger.error(f"Stage {i+1} failed!")
            logger.info(f"Successfully completed stages: {list(all_results.keys())}")
            break
        
        all_results[f'stage_{i+1}'] = results
        
        # Extract and validate parameters for next stage
        if hasattr(results, 'samples'):
            # Get weighted samples
            try:
                from dynesty import utils as dyfunc
                samples = dyfunc.resample_equal(results.samples, np.exp(results.logwt - results.logz[-1]))
            except:
                samples = results.samples
                weights = np.exp(results.logwt - results.logz[-1])
            
            # Get parameter names for this stage
            fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(stage_args)
            
            # Calculate statistics and validate
            logger.info(f"\nStage {i+1} Results:")
            all_params_valid = True
            
            for j, param in enumerate(fitted_p_names):
                if j < samples.shape[1]:
                    param_samples = samples[:, j]
                    median_val = np.median(param_samples)
                    std_val = np.std(param_samples)
                    
                    # Check physical bounds
                    if param in PHYSICAL_BOUNDS:
                        bounds = PHYSICAL_BOUNDS[param]
                        if median_val < bounds['min'] or median_val > bounds['max']:
                            if logger:
                                logger.warning(f"  ⚠️  {param}: {median_val:.3e} outside physical bounds!")
                            all_params_valid = False
                    
                    cumulative_params[param] = {
                        'median': median_val,
                        'std': std_val
                    }
                    logger.info(f"  {param}: {median_val:.3e} ± {std_val:.3e}")
            
            if not all_params_valid:
                if logger:
                    logger.warning("⚠️  Some parameters outside physical bounds. Check configuration.")
    
    logger.info(f"\n🎉 Curriculum learning complete!")
    
    # Summary
    total_calls_used = sum(stage.get('maxcall', 0) for stage in curriculum[:len(all_results)])
    logger.info(f"\n📊 Curriculum Learning Summary:")
    logger.info(f"  Total calls used: {total_calls_used:,} / {args.maxcall:,} ({total_calls_used/args.maxcall*100:.0f}%)")
    logger.info(f"  Stages completed: {len(all_results)}")
    logger.info(f"  Final parameters found: {len(cumulative_params)}")
    
    # Final validation
    if cumulative_params:
        final_params = np.array([cumulative_params[p]['median'] 
                               for p in cumulative_params.keys()])
        param_names = list(cumulative_params.keys())
        
        is_valid, reason, *_ = check_physical_plausibility(final_params, param_names, args)
        if is_valid:
            logger.info("✅ Final parameters pass all physical checks!")
        else:
            if logger:
                logger.warning(f"❌ Final parameters fail physical checks: {reason}")
    
    return all_results

def check_early_stopping(sampler, convergence_tracker, args):
    """Check if we should stop due to persistent unphysical solutions."""
    if not hasattr(sampler, 'results') or len(sampler.results.samples) < 1000:
        return False, "Not enough samples yet"
    
    # Check if we've been stuck with bad physics for too long
    if convergence_tracker and hasattr(convergence_tracker, 'health_warnings'):
        if len(convergence_tracker.health_warnings) > 5:
            # Many parameters at bounds
            bound_warnings = [w for w in convergence_tracker.health_warnings if 'bound' in w]
            if len(bound_warnings) > 4:
                return True, f"Too many parameters stuck at bounds ({len(bound_warnings)})"
    
    # Check recent sample validity
    recent_samples = sampler.results.samples[-500:]
    fitted_p_names = args.fitted_param_names  # You'll need to pass this through args
    
    n_valid = 0
    for sample in recent_samples:
        is_valid, *_ = check_physical_plausibility(sample, fitted_p_names, args)
        if is_valid:
            n_valid += 1
    
    valid_fraction = n_valid / len(recent_samples)
    if valid_fraction < 0.01:  # Less than 1% valid samples
        return True, f"Only {valid_fraction*100:.1f}% of recent samples are physical"
    
    return False, "Continuing"

def run_single_dynesty(args, gaia_data_dict, gp_surrogate=None):
    """
    Run a single Dynesty sampling loop with enhanced monitoring, convergence diagnostics,
    physical plausibility checks, and optional dashboard support.
    """

    import threading
    from io import StringIO
    global convergence_tracker

    # -----------------------------------------------------------------------
    # 1. Load Gaia rotation curve data and validate
    # -----------------------------------------------------------------------
    R_data = gaia_data_dict["R_kpc"]
    v_data = gaia_data_dict["v_obs"]
    sigma_data = gaia_data_dict["sigma_v"]

    logger.info(f"Loaded {len(R_data)} stars")
    logger.info(f"R range: {R_data.min():.2f}–{R_data.max():.2f} kpc")
    logger.info(f"v_obs range: {v_data.min():.2f}–{v_data.max():.2f} km/s")

    if not np.all(np.isfinite(R_data)):
        logger.error("Non-finite values detected in R_data")
    if not np.all(np.isfinite(v_data)):
        logger.error("Non-finite values detected in v_data")
    if not np.all(np.isfinite(sigma_data)):
        logger.error("Non-finite values detected in sigma_data")

    # -----------------------------------------------------------------------
    # 2. Load parameter configuration
    # -----------------------------------------------------------------------
    fitted_names, fitted_labels, p0_guess, p_low, p_high, log_flags = get_param_labels_and_bounds(args)
    ndim = len(fitted_names)
    convergence_tracker = ConvergenceTracker(fitted_names)

    # -----------------------------------------------------------------------
    # 3. Inject all_param_info_list if needed (safety patch)
    # -----------------------------------------------------------------------
    if not hasattr(args, "all_param_info_list") or args.all_param_info_list is None:
        logger.warning("⚠️ args.all_param_info_list was missing — injecting now")
        get_param_labels_and_bounds(args)

    # -----------------------------------------------------------------------
    # 4. Check initial likelihood + plausibility
    # -----------------------------------------------------------------------
    logger.info("Checking log-likelihood of initial parameter guess...")
    test_logl, test_blob = log_likelihood_dynesty_debug(
        p0_guess, fitted_names, args, args.all_param_info_list,
        R_data, v_data, sigma_data, args.xi, gp_surrogate
    )
    logger.info(f"Initial logL: {test_logl:.2f}, RMSE: {test_blob[0]:.2f} km/s")

    is_valid, reason, *_ = check_physical_plausibility(p0_guess, fitted_names, args)
    if not is_valid:
        logger.warning(f"Initial parameters fail physical checks: {reason}")
        logger.info("Resetting p0 to midpoint of prior bounds...")
        for i in range(ndim):
            p0_guess[i] = np.sqrt(p_low[i] * p_high[i]) if log_flags[i] else 0.5 * (p_low[i] + p_high[i])

    # -----------------------------------------------------------------------
    # 5. Initialize dynesty sampler
    # -----------------------------------------------------------------------
    ptform_args = (fitted_names, np.array(p_low), np.array(p_high), log_flags)
    logl_args = (fitted_names, args, args.all_param_info_list, R_data, v_data, sigma_data, args.xi, gp_surrogate)

    pool = None
    if args.num_threads > 1:
        try:
            pool = Pool(args.num_threads)
            logger.info(f"Initialized multiprocessing pool with {args.num_threads} threads")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize multiprocessing: {e}")

    sampler = None
    try:
        sampler = DynamicNestedSampler(
            log_likelihood_dynesty_debug,
            prior_transform_dynesty,
            ndim,
            pool=pool,
            queue_size=args.num_threads if pool else None,
            sample=args.sample_method,
            bound=args.bound_method,
            enlarge=args.enlarge_factor,
            walks=args.walks,
            ptform_args=ptform_args,
            logl_args=logl_args,
            blob=True
        )
        logger.info("Dynesty sampler initialized")
        if not hasattr(sampler, "saved_logz"):
            sampler.saved_logz = []
        if hasattr(args, '_resume_checkpoint_file'):
            sampler.restore(args._resume_checkpoint_file)
            logger.info(f"✅ Resumed from checkpoint: {args._resume_checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to create sampler: {e}")
        return None

    # Initialize saved_logz tracking
    try:
        logz_list = getattr(sampler.results, 'logz', None)
        if logz_list is not None and len(logz_list) >= 2:
            sampler.saved_logz = list(logz_list[-2:])
        else:
            sampler.saved_logz = []
    except Exception as e:
        logger.warning(f"Failed to initialize saved_logz: {e}")
        sampler.saved_logz = []

    # -----------------------------------------------------------------------
    # 6. Monitoring setup (dashboard, log files, convergence tracker)
    # -----------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_file = Path(args.output_dir) / "dynesty_checkpoint.pkl"
    dashboard_monitor = None

    if args.enable_dashboard:
        try:
            from monitor_dashboard import DynestyMonitor
            dashboard_monitor = DynestyMonitor(Path(args.output_dir))
            logger.info("Dashboard monitoring enabled")
        except Exception as e:
            logger.warning(f"Dashboard disabled due to error: {e}")

    # -----------------------------------------------------------------------
    # 7. Run either built-in nested loop or custom loop with early stopping
    # -----------------------------------------------------------------------
    run_start_time = time.time()

    if args.use_run_nested:
        logger.info("Running sampler using built-in run_nested()")
        sampler.run_nested(
            nlive_init=args.nlive_init,
            nlive_batch=args.nlive_batch,
            dlogz_init=args.dlogz_target,
            maxcall=args.maxcall,
            print_progress=True,
            checkpoint_file=str(checkpoint_file),
            checkpoint_every=args.checkpoint_every
        )
    else:
        logger.info("Running sampler using custom loop with early stopping")
        args.fitted_param_names = fitted_names
        early_stop_counter = 0
        last_monitor = last_check = time.time()

        try:
            for _ in sampler.sample_initial(nlive=args.nlive_init, maxcall=args.maxcall, save_samples=True):
                now = time.time()

                # Checkpoint
                if now - getattr(sampler, '_last_checkpoint_time', 0) > args.checkpoint_every:
                    try:
                        sampler.save(str(checkpoint_file))
                        logger.info(f"💾 Checkpoint saved at {checkpoint_file}")
                        sampler._last_checkpoint_time = now
                    except Exception as e:
                        logger.warning(f"⚠️ Checkpoint failed: {e}")

                # Monitor progress
                if now - last_monitor > args.monitor_interval_s:
                    last_monitor = now
                    enhanced_monitor_sampler_progress(sampler, fitted_names, fitted_labels,
                                                      run_start_time, logger, gp_surrogate, args,
                                                      dashboard_monitor)

                # Early stop check every 5 min
                if now - last_check > 300:
                    last_check = now
                    stop, reason = check_early_stopping(sampler, convergence_tracker, args)
                    if stop:
                        early_stop_counter += 1
                        logger.warning(f"Early stop check {early_stop_counter}/3: {reason}")
                        if early_stop_counter >= 3:
                            raise RuntimeError("Early stopping: unphysical region")
                    else:
                        early_stop_counter = 0

        except RuntimeError as e:
            logger.error(str(e))
            if hasattr(sampler, 'results'):
                np.savez(Path(args.output_dir) / "partial_results_unphysical.npz",
                         samples=sampler.results.samples,
                         logz=sampler.results.logz,
                         error=str(e))
            if pool:
                pool.close()
                pool.join()
            return None

    # -----------------------------------------------------------------------
    # 8. Return result
    # -----------------------------------------------------------------------
    return sampler.results


# ============================================================================
# Main Entry Point
# ============================================================================

def main_dynesty():
    """
    Main entry point for running the Enhanced Dynesty Sampler with full physical plausibility,
    curriculum learning, dashboard monitoring, and optional GP acceleration.
    
    This function initializes arguments, validates Gaia data, sets up the xi model,
    injects default/fixed values, and then delegates execution to a standard
    Dynesty run or curriculum-learning-based staged sampler.
    
    Includes: robust resumption, logger initialization, theory mode fixes,
    and defensive patch to ensure `args.all_param_info_list` is always populated.
    """
    global logger, debug_counter, RUN_ID
    from datetime import datetime
    import uuid
    RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    logger = get_or_create_logger()
    debug_counter = 0


    import argparse
    from data_io import load_all_sky_gaia_slices
    from pathlib import Path
    

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Enhanced Dynesty sampler for Density-Metric model with physical constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core run options
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Resume from checkpoint in output_dir/dynesty_checkpoint.pkl")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Enable verbose debug logging")
    parser.add_argument('--xi', type=str, default='power',
                            choices=['power', 'logistic', 'enhanced', 'grav_color', 'mass_threshold'],
                            help="Choice of xi(ρ) function")
    parser.add_argument('--max_sample_gaia', type=int, default=10000,
                        help="Maximum number of Gaia stars to use")
    parser.add_argument('--output_dir', type=str, default="chains_dynesty",
                        help="Output directory for results")
    parser.add_argument('--R_d_thin_high', type=float, default=None,
                    help="Override upper prior bound for R_d_thin_kpc")


    # Sampler options
    parser.add_argument('--nlive_init', type=int, default=800,
                        help="Initial number of live points")
    parser.add_argument('--nlive_batch', type=int, default=200,
                        help="Live points per batch")
    parser.add_argument('--dlogz_target', type=float, default=0.01,
                        help="Target dlogz for convergence")
    parser.add_argument('--num_threads', type=int, default=8,
                        help="Number of threads for parallelization")
    parser.add_argument('--maxcall', type=int, default=2000000,
                        help="Maximum likelihood calls")
    parser.add_argument('--monitor_interval_s', type=int, default=60,
                        help="Monitoring interval in seconds")
    parser.add_argument('--enable_dashboard', action='store_true', default=True,
                        help="Enable enhanced monitoring dashboard")
    parser.add_argument('--monitor_config', type=str, default=None,
                        help="Path to monitoring configuration file")
    parser.add_argument('--use_run_nested', action='store_true', default=False,
                        help="Use run_nested instead of custom loop with early stopping")
    parser.add_argument('--checkpoint_every', type=int, default=60,
                        help="Checkpoint interval in seconds")
    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help="Path to a specific dynesty checkpoint to resume from")
    parser.add_argument('--max_thick_thin_ratio', type=float, default=None, 
                        help="Max allowed thick/thin disk mass ratio (default: 0.7 for Milky Way)")
    parser.add_argument('--M_disk_thin_min', type=float, default=None,
                        help="Override lower prior bound for M_disk_thin_solar")
    parser.add_argument('--M_disk_thin_max', type=float, default=None, 
                        help="Override upper prior bound for M_disk_thin_solar")
    parser.add_argument('--h_z_thin_min', type=float, default=None,
                        help="Override lower prior bound for h_z_thin_kpc")
    parser.add_argument('--R_d_thick_max', type=float, default=None,
                        help="Override upper prior bound for R_d_thick_kpc")
    parser.add_argument('--M_gas_max', type=float, default=None,
                        help="Override upper prior bound for M_gas_solar")
    parser.add_argument('--force_new_query_gaia', action='store_true', default=False,
                        help="Force new Gaia query, ignoring raw cache")
    parser.add_argument('--force_reprocess_raw', action='store_true', default=False,
                        help="Force reprocessing of raw Gaia data, ignoring processed cache")


    


    # Dynesty sampler group
    dynesty_g = parser.add_argument_group('Dynesty Sampler Settings')
    dynesty_g.add_argument('--sample_method', type=str, default='rslice',
                           choices=['rwalk', 'rslice', 'hslice'],
                           help="Sampling method")
    dynesty_g.add_argument('--walks', type=int, default=25,
                           help="Number of walks for rwalk sampler (ignored if not using rwalk)")
    dynesty_g.add_argument('--enlarge_factor', type=float, default=2.5,
                           help="Bound enlargement factor")
    dynesty_g.add_argument('--bound_method', type=str, default='multi',
                           choices=['none', 'single', 'multi', 'balls', 'cubes'],
                           help="Bounding method")

    # Enhanced features
    ai_g = parser.add_argument_group('Enhanced Features')
    ai_g.add_argument('--use_curriculum_learning', action='store_true', default=False,
                      help="Use curriculum learning (recommended for many parameters)")
    ai_g.add_argument('--use_gp_surrogate', action='store_true', default=False,
                      help="Use Gaussian Process surrogate for speedup")
    ai_g.add_argument('--gp_n_initial', type=int, default=500,
                      help="Initial training points for GP")
    ai_g.add_argument('--gp_uncertainty_threshold', type=float, default=0.1,
                      help="GP uncertainty threshold")
    ai_g.add_argument('--validate_data', action='store_true', default=True,
                      help="Validate loaded data quality")
    parser.add_argument('--use_previous_best', action='store_true', default=False,
                        help="Initialize from previous best-fit parameters")
    parser.add_argument('--previous_results_file', type=str,
                        default="chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz",
                        help="Path to previous results for initialization")
    parser.add_argument('--tighten_bounds_factor', type=float, default=0.1,
                        help="Factor for tightening bounds around previous best (0.1 = 10% window)")
    parser.add_argument('--disable_dashboard', action='store_true', default=False,
                        help="Disable dashboard monitoring to avoid JSON errors")
    ai_g.add_argument('--fix_gamma', type=float, default=None,
                      help="Fix gamma exponent (theory predicts 2.7)")
    ai_g.add_argument('--fix_lambda_g', type=float, default=None,
                      help="Fix lambda_g enhancement factor (theory predicts 8.0)")
    ai_g.add_argument('--theory_mode', action='store_true', default=False,
                      help="Use theoretical values: gamma=2.7, lambda_g=8.0")

    # Model components
    mw_model_g = parser.add_argument_group('Model Components')
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True)
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)

    # Fit flags (original + grav_color extensions)
    fit_g = parser.add_argument_group('Parameters to Fit')
    fit_g.add_argument('--fit_xi_params', action='store_true',
                       help="Fit xi function parameters")
    fit_g.add_argument('--fit_rho_c', action='store_true',
                       help="Fit rho_c (for grav_color mode)")
    fit_g.add_argument('--fit_gamma', action='store_true',
                       help="Fit gamma exponent (grav_color)")
    fit_g.add_argument('--fit_lambda_g', action='store_true',
                       help="Fit lambda_g enhancement (grav_color)")
    fit_g.add_argument('--fit_disk_thin', action='store_true',
                       help="Fit thin disk parameters")
    fit_g.add_argument('--fit_disk_thick', action='store_true',
                       help="Fit thick disk parameters")
    fit_g.add_argument('--fit_bulge', action='store_true',
                       help="Fit bulge parameters")
    fit_g.add_argument('--fit_gas', action='store_true',
                       help="Fit gas parameters")

    # Fixed values
    fixed_g = parser.add_argument_group('Fixed Parameter Values')
    for p_name, p_details in MW_MULTI_COMP_PARAM_CONFIG.items():
        fixed_g.add_argument(f"--{p_details['fixed_val_from_arg']}",
                             type=float,
                             default=p_details['default_fixed'],
                             help=f"Fixed/initial value for {p_name}")


    # Parse args
    args = parser.parse_args()
    
    logger = get_or_create_logger()
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dynesty_debug.log"

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)



    logger.info("📡 Logger initialized. Writing to: %s", log_file)


    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Set to WARNING to reduce output
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True  # Override any existing configuration
    )

    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()

    # Add only ONE handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # File gets everything
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(file_handler)

    # Set logger level based on production vs debug mode
    if args.debug:  # Add --debug flag to your argparser
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)  # Only warnings and errors to console    
    
    logger.info("Starting Enhanced Dynesty Sampler v2.0")

    if not DYNESTY_AVAILABLE:
        logger.error("Dynesty library not available.")
        sys.exit(1)

    # Load Gaia data
    gaia_cache_file = Path("gaia_sky_slices") / "all_sky_gaia.csv"
    if not gaia_cache_file.exists() or args.force_new_query_gaia:
        df_all_sky = load_all_sky_gaia_slices(
            lon_bin_width=30,
            stars_per_bin=12000,
            output_dir="gaia_sky_slices",
            force_query=args.force_new_query_gaia,
            max_distance_kpc=30.0
        )
    else:
        import pandas as pd
        df_all_sky = pd.read_csv(gaia_cache_file)

    if df_all_sky.empty:
        logger.error("Gaia data load failed")
        sys.exit(1)

    gaia_data_dict = {col: df_all_sky[col].values for col in df_all_sky.columns}

    # Inject xi-mode specific parameters
    setup_xi_parameters_for_mode(args)

    # Ensure physical prior bounds match -- override if needed
    if args.R_d_thin_high is not None:
        MW_MULTI_COMP_PARAM_CONFIG['R_d_thin_kpc']['high'] = args.R_d_thin_high
    if args.M_disk_thin_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thin_solar']['low'] = args.M_disk_thin_min
    if args.M_disk_thin_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thin_solar']['high'] = args.M_disk_thin_max
    if args.h_z_thin_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['h_z_thin_kpc']['low'] = args.h_z_thin_min
    if args.R_d_thick_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['R_d_thick_kpc']['high'] = args.R_d_thick_max
    if args.M_gas_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_gas_solar']['high'] = args.M_gas_max

    if args.checkpoint_file is None:
        args.checkpoint_file = str(Path(args.output_dir) / "dynesty_checkpoint.pkl")

    if args.resume:
        checkpoint_path = Path(args.checkpoint_file)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        args._resume_checkpoint_file = str(checkpoint_path)

    # Safety patch for all_param_info_list (used by plausibility checks)
    if not hasattr(args, 'all_param_info_list') or args.all_param_info_list is None:
        logger.info("Injecting args.all_param_info_list with get_param_labels_and_bounds()")
        get_param_labels_and_bounds(args)
        
    if args.xi == 'mass_threshold':
        logger.error("WARNING: mass_threshold model CANNOT simultaneously pass Cassini and galaxy tests!")
        logger.error("This model is fundamentally incompatible with Solar System constraints.")
        logger.error("Consider using 'power', 'enhanced', or 'grav_color' instead.")
        logger.info("DEBUG: Checking M_crit_msun config:")
        if 'M_crit_msun' in MW_MULTI_COMP_PARAM_CONFIG:
            config = MW_MULTI_COMP_PARAM_CONFIG['M_crit_msun']
            logger.info(f"  Low: {config['low']}")
            logger.info(f"  High: {config['high']}")
            logger.info(f"  Default: {config['default_fixed']}")
        else:
            logger.error("  M_crit_msun NOT FOUND in config!")        

    # Theory mode overrides
    if args.theory_mode:
        logger.info("🧪 THEORY MODE: gamma=2.7, lambda_g=8.0, only fitting rho_c")
        args.fix_gamma = 2.7
        args.fix_lambda_g = 8.0
        args.gamma_fixed = 2.7
        args.lambda_g_fixed = 8.0
        args.fit_xi_params = False
        args.fit_gamma = False
        args.fit_lambda_g = False

    # Validate Gaia
    if not validate_gaia_data_for_fitting(gaia_data_dict):
        logger.error("❌ Gaia data validation failed.")
        sys.exit(1)

    # GP surrogate (optional)
    gp_surrogate = None
    if args.use_gp_surrogate:
        if not GP_AVAILABLE:
            logger.error("GP surrogate requested but scikit-learn not available.")
            sys.exit(1)
        _, _, _, p_low, p_high, _ = get_param_labels_and_bounds(args)
        param_bounds = np.column_stack([p_low, p_high])
        gp_surrogate = GPSurrogateModel(param_names=args.fitted_param_names,
                                        param_bounds=param_bounds,
                                        uncertainty_threshold=args.gp_uncertainty_threshold,
                                        n_initial=args.gp_n_initial)
        gp_surrogate.generate_initial_training_data(lambda p, a: v_model_for_dynesty(gaia_data_dict['R_kpc'], p, a.xi, a), args)

    # Sampling logic
    logger.info("Beginning sampling...")
    if args.use_curriculum_learning:
        results = run_curriculum_learning(args, gaia_data_dict, logger)
    else:
        results = run_single_dynesty(args, gaia_data_dict, gp_surrogate)

    # Save results...
    # (Unmodified saving/snapshot code continues here)

    logger.info("✅ Main dynesty run complete.")

    
    if results is None:
        # This only works if run_single_dynesty() ran far enough to initialize sampler
        try:
            if 'sampler' in locals() and hasattr(sampler, 'results') and hasattr(sampler.results, 'samples'):
                output_file = Path(args.output_dir) / "partial_results_unphysical.npz"
                np.savez(output_file,
                        samples=sampler.results.samples,
                        logz=sampler.results.logz,
                        logzerr=getattr(sampler.results, 'logzerr', None),
                        logl=sampler.results.logl,
                        blob=getattr(sampler.results, 'blob', None))
                logger.info(f"🧪 Saved partial results to: {output_file}")
            else:
                logger.warning("⚠️ Sampler did not produce valid results. No partial output saved.")
        except Exception as e:
            logger.error(f"❌ Failed to save partial results: {e}")

        logger.error("No results to save")
        finalize_record(RUN_ID, success=False,
                        logz=np.nan, logz_err=np.nan,
                        eff=0, rmse=np.nan,
                        n_samples=0, n_calls=0,
                        param_stats={}, phys_ok=False,
                        phys_reason="No results to save")
        return


    
    # Save results
    if results is None:
        logger.error("No results to save")
        finalize_record(RUN_ID, success=False,
                        logz=np.nan, logz_err=np.nan,
                        eff=0, rmse=np.nan,
                        n_samples=0, n_calls=0,
                        param_stats={}, phys_ok=False,
                        phys_reason="No results to save")
        return

    # Curriculum learning
    if isinstance(results, dict) and 'stage_1' in results:
        completed_stages = [k for k, v in results.items() if v is not None]
        if not completed_stages:
            logger.error("❌ All curriculum stages failed. No results to save.")
            return

        # Save each completed stage separately
        for stage_name in completed_stages:
            stage_results = results[stage_name]

            # --- obtain the fitted‑parameter list for THIS stage ----------------
            stage_args_local = stage_args_per_stage[stage_name]          # NEW
            fitted_p_names_stage, _, _, _, _, _ = get_param_labels_and_bounds(
                stage_args_local)                                        # NEW
            # --------------------------------------------------------------------

            output_prefix = f"dynesty_curriculum_{stage_name}_{args.xi}"
            output_npz    = Path(args.output_dir) / f"{output_prefix}_samples.npz"

            try:
                weights = np.exp(stage_results.logwt - stage_results.logz[-1])
                np.savez(
                    output_npz,
                    samples     = stage_results.samples,
                    weights     = weights,
                    param_names = np.array(fitted_p_names_stage),         # NEW
                    logl        = stage_results.logl,
                    logz        = stage_results.logz,
                    logzerr     = stage_results.logzerr,
                )
                logger.info(f"✅ Saved {stage_name} results to {output_npz}")
            except Exception as e:
                logger.error(f"❌ Failed to save {stage_name} results: {e}")

        # Use the last completed stage for any downstream processing
        final_stage = max(completed_stages)
        res = results[final_stage]

        # We also need the parameter names for that final stage               # NEW
        fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(          # NEW
            stage_args_per_stage[final_stage])                                # NEW

    else:
        # Single‑run results
        res = results

        # Parameter names come from the main run’s args                        # NEW
        fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(args)     # NEW


    # ---------------------------------------------------------------------------
    # Save the final stage (or single‑run) results
    # ---------------------------------------------------------------------------
    output_parts = ["dynesty_mw", args.xi]
    if args.include_bulge:       output_parts.append("B"  + ("f" if args.fit_bulge       else "x"))
    if args.include_disk_thin:   output_parts.append("DT" + ("f" if args.fit_disk_thin   else "x"))
    if args.include_disk_thick:  output_parts.append("DK" + ("f" if args.fit_disk_thick  else "x"))
    if args.include_gas:         output_parts.append("G"  + ("f" if args.fit_gas         else "x"))

    output_basename = "_".join(output_parts)
    output_npz      = Path(args.output_dir) / f"{output_basename}_samples.npz"

    # Effective sample size
    try:
        ess = res.effective_sample_size if hasattr(res, 'effective_sample_size') else 0
    except Exception:
        weights = np.exp(res.logwt - res.logz[-1])
        ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0

    # Save final results
    try:
        np.savez(
            output_npz,
            samples     = res.samples,
            weights     = np.exp(res.logwt - res.logz[-1]),
            param_names = np.array(fitted_p_names),                      # uses stage‑specific names
            logl        = res.logl,
            logz        = res.logz,
            logzerr     = res.logzerr,
            ess         = ess,
            blob        = res.blob if hasattr(res, 'blob') else None,
        )
        logger.info(f"\n✅ Final results saved to {output_npz}")
    except Exception as e:
        logger.error(f"❌ Failed to save final results: {e}")

    # Save final .pkl
    output_pkl = Path(args.output_dir) / f"{output_basename}_results.pkl.gz"
    try:
        with gzip.open(output_pkl, "wb") as fh:
            pickle.dump(res, fh)
        logger.info(f"✅ Full results saved to {output_pkl}")
    except Exception as e:
        logger.error(f"❌ Failed to save pickle file: {e}")
        
    if results is None or not hasattr(results, 'samples') or len(results.samples) < 10:
        logger.error("Not enough samples for finalization")
        logger.info("Run terminated early - check parameters and try again")
        return

    # === FINALIZE RUN AND SNAPSHOT ===

    try:
        # Check if we have valid results
        if res is None or not hasattr(res, 'samples') or len(res.samples) < 10:
            logger.warning("⚠️ Not enough samples for proper finalization")
            finalize_record(RUN_ID, success=False,
                           logz=np.nan, logz_err=np.nan,
                           eff=0, rmse=np.nan,
                           n_samples=0, n_calls=0,
                           param_stats={}, phys_ok=False,
                           phys_reason="Too few samples")
            return
            
        # Calculate parameter statistics safely
        param_stats = {}
        try:
            if hasattr(res, 'weights') and res.weights is not None:
                weights = res.weights
                mean_params = np.average(res.samples, weights=weights, axis=0)
                # Weighted standard deviation
                variance = np.average((res.samples - mean_params)**2, weights=weights, axis=0)
                std_params = np.sqrt(variance)
            else:
                mean_params = np.mean(res.samples, axis=0)
                std_params = np.std(res.samples, axis=0)
                
            for i, name in enumerate(fitted_p_names):
                if i < len(mean_params):
                    param_stats[name] = {
                        "median": float(mean_params[i]),
                        "sigma": float(std_params[i])
                    }
        except Exception as e:
            logger.warning(f"Could not calculate parameter statistics: {e}")
            param_stats = {}

        # Get other statistics safely
        try:
            logz = res.logz[-1] if hasattr(res, 'logz') and res.logz is not None else np.nan
            logz_err = res.logzerr[-1] if hasattr(res, 'logzerr') and res.logzerr is not None else np.nan
        except Exception as e:
            logger.error(f"LogZ extraction failed: {e}")
            logz = logz_err = np.nan
        
        
        eff = float(res.eff) if hasattr(res, 'eff') else 0.0
        
        # Calculate RMSE if blob data exists
        rmse = np.nan
        if hasattr(res, 'blob') and res.blob is not None:
            try:
                rmse = float(np.sqrt(np.mean(res.blob**2)))
            except:
                rmse = np.nan
                
        n_samples = len(res.samples) if hasattr(res, 'samples') else 0
        n_calls = int(np.sum(res.ncall)) if hasattr(res, 'ncall') else 0

        # Check physical plausibility if we have parameters
        phys_ok = False
        phys_reason = "Not checked"
        if param_stats and len(mean_params) > 0:
            try:
                is_valid, reason = check_physical_plausibility(mean_params, fitted_p_names, args)
                phys_ok = is_valid
                phys_reason = reason if not is_valid else ""
            except Exception as e:
                logger.warning(f"Could not check physical plausibility: {e}")

        # Finalize the record
        finalize_record(RUN_ID, success=True,
                        logz=logz, logz_err=logz_err,
                        eff=eff, rmse=rmse,
                        n_samples=n_samples, n_calls=n_calls,
                        param_stats=param_stats,
                        phys_ok=phys_ok, phys_reason=phys_reason)

        # Create snapshot
        snapshot_path = Path(args.output_dir) / f"run_{RUN_ID}_summary.json"
        with open(snapshot_path, "w") as fh:
            json.dump({
                "run_id": RUN_ID,
                "success": True,
                "logZ": logz,
                "params": param_stats,
                "phys_ok": phys_ok,
                "phys_reason": phys_reason,
                "n_samples": n_samples,
                "rmse": rmse,
                "cmd": " ".join(sys.argv)
            }, fh, indent=2)
        logger.info(f"📄 Snapshot saved to {snapshot_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to finalize or snapshot run: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL: Dynesty library not found")
        sys.exit(1)

    main_dynesty()