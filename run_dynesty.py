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
    'M_disk_thin_solar':   {'min': 2.4e10, 'max': 9.0e10, 'typical': 5.0e10},   # Lower min, slightly higher max
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
    'rho_c_solar_kpc3':    {'min': 1e7,  'max': 1e10,   'typical': 5e8},
    'n_exp':               {'min': 0.5,  'max': 4.0,    'typical': 2.7},
}


# Expected ranges for validation
EXPECTED_XI_AT_SOLAR = (0.7, 1.0)  # Xi should not suppress gravity too much at R_sun
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
    'rho_c_solar_kpc3': {
        'label': "rho_c (M_sun/kpc^3)", 
        'fixed_val_from_arg': 'rho_c_fixed', 
        'default_fixed': 5e6,
        'low': 1e6, 
        'high': 2e7, 
        'fit_flag_arg': 'fit_xi_params',
        'log_prior': True,
        'physical_check': True
    },
    'n_exp': {
        'label': "n", 
        'fixed_val_from_arg': 'n_exp_fixed', 
        'default_fixed': 2.7,  # Update to theoretical value
        'low': PHYSICAL_BOUNDS['n_exp']['min'], 
        'high': PHYSICAL_BOUNDS['n_exp']['max'], 
        'fit_flag_arg': 'fit_xi_params',
        'log_prior': False,
        'physical_check': True
    },
    'M_disk_thin_solar': {
        'label': "M_disk_thin (M_sun)", 
        'fixed_val_from_arg': 'M_disk_thin_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['M_disk_thin_solar']['typical'],
        'low': PHYSICAL_BOUNDS['M_disk_thin_solar']['min'], 
        'high': PHYSICAL_BOUNDS['M_disk_thin_solar']['max'], 
        'fit_flag_arg': 'fit_disk_thin', 
        'include_flag_arg': 'include_disk_thin',
        'log_prior': True,
        'physical_check': True
    },
    'R_d_thin_kpc': {
        'label': "R_d_thin (kpc)", 
        'fixed_val_from_arg': 'R_d_thin_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['R_d_thin_kpc']['typical'],
        'low': 1.5,    # Keep as is
        'high': 5.0,   # Widened again based on recommendation
        'fit_flag_arg': 'fit_disk_thin', 
        'include_flag_arg': 'include_disk_thin',
        'log_prior': False,
        'physical_check': True
    },
    'h_z_thin_kpc': {
        'label': "h_z_thin (kpc)", 
        'fixed_val_from_arg': 'h_z_thin_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['h_z_thin_kpc']['typical'],
        'low': PHYSICAL_BOUNDS['h_z_thin_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['h_z_thin_kpc']['max'], 
        'fit_flag_arg': 'fit_disk_thin', 
        'include_flag_arg': 'include_disk_thin',
        'log_prior': False,
        'physical_check': True
    },
    'M_disk_thick_solar': {
        'label': "M_disk_thick (M_sun)", 
        'fixed_val_from_arg': 'M_disk_thick_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['M_disk_thick_solar']['typical'],
        'low': PHYSICAL_BOUNDS['M_disk_thick_solar']['min'], 
        'high': PHYSICAL_BOUNDS['M_disk_thick_solar']['max'], 
        'fit_flag_arg': 'fit_disk_thick', 
        'include_flag_arg': 'include_disk_thick',
        'log_prior': True,
        'physical_check': True
    },
    'R_d_thick_kpc': {
        'label': "R_d_thick (kpc)", 
        'fixed_val_from_arg': 'R_d_thick_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['R_d_thick_kpc']['typical'],
        'low': 4.0,    # Increased from 2.5
        'high': 8.0,   # Keep as is
        'fit_flag_arg': 'fit_disk_thick', 
        'include_flag_arg': 'include_disk_thick',
        'log_prior': False,
        'physical_check': True
    },
    'h_z_thick_kpc': {
        'label': "h_z_thick (kpc)", 
        'fixed_val_from_arg': 'h_z_thick_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['h_z_thick_kpc']['typical'],
        'low': PHYSICAL_BOUNDS['h_z_thick_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['h_z_thick_kpc']['max'], 
        'fit_flag_arg': 'fit_disk_thick', 
        'include_flag_arg': 'include_disk_thick',
        'log_prior': False,
        'physical_check': True
    },
    'M_bulge_solar': {
        'label': "M_bulge (M_sun)", 
        'fixed_val_from_arg': 'M_bulge_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['M_bulge_solar']['typical'],
        'low': PHYSICAL_BOUNDS['M_bulge_solar']['min'], 
        'high': PHYSICAL_BOUNDS['M_bulge_solar']['max'], 
        'fit_flag_arg': 'fit_bulge', 
        'include_flag_arg': 'include_bulge',
        'log_prior': True,
        'physical_check': True
    },
    'a_bulge_kpc': {
        'label': "a_bulge (kpc)", 
        'fixed_val_from_arg': 'a_bulge_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['a_bulge_kpc']['typical'],
        'low': PHYSICAL_BOUNDS['a_bulge_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['a_bulge_kpc']['max'], 
        'fit_flag_arg': 'fit_bulge', 
        'include_flag_arg': 'include_bulge',
        'log_prior': False,
        'physical_check': True
    },
    'M_gas_solar': {
        'label': "M_gas (M_sun)", 
        'fixed_val_from_arg': 'M_gas_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['M_gas_solar']['typical'],
        'low': PHYSICAL_BOUNDS['M_gas_solar']['min'], 
        'high': PHYSICAL_BOUNDS['M_gas_solar']['max'], 
        'fit_flag_arg': 'fit_gas', 
        'include_flag_arg': 'include_gas',
        'log_prior': True,
        'physical_check': True
    },
    'R_d_gas_kpc': {
        'label': "R_d_gas (kpc)", 
        'fixed_val_from_arg': 'R_d_gas_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['R_d_gas_kpc']['typical'],
        'low': PHYSICAL_BOUNDS['R_d_gas_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['R_d_gas_kpc']['max'], 
        'fit_flag_arg': 'fit_gas', 
        'include_flag_arg': 'include_gas',
        'log_prior': False,
        'physical_check': True
    },
    'h_z_gas_kpc': {
        'label': "h_z_gas (kpc)", 
        'fixed_val_from_arg': 'h_z_gas_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['h_z_gas_kpc']['typical'],
        'low': PHYSICAL_BOUNDS['h_z_gas_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['h_z_gas_kpc']['max'], 
        'fit_flag_arg': 'fit_gas', 
        'include_flag_arg': 'include_gas',
        'log_prior': False,
        'physical_check': True
    },
        'gamma_exp': {                             # NEW
        'label': "γ (grav‑color)",
        'fixed_val_from_arg': 'gamma_fixed',
        'default_fixed': 2.8,
        'low': 2.0, 
        'high': 3.5,
        'fit_flag_arg': 'fit_gamma',
        'log_prior': False,
        'physical_check': False       # checked together with λ_g below
    },
    'lambda_g': {                                # NEW
        'label': "λ_g",
        'fixed_val_from_arg': 'lambda_g_fixed',
        'default_fixed': 1.2,
        'low': 0.5, 
        'high': 5.0,
        'fit_flag_arg': 'fit_lambda_g',
        'log_prior': False,
        'physical_check': False
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
    Check if parameters are physically reasonable.
    
    This function implements multiple checks:
    1. Individual parameter bounds
    2. Total mass constraints
    3. Xi reasonableness at solar radius
    4. Relative component ratios
    5. Density profile consistency
    
    Parameters
    ----------
    theta_values : np.ndarray
        Parameter values to check
    param_names : list
        Names of parameters
    args_obj : argparse.Namespace
        Additional arguments including model configuration
        
    Returns
    -------
    is_valid : bool
        True if parameters pass all checks
    reason : str
        Description of failure if not valid
    """
    params = dict(zip(param_names, theta_values))
    
    # 1. Check individual parameter bounds against PHYSICAL_BOUNDS
    for param, value in params.items():
        if param in PHYSICAL_BOUNDS:
            bounds = PHYSICAL_BOUNDS[param]
            if value < bounds['min'] or value > bounds['max']:
                return False, f"{param} = {value:.2e} outside [{bounds['min']:.2e}, {bounds['max']:.2e}]"
    
    # 2. Check total baryonic mass
    mass_components = ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']
    total_mass = sum(params.get(comp, 0) for comp in mass_components)
    
    if total_mass < PHYSICAL_BOUNDS['M_total']['min']:
        return False, f"Total mass {total_mass:.2e} < {PHYSICAL_BOUNDS['M_total']['min']:.2e} M_sun"
    if total_mass > PHYSICAL_BOUNDS['M_total']['max']:
        return False, f"Total mass {total_mass:.2e} > {PHYSICAL_BOUNDS['M_total']['max']:.2e} M_sun"
    
    # 3. Check relative mass ratios (thick disk shouldn't dominate)
    if 'M_disk_thick_solar' in params and 'M_disk_thin_solar' in params:
        if params['M_disk_thick_solar'] > 0 and params['M_disk_thin_solar'] > 0:
            thick_thin_ratio = params['M_disk_thick_solar'] / params['M_disk_thin_solar']

            # ---> NEW: choose limit from CLI or context
            if args_obj.max_thick_thin_ratio is not None:
                ratio_limit = args_obj.max_thick_thin_ratio
            elif getattr(args_obj, 'fit_target', '') == 'milkyway':
                ratio_limit = 0.7
            else:
                ratio_limit = np.inf          # no constraint

            if thick_thin_ratio > ratio_limit:
                # quadratic penalty,  Δχ² ≈ 20 for a 10 % violation
                penalty = -20.0 * (thick_thin_ratio / ratio_limit - 1.0)**2
                return (True, "Ratio penalty", penalty)   # adjust calling code to receive penalty

    # 4. Check scale length ordering (thick disk more extended)
    if 'R_d_thick_kpc' in params and 'R_d_thin_kpc' in params:
        if params['R_d_thick_kpc'] < params['R_d_thin_kpc']:
            return False, f"Thick disk scale length < thin disk ({params['R_d_thick_kpc']:.2f} < {params['R_d_thin_kpc']:.2f} kpc)"
    
    # 5. Check scale height ordering (thick disk thicker)
    if 'h_z_thick_kpc' in params and 'h_z_thin_kpc' in params:
        if params['h_z_thick_kpc'] < params['h_z_thin_kpc'] * 2:
            return False, f"Thick disk not thick enough ({params['h_z_thick_kpc']:.2f} < 2×{params['h_z_thin_kpc']:.2f} kpc)"
    
    # 6. Check xi at solar radius (shouldn't suppress gravity too much)
    xi_solar = 1.0  # <-- Default so it's always defined

    if 'rho_c_solar_kpc3' in params:
        # Estimate density at solar radius
        # For a typical disk: rho(R_sun) ~ 0.1 M_sun/pc^3 = 1e8 M_sun/kpc^3
        rho_solar_typical = 1e8

        # Add contribution from bulge if included
        if args_obj.include_bulge and 'M_bulge_solar' in params and 'a_bulge_kpc' in params:
            # Hernquist profile at R_sun
            M_b = params['M_bulge_solar']
            a_b = params['a_bulge_kpc']
            rho_bulge_solar = (M_b / (2 * np.pi)) * (a_b / (R_SUN_KPC * (R_SUN_KPC + a_b)**3))
            rho_solar_typical += rho_bulge_solar

        # Safely compute xi_solar using selected xi mode
        xi_mode = getattr(args_obj, 'xi', 'power')

        if xi_mode == 'grav_color':
            from density_metric2 import xi_gravitational_color
            gamma = params.get('gamma_exp', 2.7)
            lambda_g = params.get('lambda_g', 1.5)
            rho_c = params['rho_c_solar_kpc3']
            xi_solar = xi_gravitational_color(rho_solar_typical, rho_c, gamma, lambda_g)[0]

        elif 'n_exp' in params:
            from density_metric2 import XI_FUNCTION_MAP
            xi_func = XI_FUNCTION_MAP.get(xi_mode, XI_FUNCTION_MAP['power'])
            rho_c = params['rho_c_solar_kpc3']
            n_exp = params['n_exp']
            xi_solar = xi_func(rho_solar_typical, rho_c, n_exp)[0]

        # Validate the xi_solar value
        if xi_solar < EXPECTED_XI_AT_SOLAR[0]:
            return False, f"xi at R_sun = {xi_solar:.3f} < {EXPECTED_XI_AT_SOLAR[0]} (too much suppression)"
        if xi_solar > EXPECTED_XI_AT_SOLAR[1]:
            logger.debug(f"Note: xi at R_sun = {xi_solar:.3f} > {EXPECTED_XI_AT_SOLAR[1]} (minimal modification)")

    
    # 7. Check that we'd get reasonable rotation curve at solar radius
    if all(comp in params for comp in ['M_disk_thin_solar', 'R_d_thin_kpc']):
        # Quick estimate of Newtonian velocity
        M_enc_approx = 0.6 * total_mass  # Rough enclosed mass at R_sun
        v_newton_approx = np.sqrt(G_ASTRO_UNITS * M_enc_approx / R_SUN_KPC)
        
        # With xi modification
        if 'rho_c_solar_kpc3' in params:
            v_expected = v_newton_approx * np.sqrt(xi_solar)
            
            if v_expected < EXPECTED_V_AT_SOLAR[0] or v_expected > EXPECTED_V_AT_SOLAR[1]:
                return False, f"Estimated v(R_sun) = {v_expected:.0f} km/s outside [{EXPECTED_V_AT_SOLAR[0]}, {EXPECTED_V_AT_SOLAR[1]}]"
    
    return True, "OK"


def check_parameter_evolution(
    recent_samples: np.ndarray,
    param_names: List[str],
    logger: logging.Logger
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
    logger : logging.Logger
        Logger for warnings
        
    Returns
    -------
    dict
        Analysis results with warnings and statistics
    """
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
            'default_fixed': 5e7,  # Galaxy-appropriate
            'low': 1e6,
            'high': 1e9,
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
            'high': 4.0,
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
            'default_fixed': 5e8,
            'low': 1e7,
            'high': 1e10,
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
            'high': 4.0,
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
    dashboard_monitor=None  # Optional dashboard integration
):
    """
    Enhanced monitoring with parameter health checks, convergence diagnostics, and optional dashboard updates.
    """
    global convergence_tracker
    from density_metric2 import XI_FUNCTION_MAP, xi_power_law


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

        # Total calls
        ncall_total = np.sum(res.ncall) if isinstance(res.ncall, np.ndarray) else res.ncall
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))

        # Efficiency
        eff = 100.0 * n_samples / ncall_total if ncall_total > 0 else 0.0
        logger.info(f"⏱️  Elapsed: {elapsed_str} | 📊 Samples: {n_samples:,} | 🎲 Calls: {ncall_total:,} | 📈 Eff: {eff:.2f}%")

        if gp_surrogate is not None and GP_AVAILABLE:
            gp_stats = gp_surrogate.get_statistics()
            logger.info(f"🤖 GP Surrogate: {gp_stats['n_real_calls']:,} real, "
                        f"{gp_stats['n_surrogate_calls']:,} surrogate (speedup: {gp_stats['speedup_factor']:.1f}x)")

        if n_samples < 50:
            logger.info("⚠️  Too few samples for detailed analysis")
            return

        # LogZ stats
        current_logz = -np.inf
        if hasattr(res, 'logz') and len(res.logz) > 0:
            current_logz = res.logz[-1]
            if not np.isfinite(current_logz):
                logger.error("❌ log(Z) = -inf. All live points have likelihood = -inf.")
                return
            else:
                logger.info(f"📊 Log(Z): {current_logz:.3f}")
                if hasattr(res, 'logzerr') and len(res.logzerr) > 0:
                    logger.info(f"   Error: ±{res.logzerr[-1]:.3f}")

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
        param_issues = []

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
                    param_issues.append(f"{param_name} near lower bound")
                elif median_val > config['high'] * 0.9:
                    status = "⚠️ Near upper bound"
                    param_issues.append(f"{param_name} near upper bound")
                elif param_name in PHYSICAL_BOUNDS:
                    bounds = PHYSICAL_BOUNDS[param_name]
                    if median_val < bounds['min'] or median_val > bounds['max']:
                        status = "❌ Unphysical"
                        param_issues.append(f"{param_name} outside physical bounds")

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
            param_issues.append(f"Physical check failed: {reason}")

        # Sample validity
        n_valid = sum(
            1 for j in range(min(100, len(recent_samples)))
            if check_physical_plausibility(recent_samples[j], fitted_param_names, args_obj)[0]
        )
        valid_fraction = n_valid / min(100, len(recent_samples))
        logger.info(f"   {valid_fraction*100:.1f}% of recent samples pass physical checks")
        if valid_fraction < 0.5:
            logger.warning("⚠️  Majority of samples failing physical checks!")
            param_issues.append("Low fraction of physically valid samples")

        # Model predictions at solar radius
    logger.info(f"\n🌟 MODEL PREDICTIONS AT SOLAR RADIUS (R = {R_SUN_KPC:.2f} kpc):")
    logger.info("─" * 60)
    
    # Reconstruct the full parameter dictionary for the current median sample
    res = sampler.results
    recent_samples = res.samples[-min(1000, len(res.samples)):]
    current_params = np.median(recent_samples, axis=0)
    full_params = dict(zip(fitted_param_names, current_params))
    for p_info in args_obj.all_param_info_list:
        if not p_info['is_fitted']:
            full_params[p_info['name']] = p_info['current_val']
            
    full_params['include_disk_thin'] = args_obj.include_disk_thin
    full_params['include_disk_thick'] = args_obj.include_disk_thick
    full_params['include_bulge'] = args_obj.include_bulge
    full_params['include_gas'] = args_obj.include_gas

    try:
        r_solar = np.array([R_SUN_KPC])
        
        # Calculate Newtonian velocity (the baseline)
        v_newton_solar = v_baryon_total_newtonian_kms(r_solar, full_params)[0]
        
        # Calculate DDMM velocity
        rho_solar = rho_baryon_total_midplane_solar_kpc3(r_solar, full_params)[0]
        xi_func = XI_FUNCTION_MAP.get(args_obj.xi, XI_FUNCTION_MAP['power'])
        n_key = 'gamma_exp' if 'gamma_exp' in full_params else 'n_exp'
        A_key = 'lambda_g' if 'lambda_g' in full_params else 'A'
        xi_solar = xi_func(rho_solar, full_params['rho_c_solar_kpc3'], full_params[n_key], full_params.get(A_key, 1.0))
        xi_solar = np.minimum(xi_solar, 5.0)[0]
        v_model_solar = v_newton_solar * np.sqrt(xi_solar)

        # Print the direct comparison
        logger.info(f"   Newtonian Velocity (Baryons Only): {v_newton_solar:.1f} km/s")
        logger.info(f"   DDMM Predicted Velocity:             {v_model_solar:.1f} km/s")
        logger.info(f"   Enhancement Factor (ξ):              {xi_solar:.3f}")
        logger.info(f"   Difference (DDMM - Newtonian):       {v_model_solar - v_newton_solar:+.1f} km/s")

    except Exception as e:
        logger.error(f"❌ Error calculating model predictions: {e}")


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
    UPDATED AND COMPLETE PRIOR TRANSFORM for the DDMM model.
    This version uses physically motivated, tighter priors to prevent overshooting
    and guide the sampler to a scientifically plausible solution.
    """
    params = np.zeros_like(u_array)
    
    # Create a dictionary to map the unit cube values to their parameter names for clarity
    u_dict = dict(zip(fitted_param_names, u_array))
    
    # Define the transformation for each parameter explicitly.
    # This provides clear, readable, and easily adjustable priors.
    
    # --- Gravity / DDMM Parameters (TIGHTLY CONSTRAINED PRIORS) ---
    if 'rho_c_solar_kpc3' in u_dict:
        # Log-uniform prior between 10^7.5 and 10^9.0 M☉/kpc³
        # This focuses on the physically interesting regime for galaxies.
        log_low, log_high = 7.5, 9.0
        params[fitted_param_names.index('rho_c_solar_kpc3')] = 10**(log_low + u_dict['rho_c_solar_kpc3'] * (log_high - log_low))

    if 'n_exp' in u_dict:
        # Uniform prior between 1.5 and 3.0.
        # Avoids extreme values that can cause instability.
        low, high = 1.5, 3.0
        params[fitted_param_names.index('n_exp')] = low + u_dict['n_exp'] * (high - low)
        
    if 'gamma_exp' in u_dict:
        # Use the same tight prior for 'gamma_exp'
        low, high = 1.5, 3.0
        params[fitted_param_names.index('gamma_exp')] = low + u_dict['gamma_exp'] * (high - low)

    if 'lambda_g' in u_dict:
        # Uniform prior between 0.3 and 1.5.
        # CRITICAL: This prevents the extreme overshooting seen in the plots.
        low, high = 0.3, 1.5
        params[fitted_param_names.index('lambda_g')] = low + u_dict['lambda_g'] * (high - low)

    # --- Baryonic Component Parameters (STANDARD PRIORS) ---
    # These can be wider as they are constrained by the baryonic mass model.
    
    # Thin Disk
    if 'M_disk_thin_solar' in u_dict:
        log_low, log_high = 10.5, 11.1 # 3e10 to 1.2e11 M☉
        params[fitted_param_names.index('M_disk_thin_solar')] = 10**(log_low + u_dict['M_disk_thin_solar'] * (log_high - log_low))
        
    if 'R_d_thin_kpc' in u_dict:
        low, high = 2.0, 4.5
        params[fitted_param_names.index('R_d_thin_kpc')] = low + u_dict['R_d_thin_kpc'] * (high - low)

    if 'h_z_thin_kpc' in u_dict:
        low, high = 0.15, 0.5
        params[fitted_param_names.index('h_z_thin_kpc')] = low + u_dict['h_z_thin_kpc'] * (high - low)

    # Thick Disk
    if 'M_disk_thick_solar' in u_dict:
        log_low, log_high = 9.5, 10.7 # 3e9 to 5e10 M☉
        params[fitted_param_names.index('M_disk_thick_solar')] = 10**(log_low + u_dict['M_disk_thick_solar'] * (log_high - log_low))

    if 'R_d_thick_kpc' in u_dict:
        low, high = 3.5, 9.5
        params[fitted_param_names.index('R_d_thick_kpc')] = low + u_dict['R_d_thick_kpc'] * (high - low)
        
    if 'h_z_thick_kpc' in u_dict:
        low, high = 0.7, 1.5
        params[fitted_param_names.index('h_z_thick_kpc')] = low + u_dict['h_z_thick_kpc'] * (high - low)

    # Bulge
    if 'M_bulge_solar' in u_dict:
        log_low, log_high = 9.7, 10.4 # 5e9 to 2.5e10 M☉
        params[fitted_param_names.index('M_bulge_solar')] = 10**(log_low + u_dict['M_bulge_solar'] * (log_high - log_low))
        
    if 'a_bulge_kpc' in u_dict:
        low, high = 0.2, 2.0
        params[fitted_param_names.index('a_bulge_kpc')] = low + u_dict['a_bulge_kpc'] * (high - low)
        
    # Gas
    if 'M_gas_solar' in u_dict:
        log_low, log_high = 9.7, 10.8 # 5e9 to 6e10 M☉
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
    xi_type: str,
    gp_surrogate=None
) -> Tuple[float, List[float]]:
    """
    Enhanced log-likelihood with correct function calls and physical plausibility checks.
    """
    # 1. Combine fitted parameters with fixed ones and add boolean flags
    params = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in all_param_info_list:
        if not p_info['is_fitted']:
            params[p_info['name']] = p_info['current_val']
    
    params['include_disk_thin'] = args_dynesty_obj.include_disk_thin
    params['include_disk_thick'] = args_dynesty_obj.include_disk_thick
    params['include_bulge'] = args_dynesty_obj.include_bulge
    params['include_gas'] = args_dynesty_obj.include_gas
    
    # 2. Check for physical plausibility (e.g., mass ratios) before expensive calculations
    is_valid, reason, *_ = check_physical_plausibility(theta_values_fitted, fitted_param_names, args_dynesty_obj)
    if not is_valid:
        return -np.inf, [np.inf]

    # 3. Compute the model velocity using the correct, robust method
    try:
        v_newton = v_baryon_total_newtonian_kms(R_data, params)
        rho = rho_baryon_total_midplane_solar_kpc3(R_data, params)
        xi_func = XI_FUNCTION_MAP.get(xi_type, XI_FUNCTION_MAP['power'])
        
        n_key = 'gamma_exp' if 'gamma_exp' in params else 'n_exp'
        A_key = 'lambda_g' if 'lambda_g' in params else 'A'
        
        xi = xi_func(rho, params['rho_c_solar_kpc3'], params[n_key], params.get(A_key, 1.0))
        xi = np.minimum(xi, 5.0)
        
        v_model = v_newton * np.sqrt(xi)

        if not np.all(np.isfinite(v_model)):
            return -np.inf, [np.inf]
            
    except Exception:
        return -np.inf, [np.inf]
    
    # --- NEW: PLAUSIBILITY PENALTY TO GUIDE THE SAMPLER ---
    # This directly addresses your second question.
    # We check the model's prediction in the solar neighborhood. If it's too high,
    # we return -inf, telling dynesty this is a "forbidden" region of parameter space.
    v_model_solar = np.median(v_model[(R_data > 7.5) & (R_data < 8.5)])
    if v_model_solar > 250:  # Set a hard ceiling for plausible velocities
        return -np.inf, [np.inf]
    # --- END OF NEW SECTION ---

    # 4. Calculate the standard chi-squared likelihood
    chi2 = np.sum(((v_data - v_model) / sigma_data)**2)
    log_L = -0.5 * chi2
    
    if not np.isfinite(log_L):
        return -np.inf, [np.inf]

    # Return the RMSE as a "blob" for monitoring
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
    """
    global debug_counter
    
    # Initialize debug counter and logger
    if 'debug_counter' not in globals():
        debug_counter = 0
    
    # Get logger safely
    logger = get_or_create_logger()
    
    # Import needed functions
    from density_metric2 import XI_FUNCTION_MAP, xi_gravitational_color

    # DEBUG: Print what parameters we have (only once)
    if not hasattr(v_model_for_dynesty, "_params_logged"):
        logger.info(f"\n[PARAMS DEBUG] xi_type: {xi_type_str}")
        logger.info(f"[PARAMS DEBUG] Available parameters: {list(p_all_params_dict.keys())}")
        logger.info(f"[PARAMS DEBUG] Parameter values:")
        for k, v in p_all_params_dict.items():
            if isinstance(v, (int, float)):
                logger.info(f"   {k}: {v:.3e}")
        v_model_for_dynesty._params_logged = True
    
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
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.error(f"Error extracting parameters: {e}")
            debug_counter += 1
        return np.zeros_like(R_kpc_array)
    
    # Validate xi parameters
    if not np.isfinite(rho_c_solar_kpc3):
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.warning(f"Non-finite rho_c: {rho_c_solar_kpc3}")
            debug_counter += 1
        return np.zeros_like(R_kpc_array)
    
    # Calculate Newtonian velocities and densities
    if ARGS_obj_dynesty.fit_target == 'milkyway':
        v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict)
        rho_midplane_for_xi = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict)
    else:
        raise NotImplementedError("Only Milky Way fitting currently supported")
    
    # Validate intermediate results
    if not np.all(np.isfinite(v_n_kms)):
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite Newtonian velocities detected")
            debug_counter += 1
        v_n_kms = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(np.isfinite(rho_midplane_for_xi)):
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite densities detected")
            debug_counter += 1
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
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.error(f"Error calculating xi with {xi_type_str}: {e}")
            debug_counter += 1
        xi_raw = np.ones_like(rho_midplane_for_xi)

    # Log xi verification (only once)
    if not hasattr(v_model_for_dynesty, "_has_logged_xi") and threading.current_thread() is threading.main_thread():
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
        if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite final velocities detected")
            debug_counter += 1
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
                'default_fixed': 5e7,  # Galaxy-appropriate
                'low': 1e6,
                'high': 1e9,
                'fit_flag_arg': 'fit_rho_c',  # New flag
                'log_prior': True,
                'physical_check': True
            }

#!/usr/bin/env python3
"""
Debug patch to add to your run_dynesty.py to diagnose the crash.
Add these modifications to your existing code.
"""

# 1. Add this debug version of log_likelihood_dynesty right after the original:

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
    """Debug version with extensive logging"""
    
    # Get logger safely for multiprocessing
    logger = get_or_create_logger()
    
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
    
    # Hard constraint: thick disk scale length must be > thin disk scale length
    if ('R_d_thick_kpc' in fitted_param_names and 'R_d_thin_kpc' in fitted_param_names):
        idx_thick = fitted_param_names.index('R_d_thick_kpc')
        idx_thin = fitted_param_names.index('R_d_thin_kpc')
        R_d_thick = theta_values_fitted[idx_thick]
        R_d_thin = theta_values_fitted[idx_thin]
        
        if R_d_thick < 1.05 * R_d_thin:
            if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
                logger.warning(f"Rejecting: R_d_thick ({R_d_thick:.2f}) < 1.05 * R_d_thin ({R_d_thin:.2f})")
                log_likelihood_dynesty_debug.debug_counter += 1
            return -np.inf, [np.inf]

    # Hard constraint: thick disk scale height must be > thin disk scale height
    if ('h_z_thick_kpc' in fitted_param_names and 'h_z_thin_kpc' in fitted_param_names):
        idx_h_thick = fitted_param_names.index('h_z_thick_kpc')
        idx_h_thin = fitted_param_names.index('h_z_thin_kpc')
        h_z_thick = theta_values_fitted[idx_h_thick]
        h_z_thin = theta_values_fitted[idx_h_thin]
        
        if h_z_thick < 2.0 * h_z_thin:
            if log_likelihood_dynesty_debug.debug_counter < DEBUG_COUNTER_MAX:
                logger.warning(f"Rejecting: h_z_thick ({h_z_thick:.3f}) < 2 * h_z_thin ({h_z_thin:.3f})")
                log_likelihood_dynesty_debug.debug_counter += 1
            return -np.inf, [np.inf]
        
    # Log first few data points
    if not hasattr(log_likelihood_dynesty_debug, '_logged_data'):
        logger.info(f"DEBUG: Data shapes - R: {R_data.shape}, v: {v_data.shape}, sigma: {sigma_data.shape}")
        logger.info(f"DEBUG: First 5 R values: {R_data[:5]}")
        logger.info(f"DEBUG: First 5 v values: {v_data[:5]}")
        logger.info(f"DEBUG: First 5 sigma values: {sigma_data[:5]}")
        logger.info(f"DEBUG: R range: [{np.min(R_data):.2f}, {np.max(R_data):.2f}]")
        logger.info(f"DEBUG: v range: [{np.min(v_data):.2f}, {np.max(v_data):.2f}]")
        log_likelihood_dynesty_debug._logged_data = True
    
    # Now call the original function
    return log_likelihood_dynesty(
        theta_values_fitted, fitted_param_names, args_dynesty_obj,
        all_param_info_list, R_data, v_data, sigma_data, xi_type, gp_surrogate
    )

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
            medians = np.average(samples, weights=weights, axis=0)
            previous_best = dict(zip(param_names, medians))
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

        # Should we fit this parameter?
        is_fitted = False
        if 'fit_flag_arg' in p_details and getattr(ARGS, p_details['fit_flag_arg'], False):
            fixed_arg_name = p_details['fixed_val_from_arg']
            if f"--{fixed_arg_name}" not in sys.argv:
                is_fitted = True
            else:
                logger.info(f"  {p_name}: Using fixed value (overrides fit flag)")

        # Get current value
        current_val = getattr(ARGS, p_details['fixed_val_from_arg'])

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
    """Test likelihood at typical parameter values"""
    logger.info("\n" + "="*60)
    logger.info("TESTING LIKELIHOOD AT TYPICAL PARAMETERS")
    logger.info("="*60)
    
    # Get parameter configuration
    fitted_p_names, fitted_p_labels, p0_guess, p_low, p_high, use_log_flags = \
        get_param_labels_and_bounds(args)
    
    # Set all parameters to typical values
    typical_params = []
    for name in fitted_p_names:
        if name in PHYSICAL_BOUNDS:
            typical_params.append(PHYSICAL_BOUNDS[name]['typical'])
        else:
            # Use middle of prior range
            idx = fitted_p_names.index(name)
            typical_params.append((p_low[idx] + p_high[idx]) / 2)
    
    typical_params = np.array(typical_params)
    
    logger.info("Typical parameters:")
    for name, val in zip(fitted_p_names, typical_params):
        logger.info(f"  {name}: {val:.3e}")
    
    # CRITICAL: Also log the FIXED parameters that aren't being fitted
    logger.info("\nFixed parameters:")
    if hasattr(args, 'all_param_info_list'):
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                logger.info(f"  {p_info['name']}: {p_info['current_val']:.3e}")
    
    # For grav_color, make sure we have the xi parameters
    if args.xi == 'grav_color':
        logger.info(f"\nXi function parameters for grav_color:")
        logger.info(f"  rho_c_fixed: {args.rho_c_fixed:.3e}")
        logger.info(f"  gamma_fixed: {args.gamma_fixed:.3f}")
        logger.info(f"  lambda_g_fixed: {args.lambda_g_fixed:.3f}")
    
    # Evaluate likelihood
    logl_args_tuple = (fitted_p_names, args, args.all_param_info_list,
                       gaia_data['R_kpc'], gaia_data['v_obs'], gaia_data['sigma_v'],
                       args.xi, None)
    
    try:
        log_L, blob = log_likelihood_dynesty(typical_params, *logl_args_tuple)
    except Exception as e:
        logger.error(f"Exception during likelihood evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info(f"\nLog-likelihood at typical params: {log_L:.1f}")
    logger.info(f"RMSE: {blob[0]:.1f} km/s")
    
    if log_L == -np.inf:
        logger.error("ERROR: Typical parameters give -inf likelihood!")
        logger.error("This suggests a fundamental problem with the model or data.")
        
        # Additional debugging
        logger.error("\nDEBUGGING INFO:")
        logger.error(f"Number of data points: {len(gaia_data['R_kpc'])}")
        logger.error(f"Xi type: {args.xi}")
        logger.error(f"Fitted parameters: {fitted_p_names}")
        
        return False
    
    if log_L < -1e6:
        logger.warning(f"WARNING: Very negative log-likelihood ({log_L:.1e})")
        logger.warning("Model may be incompatible with data.")
    
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
    Run a single dynesty sampling with enhanced monitoring, convergence diagnostics,
    and optional real-time dashboard monitoring.
    """
    import threading
    from io import StringIO
    global convergence_tracker

    # Extract data
    R_data_for_run = gaia_data_dict["R_kpc"]
    v_data_for_run = gaia_data_dict["v_obs"]
    sigma_data_for_run = gaia_data_dict["sigma_v"]
    
    logger.info(f"DEBUG: Loaded {len(R_data_for_run)} data points")
    logger.info(f"DEBUG: R range: [{np.min(R_data_for_run):.2f}, {np.max(R_data_for_run):.2f}] kpc")
    logger.info(f"DEBUG: v range: [{np.min(v_data_for_run):.2f}, {np.max(v_data_for_run):.2f}] km/s")
    logger.info(f"DEBUG: sigma range: [{np.min(sigma_data_for_run):.2f}, {np.max(sigma_data_for_run):.2f}] km/s")
    
    # Check for any invalid values
    if np.any(~np.isfinite(R_data_for_run)):
        logger.error(f"ERROR: Found {np.sum(~np.isfinite(R_data_for_run))} non-finite R values!")
    if np.any(~np.isfinite(v_data_for_run)):
        logger.error(f"ERROR: Found {np.sum(~np.isfinite(v_data_for_run))} non-finite v values!")
    if np.any(~np.isfinite(sigma_data_for_run)):
        logger.error(f"ERROR: Found {np.sum(~np.isfinite(sigma_data_for_run))} non-finite sigma values!")
    
    # Get parameter configuration FIRST (before trying to use p0_guess!)
    fitted_p_names, fitted_p_labels, p0_guess, p_low, p_high, use_log_flags = \
        get_param_labels_and_bounds(args)
    ndim_dynesty = len(fitted_p_names)
    convergence_tracker = ConvergenceTracker(fitted_p_names)

    logger.info(f"Dynesty fitting {ndim_dynesty} parameters: {fitted_p_names}")
    
    # NOW we can test the likelihood function with initial parameters
    logger.info("DEBUG: Testing likelihood function with initial parameters...")
    test_logl, test_blob = log_likelihood_dynesty_debug(
        p0_guess, fitted_p_names, args, args.all_param_info_list,
        R_data_for_run, v_data_for_run, sigma_data_for_run, args.xi, None
    )
    logger.info(f"DEBUG: Test log-likelihood = {test_logl}, RMSE = {test_blob[0]}")
    
    if test_logl == -np.inf:
        logger.error("ERROR: Initial parameters give -inf likelihood! Check your model or data.")
        # Don't continue if initial params are bad

    # Validate initial guess
    is_valid, reason, *_ = check_physical_plausibility(p0_guess, fitted_p_names, args)
    if not is_valid:
        if logger:
            logger.warning(f"Initial guess fails physical checks: {reason}")
        if logger:
            logger.warning("Adjusting to center of prior range...")
        for i in range(ndim_dynesty):
            if use_log_flags[i]:
                p0_guess[i] = np.sqrt(p_low[i] * p_high[i])
            else:
                p0_guess[i] = 0.5 * (p_low[i] + p_high[i])

    # Sampler input setup
    ptform_args_tuple = (fitted_p_names, np.array(p_low), np.array(p_high), use_log_flags)
    logl_args_tuple = (fitted_p_names, args, args.all_param_info_list,
                       R_data_for_run, v_data_for_run, sigma_data_for_run,
                       args.xi, gp_surrogate)

    # Multiprocessing setup
    pool_obj, queue_size_for_sampler = None, None
    if args.num_threads > 1:
        try:
            pool_obj = Pool(args.num_threads)
            queue_size_for_sampler = args.num_threads
            logger.info(f"Dynesty will run with {args.num_threads} threads")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create Pool: {e}. Running serially.")

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Optional dashboard
    dashboard_monitor = None
    if args.enable_dashboard:
        try:
            from monitor_dashboard import DynestyMonitor  # Ensure this is importable
            dashboard_monitor = DynestyMonitor(
                Path(args.output_dir),
                Path(args.monitor_config) if args.monitor_config else None
            )
            logger.info(f"✅ Dashboard monitoring enabled. Progress file: {args.output_dir}/progress.json")
            logger.info(f"   View with: python monitor_dashboard.py {args.output_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to initialize dashboard monitor: {e}")

    # Sampler
    logger.info(f"Sampler configuration: method='{args.sample_method}', "
                f"bound='{args.bound_method}', enlarge={args.enlarge_factor}")

    try:
        # In run_single_dynesty, wrap the sampler creation in try/except:
        logger.info("Creating sampler...")
        sampler = DynamicNestedSampler(
            log_likelihood_dynesty_debug,  # Use debug version temporarily
            prior_transform_dynesty,
            ndim_dynesty,
            pool=pool_obj,
            queue_size=queue_size_for_sampler,
            sample=args.sample_method,
            bound=args.bound_method,
            enlarge=args.enlarge_factor,
            ptform_args=ptform_args_tuple,
            logl_args=logl_args_tuple,
            blob=True,
            walks=args.walks
        )
        logger.info("Sampler created successfully")
        
        # === Restore from checkpoint if resuming ===
        if hasattr(args, "_resume_checkpoint_file"):
            sampler.restore(args._resume_checkpoint_file)
            logger.info("✅ Restored sampler from checkpoint")
        
    except Exception as e:
        logger.error(f"ERROR creating sampler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Safely restore saved_logz if res already populated (e.g. checkpoint resume)
    try:
        if hasattr(sampler, 'results') and hasattr(sampler.results, 'logz') and len(sampler.results.logz) >= 2:
            sampler.saved_logz = list(sampler.results.logz[-2:])
        else:
            sampler.saved_logz = []
    except Exception as e:
        print(f"WARNING: Could not initialize saved_logz: {e}")
        sampler.saved_logz = []

    run_start_time = time.time()
    checkpoint_file = Path(args.output_dir) / "dynesty_checkpoint.pkl"

    # Rest of the function continues as before...
    if args.use_run_nested:
        # Built-in run_nested
        logger.info(f"Using run_nested() with nlive_init={args.nlive_init}, dlogz={args.dlogz_target}")
        monitor_log_path = Path(args.output_dir) / f"monitor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        monitor_file = open(monitor_log_path, 'w', buffering=1)
        logger.info(f"📝 Monitoring output: {monitor_log_path}")

        stop_monitoring = threading.Event()

        def monitor_thread():
            csv_log_path = Path(args.output_dir) / "dynesty_live_progress.csv"
            last_check = time.time()

            try:
                while not stop_monitoring.is_set():
                    time.sleep(10)
                    if time.time() - last_check > args.monitor_interval_s:
                        last_check = time.time()

                        if hasattr(sampler, 'results') and hasattr(sampler.results, 'samples'):
                            if len(sampler.results.samples) > 50:
                                # === Update terminal + monitor_log.txt ===
                                old_stdout = sys.stdout
                                sys.stdout = StringIO()

                                enhanced_monitor_sampler_progress(
                                    sampler, fitted_p_names, fitted_p_labels,
                                    run_start_time, logger, gp_surrogate, args,
                                    dashboard_monitor
                                )

                                monitor_text = sys.stdout.getvalue()
                                sys.stdout = old_stdout

                                monitor_file.write(f"\n{monitor_text}\n")
                                monitor_file.flush()

                                print("\n" + "=" * 70)
                                print(f"🔔 MONITORING UPDATE - Details in: {monitor_log_path}")

                                if hasattr(sampler.results, 'logz') and len(sampler.results.logz) > 0:
                                    current_logz = sampler.results.logz[-1]
                                    dlogz = (
                                        sampler.results.logz[-1] - sampler.results.logz[-2]
                                        if len(sampler.results.logz) > 1 else np.inf
                                    )
                                    print(f"🔔 Log(Z): {current_logz:.3f} | dlogz: {dlogz:.4f}")
                                    if dlogz < args.dlogz_target * 2:
                                        print("🔔 ✅ Approaching convergence!")
                                print("=" * 70 + "\n")

                                # === Append to dynesty_live_progress.csv ===
                                try:
                                    logz = sampler.results.logz[-1]
                                    dlogz = sampler.results.logz[-1] - sampler.results.logz[-2] if len(sampler.results.logz) > 1 else float('nan')
                                    weights = np.exp(sampler.results.logwt - logz)
                                    ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0
                                    n_samples = len(sampler.results.samples)
                                    n_calls = np.sum(sampler.results.ncall) if hasattr(sampler.results, 'ncall') else float('nan')

                                    # Get parameter medians
                                    sample_medians = np.median(sampler.results.samples, axis=0)
                                    param_values = {f"{param}": float(val) for param, val in zip(fitted_p_names, sample_medians)}

                                    # Build row with expanded data
                                    new_row = {
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "logz": round(float(logz), 6),
                                        "dlogz": round(float(dlogz), 6),
                                        "ess": round(float(ess), 2),
                                        "n_samples": n_samples,
                                        "n_calls": n_calls,
                                        **param_values
                                    }

                                    # Check if header is missing
                                    write_header = True
                                    if csv_log_path.exists():
                                        try:
                                            with open(csv_log_path, "r") as f:
                                                first_line = f.readline().strip()
                                                write_header = not (first_line and "timestamp" in first_line)
                                        except Exception:
                                            write_header = True

                                    with open(csv_log_path, "a", newline="") as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=new_row.keys())
                                        if write_header:
                                            writer.writeheader()
                                        writer.writerow(new_row)

                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to write to dynesty_live_progress.csv: {e}")

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                monitor_file.write(f"ERROR: {e}\n")
            finally:
                monitor_file.close()

        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()
        
        try:
            sampler.run_nested(
                nlive_init=args.nlive_init,
                nlive_batch=args.nlive_batch,
                dlogz_init=args.dlogz_target,
                maxcall=args.maxcall,
                print_progress=True,
                checkpoint_file=str(checkpoint_file),
                checkpoint_every=args.checkpoint_every
            )
        finally:
            stop_monitoring.set()
            monitor.join(timeout=5)
            try:
                monitor_file.close()
            except:
                pass
            
            logger.info(f"\n📊 Final monitoring report: {monitor_log_path}")

    else:
        # Custom sampling loop with early stopping
        logger.info("Using custom sampling loop with adaptive monitoring and early stopping")
        
        # Store fitted_param_names in args for check_early_stopping
        args.fitted_param_names = fitted_p_names
        
        last_monitor_time = time.time()
        last_check_time = time.time()
        early_stop_checks = 0
        
        try:
            for it, results in enumerate(sampler.sample_initial(
                nlive=args.nlive_init,
                maxcall=args.maxcall,
                save_samples=True
            )):
                # Checkpoint every N seconds
                if not hasattr(sampler, "_last_checkpoint_time"):
                    sampler._last_checkpoint_time = time.time()

                if time.time() - sampler._last_checkpoint_time > args.checkpoint_every:
                    try:
                        sampler.save(str(checkpoint_file))
                        logger.info(f"💾 Checkpoint saved to {checkpoint_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save checkpoint: {e}")
                    sampler._last_checkpoint_time = time.time()
                # Periodic monitoring
                if time.time() - last_monitor_time > args.monitor_interval_s:
                    last_monitor_time = time.time()
                    enhanced_monitor_sampler_progress(
                        sampler, fitted_p_names, fitted_p_labels,
                        run_start_time, logger, gp_surrogate, args,
                        dashboard_monitor
                    )
                
                # Early stopping check every 5 minutes
                if time.time() - last_check_time > 300:  # 5 minutes
                    last_check_time = time.time()
                    should_stop, reason = check_early_stopping(sampler, convergence_tracker, args)
                    
                    if should_stop:
                        early_stop_checks += 1
                        logger.warning(f"⚠️ Early stopping check {early_stop_checks}/3: {reason}")
                        
                        if early_stop_checks >= 3:  # Consistent failures
                            logger.error("❌ STOPPING: Model persistently finding unphysical solutions")
                            logger.error(f"   Reason: {reason}")
                            logger.error("   Suggestions:")
                            logger.error("   1. Check prior bounds are realistic")
                            logger.error("   2. Verify data quality")
                            logger.error("   3. Consider different xi function")
                            logger.error("   4. Try curriculum learning approach")
                            raise RuntimeError("Early stopping due to unphysical solutions")
                    else:
                        early_stop_checks = 0  # Reset counter if things improve
                        
        except RuntimeError as e:
            logger.error(f"Sampling terminated: {e}")
            
            # Save partial results
            if hasattr(sampler, 'results'):
                output_file = Path(args.output_dir) / "partial_results_unphysical.npz"
                np.savez(output_file,
                        samples=sampler.results.samples,
                        logz=sampler.results.logz,
                        error="Early stopped due to unphysical solutions")
                logger.info(f"Partial results saved to {output_file}")
            
            # Clean up and return None to indicate failure
            if pool_obj:
                pool_obj.close()
                pool_obj.join()
            
            return None
    
    # Return the results if we get here
    return sampler.results

# ============================================================================
# Main Entry Point
# ============================================================================

def main_dynesty():
    """Main entry point with enhanced configuration and validation."""
    global logger, debug_counter
    logger = get_or_create_logger()  # ✅ Initialize logger properly here
    
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Enhanced Dynesty sampler for Density-Metric model with physical constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    debug_counter = 0  # Reset debug counter
    from data_io import load_all_sky_gaia_slices



    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("run_dynesty") if logger is None else logger
    logger.info("Starting Enhanced Dynesty Sampler v2.0")

    if not DYNESTY_AVAILABLE:
        logger.error("Dynesty library not found")
        sys.exit(1)

    # Core run options
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Resume from checkpoint in output_dir/dynesty_checkpoint.pkl")

    parser.add_argument('--xi', type=str, default='power',
                        choices=['power', 'logistic', 'enhanced', 'grav_color'],
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

    # Parse arguments
    args = parser.parse_args()
    checkpoint_file = Path(args.output_dir) / "dynesty_checkpoint.pkl"
    RUN_ID = run_history.start_record(args)   # ✅ Now it's safe to call, because args is defined
    args.fit_target = 'milkyway'  # Currently only MW supported
    from data_io import load_all_sky_gaia_slices
    
    logger.info(f"\n📡 Loading full Gaia dataset from longitudinal slices...")

    df_all_sky = load_all_sky_gaia_slices(
        lon_bin_width=30,           # 12 longitude bins
        stars_per_bin=12000,         # per bin
        output_dir="gaia_sky_slices",
        force_query=args.force_new_query_gaia,      # honors CLI flag
        max_distance_kpc=30.0
    )

    if df_all_sky.empty:
        logger.error("❌ Full-sky Gaia loading failed")
        sys.exit(1)

    # Convert to the dictionary format expected by run_dynesty
    gaia_data_dict = {col: df_all_sky[col].values for col in df_all_sky.columns}
    print(f"Loaded {len(gaia_data_dict['R_kpc'])} stars for the fit.")
    print(f"Median v_obs at R~8kpc: {np.median(gaia_data_dict['v_obs'][(gaia_data_dict['R_kpc']>7) & (gaia_data_dict['R_kpc']<9)]):.1f} km/s")

    if "source_id" in df_all_sky.columns:
        gaia_data_dict["source_id"] = df_all_sky["source_id"].values
    if "quality_flag" in df_all_sky.columns:
        gaia_data_dict["quality_flag"] = df_all_sky["quality_flag"].values


    # Inject dynamic xi setup
    setup_xi_parameters_for_mode(args)

    # Start run logging
    RUN_ID = run_history.start_record(args)

    if args.theory_mode:
        logger.info("🧪 THEORY MODE: Using gravitational color confinement values")
        args.fix_gamma = 2.7
        args.fix_lambda_g = 8.0
        args.gamma_fixed    = 2.7
        args.lambda_g_fixed = 8.0
        # Only ρ_c is left free
        args.fit_xi_params  = False
        args.fit_gamma      = False
        args.fit_lambda_g   = False
        
        # Only fit rho_c
        logger.info("   γ (gamma) = 2.7 (fixed from β_g = -11/3)")
        logger.info("   λ (lambda_g) = 8.0 (fixed for 9x enhancement)")
        logger.info("   Only fitting ρ_c (critical density)")

    check_prior_bounds_compatibility(args)
    
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
    
        
    
    # === Resume checkpoint logic - DEFER actual restore to sampler setup ===
    if args.resume:
        checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else Path(args.output_dir) / "dynesty_checkpoint.pkl"
        
        if not checkpoint_file.exists():
            logger.error(f"No checkpoint found at {checkpoint_file}")
            sys.exit(1)

        # Flag this so we restore sampler after it's built
        args._resume_checkpoint_file = str(checkpoint_file)

        
        # Safety check: previous results file must exist if using previous best
        if args.use_previous_best and not os.path.exists(args.previous_results_file):
            logger.warning(f"⚠️ Previous results file not found: {args.previous_results_file}")
            args.use_previous_best = False

        # Safety check: disable dashboard if explicitly turned off or import fails
        if args.enable_dashboard and not args.disable_dashboard:
            try:
                from monitor_dashboard import DynestyMonitor
            except ImportError:
                logger.warning("⚠️ Dashboard module not available, disabling dashboard")
                args.enable_dashboard = False

        
    
    # Run physics self-tests
    logger.info("Running physics module self-tests...")
    run_physics_self_tests()
    logger.info("✅ Physics tests passed")
    
    # Check configuration
    temp_fitted_names, _, _, _, _, _ = get_param_labels_and_bounds(args)
    n_params = len(temp_fitted_names)
    
    logger.info(f"\nModel complexity: {n_params} free parameters")
    
    if n_params > 10:
        if logger:
            logger.warning("⚠️  Many parameters to fit!")
        if not args.use_curriculum_learning:
            if logger:
                logger.warning("   Consider using --use_curriculum_learning")
        if args.nlive_init < 50 * n_params:
            if logger:
                logger.warning(f"   Consider increasing --nlive_init (currently {args.nlive_init})")
    
    # Initialize GP surrogate if requested
    gp_surrogate = None
    if args.use_gp_surrogate:
        if not GP_AVAILABLE:
            logger.error("GP surrogate requested but scikit-learn not available")
            sys.exit(1)
        
        logger.info("\n🤖 Initializing GP surrogate...")
        _, _, _, p_low, p_high, _ = get_param_labels_and_bounds(args)
        param_bounds = np.column_stack([p_low, p_high])
        
        gp_surrogate = GPSurrogateModel(
            param_names=temp_fitted_names,
            param_bounds=param_bounds,
            uncertainty_threshold=args.gp_uncertainty_threshold,
            n_initial=args.gp_n_initial
        )
        
        # Generate training data
        def physics_wrapper(param_dict, args_obj):
            return v_model_for_dynesty(
                gaia_data_dict['R_kpc'], param_dict, args.xi, args_obj
            )
        
        gp_surrogate.generate_initial_training_data(physics_wrapper, args)
        
    # Validate data
    if not validate_gaia_data_for_fitting(gaia_data_dict):
        logger.error("Data validation failed! Check your Gaia data.")
        sys.exit(1)
    
    # Test likelihood function
    if not test_likelihood_at_typical_params(args, gaia_data_dict):
        logger.error("Likelihood test failed! Model may be incompatible with data.")
        logger.error("Suggestions:")
        logger.error("1. Check that data units match model expectations")
        logger.error("2. Try simpler model (fewer components)")
        logger.error("3. Check xi function is behaving correctly")
        sys.exit(1)
    
    # Run sampling
    if args.use_curriculum_learning:
        logger.info("\n🎓 Using curriculum learning approach")
        results = run_curriculum_learning(args, gaia_data_dict, logger)
    else:
        logger.info("\n🎯 Using standard sampling")
        results = run_single_dynesty(args, gaia_data_dict, gp_surrogate)
    
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
        
        for stage_name in completed_stages:
            stage_results = results[stage_name]
            output_prefix = f"dynesty_curriculum_{stage_name}_{args.xi}"
            output_npz = Path(args.output_dir) / f"{output_prefix}_samples.npz"

            try:
                weights = np.exp(stage_results.logwt - stage_results.logz[-1])
                np.savez(output_npz,
                        samples=stage_results.samples,
                        weights=weights,
                        param_names=np.array(fitted_p_names), 
                        logl=stage_results.logl,
                        logz=stage_results.logz,
                        logzerr=stage_results.logzerr)
                logger.info(f"Saved {stage_name} to {output_npz}")
            except Exception as e:
                logger.error(f"Failed to save {stage_name}: {e}")

        final_stage = max(completed_stages)
        res = results[final_stage]

    else:
        # Single run results
        res = results# Curriculum learning results
        
    if isinstance(results, dict) and 'stage_1' in results:
        completed_stages = [k for k, v in results.items() if v is not None]
        if not completed_stages:
            logger.error("❌ All curriculum stages failed. No results to save.")
            return

        # Save each completed stage
        for stage_name in completed_stages:
            stage_results = results[stage_name]
            output_prefix = f"dynesty_curriculum_{stage_name}_{args.xi}"
            output_npz = Path(args.output_dir) / f"{output_prefix}_samples.npz"

            try:
                weights = np.exp(stage_results.logwt - stage_results.logz[-1])
                np.savez(output_npz,
                    samples=stage_results.samples,
                    weights=weights,
                    param_names=np.array(fitted_p_names), 
                    logl=stage_results.logl,
                    logz=stage_results.logz,
                    logzerr=stage_results.logzerr)
                logger.info(f"✅ Saved {stage_name} results to {output_npz}")
            except Exception as e:
                logger.error(f"❌ Failed to save {stage_name} results: {e}")

        # Use final stage for downstream processing
        final_stage = max(completed_stages)
        res = results[final_stage]

    else:
        # Single-run results
        res = results

    # ---- Save final stage or single-run result ----
    output_parts = ["dynesty_mw", args.xi]
    if args.include_bulge:
        output_parts.append("B" + ("f" if args.fit_bulge else "x"))
    if args.include_disk_thin:
        output_parts.append("DT" + ("f" if args.fit_disk_thin else "x"))
    if args.include_disk_thick:
        output_parts.append("DK" + ("f" if args.fit_disk_thick else "x"))
    if args.include_gas:
        output_parts.append("G" + ("f" if args.fit_gas else "x"))

    output_basename = "_".join(output_parts)
    output_npz = Path(args.output_dir) / f"{output_basename}_samples.npz"

    # Compute effective sample size
    try:
        ess = res.effective_sample_size if hasattr(res, 'effective_sample_size') else 0
    except:
        weights = np.exp(res.logwt - res.logz[-1])
        ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0

    # Save final .npz
    try:
        np.savez(output_npz,
         samples=res.samples,
         weights=np.exp(res.logwt - res.logz[-1]),
         param_names=np.array(fitted_p_names),
         logl=res.logl,
         logz=res.logz,
         logzerr=res.logzerr,
         ess=ess,
         blob=res.blob if hasattr(res, 'blob') else None)
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

    # === FINALIZE RUN AND SNAPSHOT ===
    try:
        fitted_p_names = args.fitted_param_names
        param_stats = {name: {"median": float(m), "sigma": float(s)}
                    for name, m, s in zip(fitted_p_names, np.median(res.samples, axis=0), np.std(res.samples, axis=0))}

        is_valid, reason, *_ = check_physical_plausibility(np.median(res.samples, axis=0), fitted_p_names, args)
        logz = res.logz[-1] if hasattr(res, 'logz') else np.nan
        logz_err = res.logzerr[-1] if hasattr(res, 'logzerr') else np.nan
        eff = getattr(res, 'eff', 0.0)
        rmse = float(np.sqrt(np.mean(res.blob**2))) if hasattr(res, 'blob') else np.nan
        n_samples = len(res.samples)
        n_calls = int(np.sum(res.ncall)) if hasattr(res, 'ncall') else 0

        finalize_record(RUN_ID, success=True,
                        logz=logz, logz_err=logz_err,
                        eff=eff, rmse=rmse,
                        n_samples=n_samples, n_calls=n_calls,
                        param_stats=param_stats,
                        phys_ok=is_valid, phys_reason=reason if not is_valid else "")

        snapshot_path = Path(args.output_dir) / f"run_{RUN_ID}_summary.json"
        with open(snapshot_path, "w") as fh:
            json.dump({
                "run_id": RUN_ID,
                "success": True,
                "logZ": logz,
                "params": param_stats,
                "phys_ok": is_valid,
                "phys_reason": reason if not is_valid else "",
                "cmd": " ".join(sys.argv)
            }, fh, indent=2)
        logger.info(f"📄 Snapshot saved to {snapshot_path}")
    except Exception as e:
        logger.error(f"❌ Failed to finalize or snapshot run: {e}")

    logger.info("\n✨ Enhanced dynesty run complete!")



if __name__ == "__main__":
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL: Dynesty library not found")
        sys.exit(1)
    
    main_dynesty()