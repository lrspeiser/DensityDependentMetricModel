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

# Set up module logger
logger = None

# ============================================================================
# Physical Constraints and Validation
# ============================================================================

# Physical bounds for parameters (based on MW observations and theory)
PHYSICAL_BOUNDS = {
    # Mass parameters (M_sun)
    'M_disk_thin_solar':   {'min': 3e10, 'max': 8e10, 'typical': 5e10},
    'M_disk_thick_solar':  {'min': 5e9,  'max': 3e10, 'typical': 1.5e10},  # Max 50% of thin
    'M_bulge_solar':       {'min': 0.5e10, 'max': 3e10, 'typical': 1.5e10},
    'M_gas_solar':         {'min': 5e9,  'max': 5e10,   'typical': 3e10},    # Was 5e9-3e10

     # Scale lengths (kpc) - UPDATED based on what sampler wanted
    'R_d_thin_kpc':        {'min': 2.0,  'max': 4.0,    'typical': 2.6},
    'R_d_thick_kpc':       {'min': 3.0,  'max': 6.0,    'typical': 4.0},
    'R_d_gas_kpc':         {'min': 4.0,  'max': 12.0,   'typical': 7.0},
    'a_bulge_kpc':         {'min': 0.2,  'max': 1.5,    'typical': 0.7},

    # Scale heights (kpc)
    'h_z_thin_kpc':        {'min': 0.2,  'max': 0.6,    'typical': 0.3},
    'h_z_thick_kpc':       {'min': 0.6,  'max': 1.3,    'typical': 0.9},
    'h_z_gas_kpc':         {'min': 0.05, 'max': 0.3,    'typical': 0.15},

    # Total mass constraint
    'M_total':             {'min': 5e10, 'max': 2e11,   'typical': 1e11},

    # Density parameters
    'rho_c_solar_kpc3':    {'min': 1e8,  'max': 2e9,    'typical': 1.66e9},
    'n_exp':               {'min': 0.5,  'max': 2.5,    'typical': 1.5},
}

# Expected ranges for validation
EXPECTED_XI_AT_SOLAR = (0.7, 1.0)  # Xi should not suppress gravity too much at R_sun
EXPECTED_V_AT_SOLAR = (100, 250)   # TEMPORARILY RELAXED for initial exploration

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
        'default_fixed': PHYSICAL_BOUNDS['rho_c_solar_kpc3']['typical'],
        'low': PHYSICAL_BOUNDS['rho_c_solar_kpc3']['min'], 
        'high': PHYSICAL_BOUNDS['rho_c_solar_kpc3']['max'], 
        'fit_flag_arg': 'fit_xi_params',
        'log_prior': True,
        'physical_check': True
    },
    'n_exp': {
        'label': "n", 
        'fixed_val_from_arg': 'n_exp_fixed', 
        'default_fixed': PHYSICAL_BOUNDS['n_exp']['typical'],
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
        'low': PHYSICAL_BOUNDS['R_d_thin_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['R_d_thin_kpc']['max'], 
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
        'low': PHYSICAL_BOUNDS['R_d_thick_kpc']['min'], 
        'high': PHYSICAL_BOUNDS['R_d_thick_kpc']['max'], 
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
            if thick_thin_ratio > 0.7:  # Thick disk typically 10-50% of thin disk
                return False, f"Thick/thin disk ratio {thick_thin_ratio:.2f} > 0.7"
    
    # 4. Check scale length ordering (thick disk more extended)
    if 'R_d_thick_kpc' in params and 'R_d_thin_kpc' in params:
        if params['R_d_thick_kpc'] < params['R_d_thin_kpc']:
            return False, f"Thick disk scale length < thin disk ({params['R_d_thick_kpc']:.2f} < {params['R_d_thin_kpc']:.2f} kpc)"
    
    # 5. Check scale height ordering (thick disk thicker)
    if 'h_z_thick_kpc' in params and 'h_z_thin_kpc' in params:
        if params['h_z_thick_kpc'] < params['h_z_thin_kpc'] * 2:
            return False, f"Thick disk not thick enough ({params['h_z_thick_kpc']:.2f} < 2×{params['h_z_thin_kpc']:.2f} kpc)"
    
    # 6. Check xi at solar radius (shouldn't suppress gravity too much)
    if 'rho_c_solar_kpc3' in params and 'n_exp' in params:
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
        
        # Calculate xi
        xi_solar = 1.0 / (1.0 + (rho_solar_typical / params['rho_c_solar_kpc3'])**params['n_exp'])
        
        if xi_solar < EXPECTED_XI_AT_SOLAR[0]:
            return False, f"xi at R_sun = {xi_solar:.3f} < {EXPECTED_XI_AT_SOLAR[0]} (too much suppression)"
        if xi_solar > EXPECTED_XI_AT_SOLAR[1]:
            # Not necessarily bad, but log it
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
        is_valid, reason = check_physical_plausibility(current_params, fitted_param_names, args_obj)
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
        logger.info(f"\n🌟 MODEL PREDICTIONS AT SOLAR RADIUS:")
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
            'include_bulge_density': args_obj.include_bulge
        })

        try:
            v_newton_solar = v_baryon_total_newtonian_kms(np.array([R_SUN_KPC]), full_params)[0]
            rho_solar = rho_baryon_total_midplane_solar_kpc3(np.array([R_SUN_KPC]), full_params)[0]
            xi_func = XI_FUNCTION_MAP.get(args_obj.xi, xi_power_law)
            xi_result = xi_func(rho_solar, full_params['rho_c_solar_kpc3'], full_params['n_exp'])
            xi_solar = xi_result[0] if hasattr(xi_result, '__getitem__') else float(xi_result)
            v_model_solar = v_newton_solar * np.sqrt(xi_solar)

            logger.info(f"   v_Newton(R☉) = {v_newton_solar:.1f} km/s")
            logger.info(f"   ρ(R☉,z=0) = {rho_solar:.2e} M☉/kpc³")
            logger.info(f"   ξ(R☉) = {xi_solar:.3f}")
            logger.info(f"   v_model(R☉) = {v_model_solar:.1f} km/s")

            if v_model_solar < EXPECTED_V_AT_SOLAR[0] or v_model_solar > EXPECTED_V_AT_SOLAR[1]:
                logger.warning(f"⚠️  v(R☉) = {v_model_solar:.0f} km/s outside expected range")
                param_issues.append(f"v(R☉) = {v_model_solar:.0f} km/s outside expected range")

        except Exception as e:
            logger.error(f"❌ Error calculating model predictions: {e}")

        # Recommendations
        if param_issues or convergence_tracker.stuck_counter > 3:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            logger.info("─" * 60)
            if convergence_tracker.stuck_counter > 3:
                logger.info("• Sampling appears stuck. Consider curriculum learning or adjusting sampler.")
            if any("bound" in issue for issue in param_issues):
                logger.info("• Parameters hitting bounds. Review prior ranges.")
            if any("physical" in issue for issue in param_issues):
                logger.info("• Physical plausibility issues. Consider checking model and data consistency.")

        logger.info("=" * 80)

        # 🟢 DASHBOARD INTEGRATION
        if dashboard_monitor is not None:
            try:
                dashboard_state = {
                    "elapsed_time": float(elapsed_time / 3600),  # Convert to float
                    "n_samples": int(n_samples),  # Convert to int
                    "n_calls": int(ncall_total),
                    "efficiency": float(eff),
                    "logz": float(current_logz),
                    "logz_err": float(res.logzerr[-1]) if hasattr(res, 'logzerr') and len(res.logzerr) > 0 else 0,
                    "dlogz": float(dlogz),
                    "current_nlive": int(len(res.live_points)) if hasattr(res, 'live_points') else 0,
                    "parameter_estimates": {},
                    "parameter_uncertainties": {},
                    "health_warnings": convergence_tracker.health_warnings if convergence_tracker else []
                }
                
                # Ensure all parameter values are JSON serializable
                for i, name in enumerate(fitted_param_names):
                    dashboard_state["parameter_estimates"][name] = float(current_params[i])
                    dashboard_state["parameter_uncertainties"][name] = float(np.std(recent_samples[:, i]))

                dashboard_state = make_json_serializable(dashboard_state)
                dashboard_monitor.update_progress(dashboard_state)


            except Exception as e:
                logger.error(f"Failed to update dashboard: {e}")

    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
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
    Transform unit cube samples to physical prior.
    
    Implements both uniform and log-uniform priors as specified
    in parameter configuration.
    
    Parameters
    ----------
    u_array : np.ndarray
        Unit cube samples [0,1]
    fitted_param_names : list
        Parameter names
    prior_bounds_low : np.ndarray
        Lower bounds
    prior_bounds_high : np.ndarray
        Upper bounds
    use_log_prior_flags : list
        Whether to use log-uniform prior for each parameter
        
    Returns
    -------
    np.ndarray
        Transformed parameters in physical space
    """
    if any(arg is None for arg in [fitted_param_names, prior_bounds_low, prior_bounds_high]):
        raise ValueError("prior_transform_dynesty received None for essential arguments")
    
    params = np.empty_like(u_array)
    
    for i in range(len(fitted_param_names)):
        low, high = prior_bounds_low[i], prior_bounds_high[i]
        
        if use_log_prior_flags[i]:
            # Log-uniform transform: p(x) ∝ 1/x
            # Better for scale-variant parameters like masses
            log_low, log_high = np.log10(low), np.log10(high)
            params[i] = 10**(log_low + u_array[i] * (log_high - log_low))
        else:
            # Standard uniform transform
            params[i] = low + u_array[i] * (high - low)
    
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
    Enhanced log-likelihood with physical plausibility checks.
    
    This version includes:
    - Physical parameter validation
    - Robust error handling
    - Debugging output for problematic cases
    - Optional GP surrogate evaluation
    
    Parameters
    ----------
    theta_values_fitted : np.ndarray
        Fitted parameter values
    fitted_param_names : list
        Names of fitted parameters
    args_dynesty_obj : argparse.Namespace
        Configuration object
    all_param_info_list : list
        Full parameter information
    R_data, v_data, sigma_data : np.ndarray
        Observational data
    xi_type : str
        Type of xi function ('power' or 'logistic')
    gp_surrogate : GPSurrogateModel, optional
        Gaussian process surrogate
        
    Returns
    -------
    log_likelihood : float
        Log likelihood value
    blob : list
        Additional quantities (e.g., RMS)
    """
    global debug_counter
    global logger
    if 'debug_counter' not in globals():
        debug_counter = 0
    if 'logger' not in globals() or logger is None:
        import logging
        logger = logging.getLogger("run_dynesty")
        logger.setLevel(logging.INFO)

    use_logging = logger is not None

    
    # Input validation
    if any(arg is None for arg in [theta_values_fitted, fitted_param_names, args_dynesty_obj, 
                                   all_param_info_list, R_data, v_data, sigma_data, xi_type]): 
        return -np.inf, [np.inf]
    
    # Check for non-finite inputs
    if any(not np.isfinite(val) for val in theta_values_fitted):
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.warning(f"Non-finite parameter values detected:")
            for name, val in zip(fitted_param_names, theta_values_fitted):
                if not np.isfinite(val):
                    logger.warning(f"  {name}: {val}")
            debug_counter += 1
        return -np.inf, [np.inf]
    
    # Basic sanity checks only
    for i, (name, value) in enumerate(zip(fitted_param_names, theta_values_fitted)):
        if 'M_' in name and value <= 0:
            return -np.inf, [np.inf]  # No negative masses
        if 'R_d' in name and value <= 0:
            return -np.inf, [np.inf]  # No negative scale lengths
        if 'h_z' in name and value <= 0:
            return -np.inf, [np.inf]  # No negative scale heights
        if name == 'rho_c_solar_kpc3' and value <= 0:
            return -np.inf, [np.inf]  # No negative density
        if name == 'n_exp' and (value <= 0 or value > 10):
            return -np.inf, [np.inf]  # Reasonable exponent range
    
    # Reconstruct full parameter dictionary
    current_params_full_dict = dict(zip(fitted_param_names, theta_values_fitted))
    
    # Add fixed parameters
    if all_param_info_list:
        for p_info in all_param_info_list:
            if not p_info['is_fitted']:
                current_params_full_dict[p_info['name']] = p_info['current_val']
    
    # Add component flags for MW
    if args_dynesty_obj.fit_target == 'milkyway':
        for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
            if 'include_flag_arg' in p_details_cfg:
                current_params_full_dict[p_details_cfg['include_flag_arg']] = \
                    getattr(args_dynesty_obj, p_details_cfg['include_flag_arg'])
        current_params_full_dict['include_bulge_density'] = args_dynesty_obj.include_bulge
    
    # Hard constraint: R_d_thick must be > R_d_thin
    if 'R_d_thick_kpc' in current_params_full_dict and 'R_d_thin_kpc' in current_params_full_dict:
        if current_params_full_dict['R_d_thick_kpc'] <= current_params_full_dict['R_d_thin_kpc'] * 1.1:
            return -np.inf, [np.inf]  # Hard reject: violates expected scale hierarchy

        
    # Calculate model prediction
    try:
        # Use GP surrogate if available and requested
        if gp_surrogate is not None and args_dynesty_obj.use_gp_surrogate:
            def physics_func(params, args_obj):
                return v_model_for_dynesty(R_data, params, xi_type, args_obj)
            
            v_predicted, v_uncertainty = gp_surrogate.predict(
                theta_values_fitted, 
                physics_function=physics_func,
                args_obj=args_dynesty_obj
            )
        else:
            # Standard physics model evaluation
            v_predicted = v_model_for_dynesty(
                R_data, current_params_full_dict, xi_type, args_dynesty_obj
            )
        
        # Validate predictions
        if not np.all(np.isfinite(v_predicted)):
            if debug_counter < DEBUG_COUNTER_MAX:
                logger.warning(f"Non-finite v_predicted values!")
                logger.warning(f"  First few: {v_predicted[:5]}")
                logger.warning(f"  Parameters causing issue:")
                for name, val in current_params_full_dict.items():
                    if isinstance(val, (int, float, np.number)):
                        logger.warning(f"    {name}: {val:.3e}")
                debug_counter += 1
            return -np.inf, [np.inf]
            
    except Exception as e:
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.error(f"Exception in v_model_for_dynesty: {e}")
            debug_counter += 1
        return -np.inf, [np.inf]
    
    # Calculate chi-squared and likelihood
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals = v_data - v_predicted
    
    # Calculate RMS for blob
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Check RMSE reasonableness
    if rmse > 200:  # km/s - indicates very poor fit
        if debug_counter < DEBUG_COUNTER_MAX:
            if use_logging:
                logger.warning(f"Very high RMSE: {rmse:.1f} km/s")
            debug_counter += 1
    
    # Standard Gaussian likelihood
    chi_squared_terms = (residuals / sigma_data_safe)**2
    log_L_terms = chi_squared_terms + np.log(2 * np.pi * sigma_data_safe**2)
    
    if not np.all(np.isfinite(log_L_terms)): 
        return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    
    log_L = -0.5 * np.sum(log_L_terms)
    
    # Soft prior: prefer reasonable thick/thin disk mass ratio
    if 'M_disk_thick_solar' in current_params_full_dict and 'M_disk_thin_solar' in current_params_full_dict:
        ratio = current_params_full_dict['M_disk_thick_solar'] / current_params_full_dict['M_disk_thin_solar']
        if ratio > 0.5:
            penalty = -50 * (ratio - 0.5)**2  # Apply a soft Gaussian penalty
            log_L += penalty
    
    if not np.isfinite(log_L): 
        return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    
    # Reset debug counter periodically if things are working
    if debug_counter > 0 and np.isfinite(log_L):
        debug_counter = max(0, debug_counter - 1)
    
    return log_L, [rmse]


def v_model_for_dynesty(
    R_kpc_array: np.ndarray,
    p_all_params_dict: Dict[str, float],
    xi_type_str: str,
    ARGS_obj_dynesty: argparse.Namespace
) -> np.ndarray:
    """
    Calculate model velocities with density-dependent modification.
    
    Enhanced with better error handling and validation.
    
    Parameters
    ----------
    R_kpc_array : np.ndarray
        Galactocentric radii
    p_all_params_dict : dict
        All model parameters
    xi_type_str : str
        Type of xi function
    ARGS_obj_dynesty : argparse.Namespace
        Configuration object
        
    Returns
    -------
    np.ndarray
        Model circular velocities in km/s
    """
    global debug_counter
    global logger
    if 'debug_counter' not in globals():
        debug_counter = 0
    if 'logger' not in globals() or logger is None:
        import logging
        logger = logging.getLogger("run_dynesty")
        logger.setLevel(logging.INFO)

    
    # Extract xi parameters
    rho_c_solar_kpc3 = p_all_params_dict['rho_c_solar_kpc3']
    n_exp = p_all_params_dict['n_exp']
    
    # Validate xi parameters
    if not np.isfinite(rho_c_solar_kpc3) or not np.isfinite(n_exp):
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.warning(f"Non-finite xi parameters: rho_c={rho_c_solar_kpc3}, n={n_exp}")
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
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite Newtonian velocities detected — replacing with zeros")
            debug_counter += 1
        v_n_kms = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(np.isfinite(rho_midplane_for_xi)):
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite densities detected — replacing with fallback values")
            debug_counter += 1
        rho_midplane_for_xi = np.nan_to_num(rho_midplane_for_xi, nan=0.0, posinf=1e10, neginf=0.0)

    # Select ξ(ρ) function
    xi_func = XI_FUNCTION_MAP.get(xi_type_str, XI_FUNCTION_MAP['power'])

    # Add XI verification logging (only once per process)
    if not hasattr(v_model_for_dynesty, "_has_logged_xi"):
        logger.info(f"[XI VERIFICATION] Using xi_type: '{xi_type_str}'")
        logger.info(f"[XI VERIFICATION] Xi function: {xi_func.__name__}")
        
        test_densities = [1e6, 1e8, 1e10]
        summary = []
        for test_rho in test_densities:
            try:
                test_xi_raw = xi_func(test_rho, rho_c_solar_kpc3, n_exp)
                test_xi = test_xi_raw[0] if hasattr(test_xi_raw, '__getitem__') else float(test_xi_raw)
                summary.append(f"ρ={test_rho:.0e} → ξ={test_xi:.3f}")
            except Exception as e:
                summary.append(f"ρ={test_rho:.0e} → error: {e}")
        logger.info("[XI VERIFICATION SUMMARY] " + "; ".join(summary))
        
        v_model_for_dynesty._has_logged_xi = True  # Suppress future logs


    # Calculate xi safely, supporting scalar or vectorized return
    try:
        xi_raw = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp)
        if not hasattr(xi_raw, "__getitem__"):
            xi_values = np.full_like(v_n_kms, float(xi_raw))  # scalar expanded
        else:
            xi_values = np.asarray(xi_raw, dtype=np.float64)
    except Exception as e:
        logger.error(f"❌ Error in xi function '{xi_type_str}': {e}")
        xi_values = np.ones_like(v_n_kms)

    # Sanitize xi values
    xi_values = np.nan_to_num(xi_values, nan=1.0, posinf=1.0, neginf=0.0)
    xi_values_safe = np.maximum(xi_values, 0.0)

    # Apply modified gravity
    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)

    # Final velocity validation
    if not np.all(np.isfinite(v_mod_kms)):
        if debug_counter < DEBUG_COUNTER_MAX:
            logger.warning("⚠️ Non-finite final velocities detected — zeroing invalid entries")
            debug_counter += 1
        v_mod_kms = np.nan_to_num(v_mod_kms, nan=0.0, posinf=0.0, neginf=0.0)

    return v_mod_kms


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
        # Check if component is included
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
        
        is_valid, reason = check_physical_plausibility(final_params, param_names, args)
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
        is_valid, _ = check_physical_plausibility(sample, fitted_p_names, args)
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

    # Get parameter configuration
    fitted_p_names, fitted_p_labels, p0_guess, p_low, p_high, use_log_flags = \
        get_param_labels_and_bounds(args)
    ndim_dynesty = len(fitted_p_names)
    convergence_tracker = ConvergenceTracker(fitted_p_names)

    logger.info(f"Dynesty fitting {ndim_dynesty} parameters: {fitted_p_names}")

    # Validate initial guess
    is_valid, reason = check_physical_plausibility(p0_guess, fitted_p_names, args)
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
    sampler = DynamicNestedSampler(
        log_likelihood_dynesty,
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

    convergence_tracker = ConvergenceTracker(fitted_p_names)

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


# ============================================================================
# Main Entry Point
# ============================================================================

def main_dynesty():
    """Main entry point with enhanced configuration and validation."""
    global logger, debug_counter
    debug_counter = 0  # Reset debug counter

    
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
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Enhanced Dynesty sampler for Density-Metric model with physical constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic settings
    parser.add_argument('--xi', type=str, default='power',
                    choices=['power', 'logistic', 'enhanced'],
                    help="Choice of xi(ρ) function")
    parser.add_argument('--max_sample_gaia', type=int, default=10000,
                       help="Maximum number of Gaia stars to use")
    parser.add_argument('--output_dir', type=str, default="chains_dynesty",
                       help="Output directory for results")
    
    # Sampler settings
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
    
    # Dynesty sampler settings
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
    
    # Model configuration
    mw_model_g = parser.add_argument_group('Model Components')
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True)
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)
    
    # Fit flags
    fit_g = parser.add_argument_group('Parameters to Fit')
    fit_g.add_argument('--fit_xi_params', action='store_true',
                      help="Fit xi function parameters")
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
    args.fit_target = 'milkyway'  # Currently only MW supported

    
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
    
    # Load data with enhanced validation
    logger.info(f"\nLoading Gaia data from pre-processed file...")
    gaia_data_dict = load_gaia(
        sample_max=args.max_sample_gaia,
        force_new_query_gaia=False,
        force_reprocess_raw=False,
        processed_cache_filename="gaia_cache/gaia_query_cache_DR3_processed_for_fit.parquet",
        use_enhanced_query=True,
        validate_data=args.validate_data
    )
    
    if gaia_data_dict is None:
        logger.error("Failed to load Gaia data")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(gaia_data_dict['R_kpc'])} stars")
    
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
        return

    
    # Save results
    if results is None:
        logger.error("No results to save")
        return
    
    # Process results based on type
    if isinstance(results, dict) and 'stage_1' in results:
        # Curriculum learning results
        for stage_name, stage_results in results.items():
            if stage_results is None:
                continue
            
            output_prefix = f"dynesty_curriculum_{stage_name}_{args.xi}"
            output_npz = Path(args.output_dir) / f"{output_prefix}_samples.npz"
            
            try:
                weights = np.exp(stage_results.logwt - stage_results.logz[-1])
                np.savez(output_npz,
                        samples=stage_results.samples,
                        weights=weights,
                        logl=stage_results.logl,
                        logz=stage_results.logz,
                        logzerr=stage_results.logzerr)
                logger.info(f"Saved {stage_name} to {output_npz}")
            except Exception as e:
                logger.error(f"Failed to save {stage_name}: {e}")
        
        # Use final stage for analysis
        final_stage = max(results.keys())
        res = results[final_stage]
    else:
        # Single run results
        res = results
        
        # Create filename
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
        
        # Calculate effective sample size
        try:
            ess = res.effective_sample_size if hasattr(res, 'effective_sample_size') else 0
        except:
            weights = np.exp(res.logwt - res.logz[-1])
            ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0
        
        # Save
        np.savez(output_npz,
                samples=res.samples,
                weights=np.exp(res.logwt - res.logz[-1]),
                logl=res.logl,
                logz=res.logz,
                logzerr=res.logzerr,
                ess=ess,
                blob=res.blob if hasattr(res, 'blob') else None)
        
        logger.info(f"\n✅ Results saved to {output_npz}")
        
        # Save pickle
        output_pkl = Path(args.output_dir) / f"{output_basename}_results.pkl.gz"
        try:
            with gzip.open(output_pkl, "wb") as fh:
                pickle.dump(res, fh)
            logger.info(f"✅ Full results saved to {output_pkl}")
        except Exception as e:
            logger.error(f"Failed to save pickle: {e}")
    
    # Final summary
    if hasattr(res, 'logz'):
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   log(Z) = {res.logz[-1]:.3f} ± {res.logzerr[-1]:.3f}")
        logger.info(f"   Samples: {len(res.samples)}")
        if hasattr(res, 'eff'):
            logger.info(f"   Efficiency: {res.eff:.2f}%")
    
    # Validation recommendation
    logger.info("\n💡 Next steps:")
    logger.info("   1. Check convergence diagnostics")
    logger.info("   2. Run validation suite:")
    logger.info(f"      python validate_density_model.py --params_file {output_npz}")
    logger.info("   3. Generate plots and corner plots")
    
    logger.info("\n✨ Enhanced dynesty run complete!")


if __name__ == "__main__":
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL: Dynesty library not found")
        sys.exit(1)
    
    main_dynesty()