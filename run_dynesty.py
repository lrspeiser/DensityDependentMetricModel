#!/usr/bin/env python3
"""
run_dynesty.py - Enhanced dynamic nested sampling for the Density-Metric model.

Author: Leonard Speiser
Version: 2.1 (With GR Baseline Comparison and Priors)
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
import jax
import jax.numpy as jnp

RUN_ID = None


# Configure environment for JAX Metal support and debugging
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Show full traceback for errors
os.environ['JAX_DEBUG_NANS'] = '1'  # Check for NaN/Inf
os.environ['JAX_LOG_COMPILES'] = '0'  # Disable compilation logging (was '1')
os.environ['JAX_ENABLE_CHECKS'] = '1'  # Enable runtime checks
os.environ['JAX_METAL_USE_MPS'] = '1'  # Use Metal Performance Shaders
os.environ['VECLIB_MAXIMUM_THREADS'] = '3'  # Limit BLAS threads
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_DISABLE_MOST_FALLBACKS'] = '1'
os.environ['JAX_DISABLE_JIT_COMPILE_WARNINGS'] = '1'  # NEW: Disable JIT warnings

# Configure JAX settings
jax.config.update("jax_enable_x64", False)  # Metal doesn't support float64
jax.config.update("jax_platform_name", "METAL")  # Force Metal platform
jax.config.update("jax_log_compiles", False)  # Disable compilation logging (was True)

# Configure Python logging to suppress JAX compilation messages
def configure_jax_logging():
    """Configure JAX to be less verbose about compilation."""
    # Suppress JAX compilation messages
    jax_loggers = [
        'jax',
        'jax._src.dispatch', 
        'jax._src.interpreters',
        'jax._src.interpreters.pxla',
        'jax._src.xla_bridge',
        'jax._src.compiler',
        'jax._src.lib',
        'jax._src.profiler',
    ]
    
    for logger_name in jax_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Also suppress absl logging used by JAX
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

# Apply the logging configuration
configure_jax_logging()

# Test JAX is working (this will be silent now)
try:
    test_array = jax.numpy.ones(10)
    test_result = jax.numpy.sum(test_array)
    print(f"JAX test successful: sum = {test_result}")
except Exception as e:
    print(f"JAX test failed: {e}")
    
BASELINE_LOGZ_GR = -1490897.5250096943  # From GR-only with no dark matter 

# ============================================================================
# Model Comparison Functions
# ============================================================================

def interpret_jeffreys_scale(dlogz):
    """
    Interpret the Jeffreys scale for model comparison.
    
    Parameters
    ----------
    dlogz : float
        Difference in log evidence (logZ_model - logZ_baseline)
        
    Returns
    -------
    str
        Interpretation string based on Jeffreys scale
        
    Notes
    -----
    Jeffreys scale interpretation:
    - |dlogZ| < 1: Inconclusive
    - 1 <= |dlogZ| < 2.5: Weak evidence
    - 2.5 <= |dlogZ| < 5: Moderate evidence  
    - 5 <= |dlogZ| < 10: Strong evidence
    - |dlogZ| >= 10: Decisive evidence
    """
    abs_val = abs(dlogz)
    if abs_val < 1:
        return "Inconclusive"
    elif abs_val < 2.5:
        return "Weak evidence"
    elif abs_val < 5:
        return "Moderate evidence"
    elif abs_val < 10:
        return "Strong evidence"
    else:
        return "Decisive evidence"
    
def debug_jax_function(func):
    """Decorator to debug JAX function calls"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        try:
            print(f"[JAX DEBUG] Calling {func.__name__}")
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[JAX DEBUG] {func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            print(f"[JAX DEBUG] {func.__name__} failed: {e}")
            raise
    return wrapper


# --- JAX Configuration ---
# Use float32 for performance on GPUs (Metal and CUDA).
DEFAULT_DTYPE = jnp.float32
jax.config.update("jax_enable_x64", False)

# Local physics modules (now pointing to the JAX version)
try:
    from density_metric2 import (
        v_total_kms,
        v_baryon_total_newtonian_kms,
        rho_baryon_total_midplane_solar_kpc3,
        XI_FUNCTION_MAP,
        run_physics_self_tests,
        G_ASTRO_UNITS,
        R_SUN_KPC
    )
    from data_io import load_gaia
    # This was in your original file; it's good practice to keep it for compatibility.

    logging.getLogger("density_metric2").setLevel(logging.INFO)

# This is the missing clause that must follow the 'try' block.
except ImportError as e:
    # Provide a more informative error message for the user.
    print(f"CRITICAL: Could not import local JAX-based modules: {e}")
    print("Please ensure 'density_metric2.py', 'data_io.py', and 'main2.py' are in the same directory or accessible in your PYTHONPATH.")
    sys.exit(1)
    

# Save the full CLI args to a JSON file for reproducibility
def save_run_metadata(args, output_dir):
    """Enhanced version that tracks which args were explicitly provided"""
    
    # Original functionality - save standard run_config.json
    metadata_path = Path(output_dir) / "run_config.json"
    with open(metadata_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✅ Saved run configuration to {metadata_path}")
    
    # NEW: Enhanced tracking
    import sys
    raw_cmd = " ".join(sys.argv)
    
    # Track which arguments were explicitly provided
    explicitly_provided = set()
    i = 1  # Skip script name
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--'):
            # Extract parameter name
            param_name = arg[2:].replace('-', '_')
            explicitly_provided.add(param_name)
            # Skip the next item if it's the value for this parameter
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                i += 1
        i += 1
    
    # Build enhanced metadata
    enhanced_metadata = {
        "raw_command": raw_cmd,
        "timestamp": datetime.now().isoformat(),
        "explicitly_provided_flags": sorted(list(explicitly_provided)),
        "all_parameters": vars(args),
        "parameters_from_defaults": []
    }
    
    # Identify parameters that came from defaults
    for param, value in vars(args).items():
        if param not in explicitly_provided and not param.startswith('_'):
            enhanced_metadata["parameters_from_defaults"].append(param)
    
    # Save enhanced version
    enhanced_path = Path(output_dir) / "run_config_enhanced.json"
    with open(enhanced_path, "w") as f:
        json.dump(enhanced_metadata, f, indent=2)
    
    print(f"✅ Saved enhanced run configuration to {enhanced_path}")

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


print("DEBUG: run_dynesty.py loaded")
print(f"DEBUG: __name__ = {__name__}")
print(f"DEBUG: DYNESTY_AVAILABLE = {DYNESTY_AVAILABLE}")


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
    # Don't add any handlers here - let the root logger handle it
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
    'rho_c_solar_kpc3':    {'min': 1e6,   'max': 1e16,   'typical': 5e8},
    'n_exp':               {'min': 0.5,  'max': 4.0,    'typical': 2.7},
}


# Expected ranges for validation
EXPECTED_XI_AT_SOLAR = (0.5, 3.0)  # Xi should not suppress gravity too much at R_sun
EXPECTED_V_AT_SOLAR = (100, 1500)   # TEMPORARILY RELAXED for initial exploration

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
        'default_fixed': 5e8,    # FIX: Galaxy-appropriate default (was 5e13)
        'low': 1e7,              # FIX: Realistic bounds (was 5e12)
        'high': 1e9,             # FIX: Realistic bounds (was 1e16)
        'fit_flag_arg': 'fit_xi_params',
        'log_prior': True
    },
    'A': {
        'label': "A (enhancement factor)",
        'fixed_val_from_arg': 'A_fixed',
        'default_fixed': 8.0,     # Changed from 1.0 for enhanced model!
        'low': 2.0,               # Changed from 0.5
        'high': 10.0,             # Changed from 3.0
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
    'M_disk_total_solar': {
        'label': "M_disk_total (M_sun)",
        'fixed_val_from_arg': 'M_disk_total_fixed',
        'default_fixed': 9e10,  # thin + thick
        'low': 7e10,
        'high': 1.2e11,
        'fit_flag_arg': 'fit_disk_reparameterized',
        'include_flag_arg': 'include_disk_thin',  # Uses thin flag
        'log_prior': True
    },
    'thick_mass_fraction': {
        'label': "f_thick",
        'fixed_val_from_arg': 'thick_fraction_fixed', 
        'default_fixed': 0.25,
        'low': 0.15,
        'high': 0.40,
        'fit_flag_arg': 'fit_disk_reparameterized',
        'include_flag_arg': 'include_disk_thick',
        'log_prior': False
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
# ------------------------------------------------------------------
# Revised NON‑BLOCKING plausibility checker
# ------------------------------------------------------------------
def check_physical_plausibility(
        theta_values: np.ndarray,
        param_names: List[str],
        args_obj: argparse.Namespace
) -> Tuple[bool, str]:
    """
    Minimal plausibility checker - only checks for catastrophic issues.
    Everything else is handled by penalties in the likelihood.
    """
    log = get_or_create_logger()

    # Just unpack parameters
    try:
        params = dict(zip(param_names, theta_values))
    except Exception as e:
        log.error(f"Parameter unpacking failed: {e}")
        return False, "Parameter unpacking failure"

    # Only check for truly catastrophic issues that would crash the physics
    for name, value in params.items():
        if not np.isfinite(value):
            return False, f"{name} is NaN or Inf"
        
        # Check masses are positive
        if 'M_' in name and 'solar' in name and value <= 0:
            return False, f"{name} is negative or zero"
        
        # Check scale parameters are positive
        if ('R_d' in name or 'h_z' in name or 'a_bulge' in name) and value <= 0:
            return False, f"{name} is negative or zero"

    # Everything else is fine - let penalties handle the rest
    return True, "OK"

def reconstruct_physical_parameters(params_dict):
    """
    Convert reparameterized parameters to physical parameters.
    This ensures thick/thin disk ratios are always valid.
    """
    params = params_dict.copy()
    
    # Reconstruct disk masses from total + fraction
    if 'M_disk_total_solar' in params and 'thick_mass_fraction' in params:
        M_total = params['M_disk_total_solar']
        f_thick = params['thick_mass_fraction']
        
        # Calculate individual masses
        params['M_disk_thick_solar'] = M_total * f_thick
        params['M_disk_thin_solar'] = M_total * (1 - f_thick)
        
        # Remove the reparameterized versions
        del params['M_disk_total_solar']
        del params['thick_mass_fraction']
    
    return params

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
            args.rho_c_fixed = 5e13  # Default value - MUCH HIGHER!
            logger.info(f"   Setting default rho_c_fixed = {args.rho_c_fixed:.1e}")

        # Add/update rho_c (still needed!)
        if 'rho_c_solar_kpc3' not in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'] = {}
        
        MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'].update({
            'label': "rho_c (M_sun/kpc^3)",
            'fixed_val_from_arg': 'rho_c_fixed',
            'default_fixed': 5e13,      # FIXED: Galaxy+Cassini appropriate default
            'low': 1e12,                # FIXED: Minimum for Cassini compatibility
            'high': 1e15,               # FIXED: Upper bound for exploration
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
            'default_fixed': 2.7,       # Theory value
            'low': 2.0,                 # Tighter bounds around theory
            'high': 3.5,
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
            'default_fixed': 8.0,       # Theory value
            'low': 6.0,                 # Tighter bounds around theory
            'high': 10.0,               # Allow some variation
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
            'low': 1e6,
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


def xi_gr_baseline(rho, *args, **kwargs):
    """
    A special xi function for GR baseline runs. It ignores all inputs
    and returns 1.0, ensuring xi=1 everywhere.
    """
    # jnp.ones_like ensures the output has the same shape as the input density array.
    return jnp.ones_like(jnp.atleast_1d(rho))


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
    
class BimodalAnalyzer:
    """
    Separates physical vs unphysical samples using your check_physical_plausibility().
    Computes weight fractions and weighted median parameters for each mode.
    """

    def __init__(self, args_obj):
        self.args_obj = args_obj
        self.samples = None
        self.weights = None
        self.param_names = None

    def separate_physical_modes(self):
        physical_idx = []
        unphysical_idx = []

        for i, theta in enumerate(self.samples):
            param_dict = dict(zip(self.param_names, theta))
            values = [param_dict.get(name) for name in self.param_names]
            is_valid, *_ = check_physical_plausibility(values, self.param_names, self.args_obj)
            if is_valid:
                physical_idx.append(i)
            else:
                unphysical_idx.append(i)

        def gather(indices):
            return {
                "samples": self.samples[indices],
                "weights": self.weights[indices] if len(indices) > 0 else [],
                "weight_fraction": float(np.sum(self.weights[indices])) / np.sum(self.weights)
                if len(indices) > 0 else 0.0
            }

        return gather(physical_idx), gather(unphysical_idx)

    def get_mode_parameters(self, samples, weights):
        """
        Return weighted median parameter vector for a set of samples.
        """
        import numpy as np

        def weighted_median(data, weights):
            """Compute weighted median across 1D array."""
            sorted_idx = np.argsort(data)
            data_sorted = data[sorted_idx]
            weights_sorted = weights[sorted_idx]
            cum_weights = np.cumsum(weights_sorted)
            cutoff = np.sum(weights_sorted) / 2.0
            return data_sorted[np.searchsorted(cum_weights, cutoff)]

        num_params = samples.shape[1]
        medians = []
        for i in range(num_params):
            param_column = samples[:, i]
            medians.append(weighted_median(param_column, weights))

        return [np.array(medians)]


class AdaptiveModeMonitor:
    """Monitor sampling and steer strategy in real-time based on physical modes."""
    def __init__(self, param_names, switch_threshold=0.7):
        self.param_names = param_names
        self.switch_threshold = switch_threshold
        self.mode_history = []
        self.mode_action_log = []
        self.current_mode = None

    def update(self, samples, weights):
        if len(samples) < 500:
            return None

        analyzer = BimodalAnalyzer(args_obj)
        analyzer.samples = samples
        analyzer.weights = weights
        analyzer.param_names = self.param_names

        physical_mode, unphysical_mode = analyzer.separate_physical_modes()

        mode_info = {
            'iteration': len(self.mode_history),
            'physical_weight': physical_mode['weight_fraction'],
            'n_physical': len(physical_mode['samples']),
            'n_unphysical': len(unphysical_mode['samples'])
        }
        self.mode_history.append(mode_info)

        if physical_mode['weight_fraction'] > self.switch_threshold:
            if self.current_mode != 'physical':
                logger.info("🎯 Switching focus to PHYSICAL mode")
                self.current_mode = 'physical'
                new_bounds_center = analyzer.get_mode_parameters(
                    physical_mode['samples'],
                    physical_mode['weights']
                )[0]
                self.mode_action_log.append(("tighten_bounds", new_bounds_center))
                return {
                    'action': 'tighten_bounds',
                    'mode_params': new_bounds_center
                }

        elif physical_mode['weight_fraction'] < 0.3:
            logger.warning("⚠️ Sampling dominated by unphysical mode!")
            self.mode_action_log.append(("add_constraints", {'strength': 200}))
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
    args_obj,
    gp_surrogate=None,
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

        # At the start of enhanced_monitor_sampler_progress
        logger.info("="*80)
        logger.info(f"🔍 DYNESTY PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Add health assessment
        health = get_run_health_assessment(sampler, elapsed_time, logger)
        logger.info(f"\n🏥 RUN HEALTH: {health['status']}")
        logger.info(f"   {health['message']}")
        logger.info(f"   → {health['recommendation']}")
        logger.info("="*80)

        if not hasattr(res, 'samples') or len(res.samples) == 0:
            logger.info("❌ No samples available yet")
            return

        samples = res.samples
        n_samples, n_params = samples.shape
        ncall_total = np.sum(res.ncall) if isinstance(res.ncall, np.ndarray) else res.ncall
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        eff = 100.0 * n_samples / ncall_total if ncall_total > 0 else 0.0   
        
        # Determine run phase
        if n_samples < 100:
            run_phase = "INITIALIZATION"
            phase_emoji = "🚀"
        elif elapsed_time < 300:  # First 5 minutes
            run_phase = "EARLY EXPLORATION"
            phase_emoji = "🔍"
        elif dlogz > 1.0:
            run_phase = "ACTIVE EXPLORATION"
            phase_emoji = "🎯"
        elif dlogz > 0.1:
            run_phase = "REFINEMENT"
            phase_emoji = "🔧"
        else:
            run_phase = "CONVERGING"
            phase_emoji = "✅"
        
        logger.info(f"\n{phase_emoji} RUN PHASE: {run_phase}")
        logger.info(f"   Normal for this phase: {'Yes' if run_phase != 'CONVERGING' else 'Nearly complete'}")
        logger.info(f"⏱️  Elapsed: {elapsed_str} | 📊 Samples: {n_samples:,} | 🎲 Calls: {ncall_total:,} | 📈 Eff: {eff:.2f}%")

        if gp_surrogate is not None and GP_AVAILABLE:
            gp_stats = gp_surrogate.get_statistics()
            logger.info(f"🤖 GP Surrogate: {gp_stats['n_real_calls']:,} real, "
                        f"{gp_stats['n_surrogate_calls']:,} surrogate (speedup: {gp_stats['speedup_factor']:.1f}x)")

        # Track improvement rate
        if hasattr(sampler, '_logz_history'):
            sampler._logz_history.append((elapsed_time, current_logz))
            if len(sampler._logz_history) > 10:
                sampler._logz_history.pop(0)
            
            # Calculate improvement rate over last 60 seconds
            recent_history = [(t, z) for t, z in sampler._logz_history if elapsed_time - t < 60]
            if len(recent_history) >= 2:
                time_span = recent_history[-1][0] - recent_history[0][0]
                logz_improvement = recent_history[-1][1] - recent_history[0][1]
                improvement_rate = logz_improvement / time_span if time_span > 0 else 0
                
                logger.info(f"\n📈 IMPROVEMENT METRICS:")
                logger.info(f"   Log(Z) improvement rate: {improvement_rate:+.1f} per second")
                logger.info(f"   Status: {'IMPROVING RAPIDLY ✅' if improvement_rate > 100 else 'IMPROVING STEADILY ✅' if improvement_rate > 0 else 'PLATEAUING ⚠️'}")
        else:
            sampler._logz_history = []

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
            
        logger.info(f"\n🎯 CONVERGENCE STATUS:")
        if args_obj and hasattr(args_obj, 'dlogz_target'):
            logger.info(f"   dlogZ: {dlogz:.6f} (target: {args_obj.dlogz_target:.3f})")
            if np.isfinite(dlogz) and args_obj.dlogz_target > 0:
                progress_pct = (args_obj.dlogz_target / dlogz) * 100 if dlogz > 0 else 0
                logger.info(f"   Progress to target: {min(progress_pct, 100):.1f}%")
        else:
            logger.info(f"   dlogZ: {dlogz:.6f}")

        if np.isfinite(current_logz):
            delta_logz_vs_gr = current_logz - BASELINE_LOGZ_GR
            jeffreys_interpretation = interpret_jeffreys_scale(delta_logz_vs_gr)
            
            logger.info(f"\n📊 MODEL COMPARISON VS GR BASELINE:")
            logger.info(f"   GR Baseline log(Z): {BASELINE_LOGZ_GR:.2f}")
            logger.info(f"   Current DDMM log(Z): {current_logz:.2f}")
            logger.info(f"   Δlog(Z) = {delta_logz_vs_gr:+.2f}")
            logger.info(f"   Interpretation: {jeffreys_interpretation} {'favoring GR' if delta_logz_vs_gr < 0 else 'favoring DDMM'}")
            
            if delta_logz_vs_gr < -100:
                logger.warning(f"⚠️  DDMM model is {abs(delta_logz_vs_gr):.0f} log units worse than GR!")
                logger.warning("   Consider checking parameter bounds or model configuration")
            elif delta_logz_vs_gr > 100:
                logger.info(f"✅ DDMM model is {delta_logz_vs_gr:.0f} log units better than GR!")

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
        dlogz = None  # Start undefined

        if hasattr(res, 'logz') and len(res.logz) > 2:
            logz = float(res.logz[-1])
            sampler.saved_logz.append(logz)

            if len(sampler.saved_logz) > 2:
                dlogz = float(sampler.saved_logz[-1] - sampler.saved_logz[-2])
            else:
                dlogz = float("nan")

            if np.isfinite(dlogz):
                logger.info(f"\n📏 Stopping criterion: dlogz = {dlogz:.4f}")
            else:
                logger.info(f"\n📏 Stopping criterion: dlogz is not finite (yet)")

            # Check convergence
            if args_obj and hasattr(args_obj, 'dlogz_target'):
                if dlogz is not None and np.isfinite(dlogz):
                    if dlogz < args_obj.dlogz_target:
                        logger.info(f"   → Close to convergence target ({args_obj.dlogz_target})!")
                else:
                    logger.debug("   → Skipping convergence comparison: dlogz is None or NaN")
        else:
            logger.debug("   → Not enough logZ entries yet to compute dlogz")

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
        logger.info(f"\n📌 FIXED PARAMETERS:")
        logger.info("─" * 60)
        if hasattr(args_obj, 'all_param_info_list'):
            for p_info in args_obj.all_param_info_list:
                if not p_info['is_fitted'] and p_info['name'] in ['n_exp', 'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g']:
                    logger.info(f"{p_info['name']:<25} {p_info['current_val']:.3e}")

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

        global debug_counter
        if debug_counter < 5:
            logger = get_or_create_logger()
            logger.info(f"\n=== MONITORING DEBUG {debug_counter} ===")
            xi_type = getattr(args_obj, "xi", "UNKNOWN")
            logger.info(f"xi_type: {xi_type}")
            logger.info(f"fitted_param_names: {fitted_param_names}")
            
            # Use current median parameters instead of theta_values_fitted
            if len(samples) > 0:
                current_medians = np.median(samples, axis=0)
                logger.info(f"current_median_params: {current_medians}")
                
                # Build current parameter dict
                current_params = dict(zip(fitted_param_names, current_medians))
                for p_info in args_obj.all_param_info_list:
                    if not p_info['is_fitted']:
                        current_params[p_info['name']] = p_info['current_val']
                
                logger.info("Current reconstructed params:")
                for key in ['rho_c_solar_kpc3', 'n_exp', 'A']:
                    logger.info(f"  {key}: {current_params.get(key, 'MISSING')}")
            
            # Check all_param_info_list
            logger.info("all_param_info_list contents:")
            for p_info in args_obj.all_param_info_list:
                if p_info['name'] in ['rho_c_solar_kpc3', 'n_exp', 'A']:
                    logger.info(f"  {p_info['name']}: current_val={p_info['current_val']}, is_fitted={p_info['is_fitted']}")
            
            debug_counter += 1
                
        # 1b. Convert reparameterized parameters to physical parameters
        full_params = reconstruct_physical_parameters(full_params)
        
        
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

        # Dashboard update
        if dashboard_monitor is not None:
            try:
                # Calculate GR baseline comparison for dashboard
                delta_logz_vs_gr = None
                jeffreys_interpretation = "Unknown"
                model_preference = "Unknown"
                
                if np.isfinite(current_logz):
                    delta_logz_vs_gr = current_logz - BASELINE_LOGZ_GR
                    jeffreys_interpretation = interpret_jeffreys_scale(delta_logz_vs_gr)
                    model_preference = "GR preferred" if delta_logz_vs_gr < 0 else "DDMM preferred"
                                    
                            
                
                dashboard_state = {
                    "elapsed_time": float(elapsed_time / 3600),
                    "n_samples": int(n_samples),
                    "n_calls": int(ncall_total),
                    "efficiency": float(eff),
                    "logz": float(current_logz),
                    "logz_err": (
                        float(res.logzerr[-1])
                        if hasattr(res, 'logzerr') and len(res.logzerr) > 0 and np.isfinite(res.logzerr[-1])
                        else None
                    ),
                    "dlogz": float(dlogz) if isinstance(dlogz, (float, int)) and np.isfinite(dlogz) else None,
                    "current_nlive": int(len(res.live_points)) if hasattr(res, 'live_points') else 0,
                    "gr_baseline_logz": BASELINE_LOGZ_GR,
                    "delta_logz_vs_gr": float(delta_logz_vs_gr) if delta_logz_vs_gr is not None else None,
                    "jeffreys_vs_gr": jeffreys_interpretation,
                    "model_preference": model_preference,

                    "parameter_estimates": {},
                    "parameter_uncertainties": {},
                    "fixed_parameters": {},
                    "health_warnings": convergence_tracker.health_warnings if convergence_tracker else []
                }

                for i, name in enumerate(fitted_param_names):
                    dashboard_state["parameter_estimates"][name] = float(current_params[i])
                    dashboard_state["parameter_uncertainties"][name] = float(np.std(recent_samples[:, i]))

                if hasattr(args_obj, 'all_param_info_list'):
                    for p_info in args_obj.all_param_info_list:
                        if not p_info['is_fitted'] and p_info['name'] in ['n_exp', 'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g', 'A']:
                            dashboard_state["fixed_parameters"][p_info['name']] = float(p_info['current_val'])

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


def save_npz_checkpoint(sampler, fitted_names, output_dir, logger):
    """
    Save current sampling state to NPZ file.
    This can be called periodically during sampling.
    """
    try:
        res = getattr(sampler, "results", None)
        
        if res is None:
            logger.warning("⚠️ No sampler.results found — skipping .npz snapshot")
            return False
            
        if not hasattr(res, "samples") or res.samples is None or len(res.samples) == 0:
            logger.warning("⚠️ Dynesty results has no samples yet — skipping .npz snapshot")
            return False
            
        # Build timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include xi_type in filename
        xi_type = getattr(sampler, '_xi_type', 'power')  # Store xi_type in sampler
        npz_path = Path(output_dir) / f"dynesty_checkpoint_{xi_type}_{timestamp}.npz"
        
        # Also save to a fixed filename that overwrites (for easy resumption)
        npz_latest = Path(output_dir) / f"dynesty_checkpoint_{xi_type}_latest.npz"
        
        # Calculate weights if possible
        weights = None
        if hasattr(res, 'logwt') and hasattr(res, 'logz') and len(res.logz) > 0:
            weights = np.exp(res.logwt - res.logz[-1])
        
        # Get run configuration from sampler
        run_config = getattr(sampler, '_run_config', {})
        
        # Save the checkpoint with metadata
        save_data = {
            'samples': res.samples,
            'logz': getattr(res, "logz", np.array([])),
            'logzerr': getattr(res, "logzerr", np.array([])),
            'logl': getattr(res, "logl", np.array([])),
            'logwt': getattr(res, "logwt", np.array([])),
            'blob': getattr(res, "blob", None),
            'param_names': fitted_names,
            'weights': weights,
            'n_calls': getattr(res, "ncall", None),
            'timestamp': timestamp,
            'n_samples': len(res.samples),
            # Add metadata for analyzer
            'xi_type': xi_type,
            'include_bulge': run_config.get('include_bulge', False),
            'include_disk_thin': run_config.get('include_disk_thin', True),
            'include_disk_thick': run_config.get('include_disk_thick', False),
            'include_gas': run_config.get('include_gas', False),
            'fit_disk_thin': run_config.get('fit_disk_thin', False),
            'fit_disk_thick': run_config.get('fit_disk_thick', False),
            'fit_bulge': run_config.get('fit_bulge', False),
            'fit_gas': run_config.get('fit_gas', False),
            'fit_xi_params': run_config.get('fit_xi_params', False)
        }
        
        # Save with timestamp
        np.savez(npz_path, **save_data)
        logger.info(f"✅ Saved .npz checkpoint to: {npz_path}")
        
        # Save latest version (overwrites)
        np.savez(npz_latest, **save_data)
        logger.debug(f"✅ Updated latest checkpoint: {npz_latest}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to save .npz checkpoint: {e}")
        return False

# In run_dynesty.py, modify the save_progress_json function to use a unique filename:

def save_progress_json(sampler, fitted_names, args, start_time, logger):
    """Save current progress to JSON file that updates every minute."""
    try:
        res = getattr(sampler, "results", None)
        if res is None:
            return
        
        # Use a unique filename to avoid conflicts
        progress_file = Path(args.output_dir) / "dynesty_progress.json"
        
        # Get current stats
        current_time = time.time()
        elapsed = current_time - start_time
        n_samples = len(res.samples) if hasattr(res, 'samples') else 0
        n_calls = np.sum(res.ncall) if hasattr(res, 'ncall') else 0
        
        # Get logZ and dlogZ
        current_logz = -np.inf
        dlogz = np.nan
        if hasattr(res, 'logz') and len(res.logz) > 0:
            current_logz = float(res.logz[-1])
            if len(res.logz) >= 2:
                dlogz = float(res.logz[-1] - res.logz[-2])
        
        # Add phase determination (NOW we have the variables we need)
        if n_samples < 100:
            phase = "initialization"
            expected_behavior = "Random exploration, very poor fits expected"
        elif elapsed < 300:
            phase = "early_exploration"
            expected_behavior = "Rapid improvement expected, still finding good regions"
        elif dlogz > 1.0:
            phase = "active_exploration"
            expected_behavior = "Steady improvement, model learning parameter space"
        elif dlogz > 0.1:
            phase = "refinement"
            expected_behavior = "Slow improvement, fine-tuning parameters"
        else:
            phase = "converged"
            expected_behavior = "Model converged, results reliable"
        
        # Calculate improvement metrics
        improvement_metrics = {}
        if hasattr(sampler, '_logz_checkpoint'):
            minutes_ago = 5
            if elapsed > minutes_ago * 60:
                improvement_metrics['logz_improvement_5min'] = current_logz - sampler._logz_checkpoint
                improvement_metrics['improvement_rate'] = (current_logz - sampler._logz_checkpoint) / (minutes_ago * 60)
        else:
            sampler._logz_checkpoint = current_logz
        
        # GR baseline comparison
        delta_logz_vs_gr = current_logz - BASELINE_LOGZ_GR if np.isfinite(current_logz) else np.nan
        gr_diff_percent = (np.exp(delta_logz_vs_gr) - 1) * 100 if np.isfinite(delta_logz_vs_gr) else np.nan
        
        # Get current parameter estimates
        param_estimates = {}
        if hasattr(res, 'samples') and len(res.samples) > 0:
            recent_samples = res.samples[-min(1000, len(res.samples)):]
            for i, name in enumerate(fitted_names):
                param_estimates[name] = {
                    'median': float(np.median(recent_samples[:, i])),
                    'std': float(np.std(recent_samples[:, i]))
                }
        
        # Create progress_data ONCE with ALL fields
        progress_data = {
            'source': 'dynesty_sampler',
            'timestamp': datetime.now().isoformat(),
            'elapsed_hours': elapsed / 3600,
            'phase': phase,
            'phase_description': expected_behavior,
            'is_normal': phase != 'converged',
            'health_status': 'NORMAL' if improvement_metrics.get('improvement_rate', 0) > 0 or phase == 'initialization' else 'CHECK',
            'n_samples': n_samples,
            'n_calls': n_calls,
            'efficiency_percent': 100.0 * n_samples / n_calls if n_calls > 0 else 0,
            'current_logz': current_logz,
            'improvement_metrics': improvement_metrics,
            'dlogz': dlogz,
            'target_dlogz': args.dlogz_target,
            'dlogz_ratio': dlogz / args.dlogz_target if np.isfinite(dlogz) and args.dlogz_target > 0 else np.nan,
            'gr_baseline_logz': BASELINE_LOGZ_GR,
            'delta_logz_vs_gr': delta_logz_vs_gr,
            'gr_diff_percent': gr_diff_percent,
            'jeffreys_interpretation': interpret_jeffreys_scale(delta_logz_vs_gr) if np.isfinite(delta_logz_vs_gr) else "Unknown",
            'parameter_estimates': param_estimates,
            'xi_type': args.xi,
            'run_id': getattr(args, 'run_id', 'unknown')
        }
        
        # Save with atomic write to prevent corruption
        temp_file = progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(make_json_serializable(progress_data), f, indent=2)
        
        # Atomic rename
        temp_file.replace(progress_file)
        
        # Also save a timestamped backup every 10 minutes
        if not hasattr(save_progress_json, '_last_backup') or (current_time - save_progress_json._last_backup) > 600:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = Path(args.output_dir) / f"progress_backup_{timestamp}.json"
            import shutil
            shutil.copy2(progress_file, backup_file)
            save_progress_json._last_backup = current_time
            logger.debug(f"Progress backup saved to {backup_file}")
            
    except Exception as e:
        logger.debug(f"Failed to save progress.json: {e}")

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
    Prior transform that enforces physical relationships between parameters.
    This prevents the sampler from exploring unphysical regions.
    """
    params = {}
    u_dict = {name: u for name, u in zip(fitted_param_names, u_array)}
    bounds_low_dict = {name: b for name, b in zip(fitted_param_names, prior_bounds_low)}
    bounds_high_dict = {name: b for name, b in zip(fitted_param_names, prior_bounds_high)}
    log_flags_dict = {name: f for name, f in zip(fitted_param_names, use_log_prior_flags)}

    # Create a processing order to handle dependencies: thin disk first, then thick disk.
    param_order = sorted(fitted_param_names, key=lambda p: ('thick' in p, p))

    # Process all parameters, handling dependencies explicitly
    for name in param_order:
        low = bounds_low_dict[name]
        high = bounds_high_dict[name]
        u_val = u_dict[name]
        use_log = log_flags_dict[name]

        # --- Enforce physical relationships by dynamically adjusting the lower bound ---
        if name == 'R_d_thick_kpc' and 'R_d_thin_kpc' in params:
            # Ensure R_d_thick is always > R_d_thin
            low = max(low, params['R_d_thin_kpc'] * 1.01) # Use thin disk value as new lower bound

        if name == 'h_z_thick_kpc' and 'h_z_thin_kpc' in params:
            # Ensure h_z_thick is always >= 2 * h_z_thin
            low = max(low, params['h_z_thin_kpc'] * 2.0) # Use thin disk value as new lower bound

        # Transform the parameter from the unit cube [0, 1]
        if low >= high:
            # If the calculated lower bound is already too high, clip to the high bound
            params[name] = high
        else:
            if use_log:
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[name] = 10**(log_low + u_val * (log_high - log_low))
            else:
                params[name] = low + u_val * (high - low)

    # Convert the dictionary back to a numpy array in the original, expected order
    final_params_array = np.array([params[name] for name in fitted_param_names])
    
    return final_params_array

def log_likelihood_dynesty(
    theta_values_fitted: np.ndarray,
    fitted_param_names: List[str],
    args_dynesty_obj: argparse.Namespace,
    all_param_info_list: List[Dict],
    R_data_jax: jax.Array,
    v_data_jax: jax.Array,
    sigma_data_jax: jax.Array,
    xi_type: str,
    gp_surrogate=None
) -> Tuple[float, List[float]]:
    """
    Log-likelihood with penalty tracking that only prints every 1000th evaluation.
    """
    logger = get_or_create_logger()
    
    # Initialize tracking
    if not hasattr(log_likelihood_dynesty, '_eval_stats'):
        log_likelihood_dynesty._eval_stats = {
            'count': 0,
            'penalties_applied': {'cassini': 0, 'velocity': 0, 'mass': 0, 'rho_c': 0},  # Add rho_c
            'penalty_totals': {'cassini': 0.0, 'velocity': 0.0, 'mass': 0.0, 'rho_c': 0.0},  # Add rho_c
            'worst_penalties': {'cassini': 0.0, 'velocity': 0.0, 'mass': 0.0, 'rho_c': 0.0}  # Add rho_c
        }
    
    stats = log_likelihood_dynesty._eval_stats
    stats['count'] += 1

    # 1. Reconstruct full parameter dictionary
    params = dict(zip(fitted_param_names, theta_values_fitted))
    for p_info in all_param_info_list:
        if not p_info['is_fitted']:
            params[p_info['name']] = p_info['current_val']

    # ADDITIONAL FIX for grav_color: Ensure theory mode parameters are included
    if xi_type == 'grav_color':
        # Make sure these critical parameters are present
        if 'gamma_exp' not in params:
            params['gamma_exp'] = getattr(args_dynesty_obj, 'gamma_fixed', 2.7)
        if 'lambda_g' not in params:
            params['lambda_g'] = getattr(args_dynesty_obj, 'lambda_g_fixed', 8.0)
        # Still need rho_c
        if 'rho_c_solar_kpc3' not in params:
            params['rho_c_solar_kpc3'] = getattr(args_dynesty_obj, 'rho_c_fixed', 5e8)
        
        # Debug log (only once)
        if not hasattr(log_likelihood_dynesty, '_grav_color_logged'):
            logger.info(f"[GRAV_COLOR FIX] Parameters being used:")
            logger.info(f"  gamma_exp: {params.get('gamma_exp')}")
            logger.info(f"  lambda_g: {params.get('lambda_g')}")
            logger.info(f"  rho_c_solar_kpc3: {params.get('rho_c_solar_kpc3')}")
            log_likelihood_dynesty._grav_color_logged = True

    # Reconstruct disk masses if reparameterized
    if args_dynesty_obj.fit_disk_reparameterized:
        params = reconstruct_physical_parameters(params)

    # Add boolean flags for included components
    for component in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
        params[f'include_{component}'] = getattr(args_dynesty_obj, f'include_{component}', False)

    # Defensive float casting to avoid JAX issues
    try:
        params = {k: float(v) if isinstance(v, (int, float, str)) else v for k, v in params.items()}
    except (ValueError, TypeError) as e:
        return -np.inf, [np.inf, np.inf]

    # 2. Physical plausibility check
    all_param_names_for_check = [p['name'] for p in all_param_info_list]
    all_param_values_for_check = np.array([params.get(name) for name in all_param_names_for_check if params.get(name) is not None])

    if len(all_param_names_for_check) != len(all_param_values_for_check):
        all_param_names_for_check = [name for name in all_param_names_for_check if name in params]

    is_valid, reason, *_ = check_physical_plausibility(all_param_values_for_check, all_param_names_for_check, args_dynesty_obj)

    # Don't block sample — but optionally log it
    if not is_valid:
        logger.warning(f"⚠️ Physical plausibility soft fail: {reason} (penalized, not blocked)")

    # 3. Compute model velocities
    try:
        v_model = v_total_kms(R_data_jax, params, xi_type=xi_type)
        if not jnp.all(jnp.isfinite(v_model)):
            return -np.inf, [np.inf, np.inf]
    except Exception:
        return -np.inf, [np.inf, np.inf]

    # Base likelihood
    # REGIONAL BREAKDOWN FOR GRAVITY REGIMES
    # Inner: High density, xi ≈ 1 (standard gravity regime)
    # Outer: Low density, xi > 1 (enhanced gravity regime)

    inner_mask = R_data_jax < 8.0    # Inner galaxy - should behave Newtonian
    transition_mask = (R_data_jax >= 8.0) & (R_data_jax <= 12.0)  # Transition zone
    outer_mask = R_data_jax > 12.0   # Outer galaxy - needs enhancement

    # Calculate chi2 for each region
    chi2_inner = 0.0
    chi2_transition = 0.0  
    chi2_outer = 0.0

    if jnp.sum(inner_mask) > 0:
        chi2_inner = jnp.sum(((v_data_jax[inner_mask] - v_model[inner_mask]) / sigma_data_jax[inner_mask])**2)

    if jnp.sum(transition_mask) > 0:
        chi2_transition = jnp.sum(((v_data_jax[transition_mask] - v_model[transition_mask]) / sigma_data_jax[transition_mask])**2)

    if jnp.sum(outer_mask) > 0:
        chi2_outer = jnp.sum(((v_data_jax[outer_mask] - v_model[outer_mask]) / sigma_data_jax[outer_mask])**2)

    # Weight regions to emphasize different physics
    weight_inner = 1.0      # Standard weight for Newtonian regime
    weight_transition = 1.0  # Standard weight for transition
    weight_outer = 1.5      # Higher weight for enhancement regime (where DDMM should shine)

    # Combined chi2 with regional weighting
    chi2_total = (weight_inner * chi2_inner + 
                weight_transition * chi2_transition + 
                weight_outer * chi2_outer)

    log_L = -0.5 * chi2_total

    # Calculate regional RMSE for diagnostics
    rmse_inner = jnp.sqrt(chi2_inner / jnp.sum(inner_mask)) if jnp.sum(inner_mask) > 0 else 0.0
    rmse_transition = jnp.sqrt(chi2_transition / jnp.sum(transition_mask)) if jnp.sum(transition_mask) > 0 else 0.0
    rmse_outer = jnp.sqrt(chi2_outer / jnp.sum(outer_mask)) if jnp.sum(outer_mask) > 0 else 0.0
    current_penalties = {}
    
    # Penalty for poor inner galaxy fit (should be Newtonian-like)
    if rmse_inner > 25.0:  # Inner galaxy should fit well with minimal enhancement
        penalty = -50.0 * ((rmse_inner - 25.0) / 15.0)**2
        log_L += penalty
        current_penalties['inner_fit'] = penalty

    # Penalty for poor outer galaxy fit (this is where DDMM should help)
    if rmse_outer > 40.0:  # Outer galaxy is the main target for improvement
        penalty = -100.0 * ((rmse_outer - 40.0) / 20.0)**2  # Stronger penalty
        log_L += penalty
        current_penalties['outer_fit'] = penalty

    # Bonus for good outer galaxy fit (reward successful enhancement)
    if rmse_outer < 20.0:
        bonus = 10.0 * (20.0 - rmse_outer) / 20.0  # Small bonus for excellent outer fit
        log_L += bonus
        current_penalties['outer_bonus'] = bonus
    
    # Cassini penalty
        if not getattr(args_dynesty_obj, 'disable_cassini_penalty', False):
            try:
                cassini_dev, _ = check_cassini_compatibility(params, xi_type)
                if np.isfinite(cassini_dev) and cassini_dev > 2.3e-5:
                    penalty = -500.0 * ((cassini_dev - 2.3e-5) / 2.3e-5)**2
                    log_L += penalty
                    current_penalties['cassini'] = penalty
                    stats['penalties_applied']['cassini'] += 1
                    stats['penalty_totals']['cassini'] += penalty
                    stats['worst_penalties']['cassini'] = min(stats['worst_penalties']['cassini'], penalty)
            except Exception:
                return -np.inf, [np.inf, np.inf]
        else:
            # Log once that Cassini is disabled
            if not hasattr(log_likelihood_dynesty, '_cassini_disabled_logged'):
                logger.info("📌 Cassini penalty DISABLED for this run (galaxy-only fit)")
                log_likelihood_dynesty._cassini_disabled_logged = True

        # rho_c penalty (might also want to make this conditional)
        if not getattr(args_dynesty_obj, 'disable_rho_c_penalty', False):
            rho_c = params.get('rho_c_solar_kpc3', 1e13)
            if rho_c > 1e14:  # Too high for galaxy physics
                penalty = -100.0 * (np.log10(rho_c / 1e14))**2  # Logarithmic penalty
                log_L += penalty
                current_penalties['rho_c'] = penalty
                stats['penalties_applied']['rho_c'] += 1
                stats['penalty_totals']['rho_c'] += penalty
                stats['worst_penalties']['rho_c'] = min(stats['worst_penalties']['rho_c'], penalty)
    
    # Velocity penalty
    v_solar_mask = (R_data_jax > 7.5) & (R_data_jax < 8.5)
    if jnp.any(v_solar_mask):
        v_solar = jnp.median(v_model[v_solar_mask])
        if v_solar > 300.0:
            penalty = -0.5 * ((v_solar - 300.0) / 25.0)**2
            log_L += penalty
            current_penalties['velocity'] = penalty
            stats['penalties_applied']['velocity'] += 1
            stats['penalty_totals']['velocity'] += penalty
            stats['worst_penalties']['velocity'] = min(stats['worst_penalties']['velocity'], penalty)
    
    # Mass penalty
    total_mass = sum(params.get(c, 0.0) for c in 
                    ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar'])
    if total_mass > 0:
        if total_mass < 5e10 or total_mass > 2e11:
            if total_mass < 5e10:
                penalty = -50.0 * ((5e10 - total_mass) / 5e10)**2
            else:
                penalty = -50.0 * ((total_mass - 2e11) / 2e11)**2
            log_L += penalty
            current_penalties['mass'] = penalty
            stats['penalties_applied']['mass'] += 1
            stats['penalty_totals']['mass'] += penalty
            stats['worst_penalties']['mass'] = min(stats['worst_penalties']['mass'], penalty)
    
    # Print summary every 1000th evaluation
    if stats['count'] % 1000 == 0:
        logger.info(f"\n📊 PENALTY SUMMARY (Evaluations: {stats['count']:,})")
        logger.info("─" * 60)
        
        for ptype in ['cassini', 'velocity', 'mass']:
            count = stats['penalties_applied'][ptype]
            if count > 0:
                avg_penalty = stats['penalty_totals'][ptype] / count
                worst = stats['worst_penalties'][ptype]
                pct = 100.0 * count / stats['count']
                logger.info(f"   {ptype.capitalize():8s}: {pct:5.1f}% violations | "
                           f"Avg penalty: {avg_penalty:8.1f} | Worst: {worst:8.1f}")
        
        # Reset accumulators for next batch
        for ptype in ['cassini', 'velocity', 'mass']:
            stats['penalties_applied'][ptype] = 0
            stats['penalty_totals'][ptype] = 0.0
            stats['worst_penalties'][ptype] = 0.0
        
        logger.info("─" * 60)
    
    # Return
    rmse = jnp.sqrt(jnp.mean((v_data_jax - v_model)**2))
    cassini_dev_return = cassini_dev if 'cassini_dev' in locals() else 0.0

    # Add regional RMSE (calculated from the regional breakdown above)
    rmse_inner = jnp.sqrt(chi2_inner / jnp.sum(inner_mask)) if jnp.sum(inner_mask) > 0 else 0.0
    rmse_transition = jnp.sqrt(chi2_transition / jnp.sum(transition_mask)) if jnp.sum(transition_mask) > 0 else 0.0
    rmse_outer = jnp.sqrt(chi2_outer / jnp.sum(outer_mask)) if jnp.sum(outer_mask) > 0 else 0.0
    
    return float(log_L), [float(rmse), float(cassini_dev_return), float(rmse_inner), float(rmse_transition), float(rmse_outer)]

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
            
            if xi_type_str == 'enhanced':
                if 'A' not in p_all_params_dict:
                    logger.error(f"ERROR: A missing for enhanced model!")
                    logger.error(f"Available: {list(p_all_params_dict.keys())}")
                    return np.zeros_like(R_kpc_array)
                A = p_all_params_dict['A']
            else:
                A = 1.0  # Default for non-enhanced models
  
            gamma = None
            lambda_g = None           
        if not hasattr(v_model_for_dynesty, "_enhanced_logged") and xi_type_str == 'enhanced':
            logger.info("\n[ENHANCED MODEL DEBUG]")
            logger.info(f"  Parameters received: {list(p_all_params_dict.keys())}")
            logger.info(f"  A value: {p_all_params_dict.get('A', 'MISSING')}")
            logger.info(f"  rho_c: {p_all_params_dict.get('rho_c_solar_kpc3', 'MISSING')}")
            logger.info(f"  n_exp: {p_all_params_dict.get('n_exp', 'MISSING')}")
            v_model_for_dynesty._enhanced_logged = True

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
            xi_raw = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp, A)
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
            MW_MULTI_COMP_PARAM_CONFIG['rho_c_solar_kpc3'].update({
                'label': "rho_c (M_sun/kpc^3)",
                'fixed_val_from_arg': 'rho_c_fixed',
                'default_fixed': 5e8,      # Galaxy-appropriate default
                'low': 1e7,                # FIXED: Realistic galaxy minimum (was 1e6)
                'high': 1e9,               # FIXED: Realistic galaxy maximum (was 1e10)
                'fit_flag_arg': 'fit_rho_c',
                'log_prior': True,
                'physical_check': True
            })



def check_cassini_compatibility(params, xi_type):
    """
    Returns Cassini deviation for penalty-based enforcement.
    """
    from density_metric2 import XI_FUNCTION_MAP

    rho_saturn = 2.3e21  # Saturn orbit density
    cassini_precision = 2.3e-5

    xi_func = XI_FUNCTION_MAP.get(xi_type)
    if xi_func is None:
        return float("inf"), f"Unknown xi_type: {xi_type}"

    try:
        if xi_type == 'mass_threshold':
            return float("inf"), "mass_threshold cannot pass Cassini"

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

        deviation = abs(xi_saturn - 1.0)
        return deviation, f"xi({rho_saturn:.1e}) = {xi_saturn:.6f} (Δ = {deviation:.2e})"

    except Exception as e:
        return float("inf"), f"Cassini error: {e}"

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

def ensure_valid_disk_parameters(param_dict, param_names, logger=None):
    """
    Ensure thick disk parameters are physically consistent with thin disk.
    Modifies param_dict in place.
    """
    if logger:
        logger.debug("Ensuring valid disk parameter relationships...")
    
    # Get indices
    idx_map = {name: i for i, name in enumerate(param_names)}
    
    # Fix scale heights: h_z_thick must be >= 2 * h_z_thin
    if 'h_z_thin_kpc' in idx_map and 'h_z_thick_kpc' in idx_map:
        h_z_thin = param_dict[idx_map['h_z_thin_kpc']]
        h_z_thick = param_dict[idx_map['h_z_thick_kpc']]
        
        min_h_z_thick = 2.0 * h_z_thin
        if h_z_thick < min_h_z_thick:
            param_dict[idx_map['h_z_thick_kpc']] = min_h_z_thick * 1.1  # 10% buffer
            if logger:
                logger.info(f"   Adjusted h_z_thick from {h_z_thick:.3f} to {param_dict[idx_map['h_z_thick_kpc']]:.3f} kpc")
    
    # Fix scale lengths: R_d_thick must be > R_d_thin
    if 'R_d_thin_kpc' in idx_map and 'R_d_thick_kpc' in idx_map:
        R_d_thin = param_dict[idx_map['R_d_thin_kpc']]
        R_d_thick = param_dict[idx_map['R_d_thick_kpc']]
        
        min_R_d_thick = R_d_thin * 1.5  # Thick disk typically 1.5-2x larger
        if R_d_thick < min_R_d_thick:
            param_dict[idx_map['R_d_thick_kpc']] = min_R_d_thick
            if logger:
                logger.info(f"   Adjusted R_d_thick from {R_d_thick:.3f} to {param_dict[idx_map['R_d_thick_kpc']]:.3f} kpc")
    
    return param_dict

def get_param_labels_and_bounds(ARGS):
    """
    Enhanced parameter configuration with log-prior flags, optional prior tightening,
    and validation. Supports starting from previous best-fit and narrowing bounds.
    """
    import copy
    param_info_list = []
    # Work on a deep copy to avoid modifying the global dictionary during the run
    config_to_use = copy.deepcopy(MW_MULTI_COMP_PARAM_CONFIG)
    logger.info("Configuring parameters for multi-component Milky Way model")

    # === ADD RHO_C DEBUG SECTION HERE ===
    logger.info("\n🔍 RHO_C DEBUG - COMMAND LINE ANALYSIS:")
    logger.info(f"   --rho_c_fixed from args: {getattr(ARGS, 'rho_c_fixed', 'NOT SET')}")
    logger.info(f"   --fit_xi_params from args: {getattr(ARGS, 'fit_xi_params', 'NOT SET')}")
    logger.info(f"   Xi model: {ARGS.xi}")
    
    # Check if rho_c_fixed was provided via command line
    rho_c_fixed_via_cli = '--rho-c-fixed' in sys.argv or '--rho_c_fixed' in sys.argv
    logger.info(f"   rho_c_fixed provided via CLI: {rho_c_fixed_via_cli}")

    if ARGS.xi == 'enhanced':
        logger.info("🔧 Enhanced model - setting sensible exploration bounds")
        # Let A vary widely to find what works
        config_to_use['A']['default_fixed'] = 3.0  # Starting guess
        config_to_use['A']['low'] = 2.0            # Allow small enhancement
        config_to_use['A']['high'] = 10.0          # Allow large enhancement
        
        # *** CRITICAL FIX: Only modify rho_c bounds if NOT fixed via CLI ***
        if not rho_c_fixed_via_cli and not getattr(ARGS, 'rho_c_fixed', None):
            logger.info("   🔧 No --rho_c_fixed provided, setting default enhanced bounds")
            config_to_use['rho_c_solar_kpc3']['low'] = 1e13   # Minimum for Cassini
            config_to_use['rho_c_solar_kpc3']['high'] = 1e16  # Maximum reasonable
        else:
            logger.info(f"   🔒 --rho_c_fixed={getattr(ARGS, 'rho_c_fixed', None)} provided, keeping original bounds")
            logger.info(f"      Original bounds: [{config_to_use['rho_c_solar_kpc3']['low']:.1e}, {config_to_use['rho_c_solar_kpc3']['high']:.1e}]")
        
        # Let n_exp explore
        config_to_use['n_exp']['low'] = 0.5
        config_to_use['n_exp']['high'] = 3.0
        
    def _ensure_float(value, param_name):
        """Safely convert a value to float, logging errors."""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.error(f"FATAL: Could not convert value for parameter '{param_name}' to float. Got value: '{value}' of type {type(value)}.")
            # This is a critical error, so we should exit.
            sys.exit(1)

    # === NEW: TIGHTEN PRIORS BASED ON GR BASELINE IF REQUESTED ===
    if ARGS.use_gr_baseline_priors:
        logger.info("🔒 Applying tighter priors based on GR baseline results.")
        gr_baseline_results = {
            'M_disk_thin_solar': {'value': 70.82e9, 'uncertainty': 10.88e9},
            'R_d_thin_kpc': {'value': 3.06, 'uncertainty': 0.18},
            'h_z_thin_kpc': {'value': 0.29, 'uncertainty': 0.03},
            'M_disk_thick_solar': {'value': 20.78e9, 'uncertainty': 2.1e9},
            'R_d_thick_kpc': {'value': 7.08, 'uncertainty': 0.7},
            'h_z_thick_kpc': {'value': 1.14, 'uncertainty': 0.12},
            'M_bulge_solar': {'value': 8.70e9, 'uncertainty': 0.9e9},
            'a_bulge_kpc': {'value': 1.33, 'uncertainty': 0.13},
            'M_gas_solar': {'value': 52.65e9, 'uncertainty': 5.3e9},
            'R_d_gas_kpc': {'value': 7.85, 'uncertainty': 0.8},
            'h_z_gas_kpc': {'value': 0.27, 'uncertainty': 0.03},
        }

        for param_name, stats in gr_baseline_results.items():
            if param_name in config_to_use:
                center = stats['value']
                width = 3 * stats['uncertainty']  # 3-sigma range

                # Get the original wider bounds from the config copy
                original_low = config_to_use[param_name]['low']
                original_high = config_to_use[param_name]['high']

                # Calculate new tighter bounds, but ensure they don't exceed the original bounds
                new_low = max(original_low, center - width)
                new_high = min(original_high, center + width)

                # Update the configuration dictionary that will be used for this specific run
                config_to_use[param_name]['low'] = new_low
                config_to_use[param_name]['high'] = new_high
                logger.info(f"   → Bounds for {param_name} tightened to [{new_low:.2e}, {new_high:.2e}]")


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
            'enhanced': ['rho_c_solar_kpc3', 'n_exp', 'A'],  # <-- Add this line
            'mass_threshold': ['M_crit_msun', 'xi_boost', 'width'],
            'grav_color': ['rho_c_solar_kpc3', 'gamma_exp', 'lambda_g']
        }
        is_fitted = False
        if p_details.get('fit_flag_arg') == 'fit_xi_params':
            if ARGS.fit_xi_params and p_name in xi_model_params.get(ARGS.xi, []):
                is_fitted = True
        elif 'fit_flag_arg' in p_details and getattr(ARGS, p_details['fit_flag_arg'], False):
            is_fitted = True
        fixed_arg_name = p_details['fixed_val_from_arg']
        cli_flag_dashes = f"--{fixed_arg_name.replace('_', '-')}"
        cli_flag_underscores = f"--{fixed_arg_name}"
        if cli_flag_dashes in sys.argv or cli_flag_underscores in sys.argv:
            is_fitted = False
            logger.info(f"   🔒 CLI override detected - forcing {p_name} to be FIXED")
            
        if p_name == 'rho_c_solar_kpc3':
            logger.info(f"\n🔬 DETAILED RHO_C ANALYSIS:")
            logger.info(f"   Parameter: {p_name}")
            logger.info(f"   fit_flag_arg: {p_details.get('fit_flag_arg', 'NONE')}")
            logger.info(f"   ARGS.fit_xi_params: {getattr(ARGS, 'fit_xi_params', 'NOT SET')}")
            logger.info(f"   In xi_model_params: {p_name in xi_model_params.get(ARGS.xi, [])}")
            logger.info(f"   CLI flag dashes: {cli_flag_dashes}")
            logger.info(f"   CLI flag underscores: {cli_flag_underscores}")
            logger.info(f"   Dashes in sys.argv: {cli_flag_dashes in sys.argv}")
            logger.info(f"   Underscores in sys.argv: {cli_flag_underscores in sys.argv}")
            logger.info(f"   sys.argv: {sys.argv}")
            logger.info(f"   Final is_fitted decision: {is_fitted}")

        # Get current value
        if p_details.get('log_prior', False):
            current_val_guess = 10 ** (0.5 * (np.log10(p_details['low']) +
                                              np.log10(p_details['high'])))
        else:
            current_val_guess = 0.5 * (p_details['low'] + p_details['high'])

        if p_name == 'rho_c_solar_kpc3':
            logger.info(f"\n🔍 DEBUG rho_c_solar_kpc3:")
            logger.info(f"   is_fitted: {is_fitted}")
            logger.info(f"   cli_override will be: {getattr(ARGS, p_details['fixed_val_from_arg'], 'NOT FOUND')}")
            logger.info(f"   p_details['fixed_val_from_arg']: {p_details['fixed_val_from_arg']}")
            logger.info(f"   current_val_guess: {current_val_guess}")

        # Allow CLI --<param>_fixed to override the auto start
        cli_override = getattr(ARGS, p_details['fixed_val_from_arg'])

        # Determine the final current_val, ensuring it's a float
        if not is_fitted and cli_override is not None:
            # Use the value from the command line for a fixed parameter
            current_val = _ensure_float(cli_override, p_name) # <-- FIX
        else:
            # Use the calculated guess for a fitted parameter, or the default for a fixed one
            current_val = _ensure_float(current_val_guess, p_name) # <-- FIX


        # Bounds (tightened if available), ensuring they are floats
        if p_name in bounds_modified:
            low = _ensure_float(bounds_modified[p_name]['low'], p_name)   # <-- FIX
            high = _ensure_float(bounds_modified[p_name]['high'], p_name) # <-- FIX
        else:
            low = _ensure_float(p_details['low'], p_name)   # <-- FIX
            high = _ensure_float(p_details['high'], p_name) # <-- FIX

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
        
    if ARGS.include_disk_thin and ARGS.include_disk_thick:
        logger.info("Checking disk parameter relationships...")
        
        # Find the relevant parameters in param_info_list
        param_updates = {}
        param_indices = {}
        
        for i, p in enumerate(param_info_list):
            if p['name'] in ['h_z_thin_kpc', 'h_z_thick_kpc', 'R_d_thin_kpc', 'R_d_thick_kpc']:
                param_indices[p['name']] = i
        
        # Check and fix scale heights
        if 'h_z_thin_kpc' in param_indices and 'h_z_thick_kpc' in param_indices:
            h_z_thin = param_info_list[param_indices['h_z_thin_kpc']]['current_val']
            h_z_thick = param_info_list[param_indices['h_z_thick_kpc']]['current_val']
            
            min_h_z_thick = 2.0 * h_z_thin
            if h_z_thick < min_h_z_thick:
                new_h_z_thick = min_h_z_thick * 1.1  # 10% buffer
                param_info_list[param_indices['h_z_thick_kpc']]['current_val'] = new_h_z_thick
                logger.info(f"   Adjusted h_z_thick from {h_z_thick:.3f} to {new_h_z_thick:.3f} kpc (must be ≥2x thin)")
        
        # Check and fix scale lengths
        if 'R_d_thin_kpc' in param_indices and 'R_d_thick_kpc' in param_indices:
            R_d_thin = param_info_list[param_indices['R_d_thin_kpc']]['current_val']
            R_d_thick = param_info_list[param_indices['R_d_thick_kpc']]['current_val']
            
            min_R_d_thick = R_d_thin * 1.5  # Thick disk typically 1.5-2x larger
            if R_d_thick < min_R_d_thick:
                new_R_d_thick = min_R_d_thick
                param_info_list[param_indices['R_d_thick_kpc']]['current_val'] = new_R_d_thick
                logger.info(f"   Adjusted R_d_thick from {R_d_thick:.3f} to {new_R_d_thick:.3f} kpc (must be >1.5x thin)")
    

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

    logger.info("\n🔍 INITIAL PARAMETER VALUES CHECK:")
    param_dict = {p['name']: p['current_val'] for p in param_info_list}
    
    # Check disk parameters
    if 'R_d_thin_kpc' in param_dict and 'R_d_thick_kpc' in param_dict:
        ratio = param_dict['R_d_thick_kpc'] / param_dict['R_d_thin_kpc']
        logger.info(f"  R_d ratio (thick/thin): {ratio:.2f} (should be > 1)")
        if ratio < 1:
            logger.warning("  ⚠️ WARNING: Thick disk scale length < thin disk!")
    
    if 'h_z_thin_kpc' in param_dict and 'h_z_thick_kpc' in param_dict:
        ratio = param_dict['h_z_thick_kpc'] / param_dict['h_z_thin_kpc']
        logger.info(f"  h_z ratio (thick/thin): {ratio:.2f} (should be >= 2)")
        if ratio < 2:
            logger.warning("  ⚠️ WARNING: Thick disk scale height < 2x thin disk!")
    
    # Check total mass
    mass_components = ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']
    total_mass = sum(param_dict.get(comp, 0.0) for comp in mass_components)
    logger.info(f"  Total initial mass: {total_mass:.2e} M☉")
    logger.info(f"  Expected range: [{PHYSICAL_BOUNDS['M_total']['min']:.2e}, {PHYSICAL_BOUNDS['M_total']['max']:.2e}]")


    logger.info("\n📋 FINAL PARAMETER CONFIGURATION:")
    logger.info(f"Total parameters: {len(param_info_list)}")
    logger.info("Xi-related parameters:")
    for p in param_info_list:
        if p['name'] in ['rho_c_solar_kpc3', 'n_exp', 'A', 'gamma_exp', 'lambda_g']:
            logger.info(f"  {p['name']}: fitted={p['is_fitted']}, value={p['current_val']}")

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
    test_params = p0_guess
    logger.info("Testing with initial parameters (center of prior ranges):")
    for name, val in zip(fitted_p_names, test_params):
        logger.info(f"  {name}: {val:.3e}")

    logger.info("\nUsing fixed parameters:")
    if hasattr(args, 'all_param_info_list'):
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                logger.info(f"  {p_info['name']}: {p_info['current_val']:.3e}")
                
    logger.info("\n🔍 DETAILED PARAMETER CHECK:")

    # Build full parameter dict for debugging
    full_params = dict(zip(fitted_p_names, test_params))
    for p_info in args.all_param_info_list:
        if not p_info['is_fitted']:
            full_params[p_info['name']] = p_info['current_val']
    
    # Check total mass
    mass_components = ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']
    total_mass = sum(full_params.get(comp, 0.0) for comp in mass_components)
    logger.info(f"  Total baryonic mass: {total_mass:.2e} M☉")
    logger.info(f"  Mass bounds: [{PHYSICAL_BOUNDS['M_total']['min']:.2e}, {PHYSICAL_BOUNDS['M_total']['max']:.2e}]")
    
    # Check scale parameters
    if 'R_d_thick_kpc' in full_params and 'R_d_thin_kpc' in full_params:
        logger.info(f"  R_d_thin: {full_params['R_d_thin_kpc']:.2f} kpc")
        logger.info(f"  R_d_thick: {full_params['R_d_thick_kpc']:.2f} kpc")
        logger.info(f"  Ratio (thick/thin): {full_params['R_d_thick_kpc']/full_params['R_d_thin_kpc']:.2f}")
    
    if 'h_z_thick_kpc' in full_params and 'h_z_thin_kpc' in full_params:
        logger.info(f"  h_z_thin: {full_params['h_z_thin_kpc']:.3f} kpc")
        logger.info(f"  h_z_thick: {full_params['h_z_thick_kpc']:.3f} kpc")
        logger.info(f"  Ratio (thick/thin): {full_params['h_z_thick_kpc']/full_params['h_z_thin_kpc']:.2f}")
    
    # Check xi parameters
    logger.info(f"\n  Xi parameters:")
    logger.info(f"  - rho_c: {full_params.get('rho_c_solar_kpc3', 'NOT SET')}")
    logger.info(f"  - n_exp: {full_params.get('n_exp', 'NOT SET')}")
    logger.info(f"  - A: {full_params.get('A', 'NOT SET')}")

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

def run_curriculum_learning(args, gaia_data_dict, logger, R_data_jax, v_data_jax, sigma_data_jax, dashboard_monitor=None):
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

    stage_args_per_stage = {}                       # NEW – keep every stage's Namespace

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
        results = run_single_dynesty(stage_args, gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax, dashboard_monitor=dashboard_monitor)

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

    return all_results, stage_args_per_stage

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

def run_comprehensive_gpu_test(args, R_data_jax, v_data_jax, sigma_data_jax, logger):
    """
    Comprehensive test of GPU functionality before starting expensive sampling.
    Tests all major components and provides detailed progress updates.
    """
    import time
    
    logger.info("\n" + "="*60)
    logger.info("🧪 COMPREHENSIVE GPU/PHYSICS TEST")
    logger.info("="*60)
    
    all_tests_passed = True
    test_results = {}
    
    # Test 1: JAX Backend
    logger.info("\n[Test 1/6] Checking JAX backend...")
    try:
        backend = jax.default_backend()
        devices = jax.devices()
        logger.info(f"✅ Backend: {backend}")
        logger.info(f"✅ Devices: {devices}")
        test_results['backend'] = 'PASS'
    except Exception as e:
        logger.error(f"❌ Backend check failed: {e}")
        test_results['backend'] = 'FAIL'
        all_tests_passed = False
    
    # Test 2: Data Transfer to GPU
    logger.info("\n[Test 2/6] Testing data transfer to GPU...")
    try:
        test_size = min(100, len(R_data_jax))
        R_test = R_data_jax[:test_size]
        
        start_time = time.time()
        _ = jax.device_put(R_test).block_until_ready()
        transfer_time = time.time() - start_time
        
        logger.info(f"✅ Transferred {test_size} points in {transfer_time:.3f}s")
        test_results['data_transfer'] = 'PASS'
    except Exception as e:
        logger.error(f"❌ Data transfer failed: {e}")
        test_results['data_transfer'] = 'FAIL'
        all_tests_passed = False
    
    # Test 3: Basic JAX Operations
    logger.info("\n[Test 3/6] Testing basic JAX operations...")
    try:
        # Test array operations
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        y = jnp.sum(x * 2)
        assert float(y) == 12.0, f"Expected 12.0, got {float(y)}"
        
        # Test JIT compilation
        @jax.jit
        def test_func(a, b):
            return a + b * 2
        
        result = test_func(x, x).block_until_ready()
        logger.info(f"✅ JAX operations working: test result shape = {result.shape}")
        test_results['jax_ops'] = 'PASS'
    except Exception as e:
        logger.error(f"❌ JAX operations failed: {e}")
        test_results['jax_ops'] = 'FAIL'
        all_tests_passed = False
    
    # Test 4: Physics Functions
    logger.info("\n[Test 4/6] Testing physics functions...")
    try:
        # Get initial parameters
        fitted_names, _, p0_guess, _, _, _ = get_param_labels_and_bounds(args)
        params = dict(zip(fitted_names, p0_guess))
        
        # Add fixed parameters
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                params[p_info['name']] = p_info['current_val']
        
        # Add component flags
        for component in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
            params[f'include_{component}'] = getattr(args, f'include_{component}', False)
        
        # Test velocity calculation
        R_test = jnp.array([8.0], dtype=jnp.float32)  # Solar radius
        
        logger.info("  Testing Newtonian velocity...")
        v_newton = v_baryon_total_newtonian_kms(R_test, params)
        logger.info(f"  ✅ v_Newton at R_sun = {float(v_newton[0]):.1f} km/s")
        
        logger.info("  Testing density calculation...")
        rho = rho_baryon_total_midplane_solar_kpc3(R_test, params)
        logger.info(f"  ✅ ρ at R_sun = {float(rho[0]):.2e} M☉/kpc³")
        
        logger.info("  Testing xi function...")
        xi_func = XI_FUNCTION_MAP.get(args.xi, XI_FUNCTION_MAP['power'])
        xi_val = xi_func(rho, params.get('rho_c_solar_kpc3', 1e13), 
                         params.get('n_exp', 1.5), params.get('A', 1.0))
        logger.info(f"  ✅ ξ at R_sun = {float(xi_val[0]):.3f}")
        
        logger.info("  Testing full velocity model...")
        v_total = v_total_kms(R_test, params, xi_type=args.xi)
        logger.info(f"  ✅ v_total at R_sun = {float(v_total[0]):.1f} km/s")
        
        test_results['physics'] = 'PASS'
    except Exception as e:
        logger.error(f"❌ Physics functions failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        test_results['physics'] = 'FAIL'
        all_tests_passed = False
    
    # Test 5: Likelihood Function
    logger.info("\n[Test 5/6] Testing likelihood function...")
    try:
        # Test on small subset of data
        test_size = min(100, len(R_data_jax))
        R_subset = R_data_jax[:test_size]
        v_subset = v_data_jax[:test_size]
        sigma_subset = sigma_data_jax[:test_size]
        
        start_time = time.time()
        log_L, blob = log_likelihood_dynesty(
            p0_guess, fitted_names, args, args.all_param_info_list,
            R_subset, v_subset, sigma_subset, args.xi, None
        )
        likelihood_time = time.time() - start_time
        
        logger.info(f"✅ Likelihood calculation successful")
        logger.info(f"   log(L) = {log_L:.2f}")
        logger.info(f"   RMSE = {blob[0]:.1f} km/s")
        logger.info(f"   Time for {test_size} points: {likelihood_time:.3f}s")
        logger.info(f"   Estimated time for full dataset: {likelihood_time * len(R_data_jax) / test_size:.1f}s")
        
        test_results['likelihood'] = 'PASS'
    except Exception as e:
        logger.error(f"❌ Likelihood function failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        test_results['likelihood'] = 'FAIL'
        all_tests_passed = False
    
    # Test 6: Dynesty Integration
    logger.info("\n[Test 6/6] Testing Dynesty sampler initialization...")
    try:
        # Try to create a minimal sampler
        test_sampler = DynamicNestedSampler(
            log_likelihood_dynesty,
            prior_transform_dynesty,
            len(fitted_names),
            sample='rslice',
            bound='multi',
            ptform_args=(fitted_names, np.array([0.0]), np.array([1.0]), [False]),
            logl_args=(fitted_names, args, args.all_param_info_list, 
                      R_subset, v_subset, sigma_subset, args.xi, None)
        )
        logger.info("✅ Dynesty sampler creation successful")
        test_results['dynesty'] = 'PASS'
        del test_sampler  # Clean up
    except Exception as e:
        logger.error(f"❌ Dynesty initialization failed: {e}")
        test_results['dynesty'] = 'FAIL'
        all_tests_passed = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY:")
    logger.info("="*60)
    for test_name, result in test_results.items():
        symbol = "✅" if result == 'PASS' else "❌"
        logger.info(f"{symbol} {test_name}: {result}")
    
    if all_tests_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Ready to start sampling.")
    else:
        logger.error("\n❌ SOME TESTS FAILED! Check the log for details.")
        logger.error("Sampling may fail or give incorrect results.")
    
    logger.info("="*60 + "\n")
    
    return all_tests_passed

def add_early_progress_monitor(sampler, start_time, logger, args_obj, check_interval=10):
    """
    Add very frequent progress updates during the first few minutes.
    """
    current_time = time.time()
    elapsed = current_time - start_time
    
    # Only do frequent updates in first 5 minutes
    if elapsed > 300:  # 5 minutes
        return
    
    try:
        if hasattr(sampler, 'results') and hasattr(sampler.results, 'samples'):
            res = sampler.results
            n_samples = len(res.samples)
            n_calls = res.ncall if hasattr(res, 'ncall') else 0
            
            if isinstance(n_calls, np.ndarray):
                n_calls = np.sum(n_calls)
            
            # Get logZ and dlogZ
            current_logz = -np.inf
            dlogz = np.nan
            if hasattr(res, 'logz') and len(res.logz) > 0:
                current_logz = res.logz[-1]
                if len(res.logz) >= 2:
                    dlogz = res.logz[-1] - res.logz[-2]
            
            # GR comparison
            delta_vs_gr = current_logz - BASELINE_LOGZ_GR if np.isfinite(current_logz) else np.nan
            
            # Calculate rate
            if elapsed > 0:
                samples_per_sec = n_samples / elapsed
                calls_per_sec = n_calls / elapsed
                
                logger.info(f"[{elapsed:.0f}s] Samples: {n_samples} | "
                          f"Calls: {n_calls} | "
                          f"Rate: {samples_per_sec:.1f} samples/s, {calls_per_sec:.0f} calls/s | "
                          f"dlogZ: {dlogz:.4f} (target: {args_obj.dlogz_target:.3f}) | "
                          f"vs GR: {delta_vs_gr:+.1f}")
                
                # Estimate time to first checkpoint
                if n_samples > 0 and n_samples < 100:
                    est_time_to_100 = (100 - n_samples) / samples_per_sec
                    logger.info(f"   → Estimated time to 100 samples: {est_time_to_100:.0f}s")
            
    except Exception as e:
        logger.debug(f"Early monitor error (non-critical): {e}")

def run_single_dynesty(args, gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax, gp_surrogate=None, dashboard_monitor=None):
    """
    Run a single Dynesty sampling loop with enhanced monitoring, convergence diagnostics,
    physical plausibility checks, and optional dashboard support.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    gaia_data_dict : dict
        Gaia data dictionary
    gp_surrogate : GPSurrogateModel, optional
        Gaussian process surrogate model
    dashboard_monitor : DynestyMonitor, optional
        Dashboard monitor instance (pass when resuming to reuse existing)
    """


    import threading
    from io import StringIO
    global convergence_tracker

    # -----------------------------------------------------------------------
    # 1. Load Gaia rotation curve data and validate
    # -----------------------------------------------------------------------
    R_data_np = np.asarray(R_data_jax)
    v_data_np = np.asarray(v_data_jax)
    sigma_data_np = np.asarray(sigma_data_jax)

    if R_data_jax is None:
        logger.error("❌ R_data_jax is None before run — aborting.")
        sys.exit(1)

    logger.info(f"Loaded {len(R_data_jax)} stars from JAX array")
    logger.info(f"R range: {R_data_np.min():.2f}–{R_data_np.max():.2f} kpc")
    logger.info(f"v_obs range: {v_data_np.min():.2f}–{v_data_np.max():.2f} km/s")

    if not np.all(np.isfinite(R_data_np)):
        logger.error("Non-finite values detected in R_data")
    if not np.all(np.isfinite(v_data_np)):
        logger.error("Non-finite values detected in v_data")
    if not np.all(np.isfinite(sigma_data_np)):
        logger.error("Non-finite values detected in sigma_data")

    # -----------------------------------------------------------------------
    # 2. Load parameter configuration
    # -----------------------------------------------------------------------
    fitted_names, fitted_labels, p0_guess, p_low, p_high, log_flags = get_param_labels_and_bounds(args)
    ndim = len(fitted_names)
    convergence_tracker = ConvergenceTracker(fitted_names)

    # -----------------------------------------------------------------------
    # 2.5 Run comprehensive GPU and physics tests
    # -----------------------------------------------------------------------
    logger.info("About to run GPU tests. args.resume = %s", args.resume)
    tests_passed = True  # Initialize the variable with a default value
    if not args.resume:  # Only run tests on fresh starts
        logger.info("Starting comprehensive GPU tests...")
        tests_passed = run_comprehensive_gpu_test(args, R_data_jax, v_data_jax, sigma_data_jax, logger)
    else:
        logger.info("Skipping GPU tests because resuming from checkpoint")

    # Now, this 'if' statement will always have a valid 'tests_passed' variable to check
    if not tests_passed:
        response = input("\n⚠️ Some tests failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.error("Exiting due to failed tests.")
            return None
        else:
            logger.warning("Continuing despite failed tests...")


    # Instead of trying to JIT the entire likelihood function (which contains 
    # non-JAX operations), just do a test evaluation to ensure everything works
    
    logger.info("\n" + "="*60)
    logger.info("🔥 Starting warm-up test...")
    logger.info(f"   Xi type: {args.xi}")
    logger.info(f"   Components: thin={args.include_disk_thin}, thick={args.include_disk_thick}, "
            f"bulge={args.include_bulge}, gas={args.include_gas}")

    # Build complete parameter dictionary for testing
    test_params = dict(zip(fitted_names, p0_guess))
    for p_info in args.all_param_info_list:
        if not p_info['is_fitted']:
            test_params[p_info['name']] = p_info['current_val']

    # Show critical parameters
    logger.info("\n   Critical parameters:")
    logger.info(f"   - rho_c_solar_kpc3: {test_params.get('rho_c_solar_kpc3', 'NOT SET'):.2e}")
    logger.info(f"   - n_exp: {test_params.get('n_exp', 'NOT SET')}")
    logger.info(f"   - A: {test_params.get('A', 'NOT SET')}")
    
    logger.info("\n   🔍 DEBUGGING ENHANCED MODEL:")
    logger.info(f"   - xi_type: {args.xi}")
    logger.info(f"   - fit_xi_params: {args.fit_xi_params}")
    logger.info(f"   - A_fixed from args: {args.A_fixed}")

    # Check if A is in all_param_info_list
    a_in_param_list = False
    for p_info in args.all_param_info_list:
        if p_info['name'] == 'A':
            logger.info(f"   - A in param list: current_val={p_info['current_val']}, is_fitted={p_info['is_fitted']}")
            a_in_param_list = True
            
    if not a_in_param_list:
        logger.error("   ❌ A parameter NOT FOUND in all_param_info_list!")

    # Test individual components before full likelihood
    logger.info("\n   Testing individual physics components...")

    try:
        # Test 1: Simple array operations
        test_R = jnp.array([8.0])  # Solar radius
        logger.info(f"   Test radius: R = {float(test_R[0]):.1f} kpc")
        
        logger.info("\n   🔍 FULL PARAMETER SET FOR WARM-UP:")
        for key, value in sorted(test_params.items()):
            if isinstance(value, (int, float)):
                logger.info(f"      {key}: {value:.3e}")
            else:
                logger.info(f"      {key}: {value}")


        
        # Test 2: Newtonian velocity
        for comp in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
            test_params[f'include_{comp}'] = getattr(args, f'include_{comp}', False)
        
        v_newton = v_baryon_total_newtonian_kms(test_R, test_params)
        logger.info(f"   ✓ Newtonian velocity: {float(v_newton[0]):.1f} km/s")
        
        if v_newton[0] < 10 or v_newton[0] > 500:
            logger.warning(f"   ⚠️ Newtonian velocity seems unrealistic!")
        
        # Test 3: Density
        rho = rho_baryon_total_midplane_solar_kpc3(test_R, test_params)
        logger.info(f"   ✓ Density: {float(rho[0]):.2e} M☉/kpc³")
        
        if rho[0] < 1e3 or rho[0] > 1e12:
            logger.warning(f"   ⚠️ Density seems unrealistic for solar neighborhood!")
        
        # Test 4: Xi calculation (this is likely where it fails)
        logger.info("\n   Testing xi calculation...")
        xi_func = XI_FUNCTION_MAP.get(args.xi, XI_FUNCTION_MAP['power'])
        
        # Get the actual parameters used
        rho_c = test_params.get('rho_c_solar_kpc3', 1e13)
        n_exp = test_params.get('n_exp', 1.5)
        A_val = test_params.get('A', 1.0)
        
        logger.info(f"   Xi parameters: rho_c={rho_c:.2e}, n={n_exp}, A={A_val}")
        
        # Calculate the ratio that goes into xi
        ratio = float(rho[0] / rho_c)
        logger.info(f"   Density ratio (ρ/ρ_c): {ratio:.2e}")
        
        # For enhanced model: xi = 1 + A / (1 + (ρ/ρ_c)^n)
        if args.xi == 'enhanced':
            
            MW_MULTI_COMP_PARAM_CONFIG['A']['default_fixed'] = 8.0
            MW_MULTI_COMP_PARAM_CONFIG['A']['low'] = 7.9  # Very tight bounds
            MW_MULTI_COMP_PARAM_CONFIG['A']['high'] = 8.1

            denominator = 1.0 + ratio**n_exp
            logger.info(f"   Denominator [1 + (ρ/ρ_c)^n]: {denominator:.2e}")
            
            enhancement = A_val / denominator
            logger.info(f"   Enhancement term [A/denominator]: {enhancement:.2e}")
            
            xi_expected = 1.0 + enhancement
            logger.info(f"   Expected xi value: {xi_expected:.3f}")
            
            if xi_expected > 10:
                logger.warning(f"   ⚠️ Xi value is very large! This might cause numerical issues.")
        
        # Actually calculate xi
        xi_val = xi_func(rho, rho_c, n_exp, A_val)
        logger.info(f"   ✓ Calculated xi: {float(xi_val[0]):.3f}")
        rho_saturn = 2.3e21
        xi_saturn = xi_func(rho_saturn, rho_c, n_exp, A_val)
        logger.info(f"\n   🪐 CASSINI TEST:")
        logger.info(f"   - Saturn density: {rho_saturn:.1e}")
        logger.info(f"   - xi at Saturn: {float(xi_saturn[0]):.6f}")
        logger.info(f"   - |xi - 1|: {abs(float(xi_saturn[0]) - 1.0):.2e}")
        logger.info(f"   - Cassini tolerance: 2.3e-5")
        logger.info(f"   - Passes Cassini: {abs(float(xi_saturn[0]) - 1.0) < 2.3e-5}")
        
        # Test 5: Modified velocity
        v_modified = float(v_newton[0]) * np.sqrt(float(xi_val[0]))
        logger.info(f"   ✓ Modified velocity: {v_modified:.1f} km/s")
        
        if v_modified > 1000:
            logger.warning(f"   ⚠️ Modified velocity is unrealistically high!")
        
    except Exception as e:
        logger.error(f"   ❌ Component test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Now test the full likelihood
    logger.info("\n   Testing full likelihood calculation...")
    try:
        logl_args = (
            fitted_names, args, args.all_param_info_list,
            R_data_jax, v_data_jax, sigma_data_jax,
            args.xi, gp_surrogate
        )
        
        # ADD THIS DEBUG CODE BEFORE CALLING LIKELIHOOD:
        logger.info("\n   🔍 DEBUG: About to call log_likelihood_dynesty with:")
        logger.info(f"      Number of fitted params: {len(p0_guess)}")
        logger.info(f"      Fitted param names: {fitted_names}")
        logger.info(f"      Initial values: {p0_guess}")
        logger.info(f"      Xi type: {args.xi}")
        logger.info(f"      Number of data points: {len(R_data_jax)}")
        
        # Also check if A is being passed correctly
        full_params_debug = dict(zip(fitted_names, p0_guess))
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                full_params_debug[p_info['name']] = p_info['current_val']
        
        logger.info(f"\n      Fixed parameters being used:")
        logger.info(f"      - A: {full_params_debug.get('A', 'NOT FOUND')}")
        logger.info(f"      - rho_c_solar_kpc3: {full_params_debug.get('rho_c_solar_kpc3', 'NOT FOUND')}")
        logger.info(f"      - n_exp: {full_params_debug.get('n_exp', 'NOT FOUND')}")
        
        test_logL, test_blob = log_likelihood_dynesty(p0_guess, *logl_args)
        
        if np.isfinite(test_logL):
            logger.info(f"✅ Warm-up test PASSED!")
            logger.info(f"   Test log(L) = {test_logL:.2f}")
        # Existing code still works:
        logger.info(f"   Test RMSE = {test_blob[0]:.1f} km/s")  # Overall RMSE

        # Add new regional info:
        if len(test_blob) >= 5:
            logger.info(f"   📍 REGIONAL BREAKDOWN:")
            logger.info(f"      Inner (R<8):       {test_blob[2]:.1f} km/s")
            logger.info(f"      Transition (8-12): {test_blob[3]:.1f} km/s") 
            logger.info(f"      Outer (R>12):      {test_blob[4]:.1f} km/s")
            
            # Analysis of where the model is struggling
            if test_blob[4] > test_blob[2] * 1.5:
                logger.info(f"      → Model struggling more in outer galaxy (enhancement regime)")
            elif test_blob[2] > test_blob[4] * 1.5:
                logger.info(f"      → Model struggling more in inner galaxy (Newtonian regime)")
            else:
                logger.info(f"      → Balanced performance across regions")
        else:
            logger.error(f"❌ Likelihood is {test_logL}")
            
    except Exception as e:
        logger.error(f"❌ Likelihood calculation crashed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # -----------------------------------------------------------------------
    # 3. Inject all_param_info_list if needed (safety patch)
    # -----------------------------------------------------------------------
    if not hasattr(args, "all_param_info_list") or args.all_param_info_list is None:
        logger.warning("⚠️ args.all_param_info_list was missing — injecting now")
        get_param_labels_and_bounds(args)

    logger.info("\n" + "="*60)
    logger.info("PARAMETER CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Fitting {len(fitted_names)} parameters:")
    for name, val, low, high in zip(fitted_names, p0_guess, p_low, p_high):
        logger.info(f"  {name}: {val:.3e} [{low:.3e}, {high:.3e}]")

    logger.info("\nFixed parameters:")
    for p_info in args.all_param_info_list:
        if not p_info['is_fitted']:
            logger.info(f"  {p_info['name']}: {p_info['current_val']:.3e}")

    # Specifically check n_exp
    if 'n_exp' in fitted_names:
        logger.info(f"\n✓ n_exp IS being fitted")
    else:
        n_exp_val = next((p['current_val'] for p in args.all_param_info_list if p['name'] == 'n_exp'), None)
        logger.info(f"\n✗ n_exp is FIXED at: {n_exp_val}")
    logger.info("="*60 + "\n")

    # -----------------------------------------------------------------------
    # 4. Check initial likelihood + plausibility
    # -----------------------------------------------------------------------
    logger.info("Checking log-likelihood of initial parameter guess...")
    test_logl, test_blob = log_likelihood_dynesty(
        p0_guess, fitted_names, args, args.all_param_info_list,
        R_data_jax, v_data_jax, sigma_data_jax, args.xi, gp_surrogate)
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
    def validate_initial_parameters(args, fitted_names, p0_guess, logger):
        """Validate that initial parameters are reasonable."""
        issues = []
        
        # Check for any NaN or inf values
        if not np.all(np.isfinite(p0_guess)):
            issues.append("Initial parameters contain NaN or inf values")
        
        # Check specific parameter ranges
        param_dict = dict(zip(fitted_names, p0_guess))
        
        # Check masses are positive
        for param in ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']:
            if param in param_dict and param_dict[param] <= 0:
                issues.append(f"{param} is not positive: {param_dict[param]}")
        
        # Check scale lengths/heights are positive
        for param in fitted_names:
            if ('R_d' in param or 'h_z' in param or 'a_bulge' in param) and param in param_dict:
                if param_dict[param] <= 0:
                    issues.append(f"{param} is not positive: {param_dict[param]}")
        
        # Check rho_c is reasonable for galaxies
        if 'rho_c_solar_kpc3' in param_dict:
            if param_dict['rho_c_solar_kpc3'] < 1e6 or param_dict['rho_c_solar_kpc3'] > 1e20:
                issues.append(f"rho_c is unreasonable for galaxies: {param_dict['rho_c_solar_kpc3']:.2e}")
        
        if issues:
            logger.error("❌ Initial parameter validation failed:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False
        
        logger.info("✅ Initial parameters pass basic validation")
        return True
    
    
    ptform_args = (fitted_names, np.array(p_low), np.array(p_high), log_flags)
    logl_args = (fitted_names, args, args.all_param_info_list, R_data_jax, v_data_jax, sigma_data_jax, args.xi, gp_surrogate)


    pool = None
    if args.num_threads > 1:
        try:
            # Use ThreadPoolExecutor for JAX compatibility
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=args.num_threads)
            logger.info(f"Initialized ThreadPoolExecutor with {args.num_threads} threads")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize multiprocessing: {e}")
            pool = None    
            
    # --- New robust sampler creation block ---
    sampler = None  # Ensure sampler is defined before the try block
    try:
        sampler = DynamicNestedSampler(
            log_likelihood_dynesty,
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
        logger.info("✅ Dynesty sampler initialized successfully.")

        # Store configuration in sampler for checkpointing
        sampler._run_config = {
            'fitted_names': fitted_names, 'fitted_labels': fitted_labels, 'include_bulge': args.include_bulge,
            'include_disk_thin': args.include_disk_thin, 'include_disk_thick': args.include_disk_thick,
            'include_gas': args.include_gas, 'fit_disk_thin': args.fit_disk_thin,
            'fit_disk_thick': args.fit_disk_thick, 'fit_bulge': args.fit_bulge,
            'fit_gas': args.fit_gas, 'fit_xi_params': args.fit_xi_params, 'xi': args.xi
        }
        
        sampler._xi_type = args.xi
        
        # Initialize logz tracking safely
        if not hasattr(sampler, "saved_logz"):
            sampler.saved_logz = []
        if hasattr(args, '_resume_checkpoint_file') and args._resume_checkpoint_file:
            sampler.restore(args._resume_checkpoint_file)
            logger.info(f"✅ Resumed from checkpoint: {args._resume_checkpoint_file}")
            
    except Exception as e:
        logger.error("❌❌❌ CRITICAL: Failed to create Dynesty sampler! ❌❌❌")
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Error Message: {e}")
        import traceback
        logger.error("   Full Traceback:")
        logger.error(traceback.format_exc())
        if pool and hasattr(pool, 'shutdown'):
            pool.shutdown(wait=False)
        return None # Return None to indicate catastrophic failure

    # Initialize saved_logz tracking
    try:
        # First check if sampler and sampler.results exist
        if hasattr(sampler, 'results') and sampler.results is not None:
            logz_list = getattr(sampler.results, 'logz', None)
            if logz_list is not None and len(logz_list) >= 2:
                sampler.saved_logz = list(logz_list[-2:])
            else:
                sampler.saved_logz = []
        else:
            # No results yet, just initialize empty list
            sampler.saved_logz = []
    except Exception as e:
        logger.warning(f"Failed to initialize saved_logz: {e}")
        sampler.saved_logz = []

    # -----------------------------------------------------------------------
    # 6. Monitoring setup (dashboard, log files, convergence tracker)
    # -----------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_file = Path(args.output_dir) / "dynesty_checkpoint.pkl"

    # Only create new dashboard monitor if none was passed
    if dashboard_monitor is None and args.enable_dashboard:
        try:
            from monitor_dashboard import DynestyMonitor
            dashboard_monitor = DynestyMonitor(Path(args.output_dir))
            logger.info("Dashboard monitoring enabled (new instance)")
        except Exception as e:
            logger.warning(f"Dashboard disabled due to error: {e}")
            dashboard_monitor = None
    elif dashboard_monitor is not None:
        logger.info("Using existing dashboard monitor instance")






    # -----------------------------------------------------------------------
    # 7. Run either built-in nested loop or custom loop with early stopping
    # -----------------------------------------------------------------------
    run_start_time = time.time()

    if args.use_run_nested:
        logger.info("Running sampler using built-in run_nested()")
        # Create a timer thread for periodic NPZ saves
        import threading
        stop_saving = threading.Event()
        
        def periodic_npz_save():
            while not stop_saving.is_set():
                save_npz_checkpoint(sampler, fitted_names, args.output_dir, logger)
                stop_saving.wait(300)  # Wait 5 minutes or until stopped
        
        save_thread = threading.Thread(target=periodic_npz_save)
        save_thread.daemon = True
        save_thread.start()
        
        try:
            sampler.run_nested(
                nlive_init=args.nlive_init,
                nlive_batch=args.nlive_batch,
                dlogz_init=args.dlogz_target,
                maxcall=args.maxcall,
                print_progress=True,
                checkpoint_file=str(checkpoint_file),
                checkpoint_every=args.checkpoint_every,
            )
        finally:
            stop_saving.set()
            save_thread.join(timeout=1)

    else:
        logger.info("Running sampler using custom loop with early stopping")
        args.fitted_param_names = fitted_names
        early_stop_counter = 0
        last_monitor = last_check = time.time()
        last_npz_save = time.time()
        last_progress_json = time.time()  # ADD THIS
        NPZ_SAVE_INTERVAL = 300  # Save every 5 minutes (300 seconds)
        PROGRESS_JSON_INTERVAL = 60  # Save progress.json every minute

        try:
            # --- Phase 1: Initialise live points ---
            logger.info("Initializing live points with sampler.sample_initial()...")
            logger.info("First 5 minutes will have detailed progress updates...")

            early_check_time = time.time()
            EARLY_CHECK_INTERVAL = 10  # seconds

            # SINGLE LOOP - FIXED VERSION
            for _ in sampler.sample_initial(nlive=args.nlive_init, maxcall=args.maxcall, save_samples=True):
                now = time.time()
                
                # Very frequent updates in first 5 minutes
                if now - run_start_time < 300 and now - early_check_time > EARLY_CHECK_INTERVAL:
                    add_early_progress_monitor(sampler, run_start_time, logger, args)
                    early_check_time = now

                # Checkpoint
                if now - getattr(sampler, '_last_checkpoint_time', 0) > args.checkpoint_every:
                    try:
                        sampler.save(str(checkpoint_file))
                        logger.info(f"💾 Checkpoint saved at {checkpoint_file}")
                        sampler._last_checkpoint_time = now
                    except Exception as e:
                        logger.warning(f"⚠️ Checkpoint failed: {e}")

                # NPZ Snapshot
                if now - last_npz_save > NPZ_SAVE_INTERVAL:
                    save_npz_checkpoint(sampler, fitted_names, args.output_dir, logger)
                    last_npz_save = now

                # Progress JSON (NEW)
                if now - last_progress_json > PROGRESS_JSON_INTERVAL:
                    save_progress_json(sampler, fitted_names, args, run_start_time, logger)
                    last_progress_json = now

                # Monitor progress
                if now - last_monitor > args.monitor_interval_s:
                    last_monitor = now
                    enhanced_monitor_sampler_progress(sampler, fitted_names, fitted_labels,
                                  run_start_time, logger, args,
                                  gp_surrogate, dashboard_monitor)


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

            # --- Phase 2: Dynamic sampling loop ---
            logger.info("Starting dynamic sampling with sampler.sample_batch()...")
            for _ in sampler.sample_batch(dlogz=args.dlogz_target, maxcall=args.maxcall):
                now = time.time()

                # Checkpoint
                if now - getattr(sampler, '_last_checkpoint_time', 0) > args.checkpoint_every:
                    try:
                        sampler.save(str(checkpoint_file))
                        logger.info(f"💾 Checkpoint saved at {checkpoint_file}")
                        sampler._last_checkpoint_time = now
                    except Exception as e:
                        logger.warning(f"⚠️ Checkpoint failed: {e}")

                # NPZ Snapshot
                if now - last_npz_save > NPZ_SAVE_INTERVAL:
                    save_npz_checkpoint(sampler, fitted_names, args.output_dir, logger)
                    last_npz_save = now

                # Progress JSON (NEW)
                if now - last_progress_json > PROGRESS_JSON_INTERVAL:
                    save_progress_json(sampler, fitted_names, args, run_start_time, logger)
                    last_progress_json = now

                # Monitor progress
                if now - last_monitor > args.monitor_interval_s:
                    last_monitor = now
                    enhanced_monitor_sampler_progress(sampler, fitted_names, fitted_labels,
                                  run_start_time, logger, args,
                                  gp_surrogate, dashboard_monitor)


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
        except KeyboardInterrupt:
            logger.warning("\n🛑 Sampling interrupted by user (Ctrl+C)!")

            # Save current results
            if hasattr(sampler, 'results') and sampler.results is not None:
                try:
                    res = sampler.results
                    if hasattr(res, 'samples') and len(res.samples) > 0:
                        output_parts = ["dynesty_mw_interrupted", args.xi]
                        if args.include_bulge:       output_parts.append("B"  + ("f" if args.fit_bulge       else "x"))
                        if args.include_disk_thin:   output_parts.append("DT" + ("f" if args.fit_disk_thin   else "x"))
                        if args.include_disk_thick:  output_parts.append("DK" + ("f" if args.fit_disk_thick  else "x"))
                        if args.include_gas:         output_parts.append("G"  + ("f" if args.fit_gas         else "x"))

                        output_basename = "_".join(output_parts)
                        output_npz = Path(args.output_dir) / f"{output_basename}_samples.npz"
                        
                        # Calculate weights and ESS properly
                        weights = None
                        ess = 0.0
                        if hasattr(res, 'logwt') and hasattr(res, 'logz') and len(res.logz) > 0:
                            weights = np.exp(res.logwt - res.logz[-1])
                            if np.sum(weights**2) > 0:
                                ess = 1.0 / np.sum(weights**2)

                        np.savez(
                            output_npz,
                            samples=res.samples,
                            weights=weights,
                            param_names=np.array(fitted_names),  # <-- FIXED: Use fitted_names
                            logl=getattr(res, 'logl', None),
                            logz=getattr(res, 'logz', None),
                            logzerr=getattr(res, 'logzerr', None),
                            ess=ess,  # <-- FIXED: Now defined
                            blob=getattr(res, 'blob', None),
                            # Add metadata
                            xi_type=args.xi,
                            include_bulge=args.include_bulge,
                            include_disk_thin=args.include_disk_thin,
                            include_disk_thick=args.include_disk_thick,
                            include_gas=args.include_gas,
                            fit_disk_thin=args.fit_disk_thin,
                            fit_disk_thick=args.fit_disk_thick,
                            fit_bulge=args.fit_bulge,
                            fit_gas=args.fit_gas,
                            fit_xi_params=args.fit_xi_params,
                            output_dir=str(args.output_dir),
                            run_id=getattr(args, 'run_id', 'unknown')  # <-- FIXED: Handle missing RUN_ID
                        )

                        logger.info(f"✅ Saved interrupted results to {output_npz}")
                        logger.info(f"   Samples: {len(res.samples)}")
                        if hasattr(res, 'logz') and len(res.logz) > 0:
                            logger.info(f"   Current log(Z): {res.logz[-1]:.3f}")
                    else:
                        logger.warning("No samples available to save")

                except Exception as e:
                    logger.error(f"Failed to save interrupted results: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            else:
                logger.warning("No results available to save")
            
            if pool and hasattr(pool, 'shutdown'):
                pool.shutdown(wait=True)
            raise

        except RuntimeError as e:
            logger.error(f"🛑 Sampling stopped by runtime error: {e}")
            if hasattr(sampler, 'results'):
                np.savez(Path(args.output_dir) / "partial_results_error.npz",
                        samples=sampler.results.samples,
                        logz=sampler.results.logz,
                        error=str(e))
            if pool:
                pool.shutdown(wait=True)
            raise
# -----------------------------------------------------------------------
    # 8. Return result
    # -----------------------------------------------------------------------
    try:
        return sampler.results
    finally:
        if pool and hasattr(pool, 'shutdown'):
            pool.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down successfully")


def run_gr_baseline_fixed(args, gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax):
    """
    Run a pure GR/Newtonian baseline with fixed observational baryon parameters.
    No fitting - just calculate likelihood with realistic MW parameters.
    """
    logger.info("\n" + "="*80)
    logger.info("🌌 RUNNING GR BASELINE WITH FIXED OBSERVATIONAL PARAMETERS")
    logger.info("="*80)
    
    # Create a copy of args for the baseline run
    baseline_args = argparse.Namespace(**vars(args))
    
    # Set to GR mode
    baseline_args.xi = 'gr'
    baseline_args.output_dir = args.gr_baseline_dir
    
    # Fixed observational parameters from literature
    observational_params = {
        'M_disk_thin_fixed': 5.0e10,    # McMillan 2017
        'R_d_thin_fixed': 2.6,           # Bovy & Rix 2013
        'h_z_thin_fixed': 0.3,           # Jurić et al. 2008
        'M_disk_thick_fixed': 1.0e10,   # Bland-Hawthorn & Gerhard 2016
        'R_d_thick_fixed': 3.6,          # Robin et al. 2014
        'h_z_thick_fixed': 0.9,          # Jurić et al. 2008
        'M_bulge_fixed': 1.4e10,         # Portail et al. 2017
        'a_bulge_fixed': 0.5,            # Cao et al. 2013
        'M_gas_fixed': 1.5e10,           # Kalberla & Dedes 2008
        'R_d_gas_fixed': 7.0,            # Kalberla & Kerp 2009
        'h_z_gas_fixed': 0.15,           # Nakanishi & Sofue 2016
        'rho_c_fixed': 1e13,             # Not used but needed
        'n_exp_fixed': 1.5,              # Not used but needed
        'A_fixed': 1.0,                  # Not used but needed
    }
    
    # Apply fixed parameters
    for param, value in observational_params.items():
        setattr(baseline_args, param, value)
    
    # Don't fit anything - all parameters are fixed
    baseline_args.fit_xi_params = False
    baseline_args.fit_disk_thin = False
    baseline_args.fit_disk_thick = False
    baseline_args.fit_bulge = False
    baseline_args.fit_gas = False
    baseline_args.fit_disk_reparameterized = False
    
    # Include all components
    baseline_args.include_disk_thin = True
    baseline_args.include_disk_thick = True
    baseline_args.include_bulge = True
    baseline_args.include_gas = True
    
    # Quick run settings
    baseline_args.nlive_init = 500
    baseline_args.maxcall = 50000
    baseline_args.dlogz_target = 0.1
    baseline_args.use_curriculum_learning = False
    
    logger.info("\n📊 Fixed Baryon Parameters:")
    logger.info("─" * 60)
    for param, value in observational_params.items():
        logger.info(f"  {param:<20}: {value:.2e}")
    
    # Run the baseline
    logger.info("\n🚀 Starting GR baseline calculation...")
    results = run_single_dynesty(baseline_args, gaia_data_dict, R_data_jax, 
                                v_data_jax, sigma_data_jax)
    
    if results and hasattr(results, 'logz'):
        logger.info(f"\n✅ GR Baseline Complete!")
        logger.info(f"   Final log(Z): {results.logz[-1]:.2f}")
        logger.info(f"   This represents pure Newton/GR with realistic MW baryons")
        logger.info(f"   (No dark matter, no modifications)")
    
    return results


def run_gr_baseline_fixed(args, gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax):
    """
    Run a pure GR/Newtonian baseline with fixed observational baryon parameters.
    No fitting - just calculate likelihood with realistic MW parameters.
    """
    logger.info("\n" + "="*80)
    logger.info("🌌 RUNNING GR BASELINE WITH FIXED OBSERVATIONAL PARAMETERS")
    logger.info("="*80)
    
    # Create a copy of args for the baseline run
    baseline_args = argparse.Namespace(**vars(args))
    
    # Set to GR mode
    baseline_args.xi = 'gr'
    baseline_args.output_dir = args.gr_baseline_dir
    
    # Fixed observational parameters from literature
    observational_params = {
        'M_disk_thin_fixed': 5.0e10,    # McMillan 2017
        'R_d_thin_fixed': 2.6,           # Bovy & Rix 2013
        'h_z_thin_fixed': 0.3,           # Jurić et al. 2008
        'M_disk_thick_fixed': 1.0e10,   # Bland-Hawthorn & Gerhard 2016
        'R_d_thick_fixed': 3.6,          # Robin et al. 2014
        'h_z_thick_fixed': 0.9,          # Jurić et al. 2008
        'M_bulge_fixed': 1.4e10,         # Portail et al. 2017
        'a_bulge_fixed': 0.5,            # Cao et al. 2013
        'M_gas_fixed': 1.5e10,           # Kalberla & Dedes 2008
        'R_d_gas_fixed': 7.0,            # Kalberla & Kerp 2009
        'h_z_gas_fixed': 0.15,           # Nakanishi & Sofue 2016
        'rho_c_fixed': 1e13,             # Not used but needed
        'n_exp_fixed': 1.5,              # Not used but needed
        'A_fixed': 1.0,                  # Not used but needed
    }
    
    # Apply fixed parameters
    for param, value in observational_params.items():
        setattr(baseline_args, param, value)
    
    # Don't fit anything - all parameters are fixed
    baseline_args.fit_xi_params = False
    baseline_args.fit_disk_thin = False
    baseline_args.fit_disk_thick = False
    baseline_args.fit_bulge = False
    baseline_args.fit_gas = False
    baseline_args.fit_disk_reparameterized = False
    
    # Include all components
    baseline_args.include_disk_thin = True
    baseline_args.include_disk_thick = True
    baseline_args.include_bulge = True
    baseline_args.include_gas = True
    
    # Quick run settings
    baseline_args.nlive_init = 500
    baseline_args.maxcall = 50000
    baseline_args.dlogz_target = 0.1
    baseline_args.use_curriculum_learning = False
    
    logger.info("\n📊 Fixed Baryon Parameters:")
    logger.info("─" * 60)
    for param, value in observational_params.items():
        logger.info(f"  {param:<20}: {value:.2e}")
    
    # Run the baseline
    logger.info("\n🚀 Starting GR baseline calculation...")
    results = run_single_dynesty(baseline_args, gaia_data_dict, R_data_jax, 
                                v_data_jax, sigma_data_jax)
    
    if results and hasattr(results, 'logz'):
        logger.info(f"\n✅ GR Baseline Complete!")
        logger.info(f"   Final log(Z): {results.logz[-1]:.2f}")
        logger.info(f"   This represents pure Newton/GR with realistic MW baryons")
        logger.info(f"   (No dark matter, no modifications)")
    
    return results

def analyze_model_vs_gr(enhanced_results, gr_baseline_file, args, gaia_data):
    """
    Compare enhanced model results to GR baseline.
    Generate plots and statistics.
    """
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL COMPARISON: Enhanced vs GR Baseline")
    logger.info("="*80)
    
    # Load GR baseline if file provided
    if gr_baseline_file and Path(gr_baseline_file).exists():
        gr_data = np.load(gr_baseline_file)
        gr_logz = gr_data['logz'][-1]
    else:
        logger.warning("No GR baseline file found for comparison")
        return
    
    # Get enhanced model log(Z)
    enhanced_logz = enhanced_results.logz[-1]
    
    # Bayes factor
    log_bayes_factor = enhanced_logz - gr_logz
    bayes_factor = np.exp(log_bayes_factor)
    
    logger.info(f"\n🎯 EVIDENCE COMPARISON:")
    logger.info(f"   GR Baseline log(Z):    {gr_logz:.2f}")
    logger.info(f"   Enhanced Model log(Z): {enhanced_logz:.2f}")
    logger.info(f"   Δlog(Z):               {log_bayes_factor:+.2f}")
    logger.info(f"   Bayes Factor:          {bayes_factor:.2e}")
    logger.info(f"   Interpretation:        {interpret_jeffreys_scale(log_bayes_factor)}")
    
    # Generate velocity curve comparison
    try:
        import matplotlib.pyplot as plt
        
        # Test radii
        R_test = np.linspace(4, 30, 100)
        
        # Get best-fit parameters
        enhanced_params = {}
        if hasattr(enhanced_results, 'samples'):
            weights = np.exp(enhanced_results.logwt - enhanced_results.logz[-1])
            for i, name in enumerate(args.fitted_param_names):
                enhanced_params[name] = np.average(enhanced_results.samples[:, i], 
                                                 weights=weights)
        
        # Add fixed parameters
        for p_info in args.all_param_info_list:
            if not p_info['is_fitted']:
                enhanced_params[p_info['name']] = p_info['current_val']
        
        # Calculate velocities
        v_enhanced = v_total_kms(R_test, enhanced_params, xi_type=args.xi)
        
        # GR velocities with fixed parameters
        gr_params = {
            'M_disk_thin_solar': 5.0e10,
            'R_d_thin_kpc': 2.6,
            'h_z_thin_kpc': 0.3,
            'M_disk_thick_solar': 1.0e10,
            'R_d_thick_kpc': 3.6,
            'h_z_thick_kpc': 0.9,
            'M_bulge_solar': 1.4e10,
            'a_bulge_kpc': 0.5,
            'M_gas_solar': 1.5e10,
            'R_d_gas_kpc': 7.0,
            'h_z_gas_kpc': 0.15,
            'include_disk_thin': True,
            'include_disk_thick': True,
            'include_bulge': True,
            'include_gas': True
        }
        v_gr = v_total_kms(R_test, gr_params, xi_type='gr')
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Top panel: Velocity curves
        plt.subplot(211)
        plt.scatter(gaia_data['R_kpc'], gaia_data['v_obs'], 
                   alpha=0.1, s=1, c='gray', label='Gaia data')
        plt.plot(R_test, v_gr, 'b--', linewidth=2, 
                label='Newton/GR (no DM)')
        plt.plot(R_test, v_enhanced, 'r-', linewidth=2, 
                label=f'Enhanced Model (ξ={args.xi})')
        
        # Approximate flat rotation curve
        plt.axhline(y=220, color='k', linestyle=':', alpha=0.5, 
                   label='Flat rotation (220 km/s)')
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title('Milky Way Rotation Curve Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(4, 30)
        plt.ylim(0, 300)
        
        # Bottom panel: Residuals
        plt.subplot(212)
        # Interpolate to data points
        from scipy.interpolate import interp1d
        v_gr_interp = interp1d(R_test, v_gr, bounds_error=False, 
                              fill_value='extrapolate')
        v_enhanced_interp = interp1d(R_test, v_enhanced, bounds_error=False,
                                   fill_value='extrapolate')
        
        R_data = gaia_data['R_kpc']
        v_data = gaia_data['v_obs']
        
        residuals_gr = v_data - v_gr_interp(R_data)
        residuals_enhanced = v_data - v_enhanced_interp(R_data)
        
        # Binned residuals
        R_bins = np.linspace(4, 16, 13)
        R_centers = (R_bins[:-1] + R_bins[1:]) / 2
        
        res_gr_binned = []
        res_enhanced_binned = []
        
        for i in range(len(R_bins)-1):
            mask = (R_data >= R_bins[i]) & (R_data < R_bins[i+1])
            if np.sum(mask) > 0:
                res_gr_binned.append(np.median(residuals_gr[mask]))
                res_enhanced_binned.append(np.median(residuals_enhanced[mask]))
        
        plt.plot(R_centers[:len(res_gr_binned)], res_gr_binned, 
                'bo-', label='GR residuals', linewidth=2)
        plt.plot(R_centers[:len(res_enhanced_binned)], res_enhanced_binned, 
                'ro-', label='Enhanced residuals', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Residuals (km/s)')
        plt.title('Model Residuals (Data - Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = Path(args.output_dir) / 'model_comparison.png'
        plt.savefig(plot_file, dpi=150)
        logger.info(f"\n📈 Saved comparison plot to: {plot_file}")
        
        # Print summary statistics
        logger.info(f"\n📊 RESIDUAL STATISTICS:")
        logger.info(f"   GR RMS error:       {np.sqrt(np.mean(residuals_gr**2)):.1f} km/s")
        logger.info(f"   Enhanced RMS error: {np.sqrt(np.mean(residuals_enhanced**2)):.1f} km/s")
        
        # Radial breakdown
        logger.info(f"\n📍 PERFORMANCE BY RADIUS:")
        logger.info(f"   {'Radius':<12} {'GR Error':<12} {'Enhanced Error':<12} {'Improvement':<12}")
        logger.info("   " + "-"*48)
        
        for r_bin in [(4,8), (8,12), (12,16)]:
            mask = (R_data >= r_bin[0]) & (R_data < r_bin[1])
            if np.sum(mask) > 0:
                gr_rms = np.sqrt(np.mean(residuals_gr[mask]**2))
                enh_rms = np.sqrt(np.mean(residuals_enhanced[mask]**2))
                improvement = (gr_rms - enh_rms) / gr_rms * 100
                logger.info(f"   {r_bin[0]}-{r_bin[1]} kpc    "
                          f"{gr_rms:>8.1f} km/s  {enh_rms:>8.1f} km/s     "
                          f"{improvement:>+6.1f}%")
        
    except Exception as e:
        logger.error(f"Failed to generate comparison plots: {e}")
        
        
def run_multi_chain_analysis(args):
    """Run multiple chains with different seeds and combine results."""
    from scipy.special import logsumexp
    import numpy as np
    seeds = args.chain_seeds if args.chain_seeds is not None else [42, 123, 456]

    logger.info(f"🔗 MULTI-CHAIN MODE: Running {len(args.chain_seeds)} independent chains")
    
    # Store results from each chain
    chain_results = []
    
    # Original output directory
    base_output_dir = args.output_dir
    
    for i, seed in enumerate(args.chain_seeds):
        logger.info(f"\n{'='*80}")
        logger.info(f"🎲 CHAIN {i+1}/{len(args.chain_seeds)} with seed={seed}")
        logger.info(f"{'='*80}")
        
        # Create chain-specific output directory
        args.output_dir = Path(base_output_dir) / f"chain_{seed}"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for numpy and JAX
        np.random.seed(seed)
        import jax
        jax.random.PRNGKey(seed)
        
        # Run the chain
        try:
            # Load data (only once)
            if i == 0:
                gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax = load_and_prepare_data(args)
            
            # Run single chain
            results = run_single_dynesty(args, gaia_data_dict, R_data_jax, 
                                       v_data_jax, sigma_data_jax)
            
            if results and hasattr(results, 'logz'):
                chain_info = {
                    'seed': seed,
                    'logz': results.logz[-1],
                    'logzerr': results.logzerr[-1] if hasattr(results, 'logzerr') else 0,
                    'samples': results.samples,
                    'weights': np.exp(results.logwt - results.logz[-1]),
                    'output_dir': args.output_dir
                }
                chain_results.append(chain_info)
                logger.info(f"✅ Chain {i+1} complete: logZ = {chain_info['logz']:.2f}")
            else:
                logger.error(f"❌ Chain {i+1} failed!")
                
        except Exception as e:
            logger.error(f"❌ Chain {i+1} crashed: {e}")
            continue
    
    # Combine results
    if chain_results:
        combine_chain_results(chain_results, base_output_dir)
    else:
        logger.error("❌ All chains failed!")

def combine_chain_results(chain_results, output_dir):
    """Combine evidence and samples from multiple chains."""
    from scipy.special import logsumexp
    import json
    
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMBINING CHAIN RESULTS")
    logger.info(f"{'='*80}")
    
    # Extract log evidences
    logz_values = [chain['logz'] for chain in chain_results]
    logzerr_values = [chain['logzerr'] for chain in chain_results]
    
    # Combined evidence (assuming equal prior probability for each chain)
    combined_logz = logsumexp(logz_values) - np.log(len(logz_values))
    
    # Combined error (propagate errors)
    combined_logzerr = np.sqrt(np.sum([err**2 for err in logzerr_values])) / len(logzerr_values)
    
    # Print individual results
    logger.info("\nIndividual chain results:")
    for i, chain in enumerate(chain_results):
        logger.info(f"  Chain {i+1} (seed={chain['seed']}): "
                   f"logZ = {chain['logz']:.2f} ± {chain['logzerr']:.2f}")
    
    logger.info(f"\nCombined evidence: logZ = {combined_logz:.2f} ± {combined_logzerr:.2f}")
    
    # Compare to GR baseline
    delta_logz_vs_gr = combined_logz - BASELINE_LOGZ_GR
    logger.info(f"Δlog(Z) vs GR: {delta_logz_vs_gr:+.2f}")
    logger.info(f"Interpretation: {interpret_jeffreys_scale(delta_logz_vs_gr)}")
    
    # Combine samples (weighted by evidence)
    weights_per_chain = np.exp(np.array(logz_values) - logsumexp(logz_values))
    
    all_samples = []
    all_weights = []
    
    for chain, chain_weight in zip(chain_results, weights_per_chain):
        # Weight each chain's samples by its relative evidence
        chain_weights = chain['weights'] * chain_weight
        all_samples.append(chain['samples'])
        all_weights.append(chain_weights)
    
    combined_samples = np.vstack(all_samples)
    combined_weights = np.hstack(all_weights)
    combined_weights /= np.sum(combined_weights)  # Renormalize
    
    # Save combined results
    output_path = Path(output_dir)
    np.savez(output_path / 'combined_chains_results.npz',
             samples=combined_samples,
             weights=combined_weights,
             logz=combined_logz,
             logzerr=combined_logzerr,
             individual_logz=logz_values,
             chain_seeds=[c['seed'] for c in chain_results])
    
    # Save summary
    summary = {
        'n_chains': len(chain_results),
        'seeds': [c['seed'] for c in chain_results],
        'individual_logz': logz_values,
        'combined_logz': combined_logz,
        'combined_logzerr': combined_logzerr,
        'delta_logz_vs_gr': delta_logz_vs_gr,
        'interpretation': interpret_jeffreys_scale(delta_logz_vs_gr)
    }
    
    with open(output_path / 'combined_chains_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ Combined results saved to {output_path}")

def run_single_chain_analysis(args):
    """Original single-chain analysis (existing code)."""
    # This function will contain the rest of the original main_dynesty logic
    # after the multi-chain check
    pass
    
def load_and_prepare_data(args):
    """Load and prepare Gaia data (separated to avoid reloading)."""
    from data_io import load_all_sky_gaia_slices, process_gaia_data
    import pandas as pd
    
    logger = logging.getLogger("run_dynesty")
    
    logger.info("\n" + "="*60)
    logger.info("🔭 GAIA DATA LOADING & PROCESSING")
    logger.info("="*60)
    
    gaia_cache_file = Path("gaia_sky_slices") / "all_sky_gaia.csv"
    df_all_sky = None

    if not gaia_cache_file.exists() or args.force_new_query_gaia or args.force_reprocess_raw:
        if args.force_new_query_gaia:
            logger.info("Force flag enabled: Bypassing all caches to query Gaia from scratch.")
        elif args.force_reprocess_raw:
            logger.info("Force flag enabled: Bypassing merged cache to re-process raw slice files.")
        else:
            logger.info(f"Merged cache file not found at '{gaia_cache_file}'.")

        # Fallback 1: Try to merge raw slice files
        raw_dir = Path("gaia_sky_slices")
        raw_files = sorted(raw_dir.glob("raw_L*.csv"))
        logger.info(f"Searching for raw data in: '{raw_dir.resolve()}'")

        if raw_files and not args.force_new_query_gaia:
            logger.info(f"Found {len(raw_files)} raw Gaia slice files. Attempting to merge...")
            dfs = []
            for f in raw_files:
                try:
                    df_slice = pd.read_csv(f)
                    dfs.append(df_slice)
                    logger.info(f"  ✅ Successfully loaded {f.name} with {len(df_slice)} rows.")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to load or parse {f.name}: {e}")

            if not dfs:
                logger.error("❌ All raw Gaia slice files failed to load. Cannot proceed.")
                sys.exit(1)

            logger.info("Concatenating all loaded slices into a single DataFrame...")
            df_all_sky = pd.concat(dfs, ignore_index=True)
            logger.info(f"  ✅ Merged DataFrame created with {len(df_all_sky)} total rows.")
            
            try:
                logger.info(f"Attempting to cache the merged data to: {gaia_cache_file}")
                df_all_sky.to_csv(gaia_cache_file, index=False)
                logger.info(f"  💾 Cached merged Gaia data successfully.")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to write cache file: {e}")

        else:
            # Fallback 2: Query from scratch
            logger.info("No suitable raw files found or new query forced. Querying Gaia from scratch...")
            df_all_sky = load_all_sky_gaia_slices(
                lon_bin_width=30,
                stars_per_bin=12000,
                output_dir="gaia_sky_slices",
                force_query=True, # We are in this block, so we must query
                max_distance_kpc=30.0
            )
            logger.info("  ✅ Gaia query completed.")
            
    else:
        logger.info(f"✅ Found existing merged cache file. Loading data from: {gaia_cache_file}")
        try:
            df_all_sky = pd.read_csv(gaia_cache_file)
            logger.info(f"  ✅ Loaded {len(df_all_sky)} stars from cache.")
        except Exception as e:
            logger.error(f"❌ Failed to load cached Gaia data: {e}")
            logger.error("   Try running with --force_reprocess_raw to rebuild the cache from slices.")
            sys.exit(1)

    logger.info("\n--- Processing Raw Gaia Data into Physical Units ---")
    df_all_sky = process_gaia_data(df_all_sky)

    # --- Data Validation Step ---
    logger.info("\n--- Validating loaded Gaia DataFrame ---")
    if df_all_sky is None or df_all_sky.empty:
        logger.error("❌ DataFrame is empty after loading attempts. Cannot proceed.")
        sys.exit(1)

    logger.info(f"DataFrame shape: {df_all_sky.shape}")
    logger.info(f"Columns: {df_all_sky.columns.tolist()}")

    required_cols = ["R_kpc", "v_obs", "sigma_v"]
    missing_cols = [col for col in required_cols if col not in df_all_sky.columns]
    if missing_cols:
        logger.error(f"❌ Gaia data is missing required columns: {missing_cols}")
        sys.exit(1)
    
    logger.info(f"Checking for non-finite values in critical columns...")
    for col in required_cols:
        n_bad = np.sum(~np.isfinite(df_all_sky[col]))
        if n_bad > 0:
            logger.warning(f"  ⚠️ Found {n_bad} NaN/inf values in column '{col}'. These will be filtered.")
            df_all_sky.dropna(subset=[col], inplace=True)
    logger.info(f"DataFrame shape after cleaning non-finite values: {df_all_sky.shape}")
    logger.info("Validation of DataFrame contents complete.")
    
    # --- Convert to JAX arrays ---
    logger.info("\n--- Converting data to JAX arrays for GPU ---")
    try:
        gaia_data_dict = {col: df_all_sky[col].values for col in required_cols}
        
        R_data_np = gaia_data_dict["R_kpc"].astype(np.float32)
        v_data_np = gaia_data_dict["v_obs"].astype(np.float32)
        sigma_data_np = gaia_data_dict["sigma_v"].astype(np.float32)
        logger.info(f"NumPy arrays created with dtype={R_data_np.dtype}.")

        R_data_jax = jax.device_put(R_data_np)
        v_data_jax = jax.device_put(v_data_np)
        sigma_data_jax = jax.device_put(sigma_data_np)

        logger.info(f"✅ Successfully transferred {len(R_data_jax)} stars to JAX backend: {jax.default_backend()}")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"❌ An error occurred during conversion to JAX arrays: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    return gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax

def get_run_health_assessment(sampler, elapsed_time, logger):
    """Provide a clear assessment of whether the run is healthy."""
    
    if not hasattr(sampler, 'results') or len(sampler.results.samples) < 10:
        return {
            'status': 'TOO_EARLY',
            'message': 'Not enough samples yet to assess',
            'recommendation': 'Wait at least 1 minute before assessment'
        }
    
    res = sampler.results
    n_samples = len(res.samples)
    efficiency = 100.0 * n_samples / res.ncall if res.ncall > 0 else 0
    
    # Check improvement rate
    if hasattr(sampler, '_logz_history') and len(sampler._logz_history) > 2:
        recent = sampler._logz_history[-3:]
        improving = all(recent[i][1] > recent[i-1][1] for i in range(1, len(recent)))
    else:
        improving = True  # Assume improving if no history
    
    # Health criteria
    if elapsed_time < 300:  # First 5 minutes
        if efficiency < 5:
            return {
                'status': 'CONCERN',
                'message': 'Very low efficiency in early stage',
                'recommendation': 'Check prior bounds - they might be too wide'
            }
        elif not improving and elapsed_time > 120:
            return {
                'status': 'CONCERN', 
                'message': 'Not improving after 2 minutes',
                'recommendation': 'Check model configuration'
            }
        else:
            return {
                'status': 'HEALTHY',
                'message': f'Normal early exploration (efficiency: {efficiency:.1f}%)',
                'recommendation': 'Continue running - rapid improvement expected'
            }
    
    elif elapsed_time < 1800:  # First 30 minutes
        if not improving:
            return {
                'status': 'WARNING',
                'message': 'Improvement has stalled',
                'recommendation': 'May need to adjust priors or model settings'
            }
        else:
            return {
                'status': 'HEALTHY',
                'message': f'Active exploration phase (efficiency: {efficiency:.1f}%)',
                'recommendation': 'Continue running - still exploring parameter space'
            }
    
    else:  # After 30 minutes
        if res.logz[-1] < -1e6:
            return {
                'status': 'PROBLEM',
                'message': 'Log(Z) still very negative after 30 minutes',
                'recommendation': 'Model may be mis-specified - check configuration'
            }
        else:
            return {
                'status': 'HEALTHY',
                'message': 'Run progressing normally',
                'recommendation': 'Continue until convergence (dlogZ < 0.1)'
            }

# ============================================================================
# Main Entry Point
# ============================================================================

# In run_dynesty.py, replace the entire main_dynesty() function with this.

def main_dynesty():
    """
    Main entry point for running the Enhanced Dynesty Sampler...
    """
    global logger, debug_counter, RUN_ID
    from datetime import datetime
    import uuid
    import pandas as pd
    import argparse
    from data_io import load_all_sky_gaia_slices, process_gaia_data
    from pathlib import Path

    # ============================================================================
    # 1. ARGUMENT PARSING (must come first to get --debug and --output_dir)
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="Enhanced Dynesty sampler for Density-Metric model with physical constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core run options
    parser.add_argument('--resume', action='store_true', default=False, help="Resume from checkpoint")
    parser.add_argument('--debug', action='store_true', default=False, help="Enable verbose debug logging")
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic', 'enhanced', 'grav_color', 'mass_threshold', 'gr', 'deur'], help="Choice of xi(ρ) function")
    parser.add_argument('--max_sample_gaia', type=int, default=10000, help="Maximum number of Gaia stars to use")
    parser.add_argument('--output_dir', type=str, default="chains_dynesty", help="Output directory for results")
    parser.add_argument('--R_d_thin_high', type=float, default=None, help="Override upper prior bound for R_d_thin_kpc")
    parser.add_argument('--multi_chain', action='store_true', default=False, help="Run multiple chains with different seeds")
    parser.add_argument('--chain_seeds', type=int, nargs='+', default=None, help="Random seeds for multi-chain runs")

    # Sampler options
    parser.add_argument('--nlive_init', type=int, default=800, help="Initial number of live points")
    parser.add_argument('--nlive_batch', type=int, default=200, help="Live points per batch")
    parser.add_argument('--dlogz_target', type=float, default=0.01, help="Target dlogz for convergence")
    parser.add_argument('--num_threads', type=int, default=8, help="Number of threads for parallelization")
    parser.add_argument('--maxcall', type=int, default=2000000, help="Maximum likelihood calls")
    parser.add_argument('--monitor_interval_s', type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument('--enable_dashboard', action='store_true', default=True, help="Enable enhanced monitoring dashboard")
    parser.add_argument('--monitor_config', type=str, default=None, help="Path to monitoring configuration file")
    parser.add_argument('--use_run_nested', action='store_true', default=False, help="Use run_nested instead of custom loop")
    parser.add_argument('--checkpoint_every', type=int, default=60, help="Checkpoint interval in seconds")
    parser.add_argument('--checkpoint_file', type=str, default=None, help="Path to a specific dynesty checkpoint to resume from")
    parser.add_argument('--max_thick_thin_ratio', type=float, default=None, help="Max allowed thick/thin disk mass ratio")
    parser.add_argument('--M_disk_thin_min', type=float, default=None, help="Override lower prior bound for M_disk_thin_solar")
    parser.add_argument('--M_disk_thin_max', type=float, default=None, help="Override upper prior bound for M_disk_thin_solar")
    parser.add_argument('--M_disk_total_min', type=float, default=None, help="Override lower prior bound for M_disk_total_solar")
    parser.add_argument('--M_disk_total_max', type=float, default=None, help="Override upper prior bound for M_disk_total_solar")
    parser.add_argument('--M_disk_thick_min', type=float, default=None, help="Override lower prior bound for M_disk_thick_solar")
    parser.add_argument('--M_disk_thick_max', type=float, default=None, help="Override upper prior bound for M_disk_thick_solar")
    parser.add_argument('--M_bulge_max', type=float, default=None, help="Override upper prior bound for M_bulge_solar")
    parser.add_argument('--h_z_thin_min', type=float, default=None, help="Override lower prior bound for h_z_thin_kpc")
    parser.add_argument('--R_d_thick_max', type=float, default=None, help="Override upper prior bound for R_d_thick_kpc")
    parser.add_argument('--M_gas_max', type=float, default=None, help="Override upper prior bound for M_gas_solar")
    parser.add_argument('--force_new_query_gaia', action='store_true', default=False, help="Force new Gaia query")
    parser.add_argument('--force_reprocess_raw', action='store_true', default=False, help="Force reprocessing of raw Gaia data")
    parser.add_argument('--run_gr_baseline', action='store_true', default=False,
                    help="Run GR baseline with observational parameters before main run")
    parser.add_argument('--compare_to_gr', action='store_true', default=False,
                    help="Compare results to GR baseline after run")
    parser.add_argument('--gr_baseline_dir', type=str, default="chains_gr_baseline_fixed",
                    help="Directory containing GR baseline results")
    parser.add_argument('--disable_cassini_penalty', action='store_true', default=False,
                        help="Disable Cassini constraint penalty (for galaxy-only fits)")
    parser.add_argument('--disable_rho_c_penalty', action='store_true', default=False,
                        help="Disable rho_c upper bound penalty (for galaxy-only fits)")

    # Dynesty sampler group
    dynesty_g = parser.add_argument_group('Dynesty Sampler Settings')
    dynesty_g.add_argument('--sample_method', type=str, default='rslice', choices=['rwalk', 'rslice', 'hslice'], help="Sampling method")
    dynesty_g.add_argument('--walks', type=int, default=25, help="Number of walks for rwalk sampler")
    dynesty_g.add_argument('--enlarge_factor', type=float, default=2.5, help="Bound enlargement factor")
    dynesty_g.add_argument('--bound_method', type=str, default='multi', choices=['none', 'single', 'multi', 'balls', 'cubes'], help="Bounding method")

    # Enhanced features
    ai_g = parser.add_argument_group('Enhanced Features')
    ai_g.add_argument('--use_curriculum_learning', action='store_true', default=False, help="Use curriculum learning")
    ai_g.add_argument('--use_gp_surrogate', action='store_true', default=False, help="Use Gaussian Process surrogate")
    ai_g.add_argument('--gp_n_initial', type=int, default=500, help="Initial training points for GP")
    ai_g.add_argument('--gp_uncertainty_threshold', type=float, default=0.1, help="GP uncertainty threshold")
    ai_g.add_argument('--validate_data', action='store_true', default=True, help="Validate loaded data quality")
    parser.add_argument('--use_previous_best', action='store_true', default=False, help="Initialize from previous best-fit")
    parser.add_argument('--previous_results_file', type=str, default="chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz", help="Path to previous results")
    parser.add_argument('--tighten_bounds_factor', type=float, default=0.1, help="Factor for tightening bounds")
    parser.add_argument('--disable_dashboard', action='store_true', default=False, help="Disable dashboard monitoring")
    ai_g.add_argument('--fix_gamma', type=float, default=None, help="Fix gamma exponent")
    ai_g.add_argument('--fix_lambda_g', type=float, default=None, help="Fix lambda_g enhancement factor")
    ai_g.add_argument('--theory_mode', action='store_true', default=False, help="Use theoretical values")
    ai_g.add_argument('--use_gr_baseline_priors', action='store_true', default=False, help="Use tighter priors from GR baseline")
    

    # Model components
    mw_model_g = parser.add_argument_group('Model Components')
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True)
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)

    # Fit flags
    fit_g = parser.add_argument_group('Parameters to Fit')
    fit_g.add_argument('--fit_xi_params', action='store_true', help="Fit xi function parameters")
    fit_g.add_argument('--fit_rho_c', action='store_true', help="Fit rho_c (for grav_color mode)")
    fit_g.add_argument('--fit_gamma', action='store_true', help="Fit gamma exponent (grav_color)")
    fit_g.add_argument('--fit_lambda_g', action='store_true', help="Fit lambda_g enhancement (grav_color)")
    fit_g.add_argument('--fit_disk_thin', action='store_true', help="Fit thin disk parameters")
    fit_g.add_argument('--fit_disk_thick', action='store_true', help="Fit thick disk parameters")
    fit_g.add_argument('--fit_bulge', action='store_true', help="Fit bulge parameters")
    fit_g.add_argument('--fit_gas', action='store_true', help="Fit gas parameters")
    fit_g.add_argument('--fit_disk_reparameterized', action='store_true', help="Fit disk masses using total+fraction")

    # Fixed values
    fixed_g = parser.add_argument_group('Fixed Parameter Values')
    for p_name, p_details in MW_MULTI_COMP_PARAM_CONFIG.items():
        fixed_g.add_argument(f"--{p_details['fixed_val_from_arg']}", type=float, default=p_details['default_fixed'], help=f"Fixed/initial value for {p_name}")
    args = parser.parse_args()

    # ============================================================================
    # 2. LOGGING SETUP
    # ============================================================================
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dynesty_debug.log"

    # Define console level based on debug flag
    console_level = logging.DEBUG if args.debug else logging.INFO

    # Configure the root logger directly. This is the most reliable method.
    logging.basicConfig(
        level=logging.DEBUG,  # Capture ALL levels of messages
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Handler 0: Always writes DEBUG to file
            logging.StreamHandler(sys.stdout)         # Handler 1: Writes to console
        ],
        force=True  # Override any previous configurations
    )

    # Now, get the root logger and set the console handler's level specifically.
    # This prevents DEBUG messages from showing on the console unless --debug is used.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(console_level)

    # Get our specific logger for the rest of the script
    logger = logging.getLogger("run_dynesty")

    logger.info(f"📡 Full debug log initialized. Writing to: {log_file}")
    logger.info(f"Console log level set to: {logging.getLevelName(console_level)}")

    # ============================================================================
    # 3. INITIALIZATION AND CRASH RECOVERY
    # ============================================================================
    RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    debug_counter = 0

    # Check for multi-chain mode
    if args.multi_chain:
        run_multi_chain_analysis(args)
        return  # Exit after multi-chain analysis

    # Continue with single-chain analysis (original code)

    run_tracking_enabled = True
    try:
        from run_history import start_record, finalize_record as _finalize_record
        RUN_ID = start_record(args)
        logger.info(f"Run tracking initialized with ID: {RUN_ID}")
    except ImportError:
        logger.warning("run_history module not available, disabling run tracking")
        run_tracking_enabled = False
        _finalize_record = lambda *args, **kwargs: None

    def finalize_record(*args, **kwargs):
        try:
            return _finalize_record(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to finalize run record: {e}")
            return False

    checkpoint_path = Path(args.output_dir) / "dynesty_checkpoint.pkl"
    final_npz_files = list(Path(args.output_dir).glob('*_samples.npz'))
    final_summary_files = list(Path(args.output_dir).glob('*_summary.json'))
    if checkpoint_path.exists() and not final_npz_files and not final_summary_files and not args.resume:
        logger.warning("🧠 Detected potential prior crash (checkpoint exists but no final results).")
        logger.warning("   Activating recovery mode: reducing load and forcing resumption.")
        original_nlive = args.nlive_init
        original_maxcall = args.maxcall
        args.nlive_init = max(300, int(args.nlive_init * 0.75))
        args.maxcall = max(100000, int(args.maxcall * 0.75))
        args.use_run_nested = True
        args.resume = True
        args.checkpoint_file = str(checkpoint_path)
        logger.info(f"   - nlive_init reduced: {original_nlive} -> {args.nlive_init}")
        logger.info(f"   - maxcall reduced: {original_maxcall} -> {args.maxcall}")
        logger.info("   - Switched to built-in 'run_nested' loop for better stability.")
        logger.info(f"   - Forcing resume from: {args.checkpoint_file}")

    save_run_metadata(args, args.output_dir)

    logger.info("Starting Enhanced Dynesty Sampler v2.1")
    if not DYNESTY_AVAILABLE:
        logger.error("CRITICAL: Dynesty library not available. Please install it.")
        sys.exit(1)

    # ============================================================================
    # THEORY MODE AND XI PARAMETER SETUP (MOVED BEFORE DATA LOADING)
    # ============================================================================
    if args.xi == 'mass_threshold':
        logger.error("WARNING: mass_threshold model CANNOT simultaneously pass Cassini and galaxy tests!")
        logger.error("This model is fundamentally incompatible with Solar System constraints.")
        logger.error("Consider using 'power', 'enhanced', or 'grav_color' instead.")

    # Theory mode overrides - MUST come before setup_xi_parameters_for_mode
    if args.theory_mode:
        logger.info("🧪 THEORY MODE: gamma=2.7, lambda_g=8.0, only fitting rho_c")
        args.fix_gamma = 2.7
        args.fix_lambda_g = 8.0
        args.gamma_fixed = 2.7
        args.lambda_g_fixed = 8.0
        args.fit_xi_params = False
        args.fit_gamma = False
        args.fit_lambda_g = False
        if not hasattr(args, 'rho_c_fixed') or args.rho_c_fixed is None:
            logger.info("   No --rho_c_fixed provided, will fit rho_c")
            args.fit_rho_c = True
        else:
            logger.info(f"   Using fixed rho_c = {args.rho_c_fixed:.1e}")
            args.fit_rho_c = False  # Respect the user's fixed value!

    # Setup xi parameters AFTER theory mode settings
    setup_xi_parameters_for_mode(args)

    # ============================================================================
    # GAIA DATA LOADING WITH ENHANCED LOGGING
    # ============================================================================
    gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax = load_and_prepare_data(args)

    # Ensure physical prior bounds match -- override if needed
    if args.R_d_thin_high is not None:
        MW_MULTI_COMP_PARAM_CONFIG['R_d_thin_kpc']['high'] = args.R_d_thin_high
    if args.M_disk_thin_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thin_solar']['low'] = args.M_disk_thin_min
    if args.M_disk_thin_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thin_solar']['high'] = args.M_disk_thin_max
    if args.M_disk_total_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_total_solar']['low'] = args.M_disk_total_min
    if args.M_disk_total_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_total_solar']['high'] = args.M_disk_total_max
    if args.h_z_thin_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['h_z_thin_kpc']['low'] = args.h_z_thin_min
    if args.R_d_thick_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['R_d_thick_kpc']['high'] = args.R_d_thick_max
    if args.M_gas_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_gas_solar']['high'] = args.M_gas_max
    if args.M_disk_thick_min is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thick_solar']['low'] = args.M_disk_thick_min
    if args.M_disk_thick_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_disk_thick_solar']['high'] = args.M_disk_thick_max
    if args.M_bulge_max is not None:
        MW_MULTI_COMP_PARAM_CONFIG['M_bulge_solar']['high'] = args.M_bulge_max

    if args.checkpoint_file is None:
        args.checkpoint_file = str(Path(args.output_dir) / "dynesty_checkpoint.pkl")

    # ============================================================================
    # RESUME LOGIC
    # ============================================================================
    if args.resume:
        checkpoint_path = Path(args.checkpoint_file)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        # Store the checkpoint file path for the sampler to use later
        args._resume_checkpoint_file = str(checkpoint_path)

        # --- ENHANCED RESUME LOGIC V3 ---
        try:
            with open(checkpoint_path, 'rb') as f:
                import pickle
                checkpoint_data = pickle.load(f)
            
            # Check if this is a dynesty dictionary-format checkpoint
            if isinstance(checkpoint_data, dict) and 'sampler' in checkpoint_data:
                temp_sampler = checkpoint_data['sampler']
            else:
                temp_sampler = checkpoint_data
            
            config_restored = False
            if hasattr(temp_sampler, '_run_config'):
                # --- Case 1: Modern checkpoint with config inside ---
                logger.info("✅ Restoring run configuration from inside checkpoint file...")
                for key, value in temp_sampler._run_config.items():
                    setattr(args, key, value)
                
                # CRITICAL: Ensure fit flags are boolean, not stored as 0/1
                fit_flag_keys = ['fit_xi_params', 'fit_disk_thin', 'fit_disk_thick', 
                                'fit_bulge', 'fit_gas', 'fit_rho_c', 'fit_gamma', 
                                'fit_lambda_g', 'fit_disk_reparameterized']
                for flag in fit_flag_keys:
                    if hasattr(args, flag):
                        # Convert to bool if needed
                        setattr(args, flag, bool(getattr(args, flag)))
                config_restored = True
            else:
                # --- Case 2: Old checkpoint, fallback to JSON ---
                logger.warning("⚠️ Checkpoint missing configuration. Attempting to restore from JSON file...")
                
                config_filenames = ["run_config_enhanced.json", "run_config.json"]
                for filename in config_filenames:
                    config_path = Path(args.output_dir) / filename
                    if config_path.exists():
                        try:
                            logger.info(f"   Found configuration file: {config_path}")
                            with open(config_path, 'r') as f_json:
                                saved_config_data = json.load(f_json)

                            # Use .get() for safety, falling back to the top-level dict
                            saved_args = saved_config_data.get('all_parameters', saved_config_data)
                            
                            logger.info("   [DEBUG] Applying saved arguments from JSON...")
                            for key, value in saved_args.items():
                                if key != 'resume': # Don't override the intent to resume
                                    # Ensure boolean flags are properly typed
                                    if key in ['fit_xi_params', 'fit_disk_thin', 'fit_disk_thick', 
                                            'fit_bulge', 'fit_gas', 'fit_rho_c', 'fit_gamma', 
                                            'fit_lambda_g', 'fit_disk_reparameterized']:
                                        setattr(args, key, bool(value))
                                    else:
                                        setattr(args, key, value)
                            
                            logger.info(f"✅ Successfully restored run configuration from {filename}.")
                            config_restored = True
                            break # Stop after finding the first valid config
                        
                        except Exception as e:
                            logger.error(f"   Error reading or parsing {config_path}: {e}")
                            continue
            
            # --- DIAGNOSTIC CHECK ---
            if config_restored:
                logger.info("   [DIAGNOSTIC] Final state of 'fit' flags before proceeding:")
                logger.info(f"   - args.fit_xi_params: {getattr(args, 'fit_xi_params', 'MISSING')}")
                logger.info(f"   - args.fit_bulge: {getattr(args, 'fit_bulge', 'MISSING')}")
                logger.info(f"   - args.fit_gas: {getattr(args, 'fit_gas', 'MISSING')}")
                logger.info(f"   - args.fit_disk_reparameterized: {getattr(args, 'fit_disk_reparameterized', 'MISSING')}")
            else:
                logger.error("❌ CRITICAL: Could not find configuration in checkpoint or JSON file.")
                logger.error("   Please provide the full original command line flags to resume this run.")
                sys.exit(1)

        except Exception as e:
            logger.error(f"❌ Failed to load or process checkpoint file at {checkpoint_path}: {e}")
            sys.exit(1)

    # Safety patch for all_param_info_list (used by plausibility checks)
    if not hasattr(args, 'all_param_info_list') or args.all_param_info_list is None:
        logger.info("Injecting args.all_param_info_list with get_param_labels_and_bounds()")
        get_param_labels_and_bounds(args)

    # Validate Gaia
    if not validate_gaia_data_for_fitting(gaia_data_dict):
        logger.error("❌ Gaia data validation failed.")
        sys.exit(1)

    # Initialize dashboard monitor once (removed duplicate)
    dashboard_monitor = None
    if args.enable_dashboard and not args.disable_dashboard:
        try:
            from monitor_dashboard import DynestyMonitor
            dashboard_monitor = DynestyMonitor(Path(args.output_dir))
            logger.info("Dashboard monitoring initialized")
        except Exception as e:
            logger.warning(f"Dashboard disabled due to error: {e}")
            dashboard_monitor = None

    # Run GR baseline if requested
    if args.run_gr_baseline:
        gr_results = run_gr_baseline_fixed(args, gaia_data_dict, R_data_jax, 
                                        v_data_jax, sigma_data_jax)

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
    results = None
    stage_args_per_stage = {}
    try:
        if args.use_curriculum_learning:
            results, stage_args_per_stage = run_curriculum_learning(args, gaia_data_dict, logger, R_data_jax, v_data_jax, sigma_data_jax, dashboard_monitor)
        else:
            results = run_single_dynesty(args, gaia_data_dict, R_data_jax, v_data_jax, sigma_data_jax, gp_surrogate=gp_surrogate, dashboard_monitor=dashboard_monitor)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user")
        
        # Try to get partial stats if available
        partial_stats = {}
        n_samples_partial = 0
        n_calls_partial = 0
        
        if 'results' in locals() and hasattr(results, 'samples'):
            n_samples_partial = len(results.samples)
            if hasattr(results, 'ncall'):
                n_calls_partial = int(np.sum(results.ncall))
        
        finalize_record(RUN_ID, success=False,
                        logz=np.nan, logz_err=np.nan,
                        eff=0.0, rmse=np.nan,
                        n_samples=n_samples_partial, 
                        n_calls=n_calls_partial,
                        param_stats=partial_stats, 
                        phys_ok=False,
                        phys_reason="User interrupted")
        raise  # Re-raise to properly exit
    except Exception as e:
        logger.error(f"Sampling failed with a critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        finalize_record(RUN_ID, success=False, phys_reason=f"Exception: {str(e)[:200]}")
        raise  # Re-raise to properly exit

    # =================================================================================
    # FINAL RESULTS PROCESSING AND SAVING
    # This block determines the final results object and then safely processes it.
    # =================================================================================

    # First, determine the final result object ('res') and the correct parameter names
    res = None
    fitted_p_names = []

    if isinstance(results, dict) and 'stage_1' in results:
        # --- Handle Curriculum Learning Results ---
        completed_stages = [k for k, v in results.items() if v is not None and hasattr(v, 'samples') and len(v.samples) > 0]
        if not completed_stages:
            logger.error("❌ All curriculum stages failed or produced no samples. No results to save.")
            finalize_record(RUN_ID, success=False, phys_reason="All curriculum stages failed")
            return
        else:
            logger.info(f"✅ Curriculum learning completed {len(completed_stages)} stages successfully.")
            final_stage_name = max(completed_stages)
            res = results[final_stage_name]
            final_stage_args = stage_args_per_stage[final_stage_name]
            fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(final_stage_args)
    elif results is not None:
        # --- Handle Single Run Results ---
        res = results
        fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(args)
    else:
        logger.error("No valid results were produced to save.")
        finalize_record(RUN_ID, success=False,
                        logz=np.nan, logz_err=np.nan,
                        eff=0.0, rmse=np.nan,
                        n_samples=0, n_calls=0,
                        param_stats={}, phys_ok=False,
                        phys_reason="No results to save")
        return


    # =================================================================================
    # MASTER GUARD CLAUSE: Only proceed if 'res' is a valid, populated results object.
    # =================================================================================
    if res and hasattr(res, 'samples') and len(res.samples) >= 10:
        # Add imports here to guarantee they are in scope, preventing the UnboundLocalError
        import pickle
        import gzip

        logger.info("\n✨ Run complete! Finalizing and saving results...")

        # --- 1. Define Output Filenames ---
        output_parts = ["dynesty_mw", args.xi]
        if args.include_bulge:      output_parts.append("B" + ("f" if args.fit_bulge else "x"))
        if args.include_disk_thin:  output_parts.append("DT" + ("f" if args.fit_disk_thin else "x"))
        if args.include_disk_thick: output_parts.append("DK" + ("f" if args.fit_disk_thick else "x"))
        if args.include_gas:        output_parts.append("G" + ("f" if args.fit_gas else "x"))
        output_basename = "_".join(output_parts)

        # --- 2. Save Main NPZ Results File ---
        output_npz = Path(args.output_dir) / f"{output_basename}_samples.npz"
        try:
            weights = np.exp(res.logwt - res.logz[-1])
            ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0
            np.savez(
                output_npz, samples=res.samples, weights=weights,
                param_names=np.array(fitted_p_names), logl=res.logl,
                logz=res.logz, logzerr=res.logzerr, ess=ess,
                blob=getattr(res, 'blob', None),
            )
            logger.info(f"✅ Final results saved to {output_npz}")
        except Exception as e:
            logger.error(f"❌ Failed to save final .npz results: {e}")

        # --- 3. Save Full Results Object via Pickle ---
        output_pkl = Path(args.output_dir) / f"{output_basename}_results.pkl.gz"
        try:
            with gzip.open(output_pkl, "wb") as fh:
                pickle.dump(res, fh)
            logger.info(f"✅ Full results object saved to {output_pkl}")
        except Exception as e:
            logger.error(f"❌ Failed to save final pickle file: {e}")

        # --- 4. Finalize Run Record and Create Summary ---
        try:
            median_params = np.median(res.samples, axis=0)
            std_params = np.std(res.samples, axis=0)
            param_stats = {name: {"median": float(med), "sigma": float(std)}
                           for name, med, std in zip(fitted_p_names, median_params, std_params)}
            is_valid, reason, *_ = check_physical_plausibility(median_params, fitted_p_names, args)
            logz = res.logz[-1] if hasattr(res, 'logz') and len(res.logz) > 0 else np.nan
            logz_err = res.logzerr[-1] if hasattr(res, 'logzerr') and len(res.logzerr) > 0 else np.nan
            eff = float(getattr(res, 'eff', 0.0)) * 100
            rmse = float(np.sqrt(np.mean(res.blob**2))) if hasattr(res, 'blob') and res.blob is not None else np.nan
            delta_logz_vs_gr = logz - BASELINE_LOGZ_GR if np.isfinite(logz) else np.nan
            jeffreys_interp = interpret_jeffreys_scale(delta_logz_vs_gr) if np.isfinite(delta_logz_vs_gr) else "Unavailable"
            
            logger.info(f"\n📊 Model Comparison:")
            logger.info(f"   GR Baseline logZ: {BASELINE_LOGZ_GR:.2f}")
            logger.info(f"   This run logZ:    {logz:.2f} (err: ±{logz_err:.2f})")
            logger.info(f"   ΔlogZ vs GR:      {delta_logz_vs_gr:+.2f} → {jeffreys_interp}")

            finalize_record(RUN_ID, success=True, logz=logz, logz_err=logz_err, eff=eff, rmse=rmse,
                            n_samples=len(res.samples), n_calls=int(np.sum(res.ncall)),
                            param_stats=param_stats, phys_ok=is_valid,
                            phys_reason=reason if not is_valid else "OK")
            
            snapshot_path = Path(args.output_dir) / f"run_{RUN_ID}_summary.json"
            with open(snapshot_path, "w") as fh:
                json.dump(make_json_serializable({
                    "run_id": RUN_ID, "success": True, "logZ": logz, "logZ_err": logz_err,
                    "delta_logZ_vs_GR": delta_logz_vs_gr, "jeffreys_interpretation": jeffreys_interp,
                    "params": param_stats, "phys_ok": is_valid,
                    "phys_reason": reason if not is_valid else "OK", "n_samples": len(res.samples),
                    "rmse_kms": rmse, "cmd": " ".join(sys.argv)
                }), fh, indent=2)
            logger.info(f"📄 Run summary snapshot saved to {snapshot_path}")
            return
        except Exception as e:
            logger.error(f"❌ An error occurred during the final analysis and reporting step: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
    if args.compare_to_gr:
        gr_baseline_npz = Path(args.gr_baseline_dir) / "dynesty_mw_gr_samples.npz"
        if res and hasattr(res, 'samples'):
            analyze_model_vs_gr(res, gr_baseline_npz, args, gaia_data_dict)
    
    else:
        # This block executes if the run failed or was interrupted.
        logger.error("\n==========================================================")
        logger.error("❌ RUN FAILED: No valid results were produced to save.")
        logger.error("   Check the log file for a 'CRITICAL: Failed to create Dynesty sampler!' message.")
        logger.error("==========================================================")
        finalize_record(RUN_ID, success=False, phys_reason="No valid samples produced")
        return


if __name__ == "__main__":
    print("DEBUG: Entering main block")
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL: Dynesty library not found")
        sys.exit(1)
    print("DEBUG: About to call main_dynesty()")
    main_dynesty()
    print("DEBUG: main_dynesty() completed")