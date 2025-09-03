#!/usr/bin/env python3
"""
run_dynesty_stellar_fit_cupy.py - CuPy version of stellar-focused fitting.

This version uses CuPy for GPU acceleration instead of JAX, providing better
GPU utilization on NVIDIA hardware. Focuses on minimizing chi-squared to
stellar velocity data rather than maximizing Bayesian evidence.
"""

import logging
import sys
import time
import numpy as np
import cupy as cp
from datetime import datetime
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add repository root (two levels up from 'dynesty_latest') to path reliably
try:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
except Exception:
    # Fallback to previous behavior
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CuPy model functions
try:
    from core.density_metric_cupy import (
        v_baryon_total_newtonian_kms_cupy,
        volume_density_total_midplane_solar_kpc3_cupy,
        xi_power_law_cupy,
        xi_exponential_cupy,
        xi_gravitational_color_cupy,
        xi_logistic_law_cupy,
        xi_gaussian_enhancement_cupy,
        xi_mond_like_cupy,
        DEFAULT_DTYPE
    )
    logger.info("✓ CuPy physics module imported successfully")
except ImportError as e:
    logger.error(f"✗ Cannot import CuPy physics module: {e}")
    sys.exit(1)

# Import Dynesty
try:
    from dynesty import DynamicNestedSampler, utils as dyfunc
    logger.info("✓ Dynesty imported successfully")
except ImportError:
    logger.error("✗ Dynesty not found. Install with: pip install dynesty")
    sys.exit(1)

# Physical constants
G_ASTRO_UNITS = 4.30091e-6  # kpc (km/s)^2 / Msun

# Physical bounds for parameters
PHYSICAL_BOUNDS = {
    'M_disk_thin_solar':   {'min': 2.4e10, 'max': 9.0e10, 'typical': 4.0e10},
    'M_disk_thick_solar':  {'min': 5e9,    'max': 3.5e10, 'typical': 1.5e10},
    'M_bulge_solar':       {'min': 0.5e10, 'max': 2.5e10, 'typical': 1.2e10},
    'M_gas_solar':         {'min': 5e9,    'max': 6e10,   'typical': 3.0e10},
    'R_d_thin_kpc':        {'min': 2.0,    'max': 4.5,    'typical': 2.6},
    'R_d_thick_kpc':       {'min': 3.5,    'max': 9.5,    'typical': 4.5},
    'R_d_gas_kpc':         {'min': 4.0,    'max': 15.0,   'typical': 7.0},
    'a_bulge_kpc':         {'min': 0.2,    'max': 2.0,    'typical': 0.7},
    'h_z_thin_kpc':        {'min': 0.15,   'max': 0.5,    'typical': 0.3},
    'h_z_thick_kpc':       {'min': 0.7,    'max': 1.5,    'typical': 0.9},
    'h_z_gas_kpc':         {'min': 0.05,   'max': 0.4,    'typical': 0.15},
    'rho_c_solar_kpc3':    {'min': 1e6,    'max': 1e16,   'typical': 5e8},
    'n_exp':               {'min': 0.5,    'max': 4.0,    'typical': 2.7},
}

# Define a custom v_total function that uses xi functions directly
def v_total_kms_cupy(R_kpc, params, xi_type='power', allow_experimental=False):
    """
    Calculate total velocity including xi modification using CuPy.
    """
    R_gpu = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    
    # Calculate Newtonian velocity - use dictionary format
    baryon_params = {
        'M_disk_solar': params.get('M_disk_solar', params.get('M_disk_thin_solar', 4.0e10)),
        'R_d_kpc': params.get('R_d_kpc', params.get('R_d_thin_kpc', 2.6)),
        'M_bulge_solar': params.get('M_bulge_solar', 1.2e10),
        'R_b_kpc': params.get('R_b_kpc', params.get('a_bulge_kpc', 0.7)),
        'include_bulge': params.get('include_bulge', True),
        'M_gas_solar': params.get('M_gas_solar', 3.0e10),
        'R_gas_kpc': params.get('R_gas_kpc', params.get('R_d_gas_kpc', 7.0)),
        'include_gas': params.get('include_gas', True)
    }
    v_newton = v_baryon_total_newtonian_kms_cupy(R_gpu, baryon_params)
    
    # Calculate density
    rho = volume_density_total_midplane_solar_kpc3_cupy(
        R_gpu,
        params.get('M_disk_solar', params.get('M_disk_thin_solar', 4.0e10)),
        params.get('R_d_kpc', params.get('R_d_thin_kpc', 2.6)),
        params.get('hz_disk_kpc', params.get('h_z_thin_kpc', 0.3)),
        params.get('M_bulge_solar', 1.2e10),
        params.get('R_b_kpc', params.get('a_bulge_kpc', 0.7)),
        params.get('include_bulge', True),
        params.get('M_gas_solar', 3.0e10),
        params.get('R_gas_kpc', params.get('R_d_gas_kpc', 7.0)),
        params.get('hz_gas_kpc', params.get('h_z_gas_kpc', 0.15)),
        params.get('include_gas', True)
    )
    
    # Get xi parameters
    rho_c = params.get('rho_c_solar_kpc3', params.get('rho_c', 1e8))
    
    # Calculate xi based on type
    if xi_type == 'gr':
        xi = cp.ones_like(R_gpu)
    elif xi_type == 'nfw':
        # NFW uses GR xi but adds dark matter halo
        xi = cp.ones_like(R_gpu)
        # Add NFW contribution if parameters present
        if 'M_vir' in params:
            M_vir = cp.asarray(params['M_vir'], dtype=DEFAULT_DTYPE)
            c_vir = cp.asarray(params.get('c_vir', 12.0), dtype=DEFAULT_DTYPE)
            # Literature-consistent NFW implementation
            # R_vir where mean density = 200 * rho_crit; rho_crit ~ 100 Msun/kpc^3
            rho_crit = cp.asarray(100.0, dtype=DEFAULT_DTYPE)  # Msun/kpc^3 (approx)
            R_vir = cp.power(M_vir / (200.0 * rho_crit * (4.0 * cp.pi / 3.0)), 1.0/3.0)
            r_s = R_vir / c_vir
            x = cp.maximum(R_gpu / cp.maximum(r_s, cp.asarray(1e-6, dtype=DEFAULT_DTYPE)), 1e-8)
            # Enclosed mass ratio functions
            g_x = cp.log1p(x) - x / (1.0 + x)
            g_c = cp.log1p(c_vir) - c_vir / (1.0 + c_vir)
            g_c = cp.maximum(g_c, cp.asarray(1e-12, dtype=DEFAULT_DTYPE))
            M_enc = M_vir * g_x / g_c
            v_nfw_sq = G_ASTRO_UNITS * M_enc / cp.maximum(R_gpu, cp.asarray(1e-6, dtype=DEFAULT_DTYPE))
            v_total_sq = v_newton**2 * xi + cp.maximum(v_nfw_sq, 0.0)
            return cp.sqrt(cp.maximum(v_total_sq, 0.0))
    elif xi_type == 'power':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_power_law_cupy(rho, rho_c, n_exp, A)
    elif xi_type == 'exponential':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_exponential_cupy(rho, rho_c, n_exp, A)
    elif xi_type == 'grav_color':
        gamma = params.get('gamma_exp', params.get('gamma', 2.7))
        lambda_g = params.get('lambda_g', 8.0)
        xi = xi_gravitational_color_cupy(rho, rho_c, gamma, lambda_g)
    elif xi_type == 'logistic':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_logistic_law_cupy(rho, rho_c, n_exp, A)
    elif xi_type == 'gaussian':
        sigma_log = params.get('sigma_log', 1.0)
        A = params.get('A', 1.0)
        xi = xi_gaussian_enhancement_cupy(rho, rho_c, sigma_log, A)
    elif xi_type == 'mond':
        n_exp = params.get('n_exp', 2.0)
        xi = xi_mond_like_cupy(rho, rho_c, n_exp)
    elif xi_type in ['tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor', 'rar_gate', 'rar_blend', 'rar_plateau']:
        # For tidal/RAR models, use the comprehensive function from density_metric_cupy
        from core.density_metric_cupy import v_total_kms_cupy as v_total_comprehensive
        # Pass experimental flag via params dict as expected by core
        if allow_experimental:
            params = dict(params)
            params['allow_experimental'] = True
        return v_total_comprehensive(R_kpc, params, xi_type=xi_type)
    else:
        # Default to GR
        xi = cp.ones_like(R_gpu)
    
    # Apply modification
    v_model = v_newton * cp.sqrt(cp.maximum(xi, 0.0))
    return v_model

def log_likelihood_stellar_cupy(params, R_data_gpu, v_data_gpu, sigma_data_gpu, xi_type='power', allow_experimental=False):
    """
    Stellar-focused likelihood using CuPy for GPU acceleration.
    
    This likelihood prioritizes fitting the observed stellar velocities
    by minimizing chi-squared residuals with regional weighting.
    """
    # Calculate model velocities on GPU
    try:
        v_model_gpu = v_total_kms_cupy(R_data_gpu, params, xi_type, allow_experimental=allow_experimental)
    except Exception as e:
        logger.debug(f"Model evaluation failed: {e}")
        return -np.inf
    
    # Calculate chi-squared on GPU
    residuals_gpu = (v_data_gpu - v_model_gpu) / sigma_data_gpu
    chi2_gpu = cp.sum(residuals_gpu**2)
    
    # Check for extreme values that would cause overflow
    chi2_value = float(chi2_gpu)
    if not np.isfinite(chi2_value) or chi2_value > 1e10:
        return -np.inf
    
    # Base likelihood
    log_L = -0.5 * chi2_value
    
    # Regional breakdown for detailed fitting
    radial_bins = cp.array([0, 3, 5, 7, 8.5, 10, 12, 15, 20, 30], dtype=DEFAULT_DTYPE)
    
    penalties = 0.0
    for i in range(len(radial_bins) - 1):
        r_min, r_max = radial_bins[i], radial_bins[i+1]
        mask = (R_data_gpu >= r_min) & (R_data_gpu < r_max)
        n_stars = int(cp.sum(mask))
        
        if n_stars > 0:
            chi2_region = float(cp.sum(residuals_gpu[mask]**2))
            chi2_per_star = chi2_region / n_stars
            
            # Solar neighborhood needs excellent fit
            if r_min <= 8.5 and r_max >= 7.5:
                if chi2_per_star > 2.0:
                    # Clamp to prevent overflow
                    penalty_val = min(chi2_per_star - 2.0, 100.0)
                    penalties -= 100.0 * penalty_val**2
            
            # Outer regions should have reasonable velocities
            if r_min >= 12:
                v_region = v_model_gpu[mask]
                v_mean = float(cp.mean(v_region))
                if v_mean < 150:
                    penalties -= 50.0 * ((150 - v_mean) / 50)**2
                elif v_mean > 300:
                    penalties -= 50.0 * ((v_mean - 300) / 50)**2
    
    # Shape matching bonus/penalty
    test_radii = cp.array([2.0, 5.0, 8.0, 12.0, 20.0], dtype=DEFAULT_DTYPE)
    test_velocities = []
    for r in test_radii:
        idx = cp.argmin(cp.abs(R_data_gpu - r))
        test_velocities.append(float(v_model_gpu[idx]))
    
    shape_score = 0.0
    # Check for rising curve in inner galaxy
    if test_velocities[1] > test_velocities[0]:
        shape_score += 10.0
    else:
        shape_score -= 20.0
    
    # Check for reasonable value at solar radius
    solar_v = test_velocities[2]
    if 200 <= solar_v <= 250:
        shape_score += 20.0
    else:
        shape_score -= 30.0 * abs(solar_v - 225) / 225
    
    # Check for flattening in outer galaxy
    outer_gradient = (test_velocities[-1] - test_velocities[-2]) / (float(test_radii[-1]) - float(test_radii[-2]))
    if abs(outer_gradient) < 5.0:
        shape_score += 15.0
    else:
        shape_score -= 10.0 * abs(outer_gradient)
    
    # Calculate RMSE for diagnostics
    n_data = len(R_data_gpu)
    rmse_total = float(cp.sqrt(chi2_gpu / n_data))
    
    # Apply overall penalty for poor fit
    if rmse_total > 30.0:
        # Clamp to prevent overflow
        penalty_val = min((rmse_total - 30.0) / 30.0, 10.0)
        penalties -= 100.0 * penalty_val**2
    
    # Total likelihood
    log_L_total = log_L + shape_score + penalties
    
    return log_L_total

def prior_transform(u, param_bounds):
    """
    Transform unit cube to parameter space.
    
    Parameters:
    -----------
    u : array
        Unit cube values [0, 1]
    param_bounds : list of tuples
        [(min, max), ...] for each parameter
    
    Returns:
    --------
    params : array
        Transformed parameters
    """
    params = np.zeros(len(u))
    for i, (low, high) in enumerate(param_bounds):
        if low > 0 and high/low > 100:  # Log scale for large dynamic range
            log_low = np.log10(low)
            log_high = np.log10(high)
            params[i] = 10**(log_low + u[i] * (log_high - log_low))
        else:  # Linear scale
            params[i] = low + u[i] * (high - low)
    return params

def load_gaia_data(sample_max=10000, use_144k_data=False):
    """
    Load Gaia data from processed or raw sources.
    
    Parameters:
    -----------
    sample_max : int
        Maximum number of stars to sample
    use_144k_data : bool
        If True, prioritize loading the 144k star dataset
    
    Returns:
    --------
    R_data, v_data, sigma_data : numpy arrays
        Radius, velocity, and uncertainty arrays
    """
    # Data paths - prioritize based on use_144k_data flag
    if use_144k_data:
        # Prioritize the 144k dataset
        data_paths = [
            Path("../external_data/gaia_sky_slices/all_sky_gaia.csv"),
            Path("external_data/gaia_sky_slices/all_sky_gaia.csv"),
            Path("../gaia_query_cache_DR3_processed_for_fit.parquet"),
            Path("gaia_query_cache_DR3_processed_for_fit.parquet")
        ]
    else:
        # Default: prioritize smaller processed dataset
        data_paths = [
            Path("../gaia_query_cache_DR3_processed_for_fit.parquet"),
            Path("gaia_query_cache_DR3_processed_for_fit.parquet"),
            Path("../external_data/gaia_sky_slices/all_sky_gaia.csv"),
            Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
        ]
    
    for path in data_paths:
        if path.exists():
            try:
                import pandas as pd
                
                # Handle different file formats
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                    logger.info(f"Loading parquet file: {path}")
                else:
                    df = pd.read_csv(path)
                    logger.info(f"Loading CSV file: {path}")
                
                # Log available columns
                logger.info(f"Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                
                # Check for various possible column names
                r_col = None
                v_col = None
                sigma_col = None
                
                # Radius column
                for col in ['R_kpc', 'r_kpc', 'R', 'radius', 'galactocentric_distance']:
                    if col in df.columns:
                        r_col = col
                        break
                
                # Velocity column
                for col in ['v_obs', 'v_circ', 'v_rot', 'velocity', 'v_gsr', 'circular_velocity']:
                    if col in df.columns:
                        v_col = col
                        break
                
                # Uncertainty column
                for col in ['sigma_v', 'v_err', 'velocity_error', 'sigma', 'v_gsr_error']:
                    if col in df.columns:
                        sigma_col = col
                        break
                
                if r_col and v_col:
                    R_data = df[r_col].values
                    v_data = df[v_col].values
                    
                    # Handle uncertainties
                    if sigma_col:
                        sigma_data = df[sigma_col].values
                    else:
                        # Use 10% of velocity as default uncertainty if not provided
                        sigma_data = np.maximum(np.abs(v_data) * 0.1, 10.0)
                    
                    # Clean data
                    mask = np.isfinite(R_data) & np.isfinite(v_data) & (R_data > 0) & (R_data < 30)
                    R_data = R_data[mask]
                    v_data = v_data[mask]
                    sigma_data = sigma_data[mask]
                    
                    # Sample if needed
                    if len(R_data) > sample_max:
                        indices = np.random.choice(len(R_data), sample_max, replace=False)
                        R_data = R_data[indices]
                        v_data = v_data[indices]
                        sigma_data = sigma_data[indices]
                    
                    logger.info(f"✓ Loaded {len(R_data)} stars from {path}")
                    return R_data, v_data, sigma_data
                
                # If we don't have R_kpc and v_obs, check if we have raw Gaia data to process
                elif ('parallax' in df.columns and 'pmra' in df.columns and 
                      'pmdec' in df.columns and 'radial_velocity' in df.columns):
                    logger.info("Raw Gaia data detected - processing to galactocentric coordinates...")
                    
                    # Import processing function
                    try:
                        from core.data_io import process_gaia_data
                        
                        # Process the raw data
                        df_processed = process_gaia_data(df)
                        
                        if 'R_kpc' in df_processed.columns and 'v_obs' in df_processed.columns:
                            R_data = df_processed['R_kpc'].values
                            v_data = df_processed['v_obs'].values
                            sigma_data = df_processed.get('sigma_v', np.ones_like(v_data) * 10).values
                            
                            # Clean data
                            mask = np.isfinite(R_data) & np.isfinite(v_data) & (R_data > 0) & (R_data < 30)
                            R_data = R_data[mask]
                            v_data = v_data[mask]
                            sigma_data = sigma_data[mask]
                            
                            # Sample if needed
                            if len(R_data) > sample_max:
                                indices = np.random.choice(len(R_data), sample_max, replace=False)
                                R_data = R_data[indices]
                                v_data = v_data[indices]
                                sigma_data = sigma_data[indices]
                            
                            logger.info(f"✓ Processed and loaded {len(R_data)} stars from raw Gaia data")
                            return R_data, v_data, sigma_data
                    except Exception as proc_e:
                        logger.warning(f"Could not process raw Gaia data: {proc_e}")
                
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
    
    # Create mock data for testing
    logger.warning("No real data found, creating mock data for testing")
    np.random.seed(42)
    n_stars = min(1000, sample_max)
    R_data = np.random.uniform(2, 25, n_stars)
    
    # Mock rotation curve
    v_true = 220 * np.sqrt(1 - np.exp(-R_data/3))
    sigma_data = np.ones(n_stars) * 10
    v_data = v_true + np.random.normal(0, sigma_data)
    
    logger.info(f"✓ Created {n_stars} mock stars")
    return R_data, v_data, sigma_data

def run_stellar_fit_cupy(args):
    """
    Main function to run stellar-focused fitting with CuPy.
    """
    logger.info("="*80)
    logger.info("STELLAR-FOCUSED FITTING WITH CUPY")
    logger.info("="*80)
    logger.info(f"Xi type: {args.xi}")
    logger.info(f"Max iterations: {args.maxcall}")
    
    # Check CuPy GPU
    try:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        logger.info(f"✓ GPU: {gpu_props['name'].decode()} ({gpu_props['totalGlobalMem']/1e9:.1f} GB)")
    except:
        logger.warning("⚠ GPU info not available")
    
    # Load data
    R_data, v_data, sigma_data = load_gaia_data(args.sample_max, use_144k_data=args.use_144k)
    
    # Transfer to GPU
    R_data_gpu = cp.asarray(R_data, dtype=DEFAULT_DTYPE)
    v_data_gpu = cp.asarray(v_data, dtype=DEFAULT_DTYPE)
    sigma_data_gpu = cp.asarray(sigma_data, dtype=DEFAULT_DTYPE)
    
    logger.info(f"✓ Data transferred to GPU: {len(R_data)} stars")
    logger.info(f"  R range: {R_data.min():.1f} - {R_data.max():.1f} kpc")
    logger.info(f"  <v>: {v_data.mean():.1f} km/s")
    
    # Set up parameters to fit based on xi type
    if args.xi == 'gr':
        # GR baseline - need at least one dummy parameter for Dynesty
        param_names = ['dummy']
        param_bounds = [(0.99, 1.01)]  # Dummy parameter that doesn't affect results
    elif args.xi == 'nfw':
        # NFW dark matter halo
        param_names = ['M_vir', 'c_vir']
        # Literature-informed priors (MW-like)
        param_bounds = [
            (3e11, 2e12),  # Virial mass in M_sun
            (6.0, 20.0)    # Concentration parameter
        ]
    elif args.xi == 'grav_color':
        param_names = ['rho_c_solar_kpc3', 'gamma_exp', 'lambda_g']
        param_bounds = [
            (PHYSICAL_BOUNDS['rho_c_solar_kpc3']['min'], PHYSICAL_BOUNDS['rho_c_solar_kpc3']['max']),
            (0.5, 4.0),  # gamma
            (0.1, 10.0)  # lambda_g
        ]
    elif args.xi == 'gaussian':
        # Gaussian enhancement in log-density space
        param_names = ['rho_c_solar_kpc3', 'sigma_log', 'A']
        param_bounds = [
            (PHYSICAL_BOUNDS['rho_c_solar_kpc3']['min'], PHYSICAL_BOUNDS['rho_c_solar_kpc3']['max']),
            (0.3, 2.0),  # sigma_log (log10-width)
            (0.1, 5.0)   # amplitude
        ]
    elif args.xi == 'mond':
        # MOND-like modification uses only (rho_c, n_exp)
        param_names = ['rho_c_solar_kpc3', 'n_exp']
        param_bounds = [
            (PHYSICAL_BOUNDS['rho_c_solar_kpc3']['min'], PHYSICAL_BOUNDS['rho_c_solar_kpc3']['max']),
            (PHYSICAL_BOUNDS['n_exp']['min'], PHYSICAL_BOUNDS['n_exp']['max'])
        ]
    elif args.xi in ['tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor']:
        # Tidal models
        param_names = ['rho_c', 'lambda_max', 'T0', 'sigma_lnT', 'wmin']
        param_bounds = [
            (1e6, 1e9),   # rho_c in M_sun/kpc^3
            (0.5, 10.0),  # lambda_max
            (1.0, 100.0), # T0 in (km/s)^2/kpc^2  
            (0.1, 2.0),   # sigma_lnT
            (0.0, 0.1)    # wmin
        ]
        if args.xi == 'tidal_band':
            param_names.insert(1, 'gamma')
            param_bounds.insert(1, (1.0, 5.0))  # gamma
        elif args.xi == 'tidal_band2':
            param_names.extend(['gamma', 'beta', 'alpha', 'kappa'])
            param_bounds.extend([
                (1.0, 5.0),   # gamma
                (0.1, 2.0),   # beta
                (0.5, 5.0),   # alpha
                (0.1, 3.0)    # kappa
            ])
        elif args.xi == 'tidal_ratio':
            param_names.extend(['eta', 'alpha', 'kappa'])
            param_bounds.extend([
                (0.1, 2.0),   # eta
                (0.5, 5.0),   # alpha
                (0.1, 3.0)    # kappa
            ])
        elif args.xi == 'tidal_noisyor':
            param_names.extend(['gamma', 'alpha', 'kappa'])
            param_bounds.extend([
                (1.0, 5.0),   # gamma
                (0.5, 5.0),   # alpha
                (0.1, 3.0)    # kappa
            ])
    elif args.xi in ['rar_gate', 'rar_blend']:
        # RAR models
        param_names = ['a0_m_s2', 'lambda_max', 'T0', 'sigma_lnT', 'wmin']
        param_bounds = [
            (0.5e-10, 3e-10),  # a0 in m/s^2
            (0.5, 10.0),       # lambda_max
            (1.0, 100.0),      # T0
            (0.1, 2.0),        # sigma_lnT
            (0.0, 0.1)         # wmin
        ]
        if args.xi == 'rar_gate':
            param_names.insert(1, 'gamma_exp')
            param_bounds.insert(1, (1.0, 5.0))  # gamma_exp
        elif args.xi == 'rar_blend':
            param_names[1] = 'A_excess'
            param_bounds[1] = (0.1, 3.0)  # A_excess
            param_names.insert(2, 'lambda_cap')
            param_bounds.insert(2, (1.0, 10.0))  # lambda_cap
    elif args.xi == 'rar_plateau':
        # Acceleration-based RAR plateau: fit only a0 by default
        param_names = ['a0_m_s2']
        param_bounds = [
            (6e-11, 3e-10)   # a0 in m/s^2 (log-uniform handled by prior_transform)
        ]
    else:
        # Default for other xi types (power, exponential, etc.)
        param_names = ['rho_c_solar_kpc3', 'n_exp', 'A']
        param_bounds = [
            (PHYSICAL_BOUNDS['rho_c_solar_kpc3']['min'], PHYSICAL_BOUNDS['rho_c_solar_kpc3']['max']),
            (PHYSICAL_BOUNDS['n_exp']['min'], PHYSICAL_BOUNDS['n_exp']['max']),
            (0.1, 5.0)  # A parameter
        ]
    
    # Add baryonic parameters if fitting them
    if args.fit_baryons:
        param_names.extend(['M_disk_thin_solar', 'R_d_thin_kpc'])
        param_bounds.extend([
            (PHYSICAL_BOUNDS['M_disk_thin_solar']['min'], PHYSICAL_BOUNDS['M_disk_thin_solar']['max']),
            (PHYSICAL_BOUNDS['R_d_thin_kpc']['min'], PHYSICAL_BOUNDS['R_d_thin_kpc']['max'])
        ])
    
    ndim = len(param_names)
    logger.info(f"\nFitting {ndim} parameters: {param_names}")
    
    # Fixed baryonic parameters
    fixed_params = {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True
    }
    
    # Define likelihood wrapper
    def log_likelihood(theta):
        params = fixed_params.copy()
        for i, name in enumerate(param_names):
            params[name] = theta[i]
        return log_likelihood_stellar_cupy(params, R_data_gpu, v_data_gpu, sigma_data_gpu, args.xi, 
                                          allow_experimental=args.allow_experimental)
    
    # Define prior wrapper
    def prior_func(u):
        return prior_transform(u, param_bounds)
    
    # Initialize sampler
    nlive = max(args.nlive, 25 * ndim)
    logger.info(f"\nInitializing sampler with {nlive} live points...")
    
    sampler = DynamicNestedSampler(
        log_likelihood,
        prior_func,
        ndim,
        nlive=nlive,
        bound='multi',
        sample='auto',
        bootstrap=0
    )
    
    # Run sampling
    logger.info("\nStarting nested sampling...")
    start_time = time.time()
    
    try:
        # Dynamic sampler uses different API
        sampler.run_nested(
            nlive_init=nlive,
            maxcall=args.maxcall,
            print_progress=args.verbose,
            dlogz_init=0.5
        )
    except KeyboardInterrupt:
        logger.info("\nSampling interrupted by user")
    
    elapsed = time.time() - start_time
    logger.info(f"\nSampling completed in {elapsed:.1f} seconds")
    
    # Extract results
    results = sampler.results
    
    # Get best fit (maximum likelihood)
    max_like_idx = np.argmax(results.logl)
    best_params = results.samples[max_like_idx]
    best_logl = results.logl[max_like_idx]
    
    # Create full parameter set for final evaluation
    best_params_dict = fixed_params.copy()
    for i, name in enumerate(param_names):
        best_params_dict[name] = best_params[i]
    
    # Calculate final chi-squared
    v_model_gpu = v_total_kms_cupy(R_data_gpu, best_params_dict, args.xi, 
                                   allow_experimental=args.allow_experimental)
    chi2_gpu = cp.sum(((v_data_gpu - v_model_gpu) / sigma_data_gpu)**2)
    chi2_total = float(chi2_gpu)
    # Dimensionless per-star chi (sqrt of reduced chi^2 with dof ~ N)
    chi_per_star = np.sqrt(chi2_total / len(R_data))
    # RMSE in km/s (unweighted)
    rmse_kms = float(cp.sqrt(cp.mean((v_data_gpu - v_model_gpu)**2)))
    
    # Results summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"\nBest-fit parameters:")
    for i, name in enumerate(param_names):
        logger.info(f"  {name}: {best_params[i]:.3e}")
    
    logger.info(f"\nFit quality:")
    logger.info(f"  Chi²: {chi2_total:.1f}")
    logger.info(f"  sqrt(Chi²/N): {chi_per_star:.2f} (dimensionless)")
    logger.info(f"  RMSE: {rmse_kms:.1f} km/s")
    logger.info(f"  Log(L): {best_logl:.1f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"stellar_fit_cupy_{args.xi}_results.npz"
    np.savez(
        output_file,
        samples=results.samples,
        weights=np.exp(results.logwt - results.logz[-1]),
        logz=results.logz,
        logl=results.logl,
        best_params=best_params,
        param_names=param_names,
        chi2=chi2_total,
        chi_per_star=chi_per_star,
        rmse_kms=rmse_kms
    )
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Create diagnostic plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Convert GPU arrays back to CPU for plotting
            v_model_cpu = cp.asnumpy(v_model_gpu)
            
            # Main plot
            ax1.scatter(R_data, v_data, c='k', s=1, alpha=0.3, label='Data')
            ax1.scatter(R_data, v_model_cpu, c='r', s=1, alpha=0.5, label=f'Model ({args.xi})')
            ax1.set_ylabel('Velocity (km/s)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'Stellar Fit - χ² = {chi2_total:.1f}, RMSE = {rmse_kms:.1f} km/s')
            
            # Residuals
            residuals = v_data - v_model_cpu
            ax2.scatter(R_data, residuals, c='k', s=1, alpha=0.3)
            ax2.axhline(0, color='r', linestyle='--')
            ax2.axhline(rmse_kms, color='b', linestyle=':', alpha=0.5)
            ax2.axhline(-rmse_kms, color='b', linestyle=':', alpha=0.5)
            ax2.set_xlabel('R (kpc)')
            ax2.set_ylabel('Residuals (km/s)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"stellar_fit_cupy_{args.xi}_plot.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
            
            logger.info(f"Plot saved to: {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stellar-focused fitting with CuPy GPU acceleration')
    
    # Model options
    parser.add_argument('--xi', type=str, default='power',
                       choices=['gr', 'nfw', 'power', 'exponential', 'grav_color', 'logistic', 'gaussian', 'mond',
                               'tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor', 'rar_gate', 'rar_blend', 'rar_plateau'],
                       help='Xi function type')
    
    # Sampler options
    parser.add_argument('--nlive', type=int, default=100,
                       help='Number of live points')
    parser.add_argument('--maxcall', type=int, default=10000,
                       help='Maximum likelihood evaluations')
    
    # Data options
    parser.add_argument('--sample_max', type=int, default=5000,
                       help='Maximum number of stars to use')
    parser.add_argument('--use_144k', action='store_true',
                       help='Use the full 144k Gaia dataset (will process raw data if needed)')
    
    # Fitting options
    parser.add_argument('--fit_baryons', action='store_true',
                       help='Also fit baryonic parameters')
    parser.add_argument('--allow_experimental', action='store_true',
                       help='Allow experimental xi models')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='stellar_fit_cupy_results',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Create diagnostic plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Show sampling progress')
    
    args = parser.parse_args()
    
    # Run fitting
    results = run_stellar_fit_cupy(args)
    
    if results is not None:
        logger.info("\n✓ Fitting completed successfully!")
    else:
        logger.error("\n✗ Fitting failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
