#!/usr/bin/env python3
"""
Split-region dynesty analysis for Milky Way rotation curve.

This script implements a sophisticated approach to analyze the Milky Way rotation curve
by splitting the data into two regions:

1. INNER REGION: Where GR works well (stars follow expected gravitational behavior)
2. OUTER REGION: Where GR begins to break down (stars travel too fast for expected gravity)

This allows us to:
- Identify the transition radius where GR behavior changes
- Fit different models to each region
- Compare model performance across regions
- Find unified parameters that work for both scenarios

Usage:
    py run_dynesty_split_regions.py --xi gr --transition_radius 12.0
    py run_dynesty_split_regions.py --xi enhanced --transition_radius 15.0 --analyze_regions
"""

import numpy as np
import cupy as cp
import pandas as pd
import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from multiprocessing import freeze_support

# Add current directory to path
sys.path.append('.')

try:
    from data_io import load_gaia
    DATA_IO_AVAILABLE = True
except ImportError:
    DATA_IO_AVAILABLE = False
    print("Warning: data_io not available")

try:
    from density_metric_cupy import v_total_kms_cupy, v_baryon_comprehensive_kms_cupy, volume_density_comprehensive_solar_kpc3_cupy
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    print("Warning: density_metric_cupy not available")

try:
    import dynesty
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    print("Warning: dynesty not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
R_SUN_KPC = 8.122  # Solar galactocentric distance
DEFAULT_DTYPE = np.float32

class RegionSplitter:
    """Handles splitting Gaia data into inner and outer regions."""
    
    def __init__(self, transition_radius_kpc=12.0):
        """
        Initialize the region splitter.
        
        Parameters:
        -----------
        transition_radius_kpc : float
            Radius in kpc where we transition from inner to outer region
        """
        self.transition_radius = transition_radius_kpc
        logger.info(f"Initialized region splitter with transition radius: {transition_radius_kpc} kpc")
    
    def split_gaia_data(self, gaia_data):
        """
        Split Gaia data into inner and outer regions.
        
        Parameters:
        -----------
        gaia_data : dict
            Dictionary containing Gaia data with 'R_kpc', 'v_obs', 'sigma_v' keys
            
        Returns:
        --------
        dict : Dictionary with 'inner' and 'outer' regions
        """
        if not isinstance(gaia_data, dict) or 'R_kpc' not in gaia_data:
            logger.error("Invalid gaia_data format")
            return None
            
        R_kpc = np.array(gaia_data['R_kpc'])
        v_obs = np.array(gaia_data['v_obs'])
        sigma_v = np.array(gaia_data['sigma_v'])
        
        # Split data based on radius
        inner_mask = R_kpc <= self.transition_radius
        outer_mask = R_kpc > self.transition_radius
        
        inner_data = {
            'R_kpc': R_kpc[inner_mask],
            'v_obs': v_obs[inner_mask],
            'sigma_v': sigma_v[inner_mask]
        }
        
        outer_data = {
            'R_kpc': R_kpc[outer_mask],
            'v_obs': v_obs[outer_mask],
            'sigma_v': sigma_v[outer_mask]
        }
        
        # Add z_kpc if available
        if 'z_kpc' in gaia_data:
            z_kpc = np.array(gaia_data['z_kpc'])
            inner_data['z_kpc'] = z_kpc[inner_mask]
            outer_data['z_kpc'] = z_kpc[outer_mask]
        
        # Add source_id if available
        if 'source_id' in gaia_data:
            source_id = np.array(gaia_data['source_id'])
            inner_data['source_id'] = source_id[inner_mask]
            outer_data['source_id'] = source_id[outer_mask]
        
        n_inner = len(inner_data['R_kpc'])
        n_outer = len(outer_data['R_kpc'])
        
        logger.info(f"Split data: {n_inner} stars in inner region (R ≤ {self.transition_radius} kpc)")
        logger.info(f"           {n_outer} stars in outer region (R > {self.transition_radius} kpc)")
        
        # Calculate region statistics
        self._print_region_stats(inner_data, "INNER")
        self._print_region_stats(outer_data, "OUTER")
        
        return {
            'inner': inner_data,
            'outer': outer_data,
            'transition_radius': self.transition_radius,
            'n_inner': n_inner,
            'n_outer': n_outer
        }
    
    def _print_region_stats(self, region_data, region_name):
        """Print statistics for a region."""
        R_kpc = region_data['R_kpc']
        v_obs = region_data['v_obs']
        
        logger.info(f"{region_name} REGION STATS:")
        logger.info(f"  Radius range: {R_kpc.min():.2f} - {R_kpc.max():.2f} kpc")
        logger.info(f"  Velocity range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
        logger.info(f"  Mean velocity: {v_obs.mean():.1f} ± {v_obs.std():.1f} km/s")
        logger.info(f"  Median velocity: {np.median(v_obs):.1f} km/s")

def setup_parameter_bounds(xi_type):
    """Setup parameter bounds for the split-region analysis."""
    if xi_type == 'gr':
        # Full baryonic model (11 parameters)
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc',
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc'
        ]
        bounds_low = np.array([1e10, 2.0, 0.2, 1e9, 3.0, 0.6, 1e9, 0.5, 1e9, 5.0, 0.1])
        bounds_high = np.array([1e11, 4.0, 0.4, 1e10, 5.0, 1.0, 1e10, 2.0, 1e10, 10.0, 0.3])
        use_log_prior = np.array([True, False, False, True, False, False, True, False, True, False, False])
    elif xi_type == 'enhanced':
        # Enhanced model with dark matter parameters
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc',
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g'
        ]
        bounds_low = np.array([1e10, 2.0, 0.2, 1e9, 3.0, 0.6, 1e9, 0.5, 1e9, 5.0, 0.1, 1e7, 0.5, 0.1])
        bounds_high = np.array([1e11, 4.0, 0.4, 1e10, 5.0, 1.0, 1e10, 2.0, 1e10, 10.0, 0.3, 1e9, 2.0, 10.0])
        use_log_prior = np.array([True, False, False, True, False, False, True, False, True, False, False, True, False, True])
    elif xi_type == 'power':
        # Power law model
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc',
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'n_exp', 'A'
        ]
        bounds_low = np.array([1e10, 2.0, 0.2, 1e9, 3.0, 0.6, 1e9, 0.5, 1e9, 5.0, 0.1, 1e7, 0.5, 0.1])
        bounds_high = np.array([1e11, 4.0, 0.4, 1e10, 5.0, 1.0, 1e10, 2.0, 1e10, 10.0, 0.3, 1e9, 2.0, 10.0])
        use_log_prior = np.array([True, False, False, True, False, False, True, False, True, False, False, True, False, True])
    elif xi_type == 'grav_color':
        # Gravitational color model
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc',
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g'
        ]
        bounds_low = np.array([1e10, 2.0, 0.2, 1e9, 3.0, 0.6, 1e9, 0.5, 1e9, 5.0, 0.1, 1e7, 0.5, 0.1])
        bounds_high = np.array([1e11, 4.0, 0.4, 1e10, 5.0, 1.0, 1e10, 2.0, 1e10, 10.0, 0.3, 1e9, 2.0, 10.0])
        use_log_prior = np.array([True, False, False, True, False, False, True, False, True, False, False, True, False, True])
    else:
        # Fallback to simple model
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e10, 1.0])
        bounds_high = np.array([5e11, 8.0])
        use_log_prior = np.array([True, False])
    
    return param_names, bounds_low, bounds_high, use_log_prior

def prior_transform_dynesty_cupy(u, param_names, bounds_low, bounds_high, use_log_prior):
    """Prior transform with ordered mass generation."""
    theta = np.zeros_like(u)
    
    # Track thin disk mass for ordered mass generation
    thin_mass = None
    
    for i, name in enumerate(param_names):
        if name == 'M_thin_disk_solar':
            # Generate thin disk mass first
            if use_log_prior[i]:
                theta[i] = 10**(np.log10(bounds_low[i]) + u[i] * (np.log10(bounds_high[i]) - np.log10(bounds_low[i])))
            else:
                theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
            thin_mass = theta[i]
        elif name == 'M_thick_disk_solar' and thin_mass is not None:
            # Generate thick disk mass as fraction of thin disk (5-30%)
            theta[i] = thin_mass * (0.05 + 0.25 * u[i])
        elif name == 'M_bulge_solar' and thin_mass is not None:
            # Generate bulge mass as fraction of thin disk (5-20%)
            theta[i] = thin_mass * (0.05 + 0.15 * u[i])
        elif name == 'M_gas_solar':
            # Gas mass independent but reasonable range
            if use_log_prior[i]:
                theta[i] = 10**(np.log10(bounds_low[i]) + u[i] * (np.log10(bounds_high[i]) - np.log10(bounds_low[i])))
            else:
                theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
        else:
            # Standard transformation for all other parameters
            if use_log_prior[i]:
                theta[i] = 10**(np.log10(bounds_low[i]) + u[i] * (np.log10(bounds_high[i]) - np.log10(bounds_low[i])))
            else:
                theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
    
    return theta

def log_likelihood_dynesty_cupy(theta, param_names, args, R_data, v_data, sigma_data):
    """Log-likelihood function with soft prior penalty."""
    if not PHYSICS_AVAILABLE:
        return -np.inf
    
    # Convert to CuPy arrays
    R_data_cupy = cp.asarray(R_data, dtype=DEFAULT_DTYPE)
    v_data_cupy = cp.asarray(v_data, dtype=DEFAULT_DTYPE)
    sigma_data_cupy = cp.asarray(sigma_data, dtype=DEFAULT_DTYPE)
    
    # Calculate model velocity
    try:
        v_model = v_total_kms_cupy(R_data_cupy, dict(zip(param_names, theta)), xi_type=args.xi)
    except Exception as e:
        logger.warning(f"Velocity calculation failed: {e}")
        return -np.inf
    
    # Calculate chi-squared
    chi2 = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
    logl = -0.5 * float(chi2)
    
    # ADD SOFT PRIOR PENALTY for unreasonable mass ratios
    log_prior_penalty = 0.0
    if len(param_names) >= 11:  # Comprehensive model
        thin_disk_idx = param_names.index('M_thin_disk_solar') if 'M_thin_disk_solar' in param_names else -1
        thick_disk_idx = param_names.index('M_thick_disk_solar') if 'M_thick_disk_solar' in param_names else -1
        bulge_idx = param_names.index('M_bulge_solar') if 'M_bulge_solar' in param_names else -1
        
        if thin_disk_idx >= 0 and thick_disk_idx >= 0:
            thin_mass = theta[thin_disk_idx]
            thick_mass = theta[thick_disk_idx]
            if thin_mass < thick_mass:
                log_prior_penalty -= 10.0  # ~e^-10 penalty, not fatal
        
        if thin_disk_idx >= 0 and bulge_idx >= 0:
            thin_mass = theta[thin_disk_idx]
            bulge_mass = theta[bulge_idx]
            if bulge_mass > thin_mass * 0.5:
                log_prior_penalty -= 5.0   # ~e^-5 penalty for very massive bulge
    
    # Apply the soft prior penalty
    logl += log_prior_penalty
    
    return logl

def run_region_analysis(region_data, region_name, xi_type, nlive=1000, maxcall=500000, num_threads=4):
    """Run dynesty analysis on a specific region."""
    logger.info(f"Starting {region_name} region analysis with {xi_type} model...")
    
    # Setup parameter bounds
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(xi_type)
    
    # Create args object
    class Args:
        def __init__(self, xi):
            self.xi = xi
    
    args = Args(xi_type)
    
    # Setup dynesty sampler
    sampler = dynesty.DynamicNestedSampler(
        log_likelihood_dynesty_cupy,
        prior_transform_dynesty_cupy,
        ndim=len(param_names),
        logl_args=(param_names, args, region_data['R_kpc'], region_data['v_obs'], region_data['sigma_v']),
        ptform_args=(param_names, bounds_low, bounds_high, use_log_prior),
        nlive=nlive,
        bound='multi',
        sample='rslice',
        ncores=num_threads
    )
    
    # Run sampling
    sampler.run_nested(
        maxcall=maxcall,
        dlogz_target=0.01
    )
    
    # Get results
    results = sampler.results
    
    # Save results
    output_dir = Path(f"split_region_results/{region_name.lower()}_{xi_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save posterior samples
    np.savez(
        output_dir / "posterior_samples.npz",
        samples=results.samples,
        weights=results.weights,
        logz=results.logz,
        logzerr=results.logzerr,
        param_names=param_names
    )
    
    # Save progress
    progress_data = {
        'region': region_name,
        'xi_type': xi_type,
        'n_samples': len(results.samples),
        'logz': float(results.logz),
        'logzerr': float(results.logzerr),
        'efficiency': float(getattr(results, 'efficiency', 0.0)),
        'ncall': int(results.ncall),
        'param_names': param_names.tolist(),
        'transition_radius': region_data.get('transition_radius', None)
    }
    
    with open(output_dir / "progress.json", 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    logger.info(f"{region_name} region analysis complete. LogZ: {results.logz:.2f} ± {results.logzerr:.2f}")
    
    return {
        'region': region_name,
        'results': results,
        'param_names': param_names,
        'output_dir': output_dir
    }

def analyze_region_comparison(inner_results, outer_results, transition_radius):
    """Analyze and compare results between inner and outer regions."""
    logger.info("Analyzing region comparison...")
    
    # Calculate evidence ratio
    logz_inner = inner_results['results'].logz
    logz_outer = outer_results['results'].logz
    evidence_ratio = np.exp(logz_inner - logz_outer)
    
    # Calculate parameter differences
    inner_samples = inner_results['results'].samples
    outer_samples = outer_results['results'].samples
    param_names = inner_results['param_names']
    
    param_comparison = {}
    for i, name in enumerate(param_names):
        inner_median = np.median(inner_samples[:, i])
        outer_median = np.median(outer_samples[:, i])
        inner_std = np.std(inner_samples[:, i])
        outer_std = np.std(outer_samples[:, i])
        
        param_comparison[name] = {
            'inner_median': float(inner_median),
            'outer_median': float(outer_median),
            'inner_std': float(inner_std),
            'outer_std': float(outer_std),
            'difference': float(outer_median - inner_median),
            'significance': float(abs(outer_median - inner_median) / np.sqrt(inner_std**2 + outer_std**2))
        }
    
    # Save comparison results
    comparison_data = {
        'transition_radius_kpc': transition_radius,
        'evidence_ratio': float(evidence_ratio),
        'logz_inner': float(logz_inner),
        'logz_outer': float(logz_outer),
        'parameter_comparison': param_comparison,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("split_region_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "region_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Print summary
    logger.info("REGION COMPARISON SUMMARY:")
    logger.info(f"Transition radius: {transition_radius} kpc")
    logger.info(f"Evidence ratio (inner/outer): {evidence_ratio:.3f}")
    logger.info(f"LogZ inner: {logz_inner:.2f}")
    logger.info(f"LogZ outer: {logz_outer:.2f}")
    
    # Print significant parameter differences
    logger.info("Significant parameter differences (>2σ):")
    for name, stats in param_comparison.items():
        if stats['significance'] > 2.0:
            logger.info(f"  {name}: {stats['difference']:.2e} ({stats['significance']:.1f}σ)")
    
    return comparison_data

def main():
    """Main function for split-region analysis."""
    parser = argparse.ArgumentParser(description='Split-region dynesty analysis for Milky Way rotation curve')
    parser.add_argument('--xi', type=str, default='gr', choices=['gr', 'power', 'enhanced', 'grav_color'],
                       help='Xi function type')
    parser.add_argument('--transition_radius', type=float, default=12.0,
                       help='Transition radius between inner and outer regions (kpc)')
    parser.add_argument('--nlive', type=int, default=1000,
                       help='Number of live points per region')
    parser.add_argument('--maxcall', type=int, default=500000,
                       help='Maximum likelihood calls per region')
    parser.add_argument('--num_threads', type=int, default=4,
                       help='Number of threads per region')
    parser.add_argument('--analyze_regions', action='store_true',
                       help='Run analysis on both regions')
    parser.add_argument('--compare_regions', action='store_true',
                       help='Compare results between regions')
    parser.add_argument('--max_sample_gaia', type=int, default=50000,
                       help='Maximum number of Gaia stars to use')
    
    args = parser.parse_args()
    
    logger.info("Starting split-region dynesty analysis...")
    logger.info(f"Xi type: {args.xi}")
    logger.info(f"Transition radius: {args.transition_radius} kpc")
    
    # Load Gaia data
    if not DATA_IO_AVAILABLE:
        logger.error("data_io not available")
        return
    
    logger.info("Loading Gaia data...")
    gaia_data = load_gaia(sample_max=args.max_sample_gaia)
    
    if gaia_data is None:
        logger.error("Failed to load Gaia data")
        return
    
    # Split data into regions
    splitter = RegionSplitter(args.transition_radius)
    split_data = splitter.split_gaia_data(gaia_data)
    
    if split_data is None:
        logger.error("Failed to split data into regions")
        return
    
    # Run analysis on regions
    if args.analyze_regions:
        logger.info("Running analysis on both regions...")
        
        # Run inner region analysis
        inner_results = run_region_analysis(
            split_data['inner'], 'INNER', args.xi, 
            args.nlive, args.maxcall, args.num_threads
        )
        
        # Run outer region analysis
        outer_results = run_region_analysis(
            split_data['outer'], 'OUTER', args.xi,
            args.nlive, args.maxcall, args.num_threads
        )
        
        # Compare regions if requested
        if args.compare_regions:
            comparison = analyze_region_comparison(
                inner_results, outer_results, args.transition_radius
            )
            logger.info("Region comparison complete!")
    
    logger.info("Split-region analysis complete!")

if __name__ == "__main__":
    freeze_support()  # For Windows multiprocessing
    main() 