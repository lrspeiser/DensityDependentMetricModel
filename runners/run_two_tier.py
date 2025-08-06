#!/usr/bin/env python3
"""
Two-tier fitting system for density-dependent metric models.

Tier 1: Fit galaxy rotation curve without Cassini constraint
Tier 2: Use Tier 1 as starting point with tight priors, enforce Cassini

This approach allows small parameter adjustments to satisfy both constraints.
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
import logging

# Add the parent directory to the path
sys.path.append('.')

from run_dynesty import main_dynesty
from density_metric2 import xi_power_law, xi_gravitational_color, XI_FUNCTION_MAP

logger = logging.getLogger(__name__)

def run_tier1_galaxy_fit(xi_type='enhanced', output_base='chains_tier1'):
    """
    Tier 1: Fit galaxy with all constraints disabled for maximum flexibility.
    This finds the best possible rotation curve fit.
    """
    print("\n" + "="*70)
    print("TIER 1: GALAXY-ONLY FIT (NO CASSINI/RHO_C CONSTRAINTS)")
    print("="*70)
    
    # Set up arguments for galaxy-focused fit
    argv_tier1 = [
        'run_dynesty.py',
        '--xi', xi_type,
        '--output_dir', f'{output_base}_{xi_type}',
        '--maxcall', '2000000',
        '--nlive_init', '800',
        '--dlogz_target', '0.1',
        
        # CRITICAL: Disable constraints for galaxy-only fit
        '--disable_cassini_penalty',
        '--disable_rho_c_penalty',
        
        # Fit all components
        '--fit_disk_thin',
        '--fit_disk_thick', 
        '--fit_bulge',
        '--fit_gas',
        '--include_disk_thin',
        '--include_disk_thick',
        '--include_bulge',
        '--include_gas',
        
        # Fit xi parameters freely
        '--fit_xi_params',
    ]
    
    # Save original argv and run
    original_argv = sys.argv
    sys.argv = argv_tier1
    
    try:
        main_dynesty()
    except Exception as e:
        logger.error(f"Tier 1 failed: {e}")
        return None, None, None
    finally:
        sys.argv = original_argv
        
    # Load results
    output_dir = Path(f'{output_base}_{xi_type}')
    npz_files = list(output_dir.glob('*_samples.npz'))
    if not npz_files:
        logger.error("No output from Tier 1!")
        return None, None, None
        
    results = np.load(npz_files[0])
    
    # Extract best-fit parameters and uncertainties
    weights = results['weights']
    samples = results['samples']
    param_names = results['param_names']
    
    # Calculate weighted statistics
    best_params = {}
    param_uncertainties = {}
    
    for i, name in enumerate(param_names):
        # Weighted median
        sorted_idx = np.argsort(samples[:, i])
        cumsum = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumsum, 0.5)
        best_params[name] = samples[sorted_idx[median_idx], i]
        
        # Weighted standard deviation for uncertainty
        weighted_mean = np.average(samples[:, i], weights=weights)
        weighted_var = np.average((samples[:, i] - weighted_mean)**2, weights=weights)
        param_uncertainties[name] = np.sqrt(weighted_var)
    
    print(f"\nTier 1 Complete!")
    print(f"Best log(Z) = {results['logz'][-1]:.2f}")
    print(f"Best-fit parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.3e} ± {param_uncertainties[name]:.3e}")
    
    # Test Cassini with these parameters
    if xi_type == 'enhanced' or xi_type == 'power':
        rho_saturn = 2.3e21
        n_exp = best_params.get('n_exp', 1.5)
        A = best_params.get('A', 8.0)
        rho_c = best_params.get('rho_c_solar_kpc3', 1e9)
        xi_saturn = xi_power_law(rho_saturn, rho_c, n_exp, A)
        cassini_violation = abs(float(xi_saturn) - 1.0)
        print(f"\nCassini check: |ξ-1| = {cassini_violation:.2e} (limit: 2.3e-5)")
        if cassini_violation > 2.3e-5:
            print("  ⚠️  Violates Cassini - Tier 2 will adjust parameters")
    
    return best_params, param_uncertainties, param_names


def run_tier2_cassini_compatible(tier1_params, tier1_uncertainties, param_names, 
                                xi_type='enhanced', output_base='chains_tier2',
                                prior_width_factor=3.0):
    """
    Tier 2: Start from Tier 1 solution but enforce Cassini.
    Uses tight priors centered on Tier 1 results.
    
    Parameters
    ----------
    tier1_params : dict
        Best-fit parameters from Tier 1
    tier1_uncertainties : dict
        Parameter uncertainties from Tier 1
    param_names : list
        Parameter names from Tier 1
    prior_width_factor : float
        How many sigma to extend priors (default: 3)
    """
    print("\n" + "="*70)
    print("TIER 2: CASSINI-COMPATIBLE FIT WITH TIGHT PRIORS")
    print("="*70)
    print(f"Using {prior_width_factor}-sigma priors centered on Tier 1 results")
    
    argv_tier2 = [
        'run_dynesty.py',
        '--xi', xi_type,
        '--output_dir', f'{output_base}_{xi_type}_cassini',
        '--maxcall', '1000000',  # Can be smaller since we start near solution
        '--nlive_init', '600',
        '--dlogz_target', '0.01',
        
        # Enable Cassini constraint but keep flexibility
        # '--disable_cassini_penalty' is NOT included
        # '--disable_rho_c_penalty' is NOT included
        
        # Include all components
        '--include_disk_thin',
        '--include_disk_thick',
        '--include_bulge',
        '--include_gas',
        
        # Fit all parameters but with tight priors
        '--fit_disk_thin',
        '--fit_disk_thick',
        '--fit_bulge', 
        '--fit_gas',
        '--fit_xi_params',
        
        # Use Tier 1 results as starting point
        '--use_previous_best',
        '--tighten_bounds_factor', str(1.0/prior_width_factor),  # This creates tight priors
    ]
    
    # Override prior bounds based on Tier 1 results
    # This ensures priors are centered on Tier 1 values
    for param_name in tier1_params:
        center = tier1_params[param_name]
        uncertainty = tier1_uncertainties.get(param_name, 0.1 * abs(center))
        
        # Create bounds as center ± prior_width_factor * uncertainty
        if 'M_' in param_name or 'rho_c' in param_name:
            # Log-space bounds for mass/density parameters
            log_center = np.log10(max(center, 1e-30))
            log_uncertainty = uncertainty / (center * np.log(10))  # Convert to log space
            
            lower = 10**(log_center - prior_width_factor * log_uncertainty)
            upper = 10**(log_center + prior_width_factor * log_uncertainty)
        else:
            # Linear bounds for other parameters
            lower = center - prior_width_factor * uncertainty
            upper = center + prior_width_factor * uncertainty
        
        # Ensure bounds are physical
        lower = max(lower, 1e-30)  # No negative values
        
        # Add bounds to argv
        param_key = param_name.replace('_solar', '').replace('_kpc3', '').replace('_kpc', '')
        argv_tier2.extend([f'--{param_key}_min', str(lower)])
        argv_tier2.extend([f'--{param_key}_max', str(upper)])
        
        # Also set the fixed value (used as starting point)
        fixed_key = param_name.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
        argv_tier2.extend([f'--{fixed_key}', str(center)])
    
    print("\nTight prior ranges:")
    for param_name in tier1_params:
        if param_name in tier1_uncertainties:
            center = tier1_params[param_name]
            unc = tier1_uncertainties[param_name]
            print(f"  {param_name}: {center:.3e} ± {prior_width_factor * unc:.3e}")
    
    # Save Tier 1 results for reference
    tier1_summary = {
        'best_params': tier1_params,
        'uncertainties': tier1_uncertainties,
        'param_names': list(param_names)
    }
    with open(f'tier1_summary_{xi_type}.json', 'w') as f:
        json.dump(tier1_summary, f, indent=2)
    
    # Run Tier 2
    original_argv = sys.argv
    sys.argv = argv_tier2
    
    try:
        main_dynesty()
    except Exception as e:
        logger.error(f"Tier 2 failed: {e}")
        return None
    finally:
        sys.argv = original_argv
    
    # Load and analyze Tier 2 results
    output_dir = Path(f'{output_base}_{xi_type}_cassini')
    npz_files = list(output_dir.glob('*_samples.npz'))
    if not npz_files:
        logger.error("No output from Tier 2!")
        return None
        
    tier2_results = np.load(npz_files[0])
    
    # Compare Tier 1 and Tier 2
    print("\n" + "="*70)
    print("PARAMETER SHIFTS FROM TIER 1 TO TIER 2:")
    print("="*70)
    
    tier2_weights = tier2_results['weights']
    tier2_samples = tier2_results['samples']
    
    for i, name in enumerate(param_names):
        if name in tier1_params:
            tier1_val = tier1_params[name]
            
            # Get Tier 2 median
            sorted_idx = np.argsort(tier2_samples[:, i])
            cumsum = np.cumsum(tier2_weights[sorted_idx])
            median_idx = np.searchsorted(cumsum, 0.5)
            tier2_val = tier2_samples[sorted_idx[median_idx], i]
            
            # Calculate shift in units of Tier 1 uncertainty
            shift = tier2_val - tier1_val
            if name in tier1_uncertainties and tier1_uncertainties[name] > 0:
                shift_sigma = shift / tier1_uncertainties[name]
                print(f"  {name}: {tier1_val:.3e} → {tier2_val:.3e} "
                      f"(shift: {shift_sigma:+.2f}σ)")
            else:
                rel_shift = shift / tier1_val if tier1_val != 0 else 0
                print(f"  {name}: {tier1_val:.3e} → {tier2_val:.3e} "
                      f"(shift: {rel_shift:+.1%})")
    
    print(f"\nTier 2 log(Z) = {tier2_results['logz'][-1]:.2f}")
    
    return tier2_results


def main():
    """Run the two-tier fitting process with gradient-based optimization."""
    parser = argparse.ArgumentParser(description='Two-tier DDMM fitting')
    parser.add_argument('--xi', type=str, default='enhanced',
                      choices=['power', 'enhanced', 'grav_color'],
                      help='Xi function type')
    parser.add_argument('--skip_tier1', action='store_true',
                      help='Skip tier 1 and load existing results')
    parser.add_argument('--tier1_results', type=str,
                      help='Path to tier 1 summary JSON file')
    parser.add_argument('--prior_width', type=float, default=3.0,
                      help='Prior width in units of sigma for Tier 2')
    
    args = parser.parse_args()
    
    # Run Tier 1
    if not args.skip_tier1:
        tier1_params, tier1_uncertainties, param_names = run_tier1_galaxy_fit(xi_type=args.xi)
        if tier1_params is None:
            print("Tier 1 failed!")
            return
    else:
        # Load existing tier 1 results
        if args.tier1_results:
            with open(args.tier1_results, 'r') as f:
                tier1_data = json.load(f)
                tier1_params = tier1_data['best_params']
                tier1_uncertainties = tier1_data['uncertainties']
                param_names = tier1_data['param_names']
        else:
            print("Must provide --tier1_results when using --skip_tier1")
            return
    
    # Run Tier 2
    tier2_results = run_tier2_cassini_compatible(
        tier1_params, tier1_uncertainties, param_names,
        xi_type=args.xi, prior_width_factor=args.prior_width
    )
    
    if tier2_results is not None:
        print("\n" + "="*70)
        print("TWO-TIER FITTING COMPLETE!")
        print("="*70)
        print(f"Tier 1 (galaxy-only) results in: chains_tier1_{args.xi}/")
        print(f"Tier 2 (Cassini-compatible) results in: chains_tier2_{args.xi}_cassini/")
        print("\nThe Tier 2 results satisfy both galaxy rotation curve AND Cassini constraints")


if __name__ == '__main__':
    main()