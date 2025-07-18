#!/usr/bin/env python3
"""
smart_restart_dynesty.py - Restart dynesty with mode-locking and constraints
"""

import numpy as np
from run_dynesty import *  # Import your existing code
import argparse

class ModeLockingPriorTransform:
    """Prior transform that locks to a specific mode."""
    
    def __init__(self, mode_center, mode_width, param_names, 
                 prior_bounds_low, prior_bounds_high, lock_strength=0.8):
        self.mode_center = np.array(mode_center)
        self.mode_width = np.array(mode_width)
        self.param_names = param_names
        self.prior_bounds_low = prior_bounds_low
        self.prior_bounds_high = prior_bounds_high
        self.lock_strength = lock_strength  # 0=no lock, 1=full lock
        
    def __call__(self, u_array):
        """Transform with soft locking to mode."""
        params = np.empty_like(u_array)
        
        for i in range(len(self.param_names)):
            if self.lock_strength == 0:
                # Standard uniform
                params[i] = self.prior_bounds_low[i] + \
                           u_array[i] * (self.prior_bounds_high[i] - self.prior_bounds_low[i])
            else:
                # Soft locking using tanh transformation
                center = self.mode_center[i]
                width = self.mode_width[i] * (2 - self.lock_strength)  # Tighten with lock strength
                
                # Map [0,1] to [-∞,∞] then back to parameter space
                z = np.arctanh(2 * u_array[i] - 1)  # Maps [0,1] to [-∞,∞]
                offset = z * width
                
                # Ensure we stay within bounds
                params[i] = np.clip(
                    center + offset,
                    self.prior_bounds_low[i],
                    self.prior_bounds_high[i]
                )
                
        return params

class PhysicalConstraintLikelihood:
    """Wrapper that adds physical constraints to likelihood."""
    
    def __init__(self, base_likelihood_func, constraint_strength=100):
        self.base_likelihood = base_likelihood_func
        self.constraint_strength = constraint_strength
        
    def __call__(self, theta_values, *args):
        # Get base likelihood
        base_logl, blob = self.base_likelihood(theta_values, *args)
        
        if not np.isfinite(base_logl):
            return base_logl, blob
            
        # Add constraint penalties
        penalty = 0.0
        
        # Extract parameters (you'll need to map these)
        param_names = args[0]  # First arg should be param names
        params = dict(zip(param_names, theta_values))
        
        # Thick disk > thin disk constraint
        if 'R_d_thick_kpc' in params and 'R_d_thin_kpc' in params:
            if params['R_d_thick_kpc'] < params['R_d_thin_kpc'] * 1.1:
                violation = params['R_d_thin_kpc'] * 1.1 - params['R_d_thick_kpc']
                penalty += self.constraint_strength * violation**2
                
        # Add more constraints as needed...
        
        return base_logl - penalty, blob

def smart_restart_sampling(previous_results_file, output_dir, **kwargs):
    """Smart restart with mode locking."""
    
    # Analyze previous results
    analyzer = BimodalAnalyzer(previous_results_file)
    physical_mode, unphysical_mode = analyzer.separate_physical_modes()
    
    if physical_mode['weight_fraction'] < 0.1:
        logger.warning("Physical mode has very low weight - may need different strategy")
        
    # Get mode parameters
    medians, mads = analyzer.get_mode_parameters(
        physical_mode['samples'], 
        physical_mode['weights']
    )
    
    # Set up new bounds (3-sigma around physical mode)
    new_bounds = {}
    for param, median in medians.items():
        sigma = mads[param] * 1.4826  # MAD to sigma conversion
        new_bounds[param] = {
            'low': max(PHYSICAL_BOUNDS[param]['min'], median - 3*sigma),
            'high': min(PHYSICAL_BOUNDS[param]['max'], median + 3*sigma)
        }
        
    logger.info("New tightened bounds around physical mode:")
    for param, bounds in new_bounds.items():
        logger.info(f"  {param}: [{bounds['low']:.2e}, {bounds['high']:.2e}]")
        
    # Create modified args with new bounds
    args = argparse.Namespace(**kwargs)
    args.use_mode_locking = True
    args.mode_center = list(medians.values())
    args.mode_width = [3*mads[p]*1.4826 for p in medians.keys()]
    
    # Override parameter bounds in config
    for param, bounds in new_bounds.items():
        if param in MW_MULTI_COMP_PARAM_CONFIG:
            MW_MULTI_COMP_PARAM_CONFIG[param]['low'] = bounds['low']
            MW_MULTI_COMP_PARAM_CONFIG[param]['high'] = bounds['high']
            
    # Run with modified sampler
    run_single_dynesty_with_constraints(args, gaia_data_dict)