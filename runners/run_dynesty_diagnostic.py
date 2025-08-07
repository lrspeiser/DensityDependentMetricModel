#!/usr/bin/env python3
"""
run_dynesty_diagnostic.py - Version that collects detailed diagnostic data
about parameter exploration to understand what works and what doesn't.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import cupy as cp
import argparse
from pathlib import Path
import pickle
import json
from datetime import datetime
import pandas as pd
import dynesty
from functools import partial
# import h5py  # Optional, will use JSON/NPZ instead

# Import the physics functions
from core.density_metric_cupy import v_total_kms_cupy
from runners.run_dynesty_cupy import setup_parameter_bounds

# Set CuPy memory pool
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

class DiagnosticSampler:
    """Sampler that collects detailed diagnostic information."""
    
    def __init__(self, R_data, v_data, sigma_data, xi_type, output_dir):
        """Initialize with diagnostic tracking."""
        # Ensure GPU
        cp.cuda.Device(0).use()
        
        # Store data as CuPy arrays
        self.R_data = cp.asarray(R_data, dtype=cp.float32)
        self.v_data = cp.asarray(v_data, dtype=cp.float32)
        self.sigma_data = cp.asarray(sigma_data, dtype=cp.float32)
        self.xi_type = xi_type
        self.output_dir = output_dir
        
        # Tracking
        self.call_count = 0
        self.best_logl = -np.inf
        self.best_params = None
        
        # Diagnostic data storage
        self.diagnostic_data = {
            'parameters': [],
            'logl': [],
            'chi2': [],
            'failure_reason': [],
            'xi_at_solar': [],  # Xi value at solar density
            'xi_min': [],       # Minimum xi value
            'xi_max': [],       # Maximum xi value
            'v_model_min': [],  # Min model velocity
            'v_model_max': [],  # Max model velocity
            'v_model_nan_count': [],  # Number of NaN values
            'computation_time': [],
            'memory_used': []
        }
        
        # Sample subset for detailed diagnostics
        n_diagnostic = min(100, len(R_data))
        diagnostic_indices = np.random.choice(len(R_data), n_diagnostic, replace=False)
        self.R_diagnostic = self.R_data[diagnostic_indices]
        self.v_diagnostic = self.v_data[diagnostic_indices]
        self.sigma_diagnostic = self.sigma_data[diagnostic_indices]
        
        # Reference densities for xi evaluation
        self.rho_solar = 1e8  # Solar system density in M_sun/kpc^3
        self.rho_void = 1e6   # Void density
        self.rho_cluster = 1e10  # Cluster density
        
        logging.info(f"Initialized diagnostic sampler with {len(R_data)} stars")
        logging.info(f"Tracking {n_diagnostic} stars for detailed diagnostics")
        
    def compute_xi_diagnostics(self, params):
        """Compute xi function at key density values."""
        from core.density_metric_cupy import (
            xi_gravitational_color_void_safe_cupy,
            xi_power_law_cupy
        )
        
        # Test densities
        test_densities = cp.array([self.rho_void, self.rho_solar, self.rho_cluster], dtype=cp.float32)
        
        try:
            if self.xi_type == 'grav_color_void_safe':
                rho_c = params.get('rho_c_solar_kpc3', 1e13)
                gamma = params.get('gamma_exp', 2.5)
                lambda_g = params.get('lambda_g', 8.0)
                xi_values = xi_gravitational_color_void_safe_cupy(test_densities, rho_c, gamma, lambda_g)
            elif self.xi_type == 'power':
                rho_c = params.get('rho_c_solar_kpc3', 1e13)
                n_exp = params.get('n_exp', 2.0)
                A = params.get('A', 5.0)
                xi_values = xi_power_law_cupy(test_densities, rho_c, n_exp, A)
            elif self.xi_type == 'enhanced':
                rho_c = params.get('rho_c_solar_kpc3', 1e13)
                n_exp = params.get('n_exp', 2.0)
                A = params.get('A', 5.0)
                xi_values = xi_power_law_cupy(test_densities, rho_c, n_exp, A)  # Enhanced uses power_law
            else:
                xi_values = cp.ones(3)
            
            xi_void, xi_solar, xi_cluster = float(xi_values[0].get()), float(xi_values[1].get()), float(xi_values[2].get())
            
            return {
                'xi_void': xi_void,
                'xi_solar': xi_solar,
                'xi_cluster': xi_cluster,
                'cassini_violation': abs(xi_solar - 1.0) > 1e-5
            }
        except Exception as e:
            return {
                'xi_void': np.nan,
                'xi_solar': np.nan,
                'xi_cluster': np.nan,
                'cassini_violation': True
            }
    
    def log_likelihood(self, theta, param_names):
        """Compute log-likelihood with detailed diagnostics."""
        self.call_count += 1
        start_time = datetime.now()
        
        # Initialize diagnostic entry
        diagnostic_entry = {
            'call_number': self.call_count,
            'timestamp': start_time.isoformat()
        }
        
        try:
            # Parameter validation
            failure_reason = None
            for name, value in zip(param_names, theta):
                if 'M_' in name and value <= 0:
                    failure_reason = f"Non-positive mass: {name}={value}"
                    diagnostic_entry['failure_reason'] = failure_reason
                    self.save_diagnostic_entry(theta, -np.inf, diagnostic_entry)
                    return -np.inf
                if any(x in name for x in ['R_', 'r_', 'a_', 'hz_']) and value <= 0:
                    failure_reason = f"Non-positive scale: {name}={value}"
                    diagnostic_entry['failure_reason'] = failure_reason
                    self.save_diagnostic_entry(theta, -np.inf, diagnostic_entry)
                    return -np.inf
            
            # Create parameter dictionary
            params = dict(zip(param_names, theta))
            diagnostic_entry['parameters'] = params
            
            # Compute xi diagnostics
            xi_diag = self.compute_xi_diagnostics(params)
            diagnostic_entry.update(xi_diag)
            
            # Check Cassini constraint
            if xi_diag['cassini_violation']:
                failure_reason = f"Cassini violation: xi_solar={xi_diag['xi_solar']:.6f}"
                diagnostic_entry['failure_reason'] = failure_reason
                # Don't reject immediately - compute likelihood anyway for diagnostics
            
            # Compute model velocities
            v_model = v_total_kms_cupy(self.R_data, params, xi_type=self.xi_type)
            
            # Velocity diagnostics
            v_min = float(cp.min(v_model).get())
            v_max = float(cp.max(v_model).get())
            v_nan_count = int(cp.sum(~cp.isfinite(v_model)).get())
            
            diagnostic_entry['v_model_min'] = v_min
            diagnostic_entry['v_model_max'] = v_max
            diagnostic_entry['v_model_nan_count'] = v_nan_count
            
            # Check for NaN/Inf
            if v_nan_count > 0:
                failure_reason = f"NaN velocities: {v_nan_count} points"
                diagnostic_entry['failure_reason'] = failure_reason
                self.save_diagnostic_entry(theta, -np.inf, diagnostic_entry)
                return -np.inf
            
            # Check for unrealistic velocities
            if v_max > 1000 or v_min < 0:  # km/s
                failure_reason = f"Unrealistic velocities: [{v_min:.1f}, {v_max:.1f}] km/s"
                diagnostic_entry['failure_reason'] = failure_reason
                # Continue anyway for diagnostics
            
            # Compute chi-squared
            residuals = (self.v_data - v_model) / self.sigma_data
            chi2 = cp.sum(residuals**2)
            chi2_value = float(chi2.get())
            diagnostic_entry['chi2'] = chi2_value
            diagnostic_entry['chi2_per_dof'] = chi2_value / len(self.R_data)
            
            # Log-likelihood
            logl = -0.5 * chi2_value
            
            # Memory usage
            used_bytes = mempool.used_bytes()
            diagnostic_entry['memory_used_gb'] = used_bytes / 1e9
            
            # Computation time
            elapsed = (datetime.now() - start_time).total_seconds()
            diagnostic_entry['computation_time'] = elapsed
            
            # Track best
            if logl > self.best_logl:
                self.best_logl = logl
                self.best_params = theta.copy()
                diagnostic_entry['is_best'] = True
            
            # Save diagnostic entry
            self.save_diagnostic_entry(theta, logl, diagnostic_entry)
            
            # Progress reporting
            if self.call_count <= 10 or self.call_count % 1000 == 0:
                logging.info(f"Call {self.call_count}: logL = {logl:.2f} (chi2/dof = {chi2_value/len(self.R_data):.2f})")
                logging.info(f"  Xi: solar={xi_diag['xi_solar']:.4f}, void={xi_diag['xi_void']:.4f}")
                if failure_reason:
                    logging.info(f"  Issue: {failure_reason}")
            
            return logl
            
        except Exception as e:
            diagnostic_entry['failure_reason'] = f"Exception: {str(e)}"
            diagnostic_entry['exception_type'] = type(e).__name__
            self.save_diagnostic_entry(theta, -np.inf, diagnostic_entry)
            logging.error(f"Likelihood error at call {self.call_count}: {e}")
            return -np.inf
    
    def save_diagnostic_entry(self, theta, logl, entry):
        """Save diagnostic data for later analysis."""
        # Store in memory
        self.diagnostic_data['parameters'].append(theta.tolist())
        self.diagnostic_data['logl'].append(logl)
        for key in ['chi2', 'failure_reason', 'xi_at_solar', 'xi_min', 'xi_max',
                    'v_model_min', 'v_model_max', 'v_model_nan_count', 
                    'computation_time', 'memory_used']:
            self.diagnostic_data[key].append(entry.get(key, None))
        
        # Periodic save to disk
        if self.call_count % 1000 == 0:
            self.save_diagnostics_to_file()
    
    def save_diagnostics_to_file(self):
        """Save accumulated diagnostics to NPZ and JSON files."""
        # Save to NPZ
        diagnostic_file = self.output_dir / f"diagnostics_{self.xi_type}.npz"
        
        np.savez(
            diagnostic_file,
            parameters=np.array(self.diagnostic_data['parameters']),
            logl=np.array(self.diagnostic_data['logl']),
            chi2=np.array(self.diagnostic_data['chi2']),
            v_model_min=np.array(self.diagnostic_data['v_model_min']),
            v_model_max=np.array(self.diagnostic_data['v_model_max']),
            v_model_nan_count=np.array(self.diagnostic_data['v_model_nan_count']),
            computation_time=np.array(self.diagnostic_data['computation_time']),
            memory_used=np.array(self.diagnostic_data['memory_used']),
            xi_type=self.xi_type,
            n_calls=self.call_count,
            best_logl=self.best_logl,
            n_data_points=len(self.R_data)
        )
        
        logging.info(f"Saved diagnostics to {diagnostic_file}")
        
        # Also save summary JSON
        summary_file = self.output_dir / f"diagnostic_summary_{self.xi_type}.json"
        
        # Compute statistics
        valid_logl = [l for l in self.diagnostic_data['logl'] if np.isfinite(l)]
        valid_chi2 = [c for c in self.diagnostic_data['chi2'] if c and np.isfinite(c)]
        
        summary = {
            'xi_type': self.xi_type,
            'total_calls': self.call_count,
            'n_valid': len(valid_logl),
            'n_failed': self.call_count - len(valid_logl),
            'best_logl': float(self.best_logl),
            'worst_finite_logl': float(min(valid_logl)) if valid_logl else None,
            'median_logl': float(np.median(valid_logl)) if valid_logl else None,
            'best_chi2_per_dof': float(min(valid_chi2))/len(self.R_data) if valid_chi2 else None,
            'failure_reasons': {}
        }
        
        # Count failure reasons
        for reason in self.diagnostic_data['failure_reason']:
            if reason:
                # Simplify reason for counting
                if 'Cassini' in reason:
                    key = 'Cassini_violation'
                elif 'NaN' in reason:
                    key = 'NaN_velocities'
                elif 'Unrealistic' in reason:
                    key = 'Unrealistic_velocities'
                elif 'Non-positive' in reason:
                    key = 'Non_positive_parameter'
                else:
                    key = 'Other'
                
                summary['failure_reasons'][key] = summary['failure_reasons'].get(key, 0) + 1
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Saved summary to {summary_file}")

def prior_transform(u, bounds_low, bounds_high, use_log_prior):
    """Transform unit cube to prior space."""
    theta = np.zeros(len(u))
    for i in range(len(u)):
        if use_log_prior[i]:
            log_low = np.log10(bounds_low[i])
            log_high = np.log10(bounds_high[i])
            theta[i] = 10**(log_low + u[i] * (log_high - log_low))
        else:
            theta[i] = bounds_low[i] + u[i] * (bounds_high[i] - bounds_low[i])
    return theta

def main():
    """Main function with diagnostic collection."""
    parser = argparse.ArgumentParser(description="Diagnostic dynesty runner")
    parser.add_argument('--xi', type=str, required=True,
                       choices=['grav_color_void_safe', 'enhanced', 'power', 'grav_color',
                               'elastic_strain', 'hookean', 'tension_field'],
                       help='Xi function type')
    parser.add_argument('--nlive', type=int, default=200)
    parser.add_argument('--maxcall', type=int, default=50000)
    parser.add_argument('--max_sample_gaia', type=int, default=10000,
                       help='Use smaller sample for diagnostic runs')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"runs/diagnostic_{args.xi}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Running DIAGNOSTIC mode for {args.xi}")
    
    # Load Gaia data
    logger.info("Loading Gaia data...")
    cache_file = Path("external_data/gaia_sky_slices/all_sky_gaia.csv")
    
    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        return
    
    gaia_df = pd.read_csv(cache_file)
    
    # Process the data
    from core.data_io import process_gaia_data
    if 'R_kpc' not in gaia_df.columns:
        gaia_df = process_gaia_data(gaia_df)
    
    # Extract arrays
    R_data = gaia_df['R_kpc'].values.astype(np.float32)
    v_data = gaia_df['v_obs'].values.astype(np.float32)
    sigma_data = gaia_df['sigma_v'].values.astype(np.float32)
    
    # Sample for diagnostics
    if args.max_sample_gaia and len(R_data) > args.max_sample_gaia:
        logger.info(f"Sampling {args.max_sample_gaia} stars for diagnostic run")
        indices = np.random.choice(len(R_data), args.max_sample_gaia, replace=False)
        R_data = R_data[indices]
        v_data = v_data[indices]
        sigma_data = sigma_data[indices]
    
    logger.info(f"Using {len(R_data)} stars for diagnostics")
    
    # Setup parameters
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    ndim = len(param_names)
    
    logger.info(f"Fitting {ndim} parameters: {param_names}")
    
    # Create diagnostic sampler
    sampler_wrapper = DiagnosticSampler(R_data, v_data, sigma_data, args.xi, output_dir)
    
    # Setup functions
    logl_func = partial(sampler_wrapper.log_likelihood, param_names=param_names)
    prior_func = partial(prior_transform, bounds_low=bounds_low, 
                        bounds_high=bounds_high, use_log_prior=use_log_prior)
    
    # Create sampler
    logger.info("Creating DynamicNestedSampler for diagnostics...")
    sampler = dynesty.DynamicNestedSampler(
        logl_func,
        prior_func,
        ndim=ndim,
        nlive=args.nlive,
        sample='rslice',
        bound='multi'
    )
    
    logger.info("=" * 60)
    logger.info(f"Starting diagnostic sampling with {args.nlive} live points")
    logger.info(f"Max calls: {args.maxcall}")
    logger.info("=" * 60)
    
    try:
        # Run sampling
        sampler.run_nested(
            maxcall=args.maxcall,
            dlogz_init=1.0,  # Loose tolerance for diagnostics
            print_progress=True
        )
        
        # Save final diagnostics
        sampler_wrapper.save_diagnostics_to_file()
        
        # Save results
        results = sampler.results
        
        with open(output_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        np.savez(
            output_dir / "posterior_samples.npz",
            samples=results.samples,
            logl=results.logl,
            logz=results.logz if hasattr(results, 'logz') else [np.nan],
            param_names=param_names,
            xi_type=args.xi
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC RUN COMPLETE")
        logger.info(f"Total calls: {sampler_wrapper.call_count}")
        logger.info(f"Best logL: {sampler_wrapper.best_logl:.2f}")
        logger.info(f"Diagnostics saved to: {output_dir}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nDiagnostic run interrupted")
        sampler_wrapper.save_diagnostics_to_file()
    except Exception as e:
        logger.error(f"Diagnostic run failed: {e}")
        sampler_wrapper.save_diagnostics_to_file()
        raise

if __name__ == "__main__":
    main()