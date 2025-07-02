#!/usr/bin/env python3
"""
run_dynesty.py - Run dynesty dynamic nested sampling on the Density-Metric model for the Milky Way.
Saves posterior samples to specified output. Includes self-tests and advanced progress logging.
Enhanced with expert feedback: log-uniform priors, configurable sampler settings, checkpoint support.
NOW WITH INTEGRATED MONITORING for detailed progress tracking during sampling.
AI-ENHANCED VERSION: Includes surrogate modeling, curriculum learning, and adaptive strategies.
"""
import logging
import sys
import time
import numpy as np
import argparse
import os
from pathlib import Path
import pickle
import gzip
from multiprocessing import Pool, freeze_support
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Any
import json

import matplotlib.pyplot as plt
import corner

# Optional imports for advanced features
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import qmc  # For Latin Hypercube sampling
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("WARNING: scikit-learn not found. Gaussian Process surrogate modeling disabled.")

try:
    from dynesty import DynamicNestedSampler, utils as dyfunc
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    print("CRITICAL: Dynesty library not found. Please install it: pip install dynesty")
    sys.exit(1)

try:
    from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP, run_physics_self_tests
    from data_io import load_gaia
    from main2 import get_param_labels_and_bounds as get_param_config_main_module
except ImportError as e:
    print(f"CRITICAL: Could not import local modules: {e}")
    sys.exit(1)

logger = None

# --- Parameter Configuration (Moved here to be self-contained) ---
# Enhanced with log_prior flags for scale-variant parameters
MW_MULTI_COMP_PARAM_CONFIG = {
    'rho_c_solar_kpc3': {'label': r"$\rho_c$ ($M_\odot/kpc^3$)", 'fixed_val_from_arg': 'rho_c_fixed', 
                         'default_fixed': 1e7, 'low': 1e5, 'high': 2e9, 'fit_flag_arg': 'fit_xi_params',
                         'log_prior': True},  # Log-uniform for density
    'n_exp': {'label': r"$n$", 'fixed_val_from_arg': 'n_exp_fixed', 
              'default_fixed': 1.5, 'low': 0.1, 'high': 4.0, 'fit_flag_arg': 'fit_xi_params',
              'log_prior': False},  # Linear for exponent
    'M_disk_thin_solar': {'label': r"$M_{d,thin}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thin_fixed', 
                          'default_fixed': 4.0e10, 'low': 1e10, 'high': 1.5e11, 'fit_flag_arg': 'fit_disk_thin', 
                          'include_flag_arg': 'include_disk_thin', 'log_prior': True},  # Log-uniform for mass, wider prior
    'R_d_thin_kpc': {'label': r"$R_{d,thin}$ (kpc)", 'fixed_val_from_arg': 'R_d_thin_fixed', 
                     'default_fixed': 2.5, 'low': 1.5, 'high': 5.0, 'fit_flag_arg': 'fit_disk_thin', 
                     'include_flag_arg': 'include_disk_thin', 'log_prior': False},
    'h_z_thin_kpc': {'label': r"$h_{z,thin}$ (kpc)", 'fixed_val_from_arg': 'h_z_thin_fixed', 
                     'default_fixed': 0.3, 'low': 0.15, 'high': 0.7, 'fit_flag_arg': 'fit_disk_thin', 
                     'include_flag_arg': 'include_disk_thin', 'log_prior': False},  # Wider prior
    'M_disk_thick_solar': {'label': r"$M_{d,thick}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_disk_thick_fixed', 
                           'default_fixed': 1.0e10, 'low': 0.1e10, 'high': 8e10, 'fit_flag_arg': 'fit_disk_thick', 
                           'include_flag_arg': 'include_disk_thick', 'log_prior': True},  # Log-uniform for mass
    'R_d_thick_kpc': {'label': r"$R_{d,thick}$ (kpc)", 'fixed_val_from_arg': 'R_d_thick_fixed', 
                      'default_fixed': 3.5, 'low': 2.0, 'high': 6.0, 'fit_flag_arg': 'fit_disk_thick', 
                      'include_flag_arg': 'include_disk_thick', 'log_prior': False},
    'h_z_thick_kpc': {'label': r"$h_{z,thick}$ (kpc)", 'fixed_val_from_arg': 'h_z_thick_fixed', 
                      'default_fixed': 0.9, 'low': 0.5, 'high': 1.5, 'fit_flag_arg': 'fit_disk_thick', 
                      'include_flag_arg': 'include_disk_thick', 'log_prior': False},
    'M_bulge_solar': {'label': r"$M_{bulge}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_bulge_fixed', 
                      'default_fixed': 0.9e10, 'low': 0.1e10, 'high': 5e10, 'fit_flag_arg': 'fit_bulge', 
                      'include_flag_arg': 'include_bulge', 'log_prior': True},  # Log-uniform for mass
    'a_bulge_kpc': {'label': r"$a_{bulge}$ (kpc)", 'fixed_val_from_arg': 'a_bulge_fixed', 
                    'default_fixed': 0.5, 'low': 0.1, 'high': 2.0, 'fit_flag_arg': 'fit_bulge', 
                    'include_flag_arg': 'include_bulge', 'log_prior': False},
    'M_gas_solar': {'label': r"$M_{gas}$ ($M_\odot$)", 'fixed_val_from_arg': 'M_gas_fixed', 
                    'default_fixed': 1.0e10, 'low': 0.2e10, 'high': 2.0e10, 'fit_flag_arg': 'fit_gas', 
                    'include_flag_arg': 'include_gas', 'log_prior': True},  # Log-uniform for mass
    'R_d_gas_kpc': {'label': r"$R_{d,gas}$ (kpc)", 'fixed_val_from_arg': 'R_d_gas_fixed', 
                    'default_fixed': 7.0, 'low': 3.0, 'high': 15.0, 'fit_flag_arg': 'fit_gas', 
                    'include_flag_arg': 'include_gas', 'log_prior': False},
    'h_z_gas_kpc': {'label': r"$h_{z,gas}$ (kpc)", 'fixed_val_from_arg': 'h_z_gas_fixed', 
                    'default_fixed': 0.15, 'low': 0.05, 'high': 0.5, 'fit_flag_arg': 'fit_gas', 
                    'include_flag_arg': 'include_gas', 'log_prior': False},
}


# --- Gaussian Process Surrogate Model ---
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
                logger.warning(f"Failed to evaluate training point: {e}")
        
        # Train initial GP
        self._train_gp()
        logger.info(f"✅ Initial GP training complete with {len(self.X_train)} points")
    
    def _train_gp(self):
        """Train or retrain the GP model"""
        if len(self.X_train) < 10:
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


# --- Adaptive Sampling Strategy ---
class AdaptiveSamplingStrategy:
    """
    Monitors sampling progress and adapts strategy when stuck or inefficient
    """
    
    def __init__(self, 
                 initial_params: List[str],
                 param_groups: Dict[str, List[str]],
                 thresholds: Dict[str, float]):
        """
        Parameters:
        -----------
        initial_params : list
            Full list of parameters being fitted
        param_groups : dict
            Groups of related parameters (e.g., {'disk_thin': ['M_disk_thin_solar', ...]})
        thresholds : dict
            Performance thresholds for adaptation
        """
        self.initial_params = initial_params
        self.param_groups = param_groups
        self.thresholds = thresholds
        
        # Tracking metrics
        self.sampling_history = []
        self.phase = "initial"
        self.adaptations_made = []
        
    def check_sampling_health(self, sampler, elapsed_time: float) -> Dict[str, any]:
        """
        Assess current sampling performance and health
        """
        res = sampler.results
        
        # Basic metrics
        if isinstance(res.ncall, np.ndarray):
            ncall = np.sum(res.ncall)
        else:
            ncall = res.ncall
            
        n_samples = len(res.samples) if hasattr(res, 'samples') else 0
        efficiency = (n_samples / ncall * 100) if ncall > 0 else 0
        
        # Calculate rate metrics
        call_rate = ncall / elapsed_time if elapsed_time > 0 else 0
        sample_rate = n_samples / elapsed_time if elapsed_time > 0 else 0
        
        # Degeneracy detection
        degeneracy_score = self._detect_parameter_degeneracies(res.samples) if n_samples > 1000 else 0
        
        # Mass accumulation check
        mass_escalation = self._check_mass_escalation(res.samples, self.initial_params) if n_samples > 500 else False
        
        health = {
            'ncall': ncall,
            'n_samples': n_samples,
            'efficiency': efficiency,
            'call_rate': call_rate,
            'sample_rate': sample_rate,
            'degeneracy_score': degeneracy_score,
            'mass_escalation': mass_escalation,
            'phase': self.phase,
            'time_in_phase': elapsed_time
        }
        
        self.sampling_history.append(health)
        return health
    
    def _detect_parameter_degeneracies(self, samples: np.ndarray) -> float:
        """
        Detect parameter degeneracies using correlation analysis
        """
        if len(samples) < 100:
            return 0.0
            
        # Use recent samples
        recent = samples[-min(1000, len(samples)):]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(recent.T)
        
        # Find high correlations (excluding diagonal)
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        
        # Count problematic correlations
        high_corr_count = np.sum(np.abs(corr_matrix) > 0.9)
        
        degeneracy_score = max_corr + (high_corr_count / corr_matrix.size)
        return degeneracy_score
    
    def _check_mass_escalation(self, samples: np.ndarray, param_names: List[str]) -> bool:
        """
        Check if total mass is escalating beyond reasonable bounds
        """
        mass_indices = [i for i, name in enumerate(param_names) if 'M_' in name and 'solar' in name]
        
        if not mass_indices or len(samples) < 500:
            return False
            
        # Compare early vs recent total masses
        early_samples = samples[100:300]
        recent_samples = samples[-200:]
        
        early_total = np.sum(early_samples[:, mass_indices], axis=1)
        recent_total = np.sum(recent_samples[:, mass_indices], axis=1)
        
        # Check if mass is growing significantly
        mass_growth = np.median(recent_total) / np.median(early_total)
        
        return mass_growth > 1.5 or np.median(recent_total) > 1.5e11
    
    def should_restart_sampling(self, sampler, start_time: float, phase: str = 'initial') -> Tuple[bool, Optional[str]]:
        """
        Determine if sampling should be restarted with a different strategy
        """
        elapsed = time.time() - start_time
        res = sampler.results
        
        # Get current health
        health = self.check_sampling_health(sampler, elapsed)
        
        # Check various failure modes
        if phase == 'initial' and health['ncall'] > 300000 and elapsed > 7200:  # 2 hours
            
            # Mass escalation check
            if health['mass_escalation']:
                return True, "mass_escalation"
            
            # Low efficiency check
            if health['efficiency'] < 2.0:
                return True, "low_efficiency"
            
            # High degeneracy check
            if health['degeneracy_score'] > 1.5:
                return True, "high_degeneracy"
        
        # Check if completely stuck
        if health['ncall'] > 500000 and health['efficiency'] < 1.0:
            return True, "extremely_low_efficiency"
        
        return False, None


# --- Curriculum Learning Implementation ---
def run_curriculum_learning(args, gaia_data_dict, logger):
    """
    Implement curriculum learning approach: start simple, add complexity
    """
    logger.info("🎓 Starting CURRICULUM LEARNING approach")
    
    all_results = {}
    cumulative_params = {}
    
    # Define curriculum stages with smart resource allocation
    curriculum = [
        {
            'name': 'Stage 1: Xi Parameters Only',
            'fit_flags': {
                'fit_xi_params': True,
                'fit_disk_thin': False,
                'fit_disk_thick': False,
                'fit_bulge': False,
                'fit_gas': False
            },
            'fixed_values': {
                'M_disk_thin_solar': 6e10,
                'R_d_thin_kpc': 3.0,
                'h_z_thin_kpc': 0.3,
                'M_bulge_solar': 1.5e10,
                'a_bulge_kpc': 0.7,
                'M_disk_thick_solar': 1.5e10,
                'R_d_thick_kpc': 3.5,
                'h_z_thick_kpc': 0.9,
                'M_gas_solar': 1e10,
                'R_d_gas_kpc': 7.0,
                'h_z_gas_kpc': 0.15
            },
            'nlive': 500,
            'dlogz': 0.1,  # Looser convergence for quick exploration
            'maxcall': int(args.maxcall * 0.15)  # 15% of total budget
        },
        {
            'name': 'Stage 2: Xi + Both Disk Components',
            'fit_flags': {
                'fit_xi_params': True,  # Refit with disks
                'fit_disk_thin': True,
                'fit_disk_thick': True,  # Include thick disk too
                'fit_bulge': False,
                'fit_gas': False
            },
            'use_previous': ['rho_c_solar_kpc3', 'n_exp'],  # Use Stage 1 Xi as starting point
            'nlive': 800,
            'dlogz': 0.05,  # Tighter convergence
            'maxcall': int(args.maxcall * 0.35)  # 35% of total budget
        },
        {
            'name': 'Stage 3: Full Model Fine-Tuning',
            'fit_flags': {
                'fit_xi_params': True,
                'fit_disk_thin': True,
                'fit_disk_thick': True,
                'fit_bulge': True,
                'fit_gas': True
            },
            'use_previous': 'all',  # Use all previous results
            'nlive': args.nlive_init,  # Use full requested live points
            'dlogz': args.dlogz_target,  # Use final target precision
            'maxcall': int(args.maxcall * 0.50)  # 50% of total budget
        }
    ]
    
    for i, stage in enumerate(curriculum):
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 {stage['name']}")
        logger.info(f"{'='*60}")
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
                setattr(stage_args, param.replace('_solar', '_fixed').replace('_kpc', '_fixed'), value)
        
        # Use results from previous stages
        if i > 0 and 'use_previous' in stage:
            if stage['use_previous'] == 'all':
                # Use all previous results
                for param, value in cumulative_params.items():
                    fixed_name = param.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
                    setattr(stage_args, fixed_name, value['median'])
            else:
                # Use specific parameters
                for param in stage['use_previous']:
                    if param in cumulative_params:
                        fixed_name = param.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
                        setattr(stage_args, fixed_name, cumulative_params[param]['median'])
        
        # Update sampler settings with stage-specific values
        stage_args.nlive_init = stage.get('nlive', 500)
        stage_args.maxcall = stage.get('maxcall', 200000)
        stage_args.dlogz_target = stage.get('dlogz', args.dlogz_target)
        stage_args.output_dir = Path(args.output_dir) / f"stage_{i+1}"
        
        # Run this stage
        results = run_single_dynesty(stage_args, gaia_data_dict)
        
        if results is None:
            logger.error(f"Stage {i+1} failed!")
            logger.error("This could be due to:")
            logger.error("  - Prior bounds too restrictive")
            logger.error("  - Likelihood calculation errors")
            logger.error("  - Insufficient maxcall allocation")
            logger.info(f"Successfully completed stages: {list(all_results.keys())}")
            break
        
        all_results[f'stage_{i+1}'] = results
        
        # Extract parameters for next stage
        if hasattr(results, 'samples'):
            # Use dynesty's resampling to get equal-weight samples
            try:
                from dynesty import utils as dyfunc
                samples = dyfunc.resample_equal(results.samples, np.exp(results.logwt - results.logz[-1]))
            except:
                # Fallback to weighted average if dynesty utils not available
                samples = results.samples
                weights = np.exp(results.logwt - results.logz[-1])
            
            # Get parameter names for this stage
            fitted_p_names, _, _, _, _, _ = get_param_labels_and_bounds(stage_args)
            
            # Calculate weighted statistics for all fitted parameters
            for j, param in enumerate(fitted_p_names):
                if len(samples.shape) == 2 and j < samples.shape[1]:
                    param_samples = samples[:, j]
                    weighted_median = np.median(param_samples)
                    weighted_std = np.std(param_samples)
                    
                    cumulative_params[param] = {
                        'median': weighted_median,
                        'std': weighted_std
                    }
                    logger.info(f"  {param}: {weighted_median:.3e} ± {weighted_std:.3e}")
    
    logger.info(f"\n🎉 Curriculum learning complete!")
    
    # Summarize efficiency gains
    total_calls_used = sum(stage.get('maxcall', 0) for stage in curriculum)
    logger.info(f"\n📊 Curriculum Learning Summary:")
    logger.info(f"  Total maxcall budget allocated: {total_calls_used:,} / {args.maxcall:,} ({total_calls_used/args.maxcall*100:.0f}%)")
    logger.info(f"  Stages completed: {len(all_results)}")
    logger.info(f"  Final parameters found: {len(cumulative_params)}")
    
    return all_results


# --- Enhanced Monitoring Functions ---
def format_parameter_value_monitor(value, param_name):
    """Format parameter values appropriately for monitoring"""
    if 'M_' in param_name and 'solar' in param_name:
        return f"{value:.2e} M☉"
    elif 'rho_c' in param_name:
        return f"{value:.2e} M☉/kpc³"
    elif 'kpc' in param_name:
        return f"{value:.3f} kpc"
    elif 'n_exp' in param_name:
        return f"{value:.3f}"
    else:
        return f"{value:.3e}"

def monitor_sampler_progress(sampler, fitted_param_names, fitted_param_labels, start_time, logger, gp_surrogate=None):
    """
    Monitor the progress of dynesty sampling with detailed diagnostics.
    This is adapted from dynesty_monitor.py to work with live sampler object.
    """
    try:
        res = sampler.results
        
        # Basic info
        logger.info("="*60)
        logger.info(f"DYNESTY DETAILED PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        # Get samples and basic stats
        if not hasattr(res, 'samples') or len(res.samples) == 0:
            logger.info("❌ No samples available yet")
            return
            
        samples = res.samples
        n_samples, n_params = samples.shape
        
        # Handle ncall properly
        if isinstance(res.ncall, np.ndarray):
            ncall_total = np.sum(res.ncall)
        else:
            ncall_total = res.ncall
            
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        logger.info(f"📈 Current samples: {n_samples:,} × {n_params} parameters")
        logger.info(f"📊 Total likelihood calls: {ncall_total:,}")
        logger.info(f"⏱️  Elapsed time: {elapsed_str}")
        
        # GP Surrogate statistics
        if gp_surrogate is not None and GP_AVAILABLE:
            gp_stats = gp_surrogate.get_statistics()
            logger.info(f"🤖 GP Surrogate: {gp_stats['n_real_calls']:,} real calls, "
                       f"{gp_stats['n_surrogate_calls']:,} surrogate calls "
                       f"(speedup: {gp_stats['speedup_factor']:.1f}x)")
        
        if n_samples < 50:
            logger.info("⚠️  Very few samples yet - check back later")
            return
        
        # Parameter estimates
        logger.info(f"\n📊 CURRENT PARAMETER ESTIMATES (median ± MAD):")
        logger.info("─" * 60)
        
        # Use last 1000 samples for more current estimate
        recent_samples = samples[-min(1000, len(samples)):]
        
        for i, (param_name, param_label) in enumerate(zip(fitted_param_names, fitted_param_labels)):
            values = recent_samples[:, i]
            
            median_val = np.median(values)
            mad = np.median(np.abs(values - median_val))  # Median Absolute Deviation
            
            # Format nicely
            param_display = param_name.replace('_solar', '').replace('_kpc3', '').replace('_kpc', '')
            formatted_val = format_parameter_value_monitor(median_val, param_name)
            formatted_mad = format_parameter_value_monitor(mad, param_name)
            
            logger.info(f"  {param_display:<20}: {formatted_val:<15} ± {formatted_mad}")
        
        # Key indicators for xi parameters
        logger.info(f"\n🎯 KEY INDICATORS:")
        logger.info("─" * 30)
        
        # Find xi parameters
        rho_c_idx = next((i for i, name in enumerate(fitted_param_names) if 'rho_c' in name), None)
        n_exp_idx = next((i for i, name in enumerate(fitted_param_names) if 'n_exp' in name), None)
        
        if rho_c_idx is not None and n_exp_idx is not None:
            rho_c_vals = recent_samples[:, rho_c_idx]
            n_exp_vals = recent_samples[:, n_exp_idx]
            
            median_rho_c = np.median(rho_c_vals)
            median_n = np.median(n_exp_vals)
            
            logger.info(f"  Critical density: {median_rho_c:.2e} M☉/kpc³")
            logger.info(f"  Power index: {median_n:.3f}")
            
            # Estimate xi range
            rho_inner = 1e9  # Typical inner galaxy density
            rho_outer = 1e6  # Typical outer galaxy density
            
            xi_inner = 1 / (1 + (rho_inner / median_rho_c)**median_n)
            xi_outer = 1 / (1 + (rho_outer / median_rho_c)**median_n)
            
            logger.info(f"  ξ inner (~R=2kpc): ~{xi_inner:.3f}")
            logger.info(f"  ξ outer (~R=20kpc): ~{xi_outer:.3f}")
            
            if xi_inner > 0.8:
                logger.info("  ⚠️  ξ close to 1 in inner regions - weak density dependence")
            elif xi_inner < 0.3:
                logger.info("  ✅ Strong density dependence in inner regions")
            else:
                logger.info("  ✅ Moderate density dependence")
        
        # Check baryonic masses
        mass_params = [(i, name) for i, name in enumerate(fitted_param_names) if 'M_' in name and 'solar' in name]
        
        if mass_params:
            logger.info(f"\n💫 BARYONIC MASS COMPONENTS:")
            total_mass = 0
            
            for i, name in mass_params:
                mass_vals = recent_samples[:, i]
                median_mass = np.median(mass_vals)
                total_mass += median_mass
                
                component = name.replace('M_', '').replace('_solar', '')
                logger.info(f"  {component:<12}: {median_mass:.2e} M☉")
            
            logger.info(f"  {'Total':<12}: {total_mass:.2e} M☉")
            
            # Check if masses are realistic
            if total_mass > 2e11:
                logger.info("  ⚠️  High total mass - may be compensating for weak ξ")
            elif total_mass < 5e10:
                logger.info("  ⚠️  Low total mass - insufficient baryonic matter")
            else:
                logger.info("  ✅ Reasonable total baryonic mass")
        
        # Sampling efficiency and diagnostics
        if hasattr(res, 'logl') and len(res.logl) > 100:
            recent_logl = res.logl[-min(1000, len(res.logl)):]
            logl_range = np.max(recent_logl) - np.min(recent_logl)
            
            logger.info(f"\n📈 SAMPLING DIAGNOSTICS:")
            logger.info(f"  Log-likelihood range: {logl_range:.2f}")
            logger.info(f"  Best log-L: {np.max(recent_logl):.2f}")
            logger.info(f"  Current log-L: {recent_logl[-1]:.2f}")
            
            # Calculate efficiency
            eff = 100.0 * len(res.samples) / ncall_total if ncall_total > 0 else 0.0
            logger.info(f"  Sampling efficiency: {eff:.2f}%")
            
            if logl_range < 1:
                logger.info("  ⚠️  Small likelihood range - may be converged or stuck")
            elif logl_range > 100:
                logger.info("  ⚠️  Large likelihood range - still exploring")
            else:
                logger.info("  ✅ Reasonable likelihood exploration")
        
        # Evidence estimate
        if hasattr(res, 'logz') and len(res.logz) > 0:
            logger.info(f"\n🎲 EVIDENCE:")
            logger.info(f"  Current log(Z): {res.logz[-1]:.2f}")
            if hasattr(res, 'logzerr') and len(res.logzerr) > 0:
                logger.info(f"  Error estimate: ±{res.logzerr[-1]:.2f}")
        
        # Convergence check
        if n_samples > 2000:
            # Check parameter stability over recent samples
            recent_frac = 0.3  # Last 30% of samples
            split_point = int(n_samples * (1 - recent_frac))
            
            early_samples = samples[split_point//2:split_point]
            late_samples = samples[split_point:]
            
            if len(early_samples) > 100 and len(late_samples) > 100:
                logger.info(f"\n🎯 CONVERGENCE CHECK:")
                stable_params = 0
                
                for i, param_name in enumerate(fitted_param_names):
                    early_median = np.median(early_samples[:, i])
                    late_median = np.median(late_samples[:, i])
                    
                    if early_median != 0:
                        rel_change = abs(late_median - early_median) / abs(early_median)
                        if rel_change < 0.1:  # Less than 10% change
                            stable_params += 1
                
                stability = stable_params / n_params
                logger.info(f"  Parameter stability: {stability:.1%} ({stable_params}/{n_params})")
                
                if stability > 0.8:
                    logger.info("  ✅ Parameters appear to be converging")
                elif stability > 0.5:
                    logger.info("  ⚠️  Partial convergence - needs more time")
                else:
                    logger.info("  ❌ Parameters still changing significantly")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.warning(f"Error in monitoring: {e}")


# --- Likelihood and Prior Transform Functions ---
def prior_transform_dynesty(u_array, fitted_param_names, prior_bounds_low, prior_bounds_high, use_log_prior_flags):
    """
    Enhanced prior transform with log-uniform support for scale-variant parameters.
    """
    if fitted_param_names is None or prior_bounds_low is None or prior_bounds_high is None:
        raise ValueError("prior_transform_dynesty received None for essential arguments.")
    params = np.empty_like(u_array)
    for i in range(len(fitted_param_names)):
        low, high = prior_bounds_low[i], prior_bounds_high[i]
        if use_log_prior_flags[i]:
            # Log-uniform transform for scale-variant parameters
            log_low, log_high = np.log10(low), np.log10(high)
            params[i] = 10**(log_low + u_array[i] * (log_high - log_low))
        else:
            # Standard uniform transform
            params[i] = low + u_array[i] * (high - low)
    return params

def v_model_for_dynesty(R_kpc_array, p_all_params_dict, xi_type_str, ARGS_obj_dynesty):
    rho_c_solar_kpc3 = p_all_params_dict['rho_c_solar_kpc3']
    n_exp = p_all_params_dict['n_exp']
    if ARGS_obj_dynesty.fit_target == 'milkyway':
        v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict)
        rho_midplane_for_xi = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict)
    else:
        raise NotImplementedError("SPARC fitting not yet fully configured.")
    v_n_kms = np.nan_to_num(v_n_kms, nan=0.0, posinf=0.0, neginf=0.0)
    rho_midplane_for_xi = np.nan_to_num(rho_midplane_for_xi, nan=0.0, posinf=1e10, neginf=0.0)
    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    xi_values = xi_func(rho_midplane_for_xi, rho_c_solar_kpc3, n_exp)
    xi_values = np.nan_to_num(xi_values, nan=1.0, posinf=1e10, neginf=0.0)
    xi_values_safe = np.maximum(xi_values, 0.0)
    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
    return v_mod_kms

def log_likelihood_dynesty(theta_values_fitted, fitted_param_names, args_dynesty_obj,
                           all_param_info_list, R_data, v_data, sigma_data, xi_type,
                           gp_surrogate=None):
    if any(arg is None for arg in [theta_values_fitted, fitted_param_names, args_dynesty_obj, 
                                   all_param_info_list, R_data, v_data, sigma_data, xi_type]): 
        return -np.inf, [np.inf]
    
    current_params_full_dict = dict(zip(fitted_param_names, theta_values_fitted))
    if all_param_info_list:
        for p_info in all_param_info_list:
            if not p_info['is_fitted']: current_params_full_dict[p_info['name']] = p_info['current_val']
        if args_dynesty_obj.fit_target == 'milkyway':
            for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
                if 'include_flag_arg' in p_details_cfg:
                    current_params_full_dict[p_details_cfg['include_flag_arg']] = getattr(args_dynesty_obj, p_details_cfg['include_flag_arg'])
            current_params_full_dict['include_bulge_density'] = args_dynesty_obj.include_bulge
    
    # Use GP surrogate if available
    if gp_surrogate is not None and args_dynesty_obj.use_gp_surrogate:
        # Define physics function for GP
        def physics_func(params, args_obj):
            return v_model_for_dynesty(R_data, params, xi_type, args_obj)
        
        # Get prediction from GP (with potential fallback to physics)
        v_predicted, v_uncertainty = gp_surrogate.predict(
            theta_values_fitted, 
            physics_function=physics_func,
            args_obj=args_dynesty_obj
        )
    else:
        # Standard physics model evaluation
        v_predicted = v_model_for_dynesty(R_data, current_params_full_dict, xi_type, args_dynesty_obj)
    
    if not np.all(np.isfinite(v_predicted)): return -np.inf, [np.inf]
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals = v_data - v_predicted
    rmse = np.sqrt(np.mean(residuals**2))
    chi_squared_terms = (residuals / sigma_data_safe)**2
    log_L_terms = chi_squared_terms + np.log(2 * np.pi * sigma_data_safe**2)
    if not np.all(np.isfinite(log_L_terms)): return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    log_L = -0.5 * np.sum(log_L_terms)
    if not np.isfinite(log_L): return -np.inf, [rmse if np.isfinite(rmse) else np.inf]
    return log_L, [rmse]

def _check_flag_consistency(args, logger_obj):
    logger_obj.info("Performing CLI flag consistency check...")
    components_to_check = {
        "xi_params": ["rho_c_fixed", "n_exp_fixed"],
        "disk_thin": ["M_disk_thin_fixed", "R_d_thin_fixed", "h_z_thin_fixed"],
        "disk_thick": ["M_disk_thick_fixed", "R_d_thick_fixed", "h_z_thick_fixed"],
        "bulge": ["M_bulge_fixed", "a_bulge_fixed"],
        "gas": ["M_gas_fixed", "R_d_gas_fixed", "h_z_gas_fixed"]
    }
    
    # Note: The presence of --fit_<component> means "fit at least some parameters of this component"
    # Individual fixed values override the general fit flag for specific parameters
    # This allows partial fitting (e.g., fit only h_z while fixing M and R)
    logger_obj.info("Note: Fixed value arguments override fit flags for specific parameters, allowing partial fitting.")
    logger_obj.info("CLI flag consistency OK.")

def get_param_labels_and_bounds(ARGS):
    """Enhanced to return log_prior flags for each parameter and handle partial fitting."""
    param_info_list = []
    config_to_use = MW_MULTI_COMP_PARAM_CONFIG
    logger.info("Configuring parameters for NEW multi-component Milky Way model.")
    
    for p_name, p_details in config_to_use.items():
        is_included = 'include_flag_arg' not in p_details or getattr(ARGS, p_details['include_flag_arg'], False)
        if not is_included: continue
        
        # Check if this parameter should be fitted
        is_fitted = False
        if 'fit_flag_arg' in p_details and getattr(ARGS, p_details['fit_flag_arg'], False):
            # The general fit flag is set, but check if a specific fixed value was provided
            fixed_arg_name = p_details['fixed_val_from_arg']
            if f"--{fixed_arg_name}" not in sys.argv:
                # No specific fixed value provided, so fit this parameter
                is_fitted = True
            else:
                # Specific fixed value provided, override the general fit flag
                logger.info(f"  {p_name}: Using fixed value (overrides fit flag)")
        
        current_val = getattr(ARGS, p_details['fixed_val_from_arg'])
        param_info_list.append({
            'name': p_name, 
            'label': p_details['label'], 
            'current_val': current_val,
            'low': p_details['low'], 
            'high': p_details['high'], 
            'is_fitted': is_fitted,
            'log_prior': p_details.get('log_prior', False)  # Add log_prior info
        })
    
    ARGS.all_param_info_list = param_info_list
    fitted_params_info = [p for p in param_info_list if p['is_fitted']]
    if not fitted_params_info: 
        logger.error("No parameters configured to be fitted! You must use at least one --fit_* flag.")
        logger.error("Note: If you used fixed values for all parameters in a component, nothing will be fitted.")
        sys.exit(1)
    
    # Extract log_prior flags for fitted parameters
    use_log_flags = [p['log_prior'] for p in fitted_params_info]
    
    return ([p['name'] for p in fitted_params_info], [p['label'] for p in fitted_params_info],
            np.array([p['current_val'] for p in fitted_params_info]),
            np.array([p['low'] for p in fitted_params_info]), np.array([p['high'] for p in fitted_params_info]),
            use_log_flags)  # Return the log_prior flags


def run_single_dynesty(args, gaia_data_dict, gp_surrogate=None):
    """
    Run a single dynesty sampling (extracted for reuse in curriculum learning)
    """

    R_data_for_run, v_data_for_run, sigma_data_for_run = gaia_data_dict["R_kpc"], gaia_data_dict["v_obs"], gaia_data_dict["sigma_v"]
    
    # Enhanced parameter configuration with log_prior flags
    fitted_p_names, fitted_p_labels, _, p_low, p_high, use_log_flags = get_param_labels_and_bounds(args)
    ndim_dynesty = len(fitted_p_names)
    logger.info(f"Dynesty fitting {ndim_dynesty} parameters: {fitted_p_names}")
    
    # Log prior type information
    logger.info("Parameter Prior Types:")
    for name, is_log in zip(fitted_p_names, use_log_flags):
        prior_type = "Log-Uniform" if is_log else "Uniform"
        logger.info(f"  - {name:<25} | Prior: {prior_type}")
        
    # Also log fixed parameters
    fixed_params = [p for p in args.all_param_info_list if not p['is_fitted']]
    if fixed_params:
        logger.info("Fixed Parameters:")
        for p in fixed_params:
            logger.info(f"  - {p['name']:<25} | Value: {p['current_val']:.2e}")
    
    ptform_args_tuple = (fitted_p_names, np.array(p_low), np.array(p_high), use_log_flags)
    logl_args_tuple = (fitted_p_names, args, args.all_param_info_list, R_data_for_run, v_data_for_run, sigma_data_for_run, args.xi, gp_surrogate)

    pool_obj, queue_size_for_sampler = None, None
    if args.num_threads > 1:
        try:
            pool_obj = Pool(args.num_threads)
            queue_size_for_sampler = args.num_threads
            logger.info(f"Dynesty will run with {args.num_threads} threads.")
        except Exception as e:
            logger.warning(f"Failed to create Pool: {e}. Running serially.")
    
    # Enhanced sampler configuration
    logger.info(f"Sampler configuration: method='{args.sample_method}', bound='{args.bound_method}', enlarge={args.enlarge_factor}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    sampler = DynamicNestedSampler(log_likelihood_dynesty, prior_transform_dynesty, ndim_dynesty,
                                   pool=pool_obj, queue_size=queue_size_for_sampler,
                                   sample=args.sample_method,
                                   bound=args.bound_method,
                                   enlarge=args.enlarge_factor,
                                   ptform_args=ptform_args_tuple, logl_args=logl_args_tuple,
                                   blob=True)
    
    run_start_time = time.time()
    last_progress_log_time = time.time()
    last_monitor_time = time.time()
    best_rmse_so_far = np.inf
    
    # Initialize adaptive strategy
    param_groups = {
        'disk_thin': ['M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc'],
        'disk_thick': ['M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc'],
        'bulge': ['M_bulge_solar', 'a_bulge_kpc'],
        'gas': ['M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'],
        'xi': ['rho_c_solar_kpc3', 'n_exp']
    }
    
    thresholds = {
        'min_efficiency': 2.0,
        'max_degeneracy': 0.9,
        'max_total_mass': 1.5e11,
        'max_initial_calls': 500000
    }
    
    adaptive_strategy = AdaptiveSamplingStrategy(fitted_p_names, param_groups, thresholds)
    
    # Checkpoint file path
    checkpoint_file = Path(args.output_dir) / "dynesty_checkpoint.pkl"
    
    try:
        if args.use_run_nested:
            # Use run_nested for more stable sampling (recommended)
            logger.info(f"Using run_nested() with nlive_init={args.nlive_init}, dlogz_target={args.dlogz_target}, checkpoint_every={args.checkpoint_every}s")
            
            # Add periodic monitoring callback
            last_monitor_check = [time.time()]  # Use list to make it mutable in closure
            
            def monitor_callback(res):
                """Callback function for run_nested to provide monitoring"""
                current_time = time.time()
                if current_time - last_monitor_check[0] > args.monitor_interval_s:
                    last_monitor_check[0] = current_time
                    # Create a dummy sampler object with results
                    class DummySampler:
                        def __init__(self, results):
                            self.results = results
                    dummy_sampler = DummySampler(res)
                    monitor_sampler_progress(dummy_sampler, fitted_p_names, fitted_p_labels, 
                                        run_start_time, logger, gp_surrogate)
            
            # Run nested sampling with monitoring callback
            sampler.run_nested(nlive_init=args.nlive_init, 
                nlive_batch=args.nlive_batch,
                dlogz_init=args.dlogz_target,
                maxcall=args.maxcall,
                print_progress=True, 
                checkpoint_file=str(checkpoint_file),
                checkpoint_every=args.checkpoint_every)
            logger.info("run_nested() completed.")
        else:
            # Custom sampling loop with adaptive monitoring
            logger.info(f"Running initial sampling with nlive_init = {args.nlive_init}...")
            
            for _ in sampler.sample_initial(nlive=args.nlive_init, maxcall=args.maxcall, save_samples=True):
                current_time = time.time()
                
                # Check if should restart
                if args.use_adaptive_restart:
                    should_restart, reason = adaptive_strategy.should_restart_sampling(sampler, run_start_time, 'initial')
                    if should_restart:
                        logger.warning(f"🚨 Restarting sampling due to: {reason}")
                        logger.warning("💡 Switching to curriculum learning approach")
                        
                        # Save current state
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(sampler.results, f)
                        
                        # Clean up resources
                        if pool_obj:
                            pool_obj.close()
                            pool_obj.join()
                        
                        # Switch to curriculum learning
                        return run_curriculum_learning(args, gaia_data_dict, logger)
                
                # Regular progress update
                if current_time - last_progress_log_time > args.progress_update_interval_s:
                    last_progress_log_time = current_time
                    ncall_val = sampler.results.ncall
                    if isinstance(ncall_val, np.ndarray):
                        ncall_total = np.sum(ncall_val)
                        ncall_mean = np.mean(ncall_val)
                        ncall_max = np.max(ncall_val)
                        logger.info(f"Initial Sampling | Total calls: {ncall_total:,} | Mean/point: {ncall_mean:.1f} | Max/point: {ncall_max} | Live: {len(sampler.live_logl)}")
                    else:
                        logger.info(f"Initial Sampling | Calls: {ncall_val}/{args.maxcall} | Live: {len(sampler.live_logl)}")
                
                # Detailed monitoring update
                if current_time - last_monitor_time > args.monitor_interval_s:
                    last_monitor_time = current_time
                    monitor_sampler_progress(sampler, fitted_p_names, fitted_p_labels, run_start_time, logger, gp_surrogate)

            logger.info("Initial sampling complete. Starting batch processing...")
            
            # Batch processing loop
            initial_ncall = sampler.results.ncall if not isinstance(sampler.results.ncall, np.ndarray) else np.sum(sampler.results.ncall)
            
            if initial_ncall >= args.maxcall:
                logger.warning("Initial sampling used all available calls.")
            else:
                ncall_total = initial_ncall
                
                while ncall_total < args.maxcall:
                    # Calculate stopping criterion manually
                    if hasattr(sampler.results, 'logz') and len(sampler.results.logz) > 1:
                        stop_val = sampler.results.logz[-1] - sampler.results.logz[-2]
                    else:
                        stop_val = np.inf
                    
                    if stop_val < args.dlogz_target:
                        logger.info(f"Stopping criterion met: dlogz ({stop_val:.4f}) < target ({args.dlogz_target:.4f}).")
                        break

                    
                    sampler.add_batch(nlive=args.nlive_batch, maxcall=args.maxcall, save_samples=True)
                    
                    ncall_total = sampler.results.ncall if not isinstance(sampler.results.ncall, np.ndarray) else np.sum(sampler.results.ncall)
                    
                    current_time = time.time()
                    
                    # Progress and monitoring updates
                    if current_time - last_progress_log_time > args.progress_update_interval_s:
                        last_progress_log_time = current_time
                        # ... (existing progress logging code)
                    
                    if current_time - last_monitor_time > args.monitor_interval_s:
                        last_monitor_time = current_time
                        monitor_sampler_progress(sampler, fitted_p_names, fitted_p_labels, run_start_time, logger, gp_surrogate)
            
            logger.info("Sampling loop finished.")
    
    finally:
        if pool_obj:
            logger.info("Closing and joining multiprocessing Pool.")
            pool_obj.close()
            pool_obj.join()
    
    # Final monitoring report
    if hasattr(sampler, 'results'):
        logger.info("\n🏁 FINAL MONITORING REPORT:")
        monitor_sampler_progress(sampler, fitted_p_names, fitted_p_labels, run_start_time, logger, gp_surrogate)
        return sampler.results
    
    return None


def main_dynesty():
    global logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger = logging.getLogger("run_dynesty")
    logger.info("Starting main_dynesty function (AI-Enhanced Version).")

    if not DYNESTY_AVAILABLE: logger.error("Dynesty library not found."); sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Run Dynesty for Density-Metric model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'])
    parser.add_argument('--max_sample_gaia', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default="chains_dynesty")
    parser.add_argument('--nlive_init', type=int, default=800)
    parser.add_argument('--nlive_batch', type=int, default=200, help="Number of live points to add per batch.")
    parser.add_argument('--dlogz_target', type=float, default=0.01, help="Target dlogz for stopping.")
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--update_interval', type=float, default=0.6, help="Dynesty update_interval.")
    parser.add_argument('--maxcall', type=int, default=2000000, help="Hard limit on likelihood calls.")
    parser.add_argument('--progress_update_interval_s', type=int, default=60, help="Interval in seconds for printing custom progress.")
    parser.add_argument('--monitor_interval_s', type=int, default=1800, help="Interval in seconds for detailed monitoring (default 30 min).")
    parser.add_argument('--debug_likelihood_params', type=str, default=None, help="Comma-separated physical parameters to test likelihood function.")
    parser.add_argument('--use_run_nested', action='store_true', default=False, 
                        help="Use run_nested instead of custom sampling loop (recommended for stability).")
    parser.add_argument('--checkpoint_every', type=int, default=300, 
                        help="Checkpoint interval in seconds (only for run_nested).")
    
    # Enhanced Dynesty sampler settings
    dynesty_g = parser.add_argument_group('Dynesty Sampler Settings')
    dynesty_g.add_argument('--sample_method', type=str, default='rslice', choices=['rwalk', 'rslice', 'hslice'],
                           help="Dynesty's internal sampling method. rslice is recommended for difficult posteriors.")
    dynesty_g.add_argument('--enlarge_factor', type=float, default=2.5,
                           help="Bound enlargement factor. Recommended > 2.0 for difficult posteriors.")
    dynesty_g.add_argument('--bound_method', type=str, default='multi', choices=['none', 'single', 'multi', 'balls', 'cubes'],
                           help="Bounding method for live points.")

    # AI-Enhanced features
    ai_g = parser.add_argument_group('AI-Enhanced Features')
    ai_g.add_argument('--use_curriculum_learning', action='store_true', default=False,
                      help="Use curriculum learning approach (start simple, add complexity)")
    ai_g.add_argument('--use_gp_surrogate', action='store_true', default=False,
                      help="Use Gaussian Process surrogate model for speedup")
    ai_g.add_argument('--gp_n_initial', type=int, default=500,
                      help="Number of initial training points for GP")
    ai_g.add_argument('--gp_uncertainty_threshold', type=float, default=0.1,
                      help="Uncertainty threshold for GP active learning")
    ai_g.add_argument('--use_adaptive_restart', action='store_true', default=False,
                      help="Enable adaptive restart if sampling gets stuck")

    mw_model_g = parser.add_argument_group('Milky Way Model Configuration')
    mw_model_g.add_argument('--include_bulge', action='store_true', default=False)
    mw_model_g.add_argument('--include_disk_thin', action='store_true', default=True)
    mw_model_g.add_argument('--include_disk_thick', action='store_true', default=False)
    mw_model_g.add_argument('--include_gas', action='store_true', default=False)
    
    fit_g = parser.add_argument_group('Fitting Flags (specify what to fit)')
    fit_g.add_argument('--fit_xi_params', action='store_true', help="Fit the xi function parameters (rho_c, n_exp).")
    fit_g.add_argument('--fit_disk_thin', action='store_true', help="Fit the thin disk parameters.")
    fit_g.add_argument('--fit_disk_thick', action='store_true', help="Fit the thick disk parameters.")
    fit_g.add_argument('--fit_bulge', action='store_true', help="Fit the bulge parameters.")
    fit_g.add_argument('--fit_gas', action='store_true', help="Fit the gas disk parameters.")

    fixed_g = parser.add_argument_group('Fixed Value Arguments (used if not fitting)')
    for p_name_cfg, p_details_cfg in MW_MULTI_COMP_PARAM_CONFIG.items():
        fixed_g.add_argument(f"--{p_details_cfg['fixed_val_from_arg']}", type=float, default=p_details_cfg['default_fixed'],
                                help=f"Fixed/initial value for {p_name_cfg}.")
    
    args = parser.parse_args()
    
    _check_flag_consistency(args, logger)
    run_physics_self_tests()

    args.fit_target = 'milkyway'
    gaia_data_dict = load_gaia(sample_max=args.max_sample_gaia)
    if gaia_data_dict is None: logger.error("Failed to load Gaia data."); sys.exit(1)
    logger.info(f"Loaded {len(gaia_data_dict['R_kpc'])} Gaia data points.")

    # Check complexity and recommend approach
    temp_fitted_names, _, _, _, _, _ = get_param_labels_and_bounds(args)
    n_params = len(temp_fitted_names)
    
    if n_params > 10 and not args.use_curriculum_learning:
        logger.warning(f"⚠️  Fitting {n_params} parameters without curriculum learning may be slow!")
        logger.warning("   Consider adding --use_curriculum_learning")
    
    # Initialize GP surrogate if requested
    gp_surrogate = None
    if args.use_gp_surrogate:
        if not GP_AVAILABLE:
            logger.error("GP surrogate requested but scikit-learn not available!")
            sys.exit(1)
        
        logger.info("🤖 Initializing Gaussian Process surrogate model...")
        
        # Need to get parameter bounds for GP
        _, _, _, p_low, p_high, _ = get_param_labels_and_bounds(args)
        param_bounds = np.column_stack([p_low, p_high])
        
        gp_surrogate = GPSurrogateModel(
            param_names=temp_fitted_names,
            param_bounds=param_bounds,
            uncertainty_threshold=args.gp_uncertainty_threshold,
            n_initial=args.gp_n_initial
        )
        
        # Generate initial training data
        def physics_wrapper(param_dict, args_obj):
            return v_model_for_dynesty(gaia_data_dict['R_kpc'], param_dict, args.xi, args_obj)
        
        gp_surrogate.generate_initial_training_data(physics_wrapper, args)
    
    # Choose sampling strategy
    if args.use_curriculum_learning:
        logger.info("🎓 Using CURRICULUM LEARNING approach")
        results = run_curriculum_learning(args, gaia_data_dict, logger)
    else:
        logger.info("🎯 Using standard sampling approach")
        results = run_single_dynesty(args, gaia_data_dict, gp_surrogate)
    
    
    # Process and save results
    if results is None:
        logger.error("No results to process. Exiting.")
        return
    
    # If curriculum learning, results is a dict of stages
    if isinstance(results, dict) and 'stage_1' in results:
        # Save each stage
        for stage_name, stage_results in results.items():
            if stage_results is None:
                continue
            
            output_prefix = f"dynesty_curriculum_{stage_name}_{args.xi}"
            output_npz_file = Path(args.output_dir) / f"{output_prefix}_samples.npz"
            
            try:
                weights = np.exp(stage_results.logwt - stage_results.logz[-1])
                np.savez(output_npz_file, 
                         samples=stage_results.samples,
                         weights=weights,
                         logl=stage_results.logl,
                         logz=stage_results.logz,
                         logzerr=stage_results.logzerr)
                logger.info(f"Saved {stage_name} results to {output_npz_file}")
            except Exception as e:
                logger.error(f"Failed to save {stage_name} results: {e}")
        
        # Use final stage for summary
        final_stage = max(results.keys())
        res = results[final_stage]
    else:
        # Standard single run
        res = results
        
        # Save results
        output_fname_parts = ["dynesty_mw", args.xi]
        if args.include_bulge: output_fname_parts.append("B" + ("f" if args.fit_bulge else "x"))
        if args.include_disk_thin: output_fname_parts.append("DT" + ("f" if args.fit_disk_thin else "x"))
        if args.include_disk_thick: output_fname_parts.append("DK" + ("f" if args.fit_disk_thick else "x"))
        if args.include_gas: output_fname_parts.append("G" + ("f" if args.fit_gas else "x"))
        output_basename = "_".join(output_fname_parts)
        
        output_npz_file = Path(args.output_dir) / f"{output_basename}_samples.npz"
        
        try:
            ess = res.effective_sample_size if hasattr(res, 'effective_sample_size') else 0
        except:
            weights = np.exp(res.logwt - res.logz[-1])
            ess = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0.0
        
        np.savez(output_npz_file, samples=res.samples, weights=np.exp(res.logwt - res.logz[-1]),
                 logl=res.logl, logz=res.logz, logzerr=res.logzerr, ess=ess, blob=res.blob)
        logger.info(f"Results saved to {output_npz_file}")
        
        # Save pickle
        output_pickle_file = Path(args.output_dir) / f"{output_basename}_results.pkl.gz"
        try:
            with gzip.open(output_pickle_file, "wb") as fh: 
                pickle.dump(res, fh)
            logger.info(f"Full Dynesty results object saved to {output_pickle_file}")
        except Exception as e: 
            logger.error(f"Failed to save full results object: {e}")
    
    # Report GP statistics if used
    if gp_surrogate is not None:
        gp_stats = gp_surrogate.get_statistics()
        logger.info("\n🤖 GP SURROGATE STATISTICS:")
        logger.info(f"  Real physics calls: {gp_stats['n_real_calls']:,}")
        logger.info(f"  Surrogate calls: {gp_stats['n_surrogate_calls']:,}")
        logger.info(f"  Speedup factor: {gp_stats['speedup_factor']:.1f}x")
        logger.info(f"  Training points: {gp_stats['n_training_points']}")
    
    logger.info("\n✨ AI-Enhanced dynesty run complete!")


if __name__ == "__main__":
    freeze_support()
    if not DYNESTY_AVAILABLE:
        print("CRITICAL (main entry): Dynesty library not found.")
        sys.exit(1)
    
    main_dynesty()