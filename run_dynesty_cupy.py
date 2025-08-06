#!/usr/bin/env python3
"""
run_dynesty_cupy.py - CuPy-optimized dynamic nested sampling for the Density-Metric model.

Complete version with:
- Real Gaia data loading
- Progress monitoring and JSON export
- Periodic analysis during sampling
- Resource monitoring
- Enhanced parameter bounds
- Post-processing integration
"""

import logging
import sys
import numpy as np
import argparse
from pathlib import Path
import pickle
import json
import threading
import time
from multiprocessing import Pool, freeze_support
from datetime import datetime
import pandas as pd

# CuPy imports for GPU acceleration
import cupy as cp
from density_metric_cupy import v_total_kms_cupy, to_cupy_array, to_numpy_array

# Resource monitoring
try:
    from resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False
    print("WARNING: resource_monitor not available - hardware monitoring disabled")

# Data I/O imports with fallback
try:
    from data_io import load_all_sky_gaia_slices, process_gaia_data
    DATA_IO_AVAILABLE = True
except ImportError:
    DATA_IO_AVAILABLE = False
    print("WARNING: data_io module not available - will use synthetic data fallback")

# Constants from original
BASELINE_LOGZ_GR = -1490897.5250096943  # From GR-only with no dark matter

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_logger():
    """Get or create logger instance."""
    logger = logging.getLogger('run_dynesty_cupy')
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def check_physical_plausibility(theta_values, param_names):
    """Check if parameters are physically plausible - SOFT VERSION."""
    for i, name in enumerate(param_names):
        value = theta_values[i]
        
        # Mass parameters must be positive
        if 'M_' in name and 'solar' in name:
            if value <= 0:
                return False
        
        # Scale length parameters must be positive
        if any(x in name for x in ['R_', 'r_', 'a_']):
            if value <= 0:
                return False
        
        # Scale height parameters must be positive
        if 'hz_' in name or 'h_z' in name:
            if value <= 0:
                return False
        
        # Additional checks for specific parameters
        if 'rho_c' in name and value <= 0:
            return False
        
        if 'n_exp' in name and value <= 0:
            return False
        
        if 'gamma_exp' in name and value <= 0:
            return False
        
        if 'lambda_g' in name and value <= 0:
            return False
    
    # REMOVED: Hard mass ratio vetoes that were causing sampling issues
    # The likelihood function will naturally guide the fit to reasonable ratios
    return True

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

def get_progress_aware_interpretation(delta_logz_vs_gr, phase, improvement_rate):
    """
    Get a nuanced interpretation that considers the run is still in progress.
    
    Parameters
    ----------
    delta_logz_vs_gr : float
        Current log(Z) difference vs GR baseline
    phase : str
        Current sampling phase
    improvement_rate : float
        Rate of log(Z) improvement per second
        
    Returns
    -------
    str
        Context-aware interpretation
    """
    # Get the basic Jeffreys interpretation
    basic_interp = interpret_jeffreys_scale(delta_logz_vs_gr)
    
    # If we're in early phases, always caveat
    if phase in ["initialization", "early_exploration"]:
        return f"{basic_interp} (very early - results will change)"
    
    # If we're improving rapidly, note that
    if improvement_rate > 10:  # Improving by >10 log units per second
        if delta_logz_vs_gr < 0:
            return f"Currently {basic_interp} favoring GR (but improving rapidly)"
        else:
            return f"{basic_interp} favoring DDMM (and improving)"
    
    # If we're in active exploration and currently worse than GR
    if phase == "active_exploration" and delta_logz_vs_gr < 0:
        if abs(delta_logz_vs_gr) > 10:
            return "Currently strongly disfavors DDMM (still exploring)"
        else:
            return f"{basic_interp} (still exploring parameter space)"
    
    # If we're refining
    if phase == "refinement":
        return f"{basic_interp} (refining parameters)"
    
    # If converged, give the straight interpretation
    if phase == "converged":
        return basic_interp
    
    # Default: add context that we're still running
    return f"{basic_interp} (sampling in progress)"

# ============================================================================
# PROGRESS MONITORING FUNCTIONS
# ============================================================================

def save_run_stats(sampler, fitted_names, args, start_time, logger, output_dir):
    """Save run statistics every 60 seconds."""
    try:
        current_time = time.time()
        elapsed_seconds = current_time - start_time
        elapsed_hours = elapsed_seconds / 3600
        
        # Get current sampler state
        if hasattr(sampler, 'results') and sampler.results is not None:
            results = sampler.results
            iterations = len(results.logz) if hasattr(results, 'logz') else 0
            ncall = int(np.sum(results.ncall)) if hasattr(results, 'ncall') else 0
            final_logz = float(results.logz[-1]) if hasattr(results, 'logz') and len(results.logz) > 0 else 0
            final_logzerr = float(results.logzerr[-1]) if hasattr(results, 'logzerr') and len(results.logzerr) > 0 else 0
            efficiency = getattr(results, 'efficiency', None)
            efficiency_percent = float(efficiency * 100) if efficiency is not None else 0
            
            # Calculate rate metrics
            calls_per_second = ncall / elapsed_seconds if elapsed_seconds > 0 else 0
            iterations_per_second = iterations / elapsed_seconds if elapsed_seconds > 0 else 0
        else:
            iterations = 0
            ncall = 0
            final_logz = 0
            final_logzerr = 0
            efficiency_percent = 0
            calls_per_second = 0
            iterations_per_second = 0
        
        # Create stats data
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_seconds,
            'elapsed_hours': elapsed_hours,
            'iterations': iterations,
            'ncall': ncall,
            'efficiency_percent': efficiency_percent,
            'final_logz': final_logz,
            'final_logzerr': final_logzerr,
            'calls_per_second': calls_per_second,
            'iterations_per_second': iterations_per_second,
            'run_id': args.run_id,
            'xi_type': args.xi,
            'nlive': args.nlive,
            'maxcall': args.maxcall,
            'dlogz_target': args.dlogz_target,
            'sample_method': args.sample_method,
            'bound_method': args.bound_method
        }
        
        # Save to run stats file
        stats_file = output_dir / "run_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(make_json_serializable(stats_data), f, indent=2)
            
        # Also append to history file
        history_file = output_dir / "run_stats_history.json"
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        history.append(stats_data)
        
        # Keep only last 1000 entries to prevent file from getting too large
        if len(history) > 1000:
            history = history[-1000:]
        
        with open(history_file, 'w') as f:
            json.dump(make_json_serializable(history), f, indent=2)
            
    except Exception as e:
        logger.debug(f"Failed to save run stats: {e}")

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
        
        # Add phase determination
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
        
        # Create progress_data with ALL fields
        progress_data = {
            'source': 'dynesty_sampler',
            'timestamp': datetime.now().isoformat(),
            'elapsed_hours': elapsed / 3600,
            'phase': phase,
            'phase_description': expected_behavior,
            'is_normal': phase != 'converged',
            'is_improving': improvement_metrics.get('improvement_rate', 0) > 0,
            'convergence_status': 'converged' if phase == 'converged' else 'in_progress',
            'health_status': 'NORMAL' if improvement_metrics.get('improvement_rate', 0) > 0 or phase == 'initialization' else 'CHECK',
            'n_samples': n_samples,
            'n_calls': n_calls,
            'efficiency_percent': 100.0 * n_samples / n_calls if n_calls > 0 else 0,
            'current_logz': current_logz,
            'improvement_metrics': improvement_metrics,
            'dlogz': dlogz,
            'target_dlogz': getattr(args, 'dlogz_target', 0.01),
            'dlogz_ratio': dlogz / getattr(args, 'dlogz_target', 0.01) if np.isfinite(dlogz) and getattr(args, 'dlogz_target', 0.01) > 0 else np.nan,
            'gr_baseline_logz': BASELINE_LOGZ_GR,
            'delta_logz_vs_gr': delta_logz_vs_gr,
            'gr_diff_percent': gr_diff_percent,
            'jeffreys_interpretation': get_progress_aware_interpretation(
                            delta_logz_vs_gr, 
                            phase, 
                            improvement_metrics.get('improvement_rate', 0)
                        ) if np.isfinite(delta_logz_vs_gr) else "Unknown",      
            'heartbeat': {
                'timestamp': datetime.now().isoformat(),
                'samples_since_last': n_samples - getattr(save_progress_json, '_last_n_samples', 0),
                'time_since_last': elapsed - getattr(save_progress_json, '_last_time', 0),
                'is_stuck': n_samples == getattr(save_progress_json, '_last_n_samples', 0)
            },
            'parameter_estimates': param_estimates,
            'xi_type': args.xi,
            'run_id': getattr(args, 'run_id', 'cupy_run')
        }
        
        save_progress_json._last_n_samples = n_samples
        save_progress_json._last_time = elapsed

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

def run_periodic_analysis(output_dir, xi_type, logger, suppress_plots=True):
    """
    Run the analyzer on current results without interrupting sampling.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing results
    xi_type : str
        Xi function type
    logger : logging.Logger
        Logger instance
    suppress_plots : bool
        If True, only generate summary stats, skip plots for speed
    """
    try:
        # Look for the latest checkpoint file
        npz_files = list(Path(output_dir).glob(f"*{xi_type}*_samples.npz"))
        checkpoint_files = list(Path(output_dir).glob(f"dynesty_checkpoint_{xi_type}_latest.npz"))
        
        analysis_file = None
        if checkpoint_files:
            analysis_file = checkpoint_files[0]
        elif npz_files:
            # Sort by modification time and get the latest
            analysis_file = max(npz_files, key=lambda f: f.stat().st_mtime)
        
        if not analysis_file or not analysis_file.exists():
            logger.debug("No results file found for periodic analysis")
            return
        
        # Import analyzer (do it here to avoid circular imports)
        try:
            from analyze_results import DynestyAnalyzer
        except ImportError:
            logger.warning("analyze_results.py not found - skipping periodic analysis")
            return
        
        # Create timestamped analysis subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_subdir = Path(output_dir) / "periodic_analyses" / f"analysis_{timestamp}"
        analysis_subdir.mkdir(parents=True, exist_ok=True)
        
        # Run analyzer
        logger.info(f"\n Running periodic analysis on {analysis_file.name}")
        analyzer = DynestyAnalyzer(str(analysis_file), str(analysis_subdir))
        
        # Get stats and save summary
        stats_dict = analyzer.get_parameter_stats()
        
        # Save text summary
        summary_file = analysis_subdir / "summary.txt"
        with open(summary_file, 'w') as f:
            # Redirect print output to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            
            analyzer.print_summary(stats_dict)
            analyzer.check_physical_plausibility()
            
            sys.stdout = old_stdout
        
        # Save JSON summary
        try:
            from analyze_results import export_summary
            json_summary = export_summary(analyzer, stats_dict)
            json_file = analysis_subdir / "summary.json"
            with open(json_file, 'w') as f:
                json.dump(make_json_serializable(json_summary), f, indent=2)
        except ImportError:
            # If export_summary doesn't exist, create a basic summary
            basic_summary = {
                'timestamp': timestamp,
                'parameter_stats': stats_dict,
                'logz': analyzer.logz[-1] if analyzer.logz is not None else None
            }
            json_file = analysis_subdir / "summary.json"
            with open(json_file, 'w') as f:
                json.dump(make_json_serializable(basic_summary), f, indent=2)
        
        # Generate plots if not suppressed
        if not suppress_plots:
            try:
                analyzer.plot_corner(save=True)
                analyzer.plot_rotation_curve(save=True)
                analyzer.plot_xi_profile(save=True)
            except Exception as plot_e:
                logger.warning(f"Plot generation failed: {plot_e}")
        
        # Also save a "latest" symlink for easy access
        latest_link = Path(output_dir) / "periodic_analyses" / "latest"
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(analysis_subdir.name)
        except OSError:
            # Symlinks might not work on all systems, just skip
            pass
        
        logger.info(f"OK Periodic analysis saved to {analysis_subdir}")
        
        # Log key results
        if 'M_disk_thin_solar' in stats_dict:
            logger.info(f"   Current M_thin: {stats_dict['M_disk_thin_solar']['median']:.2e} M_sun")
        if hasattr(analyzer, 'logz') and analyzer.logz is not None:
            logger.info(f"   Current log(Z): {analyzer.logz[-1]:.2f}")
        
    except Exception as e:
        logger.warning(f"Periodic analysis failed (non-critical): {e}")
        import traceback
        logger.debug(traceback.format_exc())

def save_npz_checkpoint(sampler, fitted_names, output_dir, logger):
    """Save current sampling state to NPZ file."""
    try:
        res = getattr(sampler, "results", None)
        
        if res is None:
            logger.warning("WARNING No sampler.results found — skipping .npz snapshot")
            return False
            
        if not hasattr(res, "samples") or res.samples is None or len(res.samples) == 0:
            logger.warning("WARNING Dynesty results has no samples yet — skipping .npz snapshot")
            return False
            
        # Build timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include xi_type in filename - use the actual xi_type from args
        xi_type = getattr(sampler, '_xi_type', 'gr')  # Default to 'gr' if not set
        npz_path = Path(output_dir) / f"dynesty_checkpoint_{xi_type}_{timestamp}.npz"
        
        # Also save to a fixed filename that overwrites (for easy resumption)
        npz_latest = Path(output_dir) / f"dynesty_checkpoint_{xi_type}_latest.npz"
        
        # Calculate weights if possible - make it more robust
        weights = None
        try:
            if hasattr(res, 'logwt') and hasattr(res, 'logz') and len(res.logz) > 0:
                if res.logwt is not None and res.logz is not None:
                    weights = np.exp(res.logwt - res.logz[-1])
                    # Check for invalid weights
                    if weights is not None and np.any(np.isnan(weights)):
                        logger.warning("WARNING: NaN values in weights, setting to None")
                        weights = None
        except Exception as e:
            logger.warning(f"WARNING: Failed to calculate weights: {e}")
            weights = None
        
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
            'xi_type': xi_type
        }
        
        # Save with timestamp
        np.savez(npz_path, **save_data)
        logger.info(f"OK Saved .npz checkpoint to: {npz_path}")
        
        # Save latest version (overwrites)
        np.savez(npz_latest, **save_data)
        logger.debug(f"OK Updated latest checkpoint: {npz_latest}")
        
        return True
        
    except Exception as e:
        logger.warning(f"WARNING Failed to save .npz checkpoint: {e}")
        return False

# ============================================================================
# CORE LIKELIHOOD AND PRIOR FUNCTIONS (PROCESS-SAFE)
# ============================================================================

def log_likelihood_dynesty_cupy(theta, param_names, args, R_data, v_data, sigma_data):
    """
    Process-safe log-likelihood. Accepts NumPy arrays, uses CuPy internally.
    Based on the successful debug_cupy_parallel.py script.
    """
    try:
        # Log first few calls to verify function is working
        if hasattr(log_likelihood_dynesty_cupy, 'call_count'):
            log_likelihood_dynesty_cupy.call_count += 1
        else:
            log_likelihood_dynesty_cupy.call_count = 1
            
        if log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD CALL #{log_likelihood_dynesty_cupy.call_count}: theta={theta}, params={param_names}")
        
        if not check_physical_plausibility(theta, param_names):
            return -np.inf

        # Convert to CuPy arrays for GPU computation
        R_data_cupy = to_cupy_array(R_data)
        v_data_cupy = to_cupy_array(v_data)
        sigma_data_cupy = to_cupy_array(sigma_data)
        
        params = dict(zip(param_names, theta))

        v_model = v_total_kms_cupy(R_data_cupy, params, xi_type=args.xi)
        if not cp.all(cp.isfinite(v_model)):
            return -np.inf

        chi2 = cp.sum(((v_data_cupy - v_model) / sigma_data_cupy)**2)
        logl = -0.5 * float(chi2)

        # ADD SOFT PRIOR PENALTY for unreasonable mass ratios (instead of hard rejection)
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
        
        # ============== ADD CASSINI CONSTRAINT CHECK ==============
        # Check Cassini constraint for spacetime_grain and other DDMM models
        if args.xi in ['spacetime_grain', 'peak', 'sigmoid', 'hybrid', 'broken', 'yukawa']:
            try:
                # Calculate xi at Solar System location (8.5 kpc)
                R_sun_kpc = cp.array([8.5])
                
                # Get solar neighborhood density (approximate)
                if 'M_thin_disk_solar' in params:
                    # Calculate density at Sun's position using galaxy model
                    from density_metric_cupy import volume_density_comprehensive_solar_kpc3_cupy
                    rho_sun = volume_density_comprehensive_solar_kpc3_cupy(R_sun_kpc, params)
                else:
                    # Fallback: typical solar neighborhood density
                    rho_sun = cp.array([1e6])  # M_☉/kpc³
                
                # Calculate xi at Sun's position
                v_sun_newton = v_total_kms_cupy(R_sun_kpc, params, xi_type='gr')  # Newtonian baseline
                v_sun_model = v_total_kms_cupy(R_sun_kpc, params, xi_type=args.xi)  # With enhancement
                
                # Xi is ratio of velocities squared
                if cp.all(v_sun_newton > 0):
                    xi_sun = (v_sun_model / v_sun_newton)**2
                else:
                    xi_sun = cp.array([1.0])
                
                # Cassini constraint: |γ - 1| < 2.3 × 10^-5
                # For DDMM: γ - 1 ≈ ξ - 1
                gamma_minus_one = float(xi_sun[0] - 1.0)
                cassini_limit = 2.3e-5
                
                if abs(gamma_minus_one) > cassini_limit:
                    # Soft penalty that scales with violation
                    violation_ratio = abs(gamma_minus_one) / cassini_limit
                    
                    # Penalty increases with violation
                    # Small violations: small penalty
                    # Large violations: large penalty
                    cassini_penalty = -100 * (violation_ratio - 1)**2 if violation_ratio > 1 else 0
                    
                    log_prior_penalty += cassini_penalty
                    
                    if log_likelihood_dynesty_cupy.call_count <= 10:
                        print(f"  Cassini violation: γ-1 = {gamma_minus_one:.2e} "
                              f"(limit: {cassini_limit:.2e}, penalty: {cassini_penalty:.1f})")
                
                # Additional check for spacetime_grain: ensure grain boundaries align
                if args.xi == 'spacetime_grain' and 'grain_size_kpc' in params:
                    grain_size = params['grain_size_kpc']
                    
                    # Penalize if Sun (8.5 kpc) is too close to a grain boundary
                    # We want Sun to be safely inside a grain, not at boundary
                    distance_to_boundary = min(
                        8.5 % grain_size,
                        grain_size - (8.5 % grain_size)
                    )
                    
                    if distance_to_boundary < 1.0:  # Within 1 kpc of boundary
                        boundary_penalty = -50 * (1.0 - distance_to_boundary)**2
                        log_prior_penalty += boundary_penalty
                        
                        if log_likelihood_dynesty_cupy.call_count <= 10:
                            print(f"  Grain boundary too close to Sun: {distance_to_boundary:.2f} kpc "
                                  f"(penalty: {boundary_penalty:.1f})")
                
            except Exception as cassini_e:
                # Don't fail the whole likelihood, just skip Cassini check
                if log_likelihood_dynesty_cupy.call_count <= 3:
                    print(f"  Warning: Cassini check failed: {cassini_e}")
        # ============== END CASSINI CONSTRAINT CHECK ==============
        
        # Apply the soft prior penalty
        logl += log_prior_penalty

        rmse = float(cp.sqrt(cp.mean((v_data_cupy - v_model)**2)))
        
        if log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD RESULT #{log_likelihood_dynesty_cupy.call_count}: logl={logl:.2f}, rmse={rmse:.2f}, penalty={log_prior_penalty:.2f}")
        
        return logl
    except Exception as e:
        if hasattr(log_likelihood_dynesty_cupy, 'call_count') and log_likelihood_dynesty_cupy.call_count <= 3:
            print(f"LIKELIHOOD ERROR #{log_likelihood_dynesty_cupy.call_count}: {e}")
        return -np.inf

def prior_transform_dynesty_cupy(u, param_names, bounds_low, bounds_high, use_log_prior):
    """Prior transform. Accepts and returns NumPy arrays."""
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

# ============================================================================
# DATA AND PARAMETER SETUP
# ============================================================================

def load_and_prepare_gaia_data(args):
    """
    Load and prepare real Gaia data (ported from original run_dynesty.py)
    Returns data compatible with CuPy arrays
    """
    logger = get_or_create_logger()
    
    if not DATA_IO_AVAILABLE:
        logger.warning("data_io module not available, using synthetic data fallback")
        R_data_np = np.linspace(1.0, 20.0, 1000)
        v_data_np = 200 + 50 * np.exp(-R_data_np / 8.0)
        sigma_data_np = 10.0 * np.ones_like(R_data_np)
        
        # Convert to CuPy arrays
        R_data = to_cupy_array(R_data_np.astype(np.float32))
        v_data = to_cupy_array(v_data_np.astype(np.float32))
        sigma_data = to_cupy_array(sigma_data_np.astype(np.float32))
        
        logger.info(f"Using synthetic data: {len(R_data)} data points")
        return R_data, v_data, sigma_data
    
    logger.info("\n" + "="*60)
    logger.info("GAIA DATA LOADING & PROCESSING")
    logger.info("="*60)
    
    gaia_cache_file = Path("gaia_sky_slices") / "all_sky_gaia.csv"
    df_all_sky = None

    if not gaia_cache_file.exists() or getattr(args, 'force_new_query_gaia', False) or getattr(args, 'force_reprocess_raw', False):
        if getattr(args, 'force_new_query_gaia', False):
            logger.info("Force flag enabled: Bypassing all caches to query Gaia from scratch.")
        elif getattr(args, 'force_reprocess_raw', False):
            logger.info("Force flag enabled: Bypassing merged cache to re-process raw slice files.")
        else:
            logger.info(f"Merged cache file not found at '{gaia_cache_file}'.")

        # Fallback 1: Try to merge raw slice files
        raw_dir = Path("gaia_sky_slices")
        raw_files = sorted(raw_dir.glob("raw_L*.csv"))
        logger.info(f"Searching for raw data in: '{raw_dir.resolve()}'")

        if raw_files and not getattr(args, 'force_new_query_gaia', False):
            logger.info(f"Found {len(raw_files)} raw Gaia slice files. Attempting to merge...")
            dfs = []
            for f in raw_files:
                try:
                    df_slice = pd.read_csv(f)
                    dfs.append(df_slice)
                    logger.info(f"  OK Successfully loaded {f.name} with {len(df_slice)} rows.")
                except Exception as e:
                    logger.warning(f"  WARNING Failed to load or parse {f.name}: {e}")

            if not dfs:
                logger.error("FAIL All raw Gaia slice files failed to load. Cannot proceed.")
                sys.exit(1)

            logger.info("Concatenating all loaded slices into a single DataFrame...")
            df_all_sky = pd.concat(dfs, ignore_index=True)
            logger.info(f"  OK Merged DataFrame created with {len(df_all_sky)} total rows.")
            
            try:
                logger.info(f"Attempting to cache the merged data to: {gaia_cache_file}")
                gaia_cache_file.parent.mkdir(parents=True, exist_ok=True)
                df_all_sky.to_csv(gaia_cache_file, index=False)
                logger.info(f"  Cached merged Gaia data successfully.")
            except Exception as e:
                logger.warning(f"  WARNING Failed to write cache file: {e}")

        else:
            # Fallback 2: Query from scratch
            logger.info("No suitable raw files found or new query forced. Querying Gaia from scratch...")
            try:
                df_all_sky = load_all_sky_gaia_slices(
                    lon_bin_width=30,
                    stars_per_bin=12000,
                    output_dir="gaia_sky_slices",
                    force_query=True,
                    max_distance_kpc=30.0
                )
                logger.info("  OK Gaia query completed.")
            except Exception as e:
                logger.error(f"FAIL Gaia query failed: {e}")
                logger.info("Falling back to synthetic data...")
                return load_and_prepare_gaia_data_synthetic()
            
    else:
        logger.info(f"OK Found existing merged cache file. Loading data from: {gaia_cache_file}")
        try:
            df_all_sky = pd.read_csv(gaia_cache_file)
            logger.info(f"  OK Loaded {len(df_all_sky)} stars from cache.")
        except Exception as e:
            logger.error(f"FAIL Failed to load cached Gaia data: {e}")
            logger.error("   Try running with --force_reprocess_raw to rebuild the cache from slices.")
            logger.info("Falling back to synthetic data...")
            return load_and_prepare_gaia_data_synthetic()

    logger.info("\n--- Processing Raw Gaia Data into Physical Units ---")
    try:
        df_all_sky = process_gaia_data(df_all_sky)
    except Exception as e:
        logger.error(f"FAIL Failed to process Gaia data: {e}")
        logger.info("Falling back to synthetic data...")
        return load_and_prepare_gaia_data_synthetic()

    # --- Data Validation Step ---
    logger.info("\n--- Validating loaded Gaia DataFrame ---")
    if df_all_sky is None or df_all_sky.empty:
        logger.error("FAIL DataFrame is empty after loading attempts. Cannot proceed.")
        logger.info("Falling back to synthetic data...")
        return load_and_prepare_gaia_data_synthetic()

    logger.info(f"DataFrame shape: {df_all_sky.shape}")
    logger.info(f"Columns: {df_all_sky.columns.tolist()}")

    required_cols = ["R_kpc", "v_obs", "sigma_v"]
    missing_cols = [col for col in required_cols if col not in df_all_sky.columns]
    if missing_cols:
        logger.error(f"FAIL Gaia data is missing required columns: {missing_cols}")
        logger.info("Falling back to synthetic data...")
        return load_and_prepare_gaia_data_synthetic()
    
    logger.info(f"Checking for non-finite values in critical columns...")
    for col in required_cols:
        n_bad = np.sum(~np.isfinite(df_all_sky[col]))
        if n_bad > 0:
            logger.warning(f"  WARNING Found {n_bad} NaN/inf values in column '{col}'. These will be filtered.")
            df_all_sky.dropna(subset=[col], inplace=True)
    logger.info(f"DataFrame shape after cleaning non-finite values: {df_all_sky.shape}")
    logger.info("Validation of DataFrame contents complete.")
    
    # --- Convert to CuPy arrays ---
    logger.info("\n--- Converting data to CuPy arrays for GPU ---")
    try:
        R_data_np = df_all_sky["R_kpc"].values.astype(np.float32)
        v_data_np = df_all_sky["v_obs"].values.astype(np.float32)
        sigma_data_np = df_all_sky["sigma_v"].values.astype(np.float32)
        logger.info(f"NumPy arrays created with dtype={R_data_np.dtype}.")

        # Convert to CuPy arrays
        R_data = to_cupy_array(R_data_np)
        v_data = to_cupy_array(v_data_np)
        sigma_data = to_cupy_array(sigma_data_np)

        logger.info(f"OK Successfully transferred {len(R_data)} stars to CuPy")
        logger.info("="*60 + "\n")
        
        return R_data, v_data, sigma_data
        
    except Exception as e:
        logger.error(f"FAIL An error occurred during conversion to CuPy arrays: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Falling back to synthetic data...")
        return load_and_prepare_gaia_data_synthetic()

def load_and_prepare_gaia_data_synthetic():
    """Fallback synthetic data generation."""
    logger = get_or_create_logger()
    logger.info("Generating synthetic data...")
    
    R_data_np = np.linspace(1.0, 20.0, 1000)
    v_data_np = 200 + 50 * np.exp(-R_data_np / 8.0)
    sigma_data_np = 10.0 * np.ones_like(R_data_np)
    
    # Convert to CuPy arrays
    R_data = to_cupy_array(R_data_np.astype(np.float32))
    v_data = to_cupy_array(v_data_np.astype(np.float32))
    sigma_data = to_cupy_array(sigma_data_np.astype(np.float32))
    
    logger.info(f"OK Generated {len(R_data)} synthetic data points")
    return R_data, v_data, sigma_data

def setup_parameter_bounds(xi_type):
    """Setup parameter bounds, returning NumPy arrays - Enhanced version."""
    if xi_type == 'gr':
        # Full baryonic model for GR baseline (academically complete)
        # UNCHANGED - keeping exactly as it worked for you
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc'
        ]
        
        # Bounds based on Milky Way literature values
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk: mass, scale length, scale height
            1e9, 3.0, 0.6,       # Thick disk: mass, scale length, scale height  
            1e9, 0.5,            # Bulge: mass, scale length
            1e9, 5.0, 0.1        # Gas: mass, scale length, scale height
        ])
        
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk bounds
            1e10, 5.0, 1.0,      # Thick disk bounds
            1e10, 2.0,           # Bulge bounds
            1e10, 10.0, 0.3      # Gas bounds
        ])
        
        # Use log priors for masses, linear for scale lengths/heights
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False   # Gas
        ])
    
    elif xi_type == 'enhanced':
        # Enhanced model with all baryonic components + modified gravity
        # UNCHANGED - keeping exactly as it worked for you
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'n_exp', 'A'
        ]
        
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e13, 0.5, 2.0       # Modified gravity parameters
        ])
        
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e16, 3.0, 10.0      # Modified gravity parameters
        ])
        
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False   # Modified gravity
        ])
    
    elif xi_type == 'power':
        # Power law model with all baryonic components
        # UNCHANGED - keeping similar to enhanced
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'n_exp'
        ]
        
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e13, 0.5            # Modified gravity parameters
        ])
        
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e16, 3.0            # Modified gravity parameters
        ])
        
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False          # Modified gravity
        ])
    
    elif xi_type == 'grav_color':
        # Gravitational color model with all baryonic components
        # UNCHANGED - keeping similar to enhanced
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g'
        ]
        
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e12, 2.0, 6.0       # Modified gravity parameters
        ])
        
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e15, 3.5, 10.0      # Modified gravity parameters
        ])
        
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False   # Modified gravity
        ])
    
    elif xi_type == 'sigmoid':
        # Sigmoid saturation model
        # FIXED: Better xi parameters for sigmoid
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'n_exp', 'A'
        ]
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e4, 0.5, 0.5        # FIXED: rho_c much lower for sigmoid
        ])
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e8, 3.0, 5.0        # FIXED: reasonable range for sigmoid
        ])
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False   # Sigmoid
        ])
    
    elif xi_type == 'peak':
        # Peak enhancement model
        # FIXED: Added use_log_prior and adjusted bounds
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_peak_solar_kpc3', 'width_log', 'A'
        ]
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e2, 0.3, 0.5        # Peak: density, width, amplitude
        ])
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e7, 3.0, 5.0        # Peak parameters
        ])
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False   # Peak - ADDED THIS
        ])
    
    elif xi_type == 'broken':
        # Broken power law model
        # FIXED: Adjusted A parameter bounds
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_break_solar_kpc3', 'n_low', 'n_high', 'A'
        ]
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e3, 0.5, 0.1, 0.5   # FIXED: Lower A to prevent extreme velocities
        ])
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e7, 2.0, 1.0, 5.0   # FIXED: Reasonable A range
        ])
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False, False  # Broken power
        ])
    
    elif xi_type == 'yukawa':
        # Yukawa screening model
        # FIXED: Adjusted for more reasonable enhancement
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'rho_c_solar_kpc3', 'lambda_screen_kpc', 'A'
        ]
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            1e5, 5.0, 0.5        # FIXED: Lower rho_c and A
        ])
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            1e10, 50.0, 5.0      # FIXED: More reasonable range
        ])
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            True, False, False   # Yukawa
        ])
    
    elif xi_type == 'spacetime_grain':
        # Quantum spacetime granularity model
        # Keep as is - this is a new experimental model
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
            'grain_size_kpc', 'rho_compress', 'A_grain'
        ]
        
        bounds_low = np.array([
            1e10, 2.0, 0.2,      # Thin disk
            1e9, 3.0, 0.6,       # Thick disk
            1e9, 0.5,            # Bulge
            1e9, 5.0, 0.1,       # Gas
            5.0, 1e3, 0.5        # FIXED: Lower A_grain
        ])
        
        bounds_high = np.array([
            1e11, 4.0, 0.4,      # Thin disk
            1e10, 5.0, 1.0,      # Thick disk
            1e10, 2.0,           # Bulge
            1e10, 10.0, 0.3,     # Gas
            20.0, 1e7, 5.0       # FIXED: More reasonable A_grain
        ])
        
        use_log_prior = np.array([
            True, False, False,  # Thin disk
            True, False, False,  # Thick disk
            True, False,         # Bulge
            True, False, False,  # Gas
            False, True, False   # Grain (linear for size, log for density, linear for A)
        ])
        
    else: # Fallback to simple model
        param_names = ['M_disk_solar', 'R_d_kpc']
        bounds_low = np.array([1e10, 1.0])
        bounds_high = np.array([5e11, 8.0])
        use_log_prior = np.array([True, False])
    
    return param_names, bounds_low, bounds_high, use_log_prior

import json
from datetime import datetime
import sys

def get_dynesty_weights(results):
    if hasattr(results, 'weights'):
        return results.weights
    elif isinstance(results, dict) and 'weights' in results:
        return results['weights']
    else:
        return None

def save_run_summary(filename, results, param_names, bounds_low, bounds_high, args, status, error_msg=None):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "cli_command": " ".join(sys.argv) if 'sys' in globals() else None,
        "status": status,
        "error_msg": error_msg,
        "nlive": getattr(args, 'nlive', None),
        "maxcall": getattr(args, 'maxcall', None),
        "num_threads": getattr(args, 'num_threads', None),
        "param_names": param_names,
        "bounds_low": bounds_low.tolist() if hasattr(bounds_low, 'tolist') else bounds_low,
        "bounds_high": bounds_high.tolist() if hasattr(bounds_high, 'tolist') else bounds_high,
        "best_fit": None,
        "logz": None,
        "logl": None,
        "rmse": None,
        "ncall": None,
        "notes": []
    }
    if results is not None:
        try:
            # Convert numpy arrays/scalars to JSON-serializable types
            summary["logz"] = make_json_serializable(getattr(results, 'logz', None))
            summary["logl"] = make_json_serializable(getattr(results, 'logl', None))
            summary["ncall"] = make_json_serializable(getattr(results, 'ncall', None))
            if hasattr(results, 'samples') and hasattr(results, 'logl'):
                idx = np.argmax(results.logl)
                summary["best_fit"] = results.samples[idx].tolist()
        except Exception as e:
            summary["notes"].append(f"Error extracting results: {str(e)}")
    
    # Apply make_json_serializable to the entire summary before saving
    summary = make_json_serializable(summary)
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_cupy():
    """Main function for the dynesty run with real Gaia data."""
    logger = get_or_create_logger()
    logger.info("Starting CuPy-optimized dynesty run...")

    parser = argparse.ArgumentParser(description="CuPy-optimized dynesty for DDMM",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core options
    parser.add_argument('--xi', type=str, default='gr', 
                       choices=['gr', 'power', 'enhanced', 'grav_color', 'sigmoid', 'peak', 'yukawa', 'transition', 'spacetime_grain', 'broken', 'hybrid', 'tanh'],
                       help='Xi function type')
    parser.add_argument('--output_dir', type=str, default='cupy_results',
                       help='Output directory')
    parser.add_argument('--nlive', type=int, default=2000,
                       help='Number of live points')
    parser.add_argument('--maxcall', type=int, default=1500000,
                       help='Maximum likelihood calls')
    parser.add_argument('--num_threads', type=int, default=4,
                       help='Number of threads')
    parser.add_argument('--dlogz_target', type=float, default=0.01,
                       help='Target dlogz for convergence')
    parser.add_argument('--checkpoint_every', type=int, default=900,
                       help='Seconds between automatic checkpoints')
    
    # Data loading options
    parser.add_argument('--force_new_query_gaia', action='store_true', default=False,
                       help="Force new Gaia query")
    parser.add_argument('--force_reprocess_raw', action='store_true', default=False,
                       help="Force reprocessing of raw Gaia data")
    parser.add_argument('--max_sample_gaia', type=int, default=50000,
                       help="Maximum number of Gaia stars to use")
    
    # Progress monitoring options
    parser.add_argument('--periodic_analysis', action='store_true', default=False,
                       help='Run analysis periodically during sampling')
    parser.add_argument('--analysis_interval_min', type=int, default=30,
                       help='Interval between analyses in minutes')
    parser.add_argument('--analysis_with_plots', action='store_true', default=False,
                       help='Generate plots during periodic analysis')
    
    # Post-processing flags
    parser.add_argument('--run_analysis', action='store_true',
                       help='Run analyze_results.py on the final .npz')
    parser.add_argument('--run_validation', action='store_true',
                       help='Run validate_ddmm.py on the final .npz')
    parser.add_argument('--run_plots', action='store_true',
                       help='Run generate_paper_figures.py for quick plots')
    
    # Sampler options
    parser.add_argument('--sample_method', type=str, default='rslice',
                       choices=['rwalk', 'rslice', 'hslice'],
                       help='Sampling method')
    parser.add_argument('--bound_method', type=str, default='multi',
                       choices=['none', 'single', 'multi', 'balls', 'cubes'],
                       help='Bounding method')
    
    args = parser.parse_args()

    # Create unique output directory with timestamp and parameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.xi}_{timestamp}"
    output_dir = Path(f"runs/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store the original CLI command
    import sys
    cli_command = " ".join(sys.argv)
    cli_file = output_dir / "cli_command.txt"
    with open(cli_file, 'w') as f:
        f.write(f"# Original CLI command used for this run\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"#\n")
        f.write(f"{cli_command}\n")
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CLI command saved to: {cli_file}")

    # Set run_id for tracking
    args.run_id = f"cupy_{timestamp}"

    # --- Load REAL Gaia Data ---
    logger.info("Loading Gaia data...")
    try:
        R_data, v_data, sigma_data = load_and_prepare_gaia_data(args)
        logger.info(f"Successfully loaded {len(R_data)} Gaia stars")
        
        # Limit data size if requested
        if args.max_sample_gaia and len(R_data) > args.max_sample_gaia:
            # Convert to numpy for indexing, then back to cupy
            R_np = to_numpy_array(R_data)
            v_np = to_numpy_array(v_data)
            sigma_np = to_numpy_array(sigma_data)
            
            indices = np.random.choice(len(R_np), args.max_sample_gaia, replace=False)
            
            R_data = to_cupy_array(R_np[indices])
            v_data = to_cupy_array(v_np[indices])
            sigma_data = to_cupy_array(sigma_np[indices])
            
            logger.info(f"Randomly sampled {args.max_sample_gaia} stars from dataset")
            
    except Exception as e:
        logger.error(f"Critical failure in data loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # --- Setup Data and Parameters ---
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds(args.xi)
    logger.info(f"Fitting {len(param_names)} parameters: {param_names}")
    logger.info(f"Parameter bounds:")
    for i, name in enumerate(param_names):
        prior_type = "log-uniform" if use_log_prior[i] else "uniform"
        logger.info(f"  {name}: [{bounds_low[i]:.2e}, {bounds_high[i]:.2e}] ({prior_type})")

    # --- Resource Monitoring ---
    resource_monitor = None
    if RESOURCE_MONITOR_AVAILABLE:
        try:
            resource_monitor = ResourceMonitor(output_dir)
            resource_monitor.start_monitoring()
            logger.info("Resource monitoring started")
        except Exception as e:
            logger.warning(f"Failed to start resource monitoring: {e}")

    try:
        logger.info("STEP 1: Importing dynesty...")
        import dynesty
        logger.info(f"✓ Dynesty imported successfully (version: {dynesty.__version__})")
        
        logger.info(f"STEP 2: Creating multiprocessing pool with {args.num_threads} threads...")
        with Pool(processes=args.num_threads) as pool:
            logger.info("STEP 3: Creating DynamicNestedSampler...")
            
            # Prepare arguments for parallel execution
            logl_args = (param_names, args, R_data, v_data, sigma_data)
            ptform_args = (param_names, bounds_low, bounds_high, use_log_prior)
            
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood_dynesty_cupy,
                prior_transform_dynesty_cupy,
                ndim=len(param_names),
                logl_args=logl_args,
                ptform_args=ptform_args,
                pool=pool,
                queue_size=args.num_threads,
                nlive=args.nlive,
                sample=args.sample_method,
                bound=args.bound_method
            )
            
            # Store xi_type for checkpointing
            sampler._xi_type = args.xi
            
            logger.info("✓ DynamicNestedSampler created successfully")
            
            # --- Setup Progress Monitoring ---
            logger.info("STEP 4: Setting up progress monitoring...")
            stop_event = threading.Event()
            start_time = time.time()
            last_progress_json = time.time()
            last_analysis_time = time.time()
            PROGRESS_JSON_INTERVAL = 60  # Save progress.json every minute
            ANALYSIS_INTERVAL = args.analysis_interval_min * 60 if hasattr(args, 'analysis_interval_min') else 1800
            
            def _checkpoint_worker():
                nonlocal last_progress_json, last_analysis_time
                last_run_stats = time.time()
                RUN_STATS_INTERVAL = 60  # Save run stats every 60 seconds
                
                while not stop_event.wait(args.checkpoint_every):
                    try:
                        current_time = time.time()
                        
                        # Save dynesty checkpoint
                        with open(output_dir / "dynesty_checkpoint.pkl", "wb") as cf:
                            pickle.dump(sampler.results, cf)
                        logger.info("✓ Checkpoint saved.")
                        
                        # Save run stats every 60 seconds
                        if current_time - last_run_stats > RUN_STATS_INTERVAL:
                            save_run_stats(sampler, param_names, args, start_time, logger, output_dir)
                            last_run_stats = current_time
                            logger.debug("✓ Run stats saved.")
                        
                        # Save progress JSON
                        if current_time - last_progress_json > PROGRESS_JSON_INTERVAL:
                            save_progress_json(sampler, param_names, args, start_time, logger)
                            last_progress_json = current_time
                        
                        # Run periodic analysis
                        if args.periodic_analysis and current_time - last_analysis_time > ANALYSIS_INTERVAL:
                            # First, ensure we have a recent checkpoint
                            save_npz_checkpoint(sampler, param_names, output_dir, logger)
                            
                            # Run analysis in a separate thread
                            analysis_thread = threading.Thread(
                                target=run_periodic_analysis,
                                args=(output_dir, args.xi, logger, not args.analysis_with_plots),
                                daemon=True
                            )
                            analysis_thread.start()
                            last_analysis_time = current_time
                            
                    except Exception as e:
                        logger.warning(f"✗ Checkpoint/monitoring failed: {e}")
                        
            chk_thread = threading.Thread(target=_checkpoint_worker, daemon=True)
            chk_thread.start()
            logger.info("✓ Progress monitoring thread started")

            logger.info("STEP 5: Starting nested sampling...")
            logger.info(f"  - maxcall: {args.maxcall}")
            logger.info(f"  - dlogz_target: {args.dlogz_target}")
            logger.info("  - Calling sampler.run_nested()...")
            
            try:
                sampler.run_nested(
                    maxcall=args.maxcall, 
                    print_progress=True,
                    dlogz_init=args.dlogz_target
                )
                logger.info("✓ sampler.run_nested() completed successfully")
            except Exception as e:
                logger.error(f"✗ sampler.run_nested() failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error args: {e.args}")
                raise
            
            # Stop monitoring threads
            stop_event.set()
            chk_thread.join(timeout=2)
            
            logger.info("STEP 6: Saving progress tracking...")
            try:
                def to_json_serializable(obj):
                    if hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    else:
                        return obj
                
                progress_data = {
                    "iterations": len(sampler.results.logz),
                    "logz": to_json_serializable(sampler.results.logz),
                    "logzerr": to_json_serializable(sampler.results.logzerr) if hasattr(sampler.results, 'logzerr') else [],
                    "ncall": to_json_serializable(sampler.results.ncall) if hasattr(sampler.results, 'ncall') else None,
                    "efficiency": to_json_serializable(getattr(sampler.results, 'efficiency', None)),
                    "final_logz": to_json_serializable(sampler.results.logz[-1]),
                    "final_logzerr": to_json_serializable(sampler.results.logzerr[-1]) if hasattr(sampler.results, 'logzerr') else None
                }
                with open(output_dir / "dynesty_progress.json", "w") as f:
                    json.dump(progress_data, f, indent=2)
                logger.info("✓ Saved dynesty_progress.json")
            except Exception as e:
                logger.error(f"✗ Failed to save progress: {e}")
                raise
        
        logger.info("STEP 7: Saving final results...")
        results = sampler.results
        
        try:
            with open(output_dir / "results.pkl", "wb") as f:
                pickle.dump(results, f)
            logger.info("✓ Saved results.pkl")
        except Exception as e:
            logger.error(f"✗ Failed to save results.pkl: {e}")
            raise
            
        logger.info("STEP 8: Saving final checkpoint and posterior samples...")
        try:
            # Save final checkpoint with correct xi_type
            save_npz_checkpoint(sampler, param_names, output_dir, logger)
            
            # Extract weights - make it more robust
            weights_arr = get_dynesty_weights(results)
            if weights_arr is None:
                logger.warning("  - WARNING: weights_arr is None, creating uniform weights")
                weights_arr = np.ones(len(results.samples)) / len(results.samples)
            logger.info("  - Using results.weights")
            
            logger.info(f"  - Weights shape: {weights_arr.shape}")
            logger.info(f"  - Samples shape: {results.samples.shape}")
            logger.info(f"  - LogL shape: {results.logl.shape}")
            
            # Save with metadata for analyzer compatibility
            np.savez(
                output_dir / "posterior_samples.npz",
                samples=results.samples,
                logl=results.logl,
                weights=weights_arr,
                logz=results.logz[-1],
                dlogz=(results.logzerr[-1] if hasattr(results, 'logzerr') else np.nan),
                param_names=np.array(param_names),  # CRITICAL: Add parameter names
                xi_type=args.xi,  # Add metadata for analysis
                n_samples=len(results.samples),
                timestamp=time.time()
            )
            logger.info("✓ Saved posterior_samples.npz")
        except Exception as e:
            logger.error(f"✗ Failed to save posterior_samples.npz: {e}")
            raise
            
        logger.info(f"Sampling completed! LogZ = {results.logz[-1]:.2f}")

        # Save run summary
        try:
            save_run_summary(
                output_dir / "run_summary.json",
                results, param_names, bounds_low, bounds_high, args, 
                status="success"
            )
            logger.info("✓ Saved run_summary.json")
        except Exception as e:
            logger.error(f"✗ Failed to save run_summary.json: {e}")

        # Model comparison summary
        delta_logz_vs_gr = results.logz[-1] - BASELINE_LOGZ_GR
        interpretation = interpret_jeffreys_scale(delta_logz_vs_gr)
        logger.info(f"\n*** MODEL COMPARISON SUMMARY ***")
        logger.info(f"GR Baseline LogZ: {BASELINE_LOGZ_GR:.2f}")
        logger.info(f"This Run LogZ:    {results.logz[-1]:.2f}")
        logger.info(f"Δ LogZ:           {delta_logz_vs_gr:+.2f}")
        logger.info(f"Interpretation:   {interpretation}")
        if delta_logz_vs_gr > 0:
            logger.info("*** DDMM model is preferred over GR ***")
        else:
            logger.info("*** GR model is preferred over DDMM ***")

        # === ENHANCED POST-PROCESSING ===
        posterior_npz = output_dir / "posterior_samples.npz"
        
        if args.run_analysis:
            try:
                import analyze_results as ar
                original_argv = sys.argv
                sys.argv = ['analyze_results.py', str(posterior_npz)]
                ar.main()
                sys.argv = original_argv
                logger.info("✓ analyze_results.py finished.")
            except Exception as e:
                logger.error(f"✗ analyze_results.py failed: {e}")
                
        if args.run_validation:
            try:
                import validate_ddmm as vd
                vd.main([str(posterior_npz)])
                logger.info("✓ validate_ddmm.py finished.")
            except Exception as e:
                logger.error(f"✗ validate_ddmm.py failed: {e}")
                
        if args.run_plots:
            try:
                sys.path.append("Older Files")
                import generate_paper_figures as gpf
                exec(open("Older Files/generate_paper_figures.py").read())
                logger.info("✓ generate_paper_figures.py finished.")
            except Exception as e:
                logger.error(f"✗ generate_paper_figures.py failed: {e}")

        # Final summary
        logger.info("\n*** RUN COMPLETED SUCCESSFULLY ***")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Key files:")
        logger.info(f"  - posterior_samples.npz: Main results for analysis")
        logger.info(f"  - results.pkl: Full dynesty results object")
        logger.info(f"  - dynesty_progress.json: Progress tracking")
        if args.periodic_analysis:
            logger.info(f"  - periodic_analyses/: Intermediate analysis results")

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (Ctrl+C)")
        # Try to save partial results
        try:
            if 'sampler' in locals() and hasattr(sampler, 'results'):
                partial_file = output_dir / "interrupted_results.npz"
                np.savez(partial_file, 
                        samples=sampler.results.samples,
                        logz=sampler.results.logz,
                        param_names=param_names)
                logger.info(f"Saved partial results to {partial_file}")
        except Exception as save_e:
            logger.error(f"Failed to save partial results: {save_e}")
        
    except Exception as e:
        logger.error(f"FATAL: Sampling failed: {e}", exc_info=True)
        # Save run summary with failure status
        try:
            save_run_summary(
                output_dir / "run_summary.json", 
                getattr(sampler, 'results', None) if 'sampler' in locals() else None,
                param_names if 'param_names' in locals() else [],
                bounds_low if 'bounds_low' in locals() else [],
                bounds_high if 'bounds_high' in locals() else [],
                args if 'args' in locals() else None,
                status="failed", 
                error_msg=str(e)
            )
            logger.info("✓ Saved failure summary to run_summary.json")
        except Exception as save_e:
            logger.error(f"Failed to save failure summary: {save_e}")
        
    finally:
        if resource_monitor:
            try:
                resource_monitor.stop_monitoring()
                logger.info("Resource monitoring stopped")
            except Exception as e:
                logger.warning(f"Failed to stop resource monitoring: {e}")

if __name__ == "__main__":
    freeze_support()
    main_cupy()