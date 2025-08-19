#!/usr/bin/env python3
"""
enhanced_summary.py - Enhanced summary generation for Dynesty runs

This module provides comprehensive yet concise summaries of Dynesty sampling runs,
focusing on the key metrics needed to evaluate run quality and results.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple
import sys

logger = logging.getLogger(__name__)

class DynestyRunSummary:
    """Enhanced summary generator for Dynesty runs."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        # Ensure the output directory exists to avoid file-not-found errors
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort: continue; writes may still fail but won't crash caller
            pass
        self.summary_file = self.output_dir / "run_summary_enhanced.json"
        self.console_summary_file = self.output_dir / "run_summary_console.txt"
        
    def generate_summary(self, sampler, param_names: List[str], args, 
                        start_time: float, status: str = "running") -> Dict[str, Any]:
        """Generate comprehensive summary of the current run state."""
        
        results = getattr(sampler, 'results', None)
        if results is None:
            return {"error": "No results available"}
            
        current_time = datetime.now()
        elapsed_seconds = current_time.timestamp() - start_time
        
        # Extract key metrics
        summary = {
            "metadata": {
                "timestamp": current_time.isoformat(),
                "elapsed_time": str(timedelta(seconds=int(elapsed_seconds))),
                "status": status,
                "xi_type": args.xi,
                "output_dir": str(self.output_dir)
            },
            
            "sampling_config": {
                "nlive": args.nlive,
                "maxcall": args.maxcall,
                "dlogz_target": args.dlogz_target,
                "sample_method": getattr(args, 'sample_method', 'rwalk'),
                "bound_method": getattr(args, 'bound_method', 'multi')
            },
            
            "convergence_metrics": self._get_convergence_metrics(results),
            "parameter_estimates": self._get_parameter_estimates(results, param_names),
            "evidence_metrics": self._get_evidence_metrics(results),
            "performance_metrics": self._get_performance_metrics(results, elapsed_seconds),
            "model_comparison": (self._get_model_comparison(results) if getattr(args, 'xi', None) != 'gr' else {}),
            "quality_assessment": self._assess_quality(results)
        }
        
        return summary
    
    def _get_convergence_metrics(self, results) -> Dict[str, Any]:
        """Extract convergence-related metrics."""
        metrics = {}
        
        # LogZ convergence
        if hasattr(results, 'logz') and len(results.logz) > 0:
            logz_array = np.array(results.logz)
            metrics['current_logz'] = float(logz_array[-1])
            
            if len(logz_array) > 1:
                # Incremental change in evidence between last two updates
                metrics['dlogz_increment'] = float(logz_array[-1] - logz_array[-2])
                # Back-compat alias (old name used in some tools)
                metrics['dlogz_current'] = metrics['dlogz_increment']
                
                # Moving average of incremental dlogz over last 10 iterations
                if len(logz_array) > 10:
                    recent_dlogz = np.diff(logz_array[-11:])
                    metrics['dlogz_avg_10'] = float(np.mean(recent_dlogz))
                    metrics['dlogz_std_10'] = float(np.std(recent_dlogz))
                    metrics['converging'] = metrics['dlogz_avg_10'] < 0.1
        
        # Dynesty's remaining-evidence estimate (stopping metric)
        if hasattr(results, 'logzerr') and len(results.logzerr) > 0:
            metrics['remaining_dlogz'] = float(results.logzerr[-1])
            # Keep original key for compatibility
            metrics['logz_error'] = metrics['remaining_dlogz']
        
        # Number of iterations
        metrics['iterations'] = len(results.logz) if hasattr(results, 'logz') else 0
        
        # Effective sample size
        if hasattr(results, 'samples'):
            metrics['n_samples'] = len(results.samples)
            
            # Calculate effective sample size from weights
            if hasattr(results, 'weights'):
                weights = np.array(results.weights)
                if len(weights) > 0 and np.sum(weights) > 0:
                    weights_norm = weights / np.sum(weights)
                    metrics['eff_samples'] = float(1.0 / np.sum(weights_norm**2))
                    metrics['eff_ratio'] = metrics['eff_samples'] / len(weights)
        
        return metrics
    
    def _get_parameter_estimates(self, results, param_names: List[str]) -> Dict[str, Any]:
        """Extract parameter estimates and uncertainties."""
        estimates = {}
        
        if not hasattr(results, 'samples') or not hasattr(results, 'logl'):
            return estimates
            
        # Best-fit parameters (maximum likelihood)
        idx_best = np.argmax(results.logl)
        best_params = results.samples[idx_best]
        
        estimates['best_fit'] = {
            name: float(val) for name, val in zip(param_names, best_params)
        }
        estimates['best_logl'] = float(results.logl[idx_best])
        
        # Weighted statistics if weights available
        if hasattr(results, 'weights'):
            weights = np.array(results.weights)
            if len(weights) > 0 and np.sum(weights) > 0:
                weights_norm = weights / np.sum(weights)
                
                # Weighted mean and std for each parameter
                estimates['weighted_mean'] = {}
                estimates['weighted_std'] = {}
                estimates['percentiles'] = {}
                
                for i, name in enumerate(param_names):
                    values = results.samples[:, i]
                    
                    # Weighted mean
                    mean_val = np.sum(values * weights_norm)
                    estimates['weighted_mean'][name] = float(mean_val)
                    
                    # Weighted std
                    var_val = np.sum((values - mean_val)**2 * weights_norm)
                    estimates['weighted_std'][name] = float(np.sqrt(var_val))
                    
                    # Percentiles
                    sorted_idx = np.argsort(values)
                    sorted_vals = values[sorted_idx]
                    sorted_weights = weights_norm[sorted_idx]
                    cumsum = np.cumsum(sorted_weights)
                    
                    percentiles = {}
                    for p in [16, 50, 84]:
                        idx = np.searchsorted(cumsum, p/100.0)
                        if idx < len(sorted_vals):
                            percentiles[f'p{p}'] = float(sorted_vals[idx])
                    estimates['percentiles'][name] = percentiles
        
        # Key parameters of interest
        if 'rho_c_solar_kpc3' in param_names:
            idx = param_names.index('rho_c_solar_kpc3')
            estimates['rho_c_log10'] = float(np.log10(best_params[idx]))
        
        return estimates
    
    def _get_evidence_metrics(self, results) -> Dict[str, Any]:
        """Extract evidence-related metrics."""
        metrics = {}
        
        # Current evidence
        if hasattr(results, 'logz') and len(results.logz) > 0:
            metrics['logz'] = float(results.logz[-1])
            
            # Evidence uncertainty
            if hasattr(results, 'logzerr') and len(results.logzerr) > 0:
                metrics['logz_error'] = float(results.logzerr[-1])
                metrics['logz_snr'] = abs(metrics['logz'] / metrics['logz_error']) if metrics['logz_error'] > 0 else np.inf
        
        # Information gain (H)
        if hasattr(results, 'information'):
            metrics['information'] = float(results.information[-1])
        
        return metrics
    
    def _get_performance_metrics(self, results, elapsed_seconds: float) -> Dict[str, Any]:
        """Extract performance-related metrics."""
        metrics = {}
        
        # Function calls
        if hasattr(results, 'ncall'):
            total_calls = np.sum(results.ncall)
            metrics['total_calls'] = int(total_calls)
            metrics['calls_per_second'] = float(total_calls / elapsed_seconds) if elapsed_seconds > 0 else 0
        
        # Efficiency
        if hasattr(results, 'samples') and hasattr(results, 'ncall'):
            n_samples = len(results.samples)
            total_calls = np.sum(results.ncall)
            if total_calls > 0:
                metrics['efficiency'] = float(n_samples / total_calls) * 100  # as percentage
        
        # Time estimates
        metrics['elapsed_time_seconds'] = elapsed_seconds
        metrics['elapsed_time_formatted'] = str(timedelta(seconds=int(elapsed_seconds)))
        
        # Estimate time to completion
        if hasattr(results, 'logz') and len(results.logz) > 10:
            recent_dlogz = np.diff(results.logz[-11:])
            avg_dlogz = np.mean(recent_dlogz)
            if avg_dlogz > 0:
                # Assuming exponential convergence
                target_dlogz = 0.01  # Target convergence
                current_dlogz = recent_dlogz[-1] if len(recent_dlogz) > 0 else 1.0
                if current_dlogz > target_dlogz:
                    # Simple linear estimate
                    iterations_needed = (current_dlogz - target_dlogz) / avg_dlogz * 10
                    time_per_iteration = elapsed_seconds / len(results.logz)
                    estimated_seconds = iterations_needed * time_per_iteration
                    metrics['estimated_time_remaining'] = str(timedelta(seconds=int(estimated_seconds)))
        
        return metrics
    
    def _get_model_comparison(self, results) -> Dict[str, Any]:
        """Compare with baseline models."""
        comparison = {}
        
        # GR baseline comparison
        BASELINE_LOGZ_GR = -1490897.5250096943
        
        if hasattr(results, 'logz') and len(results.logz) > 0:
            current_logz = float(results.logz[-1])
            delta_logz = current_logz - BASELINE_LOGZ_GR
            
            comparison['vs_gr'] = {
                'gr_baseline_logz': BASELINE_LOGZ_GR,
                'current_logz': current_logz,
                'delta_logz': delta_logz,
                'bayes_factor_log10': delta_logz / np.log(10),
                'interpretation': self._interpret_bayes_factor(delta_logz)
            }
        
        return comparison
    
    def _interpret_bayes_factor(self, delta_logz: float) -> str:
        """Interpret Bayes factor using Jeffreys scale."""
        if delta_logz < 0:
            return "Negative evidence (baseline preferred)"
        elif delta_logz < 1:
            return "Barely worth mentioning"
        elif delta_logz < 2.5:
            return "Substantial evidence"
        elif delta_logz < 5:
            return "Strong evidence"
        elif delta_logz < 10:
            return "Very strong evidence"
        else:
            return "Decisive evidence"
    
    def _assess_quality(self, results) -> Dict[str, Any]:
        """Assess overall run quality."""
        assessment = {
            'status': 'unknown',
            'warnings': [],
            'recommendations': []
        }
        
        # Check convergence
        if hasattr(results, 'logz') and len(results.logz) > 10:
            recent_dlogz = np.diff(results.logz[-11:])
            avg_dlogz = np.mean(recent_dlogz)
            
            if avg_dlogz < 0.01:
                assessment['status'] = 'converged'
            elif avg_dlogz < 0.1:
                assessment['status'] = 'converging'
            else:
                assessment['status'] = 'exploring'
                assessment['recommendations'].append("Run needs more iterations for convergence")
        
        # Check efficiency
        if hasattr(results, 'samples') and hasattr(results, 'ncall'):
            n_samples = len(results.samples)
            total_calls = np.sum(results.ncall)
            if total_calls > 0:
                efficiency = (n_samples / total_calls) * 100
                if efficiency < 1:
                    assessment['warnings'].append(f"Low efficiency: {efficiency:.2f}%")
                    assessment['recommendations'].append("Consider adjusting sampling parameters")
        
        # Check effective sample size
        if hasattr(results, 'weights'):
            weights = np.array(results.weights)
            if len(weights) > 0 and np.sum(weights) > 0:
                weights_norm = weights / np.sum(weights)
                eff_samples = 1.0 / np.sum(weights_norm**2)
                if eff_samples < 100:
                    assessment['warnings'].append(f"Low effective sample size: {eff_samples:.0f}")
                    assessment['recommendations'].append("Increase nlive or run longer")
        
        return assessment
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save summary to JSON file."""
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_console_summary(self, summary: Dict[str, Any]):
        """Save human-readable summary for console output."""
        lines = []
        lines.append("=" * 80)
        lines.append("DYNESTY RUN SUMMARY")
        lines.append("=" * 80)
        
        # Metadata
        lines.append(f"\nRun Information:")
        lines.append(f"  Timestamp: {summary['metadata']['timestamp']}")
        lines.append(f"  Elapsed: {summary['metadata']['elapsed_time']}")
        lines.append(f"  Status: {summary['metadata']['status']}")
        lines.append(f"  Xi Type: {summary['metadata']['xi_type']}")
        if 'output_dir' in summary.get('metadata', {}):
            lines.append(f"  Run Directory: {summary['metadata']['output_dir']}")
        
        # Convergence
        conv = summary['convergence_metrics']
        lines.append(f"\nConvergence Metrics:")
        lines.append(f"  Current LogZ: {conv.get('current_logz', 'N/A'):.2f}")
        lines.append(f"  Remaining dLogZ (Dynesty): {conv.get('remaining_dlogz', 'N/A'):.4f}")
        lines.append(f"  Incremental dLogZ (last step): {conv.get('dlogz_increment', 'N/A'):.4f}")
        lines.append(f"  Avg incremental dLogZ (last 10): {conv.get('dlogz_avg_10', 'N/A'):.4f}")
        lines.append(f"  Converged: {conv.get('converging', False)}")
        lines.append(f"  Samples: {conv.get('n_samples', 0)}")
        lines.append(f"  Effective Samples: {conv.get('eff_samples', 0):.0f}")
        
        # Key parameters
        params = summary['parameter_estimates']
        if 'best_fit' in params:
            lines.append(f"\nKey Parameters (Best Fit):")
            for key in ['rho_c_solar_kpc3', 'n_exp', 'A', 'lambda_g', 'gamma_exp']:
                if key in params['best_fit']:
                    val = params['best_fit'][key]
                    if 'rho_c' in key:
                        lines.append(f"  {key}: {val:.2e} (log10: {np.log10(val):.2f})")
                    else:
                        lines.append(f"  {key}: {val:.4f}")
        
        # Model comparison (skip for pure GR runs to avoid GR-vs-GR nonsense)
        if summary.get('metadata', {}).get('xi_type') != 'gr' and 'model_comparison' in summary and 'vs_gr' in summary['model_comparison']:
            comp = summary['model_comparison']['vs_gr']
            lines.append(f"\nModel Comparison vs GR:")
            lines.append(f"  Delta LogZ: {comp['delta_logz']:+.2f}")
            lines.append(f"  Bayes Factor (log10): {comp['bayes_factor_log10']:+.2f}")
            lines.append(f"  Interpretation: {comp['interpretation']}")
        
        # Performance
        perf = summary['performance_metrics']
        lines.append(f"\nPerformance:")
        lines.append(f"  Total Calls: {perf.get('total_calls', 0):,}")
        lines.append(f"  Efficiency: {perf.get('efficiency', 0):.2f}%")
        lines.append(f"  Calls/sec: {perf.get('calls_per_second', 0):.1f}")
        if 'estimated_time_remaining' in perf:
            lines.append(f"  Est. Time Remaining: {perf['estimated_time_remaining']}")
        
        # Quality assessment
        quality = summary['quality_assessment']
        lines.append(f"\nQuality Assessment:")
        lines.append(f"  Status: {quality['status']}")
        if quality['warnings']:
            lines.append(f"  Warnings:")
            for w in quality['warnings']:
                lines.append(f"    - {w}")
        if quality['recommendations']:
            lines.append(f"  Recommendations:")
            for r in quality['recommendations']:
                lines.append(f"    - {r}")
        
        lines.append("\n" + "=" * 80)
        
        # Save to file
        with open(self.console_summary_file, 'w') as f:
            f.write('\n'.join(lines))
        
        # Also print to console
        print('\n'.join(lines))
        
        return '\n'.join(lines)


def create_periodic_summary(sampler, param_names, args, start_time):
    """Create a summary during the run."""
    output_dir = Path(args.output_dir)
    summarizer = DynestyRunSummary(output_dir)
    
    summary = summarizer.generate_summary(sampler, param_names, args, start_time, status="running")
    summarizer.save_summary(summary)
    summarizer.save_console_summary(summary)
    
    return summary


def create_final_summary(sampler, param_names, args, start_time, status="completed"):
    """Create final summary after run completion."""
    output_dir = Path(args.output_dir)
    summarizer = DynestyRunSummary(output_dir)
    
    summary = summarizer.generate_summary(sampler, param_names, args, start_time, status=status)
    summarizer.save_summary(summary)
    console_output = summarizer.save_console_summary(summary)
    
    # Also save a compact version for quick reference
    compact_file = output_dir / "run_summary_compact.json"
    compact = {
        'timestamp': summary['metadata']['timestamp'],
        'xi_type': summary['metadata']['xi_type'],
        'logz': summary['convergence_metrics'].get('current_logz'),
        'remaining_dlogz': summary['convergence_metrics'].get('remaining_dlogz'),
        'dlogz_increment': summary['convergence_metrics'].get('dlogz_increment'),
        'converged': summary['convergence_metrics'].get('converging'),
        'best_fit_rho_c': summary['parameter_estimates'].get('best_fit', {}).get('rho_c_solar_kpc3'),
        'delta_logz_vs_gr': summary['model_comparison'].get('vs_gr', {}).get('delta_logz'),
        'efficiency': summary['performance_metrics'].get('efficiency'),
        'status': summary['quality_assessment']['status']
    }
    
    with open(compact_file, 'w') as f:
        json.dump(compact, f, indent=2)
    
    return summary, console_output