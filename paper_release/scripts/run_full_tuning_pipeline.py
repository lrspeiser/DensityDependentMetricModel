#!/usr/bin/env python3
"""
run_full_tuning_pipeline.py - Comprehensive Parameter Tuning and Ablation Pipeline

This script provides the missing "full tuning pipeline" mentioned in the conversation history,
implementing systematic ablation studies and sensitivity analysis for the paper_release folder.

Features:
- Parameter sweep and ablation studies 
- Gate disabling/enabling (bulge, shear, bar gates)
- Sensitivity analysis (±50% parameter variations)
- Holdout validation with cluster lensing
- Statistical summary generation
- Reproducible pipeline with seeded runs

Usage:
    # Full ablation study
    python paper_release/scripts/run_full_tuning_pipeline.py --mode ablation --output results/ablation_study/
    
    # Sensitivity sweep
    python paper_release/scripts/run_full_tuning_pipeline.py --mode sensitivity --params L_0,beta_bulge,alpha_shear --output results/sensitivity/
    
    # Combined analysis
    python paper_release/scripts/run_full_tuning_pipeline.py --mode combined --output results/combined_analysis/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import subprocess
from dataclasses import dataclass, asdict
import itertools

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation_suite import ValidationSuite, ValidationResults
from scripts.run_mass_scaled_hierarchical_inference import load_cluster_catalog
from scripts.run_holdout_validation import validate_holdouts

@dataclass
class ParameterSet:
    """Parameter configuration for a single run"""
    run_id: str
    L_0: float = 2.5          # Coherence length scale [kpc]  
    beta_bulge: float = 1.0   # Bulge gate strength
    alpha_shear: float = 0.05 # Shear gate strength
    gamma_bar: float = 1.0    # Bar gate strength
    n_coh: int = 3            # Coherence index
    p: float = 0.5            # Power law index
    
    # Gate enables/disables for ablation
    enable_bulge_gate: bool = True
    enable_shear_gate: bool = True  
    enable_bar_gate: bool = True
    
    # Additional experimental parameters
    a0_m_s2: float = 1.2e-10
    gamma_exp: float = 3.0
    lambda_max: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_gate_config(self) -> str:
        """Get human-readable gate configuration"""
        gates = []
        if self.enable_bulge_gate: gates.append("bulge")
        if self.enable_shear_gate: gates.append("shear") 
        if self.enable_bar_gate: gates.append("bar")
        return "+".join(gates) if gates else "no_gates"

@dataclass 
class RunResults:
    """Results from a single pipeline run"""
    run_id: str
    params: ParameterSet
    galaxy_rar_scatter: float
    galaxy_btfr_scatter: float
    cluster_median_error: float
    cluster_holdout_pass: bool
    solar_system_pass: bool
    
    # Detailed metrics
    galaxy_metrics: Dict[str, float]
    cluster_metrics: Dict[str, float]
    validation_results: ValidationResults
    
    elapsed_time: float
    status: str = "completed"
    error_msg: Optional[str] = None

class TuningPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, output_dir: Path, random_seed: int = 42):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(output_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_baseline_params(self) -> ParameterSet:
        """Generate baseline parameter set"""
        return ParameterSet(
            run_id="baseline",
            L_0=2.5,
            beta_bulge=1.0,
            alpha_shear=0.05,
            gamma_bar=1.0,
            n_coh=3,
            p=0.5
        )
    
    def generate_ablation_configs(self) -> List[ParameterSet]:
        """Generate parameter configs for ablation study"""
        baseline = self.generate_baseline_params()
        configs = []
        
        # Baseline (all gates enabled)
        configs.append(baseline)
        
        # Single gate ablations
        for rid, updates in [
            ("no_bulge", {'enable_bulge_gate': False}),
            ("no_shear", {'enable_shear_gate': False}),
            ("no_bar", {'enable_bar_gate': False}),
        ]:
            cfg = baseline.to_dict()
            cfg.update(updates)
            cfg['run_id'] = rid
            configs.append(ParameterSet(**cfg))
        
        # Double gate ablations
        for rid, updates in [
            ("no_bulge_shear", {'enable_bulge_gate': False, 'enable_shear_gate': False}),
            ("no_bulge_bar", {'enable_bulge_gate': False, 'enable_bar_gate': False}),
            ("no_shear_bar", {'enable_shear_gate': False, 'enable_bar_gate': False}),
        ]:
            cfg = baseline.to_dict()
            cfg.update(updates)
            cfg['run_id'] = rid
            configs.append(ParameterSet(**cfg))
        
        # All gates disabled (GR baseline)
        cfg = baseline.to_dict()
        cfg.update({'enable_bulge_gate': False, 'enable_shear_gate': False, 'enable_bar_gate': False})
        cfg['run_id'] = 'no_gates'
        configs.append(ParameterSet(**cfg))
        
        return configs
    
    def generate_sensitivity_configs(self, param_names: List[str], 
                                   variations: List[float] = [-0.5, -0.25, 0.25, 0.5]) -> List[ParameterSet]:
        """Generate parameter configs for sensitivity analysis"""
        baseline = self.generate_baseline_params()
        configs = [baseline]  # Include baseline
        
        for param_name in param_names:
            if not hasattr(baseline, param_name):
                self.logger.warning(f"Parameter {param_name} not found in baseline config")
                continue
                
            baseline_value = getattr(baseline, param_name)
            
            for var in variations:
                new_value = baseline_value * (1 + var)
                # Ensure positive values
                if isinstance(new_value, (int, float)) and new_value <= 0:
                    new_value = baseline_value * 0.1  # Minimum positive value
                    
                run_id = f"{param_name}_{var:+.0%}"
                config_dict = baseline.to_dict()
                config_dict[param_name] = new_value
                config_dict['run_id'] = run_id
                
                configs.append(ParameterSet(**config_dict))
        
        return configs
    
    def run_single_validation(self, params: ParameterSet) -> RunResults:
        """Run validation suite for a single parameter configuration"""
        self.logger.info(f"Running validation for {params.run_id}")
        start_time = datetime.now()
        
        try:
            # Create parameter dictionary for validation suite
            validation_params = {
                'L_0': params.L_0,
                'beta_bulge': params.beta_bulge if params.enable_bulge_gate else 0.0,
                'alpha_shear': params.alpha_shear if params.enable_shear_gate else 0.0,
                'gamma_bar': params.gamma_bar if params.enable_bar_gate else 0.0,
                'n_coh': params.n_coh,
                'p': params.p,
                'a0_m_s2': params.a0_m_s2,
                'gamma_exp': params.gamma_exp,
                'lambda_max': params.lambda_max,
                'allow_experimental': True
            }
            
            # Run validation suite
            run_output_dir = self.output_dir / f"runs/{params.run_id}"
            suite = ValidationSuite(run_output_dir)
            
            # Quick validation checks  
            suite.test_newtonian_limit()
            suite.test_energy_conservation()
            suite.test_symmetry()
            
            train_df, test_df = suite.perform_train_test_split()
            suite.evaluate_model_selection()
            
            btfr_scatter, rar_scatter = suite.compute_btfr_rar(suite.sparc_data)
            outliers_df = suite.identify_problematic_galaxies(suite.sparc_data)
            
            suite.generate_validation_report()
            
            # Solar system check (Newtonian limit proxy)
            solar_system_pass = suite.results.newtonian_limit_passed
            
            # Cluster validation (real metrics if artifacts exist)
            cluster_median_error = float('nan')
            cluster_holdout_pass = False
            cluster_metrics = {}
            
            try:
                catalog_path = REPO_ROOT / 'data' / 'cluster_lensing_catalog.csv'
                posterior_mass_scaled = REPO_ROOT / 'output' / 'mass_scaled' / 'trace.netcdf'
                posterior_fixed = REPO_ROOT / 'output' / 'fixed_scale' / 'trace.netcdf'
                out_eval = run_output_dir / 'cluster_eval'
                
                if catalog_path.exists() and posterior_mass_scaled.exists():
                    # Import lazily to avoid heavy deps if not installed
                    import arviz as az
                    from scripts.run_holdout_validation import load_holdout_clusters, validate_holdouts
                    
                    holdouts = ['Abell1689','MACSJ1149.5+2223']
                    df_holdouts = load_holdout_clusters(holdouts, catalog_path=catalog_path)
                    trace = az.from_netcdf(posterior_mass_scaled)
                    summary = validate_holdouts(df_holdouts, trace, use_mass_scaling=True, output_dir=out_eval)
                    
                    cluster_median_error = float(summary.get('median_frac_error', float('nan')))
                    cluster_holdout_pass = bool(summary.get('pass_overall', False))
                    cluster_metrics = {
                        'median_fractional_error': cluster_median_error,
                        'inside_68_frac': float(summary.get('inside_68_frac', 0.0)),
                        'systematic_bias': summary.get('systematic_bias', 'none'),
                        'pass_overall': cluster_holdout_pass,
                        'n_holdouts': int(summary.get('n_holdouts', 0))
                    }
                else:
                    # No artifacts present; skip cluster eval (do not simulate)
                    cluster_metrics = {
                        'skipped': True,
                        'reason': 'Missing catalog or posterior (output/mass_scaled/trace.netcdf)'
                    }
            except Exception as ce:
                cluster_metrics = {
                    'error': str(ce)
                }
            
            # Compile results
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            galaxy_metrics = {
                'rar_scatter_dex': rar_scatter,
                'btfr_scatter_dex': btfr_scatter,
                'outliers_flagged': suite.results.outliers_flagged,
                'n_galaxies': len(suite.sparc_data)
            }
            
            return RunResults(
                run_id=params.run_id,
                params=params,
                galaxy_rar_scatter=rar_scatter,
                galaxy_btfr_scatter=btfr_scatter,
                cluster_median_error=cluster_median_error,
                cluster_holdout_pass=cluster_holdout_pass,
                solar_system_pass=solar_system_pass,
                galaxy_metrics=galaxy_metrics,
                cluster_metrics=cluster_metrics,
                validation_results=suite.results,
                elapsed_time=elapsed_time,
                status="completed"
            )
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in run {params.run_id}: {str(e)}")
            
            # Return failed result
            return RunResults(
                run_id=params.run_id,
                params=params,
                galaxy_rar_scatter=np.nan,
                galaxy_btfr_scatter=np.nan,
                cluster_median_error=np.nan,
                cluster_holdout_pass=False,
                solar_system_pass=False,
                galaxy_metrics={},
                cluster_metrics={},
                validation_results=ValidationResults(),
                elapsed_time=elapsed_time,
                status="failed",
                error_msg=str(e)
            )
    
    def run_ablation_study(self) -> List[RunResults]:
        """Run complete ablation study"""
        self.logger.info("Starting ablation study")
        
        configs = self.generate_ablation_configs()
        results = []
        
        for config in configs:
            result = self.run_single_validation(config)
            results.append(result)
            
            # Log progress
            status_symbol = "✓" if result.status == "completed" else "✗"
            self.logger.info(f"{status_symbol} {config.run_id}: "
                           f"RAR={result.galaxy_rar_scatter:.3f}, "
                           f"BTFR={result.galaxy_btfr_scatter:.3f}, "
                           f"Cluster={result.cluster_median_error:.3f}")
        
        return results
    
    def run_sensitivity_analysis(self, param_names: List[str]) -> List[RunResults]:
        """Run sensitivity analysis for specified parameters"""
        self.logger.info(f"Starting sensitivity analysis for: {param_names}")
        
        configs = self.generate_sensitivity_configs(param_names)
        results = []
        
        for config in configs:
            result = self.run_single_validation(config)
            results.append(result)
            
            status_symbol = "✓" if result.status == "completed" else "✗"
            self.logger.info(f"{status_symbol} {config.run_id}: "
                           f"RAR={result.galaxy_rar_scatter:.3f}")
        
        return results
    
    def generate_summary_report(self, results: List[RunResults], 
                              report_type: str = "combined") -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Filter successful results
        success_results = [r for r in results if r.status == "completed"]
        failed_results = [r for r in results if r.status == "failed"]
        
        if not success_results:
            return {"error": "No successful runs to analyze"}
        
        # Find best performing configurations
        best_rar = min(success_results, key=lambda r: r.galaxy_rar_scatter)
        best_combined = min(success_results, 
                           key=lambda r: r.galaxy_rar_scatter + r.cluster_median_error)
        
        # Gate ablation analysis
        gate_analysis = {}
        baseline_result = next((r for r in success_results if r.run_id == "baseline"), None)
        
        if baseline_result:
            baseline_rar = baseline_result.galaxy_rar_scatter
            
            for result in success_results:
                if result.run_id.startswith("no_"):
                    gate_config = result.params.get_gate_config()
                    delta_rar = result.galaxy_rar_scatter - baseline_rar
                    
                    gate_analysis[result.run_id] = {
                        'gate_config': gate_config,
                        'rar_scatter': result.galaxy_rar_scatter,
                        'delta_rar': delta_rar,
                        'degradation_pct': (delta_rar / baseline_rar) * 100,
                        'still_viable': result.galaxy_rar_scatter < 0.15  # Target threshold
                    }
        
        # Parameter sensitivity analysis
        sensitivity_analysis = {}
        if any("_+" in r.run_id or "_-" in r.run_id for r in success_results):
            baseline_rar = baseline_result.galaxy_rar_scatter if baseline_result else np.nan
            
            for result in success_results:
                if "_+" in result.run_id or "_-" in result.run_id:
                    param_name = result.run_id.split("_")[0]
                    if param_name not in sensitivity_analysis:
                        sensitivity_analysis[param_name] = []
                    
                    sensitivity_analysis[param_name].append({
                        'variation': result.run_id,
                        'rar_scatter': result.galaxy_rar_scatter,
                        'delta_rar': result.galaxy_rar_scatter - baseline_rar,
                        'param_value': getattr(result.params, param_name)
                    })
        
        # Overall statistics
        rar_scatters = [r.galaxy_rar_scatter for r in success_results if np.isfinite(r.galaxy_rar_scatter)]
        cluster_errors = [r.cluster_median_error for r in success_results if np.isfinite(r.cluster_median_error)]
        
        summary = {
            'report_type': report_type,
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(results),
            'successful_runs': len(success_results),
            'failed_runs': len(failed_results),
            
            'performance_summary': {
                'rar_scatter_range': [float(np.min(rar_scatters)), float(np.max(rar_scatters))],
                'rar_scatter_mean': float(np.mean(rar_scatters)),
                'cluster_error_range': [float(np.min(cluster_errors)), float(np.max(cluster_errors))],
                'cluster_error_mean': float(np.mean(cluster_errors)),
                'solar_system_pass_rate': sum(r.solar_system_pass for r in success_results) / len(success_results)
            },
            
            'best_performers': {
                'best_rar_model': {
                    'run_id': best_rar.run_id,
                    'rar_scatter': best_rar.galaxy_rar_scatter,
                    'gate_config': best_rar.params.get_gate_config()
                },
                'best_combined_model': {
                    'run_id': best_combined.run_id,
                    'combined_score': best_combined.galaxy_rar_scatter + best_combined.cluster_median_error,
                    'gate_config': best_combined.params.get_gate_config()
                }
            },
            
            'gate_ablation_analysis': gate_analysis,
            'parameter_sensitivity_analysis': sensitivity_analysis,
            
            'detailed_results': [
                {
                    'run_id': r.run_id,
                    'status': r.status,
                    'gate_config': r.params.get_gate_config(),
                    'rar_scatter': r.galaxy_rar_scatter,
                    'btfr_scatter': r.galaxy_btfr_scatter,
                    'cluster_error': r.cluster_median_error,
                    'solar_system_pass': r.solar_system_pass,
                    'elapsed_time': r.elapsed_time,
                    'error_msg': r.error_msg
                }
                for r in results
            ]
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive parameter tuning and ablation pipeline")
    
    parser.add_argument('--mode', choices=['ablation', 'sensitivity', 'combined'], 
                       default='combined', help='Type of analysis to run')
    parser.add_argument('--params', type=str, 
                       default='L_0,beta_bulge,alpha_shear,gamma_bar',
                       help='Comma-separated parameter names for sensitivity analysis')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    output_dir = Path(args.output)
    pipeline = TuningPipeline(output_dir, random_seed=args.seed)
    
    # Run analysis based on mode
    all_results = []
    
    if args.mode in ['ablation', 'combined']:
        pipeline.logger.info("Running ablation study...")
        ablation_results = pipeline.run_ablation_study()
        all_results.extend(ablation_results)
    
    if args.mode in ['sensitivity', 'combined']:
        param_names = [p.strip() for p in args.params.split(',')]
        pipeline.logger.info(f"Running sensitivity analysis for: {param_names}")
        sensitivity_results = pipeline.run_sensitivity_analysis(param_names)
        all_results.extend(sensitivity_results)
    
    # Generate summary report
    pipeline.logger.info("Generating summary report...")
    summary = pipeline.generate_summary_report(all_results, args.mode)
    
    # Save results
    summary_file = output_dir / 'pipeline_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed CSV
    csv_data = []
    for result in all_results:
        row = {
            'run_id': result.run_id,
            'status': result.status,
            'gate_config': result.params.get_gate_config(),
            **result.params.to_dict(),
            'rar_scatter': result.galaxy_rar_scatter,
            'btfr_scatter': result.galaxy_btfr_scatter,
            'cluster_error': result.cluster_median_error,
            'solar_system_pass': result.solar_system_pass,
            'elapsed_time': result.elapsed_time
        }
        csv_data.append(row)
    
    csv_file = output_dir / 'detailed_results.csv'
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("TUNING PIPELINE SUMMARY")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful: {summary['successful_runs']}")
    print(f"Failed: {summary['failed_runs']}")
    
    if summary['successful_runs'] > 0:
        perf = summary['performance_summary']
        print(f"\nRAR scatter range: {perf['rar_scatter_range'][0]:.3f} - {perf['rar_scatter_range'][1]:.3f}")
        print(f"Best RAR model: {summary['best_performers']['best_rar_model']['run_id']} "
              f"({summary['best_performers']['best_rar_model']['rar_scatter']:.3f})")
        
        if 'gate_ablation_analysis' in summary and summary['gate_ablation_analysis']:
            print(f"\nGate ablation findings:")
            for gate_test, analysis in summary['gate_ablation_analysis'].items():
                print(f"  {gate_test}: {analysis['degradation_pct']:+.1f}% change "
                      f"({'viable' if analysis['still_viable'] else 'problematic'})")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Details: {csv_file}")
    print("="*80)

if __name__ == "__main__":
    main()