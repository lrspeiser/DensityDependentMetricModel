#!/usr/bin/env python3
"""
Run all models in parallel for maximum GPU utilization and store complete results.
This script uses multiprocessing to run multiple models simultaneously.
"""

import subprocess
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Full model configurations for production run
MODELS = {
    # Baseline models - MUST match academic literature
    'gr': {
        'description': 'General Relativity (no dark matter)',
        'sample_max': 50000,
        'maxcall': 20000,
        'nlive': 200,
        'priority': 1  # Run first for baseline
    },
    'nfw': {
        'description': 'NFW Dark Matter Halo (Navarro-Frenk-White)',
        'sample_max': 50000,
        'maxcall': 30000,
        'nlive': 250,
        'priority': 1  # Run first for comparison
    },
    
    # Density-dependent models
    'power': {
        'description': 'Power Law ξ(ρ/ρ_c)^n',
        'sample_max': 75000,
        'maxcall': 40000,
        'nlive': 300,
        'priority': 2
    },
    'exponential': {
        'description': 'Exponential ξ = A*exp(-(ρ/ρ_c)^n)',
        'sample_max': 75000,
        'maxcall': 40000,
        'nlive': 300,
        'priority': 2
    },
    'logistic': {
        'description': 'Logistic ξ = 1 + A/(1 + exp(n*(ρ/ρ_c - 1)))',
        'sample_max': 75000,
        'maxcall': 40000,
        'nlive': 300,
        'priority': 2
    },
    'gaussian': {
        'description': 'Gaussian Enhancement ξ',
        'sample_max': 50000,
        'maxcall': 30000,
        'nlive': 250,
        'priority': 3
    },
    
    # Advanced models
    'grav_color': {
        'description': 'Gravitational Color (wavelength-dependent)',
        'sample_max': 75000,
        'maxcall': 40000,
        'nlive': 300,
        'priority': 2
    },
    'mond': {
        'description': 'MOND-like modification',
        'sample_max': 50000,
        'maxcall': 30000,
        'nlive': 250,
        'priority': 3
    }
}

def run_model_worker(args):
    """Worker function to run a single model."""
    model_name, config, output_dir, timestamp, cmd_args = args
    
    logger.info(f"Starting {model_name}: {config['description']}")
    
    # Create model-specific output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Build command
    cmd = [
        sys.executable,
        'run_dynesty_stellar_fit_cupy.py',
        '--xi', model_name,
        '--sample_max', str(cmd_args.n_samples if cmd_args.n_samples else config['sample_max']),
'--maxcall', str(cmd_args.maxcall if cmd_args.maxcall else config['maxcall']),
        '--nlive', str(cmd_args.n_live if cmd_args.n_live else config['nlive']),
        '--use_144k',  # Always use full dataset
        '--output_dir', str(model_dir),
        '--plot',  # Generate plots
        '--verbose'  # Get detailed output
    ]
    
    # Note: dlogz is not currently supported by run_dynesty_stellar_fit_cupy.py
    # It would need to be added to that script if needed
    
    # Add experimental flag if needed
    if config.get('experimental', False):
        cmd.append('--allow_experimental')
    
    # Run the model
    start_time = time.time()
    
    # Write command to log file
    log_file = model_dir / 'run_log.txt'
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write("-" * 80 + "\n")
    
    try:
        # Run with output capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Save full output
        with open(model_dir / 'stdout.txt', 'w') as f:
            f.write(result.stdout)
        with open(model_dir / 'stderr.txt', 'w') as f:
            f.write(result.stderr)
        
        # Parse results
        output = result.stdout + result.stderr
        
        # Extract key metrics
        chi2 = None
        rmse = None
        best_params = {}
        
        for line in output.split('\n'):
            if 'Chi²:' in line or 'Chi2:' in line:
                try:
                    chi2 = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'RMSE:' in line:
                try:
                    rmse = float(line.split('RMSE:')[1].split('km/s')[0].strip())
                except:
                    pass
            # Capture all parameter values
            for param in ['rho_c', 'n_exp', 'A', 'M_vir', 'c_vir', 'gamma', 'lambda_g']:
                if param in line and ':' in line and 'e' in line.split(':')[1]:
                    try:
                        value = float(line.split(':')[1].strip().split()[0])
                        best_params[param] = value
                    except:
                        pass
        
        # Load NPZ results for complete data
        npz_file = model_dir / f"stellar_fit_cupy_{model_name}_results.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            
            # Prefer metrics from NPZ over parsed logs
            chi2_npz = float(data['chi2']) if 'chi2' in data else None
            rmse_kms_npz = float(data['rmse_kms']) if 'rmse_kms' in data else None
            chi_per_star_npz = float(data['chi_per_star']) if 'chi_per_star' in data else None

            # Build best_params from NPZ if available
            best_params_npz = {}
            if 'param_names' in data and 'best_params' in data:
                names = data['param_names'].tolist()
                vals = data['best_params']
                for n, v in zip(names, vals):
                    try:
                        best_params_npz[n] = float(v)
                    except Exception:
                        pass

            # Merge with any parsed params
            if best_params_npz:
                best_params.update(best_params_npz)

            # Store complete sampling data
            results = {
                'success': True,
                'model_name': model_name,
                'description': config['description'],
                'elapsed_time': elapsed,
                'chi2': chi2_npz if chi2_npz is not None else chi2,
                'chi_per_star': chi_per_star_npz,
                'rmse': rmse_kms_npz if rmse_kms_npz is not None else rmse,
                'best_params': best_params,
                'samples': data['samples'] if 'samples' in data else None,
                'weights': data['weights'] if 'weights' in data else None,
                'logz': float(data['logz'][-1]) if 'logz' in data else None,
                'param_names': data['param_names'].tolist() if 'param_names' in data else [],
                'error': None
            }
            
            # Save as JSON for easy loading
            with open(model_dir / 'results_summary.json', 'w') as f:
                json_safe = {k: v for k, v in results.items() 
                            if k not in ['samples', 'weights']}
                json.dump(json_safe, f, indent=2)
        else:
            results = {
                'success': False,
                'model_name': model_name,
                'elapsed_time': elapsed,
                'error': 'No output file generated'
            }
        
        # Prefer RMSE from results dict (km/s if present)
        rmse_out = results.get('rmse', rmse)
        if rmse_out is not None:
            logger.info(f"Completed {model_name}: RMSE={rmse_out:.2f} km/s, Time={elapsed:.1f}s")
        else:
            logger.info(f"Completed {model_name}: Time={elapsed:.1f}s (no RMSE)")
        
    except subprocess.TimeoutExpired:
        elapsed = 1800
        results = {
            'success': False,
            'model_name': model_name,
            'elapsed_time': elapsed,
            'error': 'Timeout after 30 minutes'
        }
        logger.error(f"Timeout for {model_name}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        results = {
            'success': False,
            'model_name': model_name,
            'elapsed_time': elapsed,
            'error': str(e)
        }
        logger.error(f"Error in {model_name}: {e}")
    
    # Update log file
    with open(log_file, 'a') as f:
        f.write(f"\nEnd time: {datetime.now()}\n")
        f.write(f"Elapsed: {elapsed:.1f} seconds\n")
        f.write(f"Success: {results['success']}\n")
        if results.get('rmse'):
            f.write(f"RMSE: {results['rmse']:.2f} km/s\n")
    
    return results

def main(args):
    """Main function to orchestrate parallel model runs."""
    
    print("\n" + "=" * 80)
    print("FULL PARALLEL ANALYSIS - 144K GAIA DATASET")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Available CPUs: {cpu_count()}")
    
    # Show configuration if custom values
    if args.n_samples or args.n_live or args.dlogz:
        print("\nCustom parameters:")
        if args.n_samples:
            print(f"  Sample max: {args.n_samples}")
        if args.n_live:
            print(f"  Live points: {args.n_live}")
        if args.dlogz:
            print(f"  dlogz: {args.dlogz}")
    
    # Create main output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'full_analysis_{timestamp}')
    output_dir.mkdir(exist_ok=True)
    
    # Determine number of parallel workers
    # Use fewer workers than CPUs to leave room for GPU operations
    n_workers = min(4, cpu_count() - 2)  # Leave 2 cores for system
    print(f"Using {n_workers} parallel workers")
    
    # Sort models by priority
    model_list = sorted(MODELS.items(), key=lambda x: x[1].get('priority', 99))
    
    # Prepare arguments for workers
    worker_args = [
        (name, config, output_dir, timestamp, args) 
        for name, config in model_list
    ]
    
    # Run models in parallel
    print(f"\nRunning {len(MODELS)} models in parallel...")
    print("-" * 80)
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_model_worker, worker_args)
    
    # Compile results
    all_results = {r['model_name']: r for r in results if r.get('model_name')}
    
    # Save combined results
    combined_file = output_dir / 'all_results.json'
    with open(combined_file, 'w') as f:
        json_safe = {}
        for name, result in all_results.items():
            json_safe[name] = {
                k: v for k, v in result.items() 
                if k not in ['samples', 'weights']
            }
        json.dump(json_safe, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Sort by RMSE
    successful = [(name, r) for name, r in all_results.items() 
                  if r.get('success') and r.get('rmse')]
    successful.sort(key=lambda x: x[1]['rmse'])
    
    print("\nModel Rankings (by RMSE):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<15} {'RMSE (km/s)':<12} {'χ²':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    for i, (name, result) in enumerate(successful, 1):
        chi2_str = f"{result.get('chi2', 0):.1f}" if result.get('chi2') else "—"
        print(f"{i:<6} {name:<15} {result['rmse']:<12.2f} {chi2_str:<15} "
              f"{result['elapsed_time']:<10.1f}")
    
    # Print failed models
    failed = [name for name, r in all_results.items() if not r.get('success')]
    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
    
    # Highlight baseline comparisons
    print("\nAcademic Baseline Comparison:")
    print("-" * 40)
    if 'gr' in all_results and all_results['gr'].get('rmse'):
        print(f"GR (no DM):     {all_results['gr']['rmse']:.2f} km/s")
    if 'nfw' in all_results and all_results['nfw'].get('rmse'):
        print(f"NFW (with DM):  {all_results['nfw']['rmse']:.2f} km/s")
    
    # Best alternative model
    alt_models = [(n, r) for n, r in successful if n not in ['gr', 'nfw']]
    if alt_models:
        best_alt = min(alt_models, key=lambda x: x[1]['rmse'])
        print(f"\nBest Alternative: {best_alt[0]} ({best_alt[1]['rmse']:.2f} km/s)")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total runtime: {sum(r['elapsed_time'] for r in all_results.values()):.1f} seconds")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run parallel analysis of all models')
    parser.add_argument('--use_144k', action='store_true', help='Use 144k dataset')
    parser.add_argument('--n_samples', type=int, help='Override sample_max for all models')
    parser.add_argument('--n_live', type=int, help='Override number of live points')
    parser.add_argument('--dlogz', type=float, help='Stopping criterion for nested sampling')
    parser.add_argument('--maxcall', type=int, help='Override maxcall for all models')
    
    args = parser.parse_args()
    output_dir = main(args)
