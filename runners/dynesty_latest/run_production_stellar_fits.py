#!/usr/bin/env python3
"""
run_production_stellar_fits.py - Production runs for all models including dark matter.

This script runs comprehensive stellar fitting for:
1. GR (General Relativity baseline)
2. NFW (Dark matter halo)
3. Power law xi modification
4. Exponential xi modification  
5. Gravitational color xi modification
6. MOND-like xi modification

Each model is run with production-quality settings for publication.
"""

import logging
import sys
import os
import json
import time
import numpy as np
import cupy as cp
from datetime import datetime
from pathlib import Path
import subprocess
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_runs_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production settings
PRODUCTION_SETTINGS = {
    'sample_max': 50000,      # Use up to 50k stars
    'maxcall': 2000000,       # 2 million likelihood calls
    'nlive': 1000,            # 1000 live points for good exploration
    'verbose': True,          # Show progress
}

# Models to test
MODELS = {
    'gr': {
        'xi': 'gr',
        'description': 'General Relativity (no modification)',
        'params': {},
        'fit_nfw': False
    },
    'nfw': {
        'xi': 'gr',  # GR with NFW halo
        'description': 'NFW Dark Matter Halo (ΛCDM)',
        'params': {},
        'fit_nfw': True  # This will fit NFW parameters
    },
    'power': {
        'xi': 'power',
        'description': 'Power Law Xi Modification',
        'params': {},
        'fit_nfw': False
    },
    'exponential': {
        'xi': 'exponential',
        'description': 'Exponential Xi Modification',
        'params': {},
        'fit_nfw': False,
        'maxcall': 1000000  # Reduced for exponential due to numerical issues
    },
    'grav_color': {
        'xi': 'grav_color',
        'description': 'Gravitational Color Confinement',
        'params': {},
        'fit_nfw': False
    },
    'mond': {
        'xi': 'mond',
        'description': 'MOND-like Xi Modification',
        'params': {},
        'fit_nfw': False
    }
}

def add_nfw_support_to_cupy_script():
    """
    Add NFW dark matter halo support to the CuPy stellar fitting script.
    This modifies the v_total_kms_cupy function to include NFW contribution.
    """
    logger.info("Adding NFW dark matter support to CuPy script...")
    
    # Create an enhanced version with NFW support
    nfw_code = '''
def v_nfw_kms_cupy(R_kpc, M_vir, c_vir):
    """
    Calculate NFW dark matter halo circular velocity.
    
    Parameters:
    -----------
    R_kpc : array
        Galactocentric radius in kpc
    M_vir : float
        Virial mass in M_sun
    c_vir : float
        Concentration parameter
    
    Returns:
    --------
    v_nfw : array
        NFW circular velocity in km/s
    """
    R_gpu = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    
    # NFW scale radius
    R_vir = (M_vir / (100 * 4.3e6))**(1/3)  # Approximate virial radius
    r_s = R_vir / c_vir
    
    # NFW enclosed mass function
    x = R_gpu / r_s
    g_x = cp.log(1 + x) - x / (1 + x)
    g_c = cp.log(1 + c_vir) - c_vir / (1 + c_vir)
    
    M_enc = M_vir * g_x / g_c
    
    # Circular velocity
    v_nfw_sq = G_ASTRO_UNITS * M_enc / R_gpu
    v_nfw = cp.sqrt(cp.maximum(v_nfw_sq, 0.0))
    
    return v_nfw

def v_total_kms_cupy_with_nfw(R_kpc, params, xi_type='power', include_nfw=False):
    """
    Extended version that includes optional NFW dark matter halo.
    """
    # Get baryonic + xi contribution
    v_baryon_xi = v_total_kms_cupy(R_kpc, params, xi_type)
    
    if include_nfw and 'M_vir' in params:
        # Add NFW contribution in quadrature
        M_vir = params.get('M_vir', 1e12)  # Virial mass
        c_vir = params.get('c_vir', 10.0)   # Concentration
        v_nfw = v_nfw_kms_cupy(R_kpc, M_vir, c_vir)
        
        # Total velocity: sqrt(v_baryon^2 + v_nfw^2)
        v_total = cp.sqrt(v_baryon_xi**2 + v_nfw**2)
        return v_total
    else:
        return v_baryon_xi
'''
    
    # Save enhanced version
    output_file = Path("run_dynesty_stellar_fit_cupy_nfw.py")
    
    # Read original file
    with open("run_dynesty_stellar_fit_cupy.py", 'r') as f:
        original_code = f.read()
    
    # Insert NFW code after the imports but before v_total_kms_cupy
    insert_pos = original_code.find("def v_total_kms_cupy")
    enhanced_code = original_code[:insert_pos] + nfw_code + "\n" + original_code[insert_pos:]
    
    # Modify the likelihood to use NFW version when needed
    enhanced_code = enhanced_code.replace(
        "v_model_gpu = v_total_kms_cupy(R_data_gpu, params, xi_type)",
        "v_model_gpu = v_total_kms_cupy_with_nfw(R_data_gpu, params, xi_type, include_nfw=params.get('include_nfw', False))"
    )
    
    with open(output_file, 'w') as f:
        f.write(enhanced_code)
    
    logger.info(f"Created NFW-enhanced version: {output_file}")
    return output_file

def run_single_model(model_name, model_config, settings):
    """
    Run stellar fitting for a single model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'gr', 'nfw', 'power')
    model_config : dict
        Configuration for the model
    settings : dict
        Production settings
    
    Returns:
    --------
    result : dict
        Results including chi2, RMSE, best parameters
    """
    logger.info("="*80)
    logger.info(f"Running {model_name.upper()}: {model_config['description']}")
    logger.info("="*80)
    
    # Build command
    if model_config.get('fit_nfw', False):
        # Use NFW-enhanced version
        script = "run_dynesty_stellar_fit_cupy_nfw.py"
        extra_args = ["--fit_nfw"]
    else:
        script = "run_dynesty_stellar_fit_cupy.py"
        extra_args = []
    
    # Check if script exists
    if not Path(script).exists():
        if 'nfw' in script:
            # Create NFW version if needed
            script = add_nfw_support_to_cupy_script()
    
    cmd = [
        "python", script,
        "--xi", model_config.get('xi', 'gr'),
        "--sample_max", str(settings.get('sample_max', PRODUCTION_SETTINGS['sample_max'])),
        "--maxcall", str(model_config.get('maxcall', settings.get('maxcall', PRODUCTION_SETTINGS['maxcall']))),
        "--nlive", str(settings.get('nlive', PRODUCTION_SETTINGS['nlive'])),
        "--output_dir", f"production_results/{model_name}",
        "--plot"
    ]
    
    if settings.get('verbose', False):
        cmd.append("--verbose")
    
    cmd.extend(extra_args)
    
    # Add any model-specific parameters
    for key, value in model_config.get('params', {}).items():
        cmd.extend([f"--{key}", str(value)])
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the fitting
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ {model_name} completed in {elapsed/60:.1f} minutes")
        
        # Parse results from output
        output_lines = result.stdout.split('\n')
        chi2 = None
        rmse = None
        for line in output_lines:
            if 'Chi²:' in line:
                chi2 = float(line.split('Chi²:')[1].strip())
            elif 'RMSE:' in line:
                rmse = float(line.split('RMSE:')[1].split('km/s')[0].strip())
        
        # Load detailed results from npz file
        results_file = Path(f"production_results/{model_name}") / f"stellar_fit_cupy_{model_config.get('xi', 'gr')}_results.npz"
        if results_file.exists():
            data = np.load(results_file)
            best_params = data['best_params']
            param_names = data['param_names']
        else:
            best_params = None
            param_names = None
        
        return {
            'success': True,
            'chi2': chi2,
            'rmse': rmse,
            'elapsed_minutes': elapsed/60,
            'best_params': best_params,
            'param_names': param_names,
            'output': result.stdout,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {model_name} failed after {elapsed/60:.1f} minutes")
        logger.error(f"Error: {e.stderr}")
        
        return {
            'success': False,
            'chi2': None,
            'rmse': None,
            'elapsed_minutes': elapsed/60,
            'best_params': None,
            'param_names': None,
            'output': e.stdout,
            'error': e.stderr
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {model_name} failed with exception: {e}")
        
        return {
            'success': False,
            'chi2': None,
            'rmse': None,
            'elapsed_minutes': elapsed/60,
            'best_params': None,
            'param_names': None,
            'output': None,
            'error': str(e)
        }

def create_comparison_table(results):
    """
    Create a comparison table of all model results.
    
    Parameters:
    -----------
    results : dict
        Results from all models
    
    Returns:
    --------
    table_str : str
        Formatted comparison table
    """
    # Sort models by chi2 (best first)
    sorted_models = sorted(
        [(k, v) for k, v in results.items() if v['success'] and v['chi2'] is not None],
        key=lambda x: x[1]['chi2']
    )
    
    table_lines = []
    table_lines.append("="*100)
    table_lines.append("MODEL COMPARISON RESULTS")
    table_lines.append("="*100)
    table_lines.append("")
    table_lines.append(f"{'Model':<15} {'Description':<35} {'Chi²':<12} {'RMSE (km/s)':<12} {'Time (min)':<10} {'Status':<10}")
    table_lines.append("-"*100)
    
    # Best model first
    if sorted_models:
        best_model, best_result = sorted_models[0]
        table_lines.append(
            f"{best_model:<15} {MODELS[best_model]['description']:<35} "
            f"{best_result['chi2']:<12.1f} {best_result['rmse']:<12.1f} "
            f"{best_result['elapsed_minutes']:<10.1f} {'✓ BEST':<10}"
        )
        
        # Other successful models
        for model, result in sorted_models[1:]:
            improvement = ((result['chi2'] - best_result['chi2']) / best_result['chi2']) * 100
            table_lines.append(
                f"{model:<15} {MODELS[model]['description']:<35} "
                f"{result['chi2']:<12.1f} {result['rmse']:<12.1f} "
                f"{result['elapsed_minutes']:<10.1f} {f'+{improvement:.1f}%':<10}"
            )
    
    # Failed models
    for model, result in results.items():
        if not result['success']:
            table_lines.append(
                f"{model:<15} {MODELS[model]['description']:<35} "
                f"{'--':<12} {'--':<12} "
                f"{result['elapsed_minutes']:<10.1f} {'✗ FAILED':<10}"
            )
    
    table_lines.append("-"*100)
    table_lines.append("")
    
    # Add parameter details for best model
    if sorted_models:
        best_model, best_result = sorted_models[0]
        if best_result['param_names'] is not None:
            table_lines.append(f"BEST MODEL PARAMETERS ({best_model}):")
            table_lines.append("-"*50)
            for i, name in enumerate(best_result['param_names']):
                if best_result['best_params'] is not None and i < len(best_result['best_params']):
                    table_lines.append(f"  {name:<30} {best_result['best_params'][i]:.3e}")
            table_lines.append("")
    
    return "\n".join(table_lines)

def main():
    """Main entry point for production runs."""
    parser = argparse.ArgumentParser(description='Run production stellar fits for all models')
    
    # Allow overriding specific settings
    parser.add_argument('--sample_max', type=int, default=PRODUCTION_SETTINGS['sample_max'],
                       help='Maximum number of stars to use')
    parser.add_argument('--maxcall', type=int, default=PRODUCTION_SETTINGS['maxcall'],
                       help='Maximum likelihood evaluations')
    parser.add_argument('--nlive', type=int, default=PRODUCTION_SETTINGS['nlive'],
                       help='Number of live points')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()) + ['all'],
                       default=['all'], help='Models to run')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with reduced settings')
    
    args = parser.parse_args()
    
    # Adjust settings
    settings = PRODUCTION_SETTINGS.copy()
    if args.quick:
        logger.info("Running in QUICK TEST mode with reduced settings")
        settings = {
            'sample_max': 1000,
            'maxcall': 10000,
            'nlive': 100,
            'verbose': True
        }
    else:
        settings['sample_max'] = args.sample_max
        settings['maxcall'] = args.maxcall
        settings['nlive'] = args.nlive
    
    # Determine which models to run
    if 'all' in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models
    
    logger.info("="*100)
    logger.info("PRODUCTION STELLAR FITTING RUN")
    logger.info("="*100)
    logger.info(f"Settings:")
    logger.info(f"  Max stars: {settings['sample_max']:,}")
    logger.info(f"  Max calls: {settings['maxcall']:,}")
    logger.info(f"  Live points: {settings['nlive']:,}")
    logger.info(f"  Models: {', '.join(models_to_run)}")
    logger.info("")
    
    # Create output directory
    Path("production_results").mkdir(exist_ok=True)
    
    # Run each model
    results = {}
    total_start = time.time()
    
    for model_name in models_to_run:
        if model_name not in MODELS:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue
        
        results[model_name] = run_single_model(
            model_name,
            MODELS[model_name],
            settings
        )
        
        # Save intermediate results
        with open("production_results/results_summary.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    total_elapsed = time.time() - total_start
    
    # Create comparison table
    table = create_comparison_table(results)
    logger.info("\n" + table)
    
    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'settings': settings,
        'models_run': models_to_run,
        'total_time_minutes': total_elapsed/60,
        'results': results,
        'comparison_table': table
    }
    
    with open("production_results/final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open("production_results/comparison_table.txt", 'w') as f:
        f.write(table)
    
    logger.info(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results saved to production_results/")
    
    # Determine winner
    successful = [(k, v) for k, v in results.items() if v['success'] and v['chi2'] is not None]
    if successful:
        best_model = min(successful, key=lambda x: x[1]['chi2'])
        logger.info(f"\n🏆 BEST MODEL: {best_model[0].upper()} with Chi² = {best_model[1]['chi2']:.1f}")
    
    return results

if __name__ == '__main__':
    results = main()
