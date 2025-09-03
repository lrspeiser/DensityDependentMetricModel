#!/usr/bin/env python3
"""
run_production_stellar_fits_all.py - Comprehensive production runs for all models.

This script runs stellar fitting for:
1. GR (General Relativity baseline)
2. NFW (Dark matter halo)
3. Power law xi modification
4. Exponential xi modification
5. Gravitational color xi modification
6. MOND-like xi modification
7. Tidal band models (tidal_band, tidal_band2, tidal_ratio, tidal_noisyor)
8. RAR-anchored models (rar_gate, rar_blend)

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
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_runs_all_{datetime.now():%Y%m%d_%H%M%S}.log'),
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

# Quick test settings (for debugging)
TEST_SETTINGS = {
    'sample_max': 1000,
    'maxcall': 10000,
    'nlive': 100,
    'verbose': True,
}

# Models to test
MODELS = {
    'gr': {
        'xi': 'gr',
        'description': 'General Relativity (no modification)',
        'allow_experimental': False,
        'expected_chi2_range': (1000, 5000)  # Expected chi2 range for quality check
    },
    'nfw': {
        'xi': 'nfw',
        'description': 'NFW Dark Matter Halo (ΛCDM)',
        'allow_experimental': False,
        'expected_chi2_range': (500, 2000)
    },
    'power': {
        'xi': 'power',
        'description': 'Power Law Xi Modification',
        'allow_experimental': False,
        'expected_chi2_range': (400, 1500)
    },
    'exponential': {
        'xi': 'exponential',
        'description': 'Exponential Xi Modification',
        'allow_experimental': False,
        'maxcall': 1000000,  # Reduced for exponential due to numerical issues
        'expected_chi2_range': (400, 1500)
    },
    'grav_color': {
        'xi': 'grav_color',
        'description': 'Gravitational Color Confinement',
        'allow_experimental': False,
        'expected_chi2_range': (350, 1200)
    },
    'mond': {
        'xi': 'mond',
        'description': 'MOND-like Xi Modification',
        'allow_experimental': False,
        'expected_chi2_range': (400, 1500)
    },
    'tidal_band': {
        'xi': 'tidal_band',
        'description': 'Tidal Band with Density Screening',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_band2': {
        'xi': 'tidal_band2',
        'description': 'Tidal Band v2 (Logistic lnT onset)',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_ratio': {
        'xi': 'tidal_ratio',
        'description': 'Tidal Ratio Trigger',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_noisyor': {
        'xi': 'tidal_noisyor',
        'description': 'Tidal Noisy-OR Aggregator',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'rar_gate': {
        'xi': 'rar_gate',
        'description': 'RAR-anchored Gating on g_bar',
        'allow_experimental': True,
        'expected_chi2_range': (250, 900)
    },
    'rar_blend': {
        'xi': 'rar_blend',
        'description': 'RAR Blend (amplitude × RAR excess)',
        'allow_experimental': True,
        'expected_chi2_range': (250, 900)
    }
}

def run_single_model(model_name, model_config, settings):
    """
    Run stellar fitting for a single model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'gr', 'nfw', 'tidal_band')
    model_config : dict
        Configuration for the model
    settings : dict
        Production or test settings
    
    Returns:
    --------
    result : dict
        Results including chi2, RMSE, best parameters
    """
    logger.info("="*80)
    logger.info(f"Running {model_name.upper()}: {model_config['description']}")
    logger.info("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "run_dynesty_stellar_fit_cupy.py",
        "--xi", model_config['xi'],
        "--sample_max", str(settings['sample_max']),
        "--maxcall", str(model_config.get('maxcall', settings['maxcall'])),
        "--nlive", str(settings['nlive']),
        "--output_dir", f"production_results_{model_name}"
    ]
    
    if model_config.get('allow_experimental', False):
        cmd.append("--allow_experimental")
    
    if settings['verbose']:
        cmd.append("--verbose")
    
    # Add plot flag
    cmd.append("--plot")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the model
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        
        # Parse output for results
        output_lines = result.stdout.split('\n')
        chi2 = None
        rmse = None
        logl = None
        
        for line in output_lines:
            if 'Chi²:' in line:
                chi2 = float(line.split(':')[1].strip())
            elif 'RMSE:' in line:
                rmse = float(line.split(':')[1].strip().split()[0])
            elif 'Log(L):' in line:
                logl = float(line.split(':')[1].strip())
        
        # Check if results are in expected range
        if chi2 and 'expected_chi2_range' in model_config:
            min_chi2, max_chi2 = model_config['expected_chi2_range']
            if not (min_chi2 <= chi2 <= max_chi2):
                logger.warning(f"⚠ Chi² = {chi2:.1f} is outside expected range [{min_chi2}, {max_chi2}]")
        
        logger.info(f"✓ {model_name} completed in {elapsed:.1f}s")
        logger.info(f"  Chi²: {chi2:.1f}")
        logger.info(f"  RMSE: {rmse:.1f} km/s")
        logger.info(f"  Log(L): {logl:.1f}")
        
        return {
            'model': model_name,
            'chi2': chi2,
            'rmse': rmse,
            'logl': logl,
            'runtime': elapsed,
            'success': True
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {model_name} failed after {elapsed:.1f}s")
        logger.error(f"  Error: {e.stderr}")
        
        return {
            'model': model_name,
            'chi2': None,
            'rmse': None,
            'logl': None,
            'runtime': elapsed,
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {model_name} failed with unexpected error after {elapsed:.1f}s")
        logger.error(f"  Error: {str(e)}")
        
        return {
            'model': model_name,
            'chi2': None,
            'rmse': None,
            'logl': None,
            'runtime': elapsed,
            'success': False,
            'error': str(e)
        }

def create_summary_table(results):
    """
    Create a summary table of all results.
    
    Parameters:
    -----------
    results : list of dict
        Results from all models
    
    Returns:
    --------
    df : pandas.DataFrame
        Summary table
    """
    df = pd.DataFrame(results)
    
    # Sort by chi2 (best first)
    df = df.sort_values('chi2', na_position='last')
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['rank', 'model', 'chi2', 'rmse', 'logl', 'runtime', 'success']
    df = df[cols]
    
    return df

def create_comparison_plot(results):
    """
    Create comparison plots of model performance.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Filter successful results
        successful = [r for r in results if r['success'] and r['chi2'] is not None]
        
        if not successful:
            logger.warning("No successful results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Chi2 comparison
        models = [r['model'] for r in successful]
        chi2_values = [r['chi2'] for r in successful]
        
        ax1.bar(range(len(models)), chi2_values, color='steelblue')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Chi²')
        ax1.set_title('Model Comparison - Chi²')
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_values = [r['rmse'] for r in successful]
        
        ax2.bar(range(len(models)), rmse_values, color='darkgreen')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('RMSE (km/s)')
        ax2.set_title('Model Comparison - RMSE')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'model_comparison_{datetime.now():%Y%m%d_%H%M%S}.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        logger.info(f"Comparison plot saved to: {plot_file}")
        
    except Exception as e:
        logger.warning(f"Could not create comparison plot: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run production stellar fits for all models')
    
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced settings')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to run (default: all)')
    parser.add_argument('--skip_published', action='store_true',
                       help='Skip published models and only run experimental ones')
    
    args = parser.parse_args()
    
    # Select settings
    if args.test:
        logger.info("Running in TEST mode with reduced settings")
        settings = TEST_SETTINGS
    else:
        logger.info("Running in PRODUCTION mode")
        settings = PRODUCTION_SETTINGS
    
    # Select models to run
    if args.models:
        models_to_run = {k: v for k, v in MODELS.items() if k in args.models}
    elif args.skip_published:
        models_to_run = {k: v for k, v in MODELS.items() 
                        if v.get('allow_experimental', False)}
    else:
        models_to_run = MODELS
    
    logger.info(f"Will run {len(models_to_run)} models: {list(models_to_run.keys())}")
    
    # Run all models
    results = []
    total_start = time.time()
    
    for i, (model_name, model_config) in enumerate(models_to_run.items(), 1):
        logger.info(f"\n[{i}/{len(models_to_run)}] Processing {model_name}...")
        result = run_single_model(model_name, model_config, settings)
        results.append(result)
        
        # Save intermediate results
        df = create_summary_table(results)
        df.to_csv(f'results_summary_{datetime.now():%Y%m%d_%H%M%S}.csv', index=False)
    
    total_elapsed = time.time() - total_start
    
    # Create final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    df = create_summary_table(results)
    logger.info(f"\n{df.to_string()}")
    
    # Save final results
    output_file = f'final_results_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Create comparison plot
    create_comparison_plot(results)
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    logger.info(f"\nCompleted {len(successful)}/{len(results)} models successfully")
    logger.info(f"Total runtime: {total_elapsed/60:.1f} minutes")
    
    if successful:
        best_model = min(successful, key=lambda x: x['chi2'] if x['chi2'] else float('inf'))
        logger.info(f"\nBest model: {best_model['model']}")
        logger.info(f"  Chi²: {best_model['chi2']:.1f}")
        logger.info(f"  RMSE: {best_model['rmse']:.1f} km/s")

if __name__ == '__main__':
    main()
