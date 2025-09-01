#!/usr/bin/env python3
"""
run_all_stellar_fits.py - Unified script to run all stellar fitting models.

This script properly handles the different parameter naming conventions and
function signatures used by the various components of the DDMM codebase.
"""

import logging
import sys
import os
import time
import json
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
        logging.FileHandler(f'all_stellar_fits_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production settings
PRODUCTION_SETTINGS = {
    'sample_max': 200000,      # Use ALL available stars (144k in dataset)
    'maxcall': 40000000,       # 40 million likelihood calls for thorough exploration
    'nlive': 2000,             # 2000 live points for excellent exploration
}

# Quick test settings (for debugging)
TEST_SETTINGS = {
    'sample_max': 1000,
    'maxcall': 10000,
    'nlive': 100,
}

# Models to test
MODELS = {
    'gr': {
        'description': 'General Relativity (no modification)',
        'allow_experimental': False,
        'expected_chi2_range': (1000, 5000)
    },
    'nfw': {
        'description': 'NFW Dark Matter Halo (ΛCDM)',
        'allow_experimental': False,
        'expected_chi2_range': (500, 2000)
    },
    'power': {
        'description': 'Power Law Xi Modification',
        'allow_experimental': False,
        'expected_chi2_range': (400, 1500)
    },
    'exponential': {
        'description': 'Exponential Xi Modification',
        'allow_experimental': False,
        'maxcall': 1000000,  # Reduced for exponential due to numerical issues
        'expected_chi2_range': (400, 1500)
    },
    'grav_color': {
        'description': 'Gravitational Color Confinement',
        'allow_experimental': False,
        'expected_chi2_range': (350, 1200)
    },
    'mond': {
        'description': 'MOND-like Xi Modification',
        'allow_experimental': False,
        'expected_chi2_range': (400, 1500)
    },
    'tidal_band': {
        'description': 'Tidal Band with Density Screening',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_band2': {
        'description': 'Tidal Band v2 (Logistic lnT onset)',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_ratio': {
        'description': 'Tidal Ratio Trigger',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'tidal_noisyor': {
        'description': 'Tidal Noisy-OR Aggregator',
        'allow_experimental': True,
        'expected_chi2_range': (300, 1000)
    },
    'rar_gate': {
        'description': 'RAR-anchored Gating on g_bar',
        'allow_experimental': True,
        'expected_chi2_range': (250, 900)
    },
    'rar_blend': {
        'description': 'RAR Blend (amplitude × RAR excess)',
        'allow_experimental': True,
        'expected_chi2_range': (250, 900)
    }
}

def run_single_model(model_name, settings, output_dir='stellar_fit_results'):
    """
    Run stellar fitting for a single model using the fixed CuPy script.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'gr', 'nfw', 'tidal_band')
    settings : dict
        Run settings (production or test)
    output_dir : str
        Base output directory
    
    Returns:
    --------
    result : dict
        Results including chi2, RMSE, best parameters
    """
    model_config = MODELS[model_name]
    
    logger.info("="*80)
    logger.info(f"Running {model_name.upper()}: {model_config['description']}")
    logger.info("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "run_dynesty_stellar_fit_cupy.py",
        "--xi", model_name,
        "--sample_max", str(settings['sample_max']),
        "--maxcall", str(model_config.get('maxcall', settings['maxcall'])),
        "--nlive", str(settings['nlive']),
        "--output_dir", f"{output_dir}/{model_name}",
        "--plot"
    ]
    
    if model_config.get('allow_experimental', False):
        cmd.append("--allow_experimental")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the model
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        
        # Parse output for results
        output_lines = result.stdout.split('\n')
        chi2 = None
        rmse = None
        logl = None
        
        for line in output_lines:
            if 'Chi²:' in line or 'Chi2:' in line:
                try:
                    chi2 = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'RMSE:' in line:
                try:
                    rmse = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Log(L):' in line:
                try:
                    logl = float(line.split(':')[1].strip())
                except:
                    pass
        
        # Check if the run succeeded
        success = result.returncode == 0
        
        # If we got results, check if they're in expected range
        if chi2 and 'expected_chi2_range' in model_config:
            min_chi2, max_chi2 = model_config['expected_chi2_range']
            if not (min_chi2 <= chi2 <= max_chi2):
                logger.warning(f"⚠ Chi² = {chi2:.1f} is outside expected range [{min_chi2}, {max_chi2}]")
        
        if success and chi2 is not None:
            logger.info(f"✓ {model_name} completed in {elapsed:.1f}s")
            logger.info(f"  Chi²: {chi2:.1f}")
            if rmse:
                logger.info(f"  RMSE: {rmse:.1f} km/s")
            if logl:
                logger.info(f"  Log(L): {logl:.1f}")
        else:
            logger.warning(f"⚠ {model_name} completed with issues in {elapsed:.1f}s")
            if result.stderr:
                logger.debug(f"  Stderr: {result.stderr[:500]}")
        
        return {
            'model': model_name,
            'chi2': chi2,
            'rmse': rmse,
            'logl': logl,
            'runtime': elapsed,
            'success': success,
            'return_code': result.returncode
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {model_name} failed with exception after {elapsed:.1f}s")
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
    
    # Sort by chi2 (best first), putting None values last
    df['chi2_sort'] = df['chi2'].fillna(float('inf'))
    df = df.sort_values('chi2_sort').drop('chi2_sort', axis=1)
    
    # Add rank for successful models
    df['rank'] = None
    successful_mask = df['chi2'].notna()
    df.loc[successful_mask, 'rank'] = range(1, successful_mask.sum() + 1)
    
    # Reorder columns
    cols = ['rank', 'model', 'chi2', 'rmse', 'logl', 'runtime', 'success']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    return df

def create_comparison_plot(results, output_dir='stellar_fit_results'):
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
        
        bars1 = ax1.bar(range(len(models)), chi2_values, color='steelblue')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Chi²')
        ax1.set_title('Model Comparison - Chi²')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, chi2_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        # RMSE comparison
        rmse_values = [r.get('rmse', 0) for r in successful]
        
        bars2 = ax2.bar(range(len(models)), rmse_values, color='darkgreen')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('RMSE (km/s)')
        ax2.set_title('Model Comparison - RMSE')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars2, rmse_values):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_file = f'{output_dir}/model_comparison_{datetime.now():%Y%m%d_%H%M%S}.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        logger.info(f"Comparison plot saved to: {plot_file}")
        
    except Exception as e:
        logger.warning(f"Could not create comparison plot: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run stellar fits for all models')
    
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced settings')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to run (default: all)')
    parser.add_argument('--skip_experimental', action='store_true',
                       help='Skip experimental models (tidal and RAR)')
    parser.add_argument('--output_dir', default='stellar_fit_results',
                       help='Output directory for results')
    
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
        models_to_run = args.models
        # Validate model names
        invalid = [m for m in models_to_run if m not in MODELS]
        if invalid:
            logger.error(f"Invalid models: {invalid}")
            logger.info(f"Available models: {list(MODELS.keys())}")
            sys.exit(1)
    elif args.skip_experimental:
        models_to_run = [k for k in MODELS.keys() 
                        if not MODELS[k].get('allow_experimental', False)]
    else:
        models_to_run = list(MODELS.keys())
    
    logger.info(f"Will run {len(models_to_run)} models: {models_to_run}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all models
    results = []
    total_start = time.time()
    
    for i, model_name in enumerate(models_to_run, 1):
        logger.info(f"\n[{i}/{len(models_to_run)}] Processing {model_name}...")
        result = run_single_model(model_name, settings, str(output_dir))
        results.append(result)
        
        # Save intermediate results
        df = create_summary_table(results)
        df.to_csv(output_dir / f'results_intermediate_{datetime.now():%Y%m%d_%H%M%S}.csv', index=False)
    
    total_elapsed = time.time() - total_start
    
    # Create final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    df = create_summary_table(results)
    logger.info(f"\n{df.to_string()}")
    
    # Save final results
    output_file = output_dir / f'final_results_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Save JSON for programmatic access
    json_file = output_dir / f'final_results_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON results saved to: {json_file}")
    
    # Create comparison plot
    create_comparison_plot(results, str(output_dir))
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    logger.info(f"\nCompleted {len(successful)}/{len(results)} models successfully")
    logger.info(f"Total runtime: {total_elapsed/60:.1f} minutes")
    
    if successful:
        best_model = min(successful, key=lambda x: x['chi2'] if x['chi2'] else float('inf'))
        logger.info(f"\nBest model: {best_model['model']}")
        if best_model['chi2']:
            logger.info(f"  Chi²: {best_model['chi2']:.1f}")
        if best_model.get('rmse'):
            logger.info(f"  RMSE: {best_model['rmse']:.1f} km/s")
        if best_model.get('logl'):
            logger.info(f"  Log(L): {best_model['logl']:.1f}")

if __name__ == '__main__':
    main()
