#!/usr/bin/env python3
"""
run_production_fits.py - Production-grade stellar fitting with full dataset.

This script is optimized for running high-quality fits using all available
stellar data with extensive sampling.
"""

import logging
import sys
import os
import time
import numpy as np
import cupy as cp
from datetime import datetime
from pathlib import Path
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_fits_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Production configurations for different model types
CONFIGS = {
    'high_precision': {
        'description': 'Highest precision for publication-quality results',
        'sample_max': 200000,      # All available stars
        'maxcall': 40000000,       # 40M calls
        'nlive': 3000,             # 3000 live points
        'dlogz': 0.01,             # Very tight convergence
    },
    'standard': {
        'description': 'Standard production run with good accuracy',
        'sample_max': 144000,      # Full Gaia dataset
        'maxcall': 20000000,       # 20M calls
        'nlive': 2000,             # 2000 live points
        'dlogz': 0.05,             # Standard convergence
    },
    'fast': {
        'description': 'Fast production run for preliminary results',
        'sample_max': 100000,      # 100k stars
        'maxcall': 10000000,       # 10M calls
        'nlive': 1500,             # 1500 live points
        'dlogz': 0.1,              # Looser convergence
    },
    'benchmark': {
        'description': 'Benchmark configuration for performance testing',
        'sample_max': 50000,       # 50k stars
        'maxcall': 5000000,        # 5M calls
        'nlive': 1000,             # 1000 live points
        'dlogz': 0.2,              # Quick convergence
    }
}

# Priority models for production runs
PRIORITY_MODELS = [
    'power',       # Most promising based on tests
    'grav_color',  # Gravitational color confinement
    'tidal_band',  # Tidal screening
    'rar_gate',    # RAR-based modification
    'nfw',         # Standard ΛCDM comparison
]

def check_gpu_memory():
    """Check available GPU memory and recommend configuration."""
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Get GPU properties
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        total_memory = gpu_props['totalGlobalMem'] / 1e9
        
        # Get current usage
        used_memory = mempool.used_bytes() / 1e9
        available_memory = total_memory - used_memory
        
        logger.info(f"GPU Memory Status:")
        logger.info(f"  Total: {total_memory:.1f} GB")
        logger.info(f"  Used: {used_memory:.1f} GB")
        logger.info(f"  Available: {available_memory:.1f} GB")
        
        # Recommend configuration based on available memory
        if available_memory > 30:
            recommended = 'high_precision'
            logger.info(f"  ✓ Sufficient memory for high_precision configuration")
        elif available_memory > 20:
            recommended = 'standard'
            logger.info(f"  ✓ Sufficient memory for standard configuration")
        elif available_memory > 10:
            recommended = 'fast'
            logger.info(f"  ⚠ Limited memory - recommend fast configuration")
        else:
            recommended = 'benchmark'
            logger.warning(f"  ⚠ Low memory - using benchmark configuration")
        
        return recommended
        
    except Exception as e:
        logger.warning(f"Could not check GPU memory: {e}")
        return 'standard'

def optimize_gpu_settings():
    """Optimize CuPy/CUDA settings for maximum performance."""
    try:
        # Set memory pool growth
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Pre-allocate memory to avoid fragmentation
        mempool.set_limit(size=30 * 1024**3)  # 30 GB limit
        
        # Enable TF32 for Ampere GPUs (RTX 30xx, 40xx, 50xx)
        cp.cuda.set_cublas_tf32(True)
        cp.cuda.set_cudnn_tf32(True)
        
        # Set CUDA device flags for better performance
        cp.cuda.runtime.setDeviceFlags(cp.cuda.runtime.deviceScheduleYield)
        
        logger.info("✓ GPU optimizations applied")
        
    except Exception as e:
        logger.warning(f"Some GPU optimizations failed: {e}")

def estimate_runtime(config, n_models):
    """Estimate total runtime for the production run."""
    # Rough estimates based on benchmarks (seconds per million calls)
    time_per_mcall = {
        'high_precision': 15,  # ~15 seconds per million calls with 3000 live points
        'standard': 12,         # ~12 seconds per million calls with 2000 live points
        'fast': 10,            # ~10 seconds per million calls with 1500 live points
        'benchmark': 8         # ~8 seconds per million calls with 1000 live points
    }
    
    config_name = config
    if isinstance(config, dict):
        # Find matching config by parameters
        for name, cfg in CONFIGS.items():
            if cfg['maxcall'] == config.get('maxcall'):
                config_name = name
                break
        else:
            config_name = 'standard'
    
    mcalls = CONFIGS[config_name]['maxcall'] / 1e6
    time_per_model = mcalls * time_per_mcall.get(config_name, 12)
    total_time = time_per_model * n_models
    
    logger.info(f"\nRuntime Estimate:")
    logger.info(f"  Per model: {time_per_model/60:.1f} minutes")
    logger.info(f"  Total ({n_models} models): {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    return total_time

def run_production_fit(model_name, config, output_dir='production_results'):
    """
    Run a single model with production settings.
    """
    import subprocess
    
    logger.info("="*80)
    logger.info(f"Starting production fit for {model_name.upper()}")
    logger.info(f"Configuration: {config.get('description', 'custom')}")
    logger.info("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "run_dynesty_stellar_fit_cupy.py",
        "--xi", model_name,
        "--sample_max", str(config['sample_max']),
        "--maxcall", str(config['maxcall']),
        "--nlive", str(config['nlive']),
        "--output_dir", f"{output_dir}/{model_name}",
        "--plot",
        "--verbose"
    ]
    
    # Add experimental flag if needed
    if model_name in ['tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor', 'rar_gate', 'rar_blend']:
        cmd.append("--allow_experimental")
    
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Starting at {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Run with real-time output
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )
        
        # Stream output in real-time
        for line in process.stdout:
            if line.strip():
                print(line.rstrip())
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            logger.info(f"✓ {model_name} completed successfully in {elapsed/60:.1f} minutes")
            return {'model': model_name, 'success': True, 'runtime': elapsed}
        else:
            logger.error(f"✗ {model_name} failed with return code {process.returncode}")
            return {'model': model_name, 'success': False, 'runtime': elapsed}
            
    except KeyboardInterrupt:
        logger.warning(f"Interrupted {model_name} after {(time.time()-start_time)/60:.1f} minutes")
        process.terminate()
        raise
    except Exception as e:
        logger.error(f"Error running {model_name}: {e}")
        return {'model': model_name, 'success': False, 'error': str(e)}

def main():
    """Main entry point for production runs."""
    parser = argparse.ArgumentParser(
        description='Production-grade stellar fitting with full dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with automatic configuration selection
  python run_production_fits.py --auto
  
  # Run specific models with high precision
  python run_production_fits.py --models power grav_color --config high_precision
  
  # Run priority models with standard settings
  python run_production_fits.py --priority --config standard
  
  # Custom configuration
  python run_production_fits.py --models power --sample_max 144000 --maxcall 30000000 --nlive 2500
        """
    )
    
    parser.add_argument('--models', nargs='+', 
                       help='Specific models to run')
    parser.add_argument('--priority', action='store_true',
                       help='Run priority models only')
    parser.add_argument('--config', choices=list(CONFIGS.keys()),
                       help='Use predefined configuration')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically select best configuration based on GPU')
    
    # Custom configuration options
    parser.add_argument('--sample_max', type=int,
                       help='Maximum number of stars to use')
    parser.add_argument('--maxcall', type=int,
                       help='Maximum likelihood evaluations')
    parser.add_argument('--nlive', type=int,
                       help='Number of live points')
    
    parser.add_argument('--output_dir', default='production_results',
                       help='Output directory for results')
    parser.add_argument('--continue_from', 
                       help='Continue from a specific model (skip previous ones)')
    
    args = parser.parse_args()
    
    # Check GPU and optimize settings
    logger.info("Checking GPU status...")
    optimize_gpu_settings()
    
    # Determine configuration
    if args.auto:
        config_name = check_gpu_memory()
        config = CONFIGS[config_name]
        logger.info(f"Auto-selected configuration: {config_name}")
    elif args.config:
        config = CONFIGS[args.config]
        logger.info(f"Using configuration: {args.config}")
    elif args.sample_max or args.maxcall or args.nlive:
        # Custom configuration
        config = {
            'description': 'custom',
            'sample_max': args.sample_max or 144000,
            'maxcall': args.maxcall or 20000000,
            'nlive': args.nlive or 2000,
            'dlogz': 0.05
        }
        logger.info("Using custom configuration")
    else:
        # Default to standard
        config = CONFIGS['standard']
        logger.info("Using standard configuration")
    
    # Log configuration details
    logger.info(f"\nConfiguration Details:")
    logger.info(f"  Stars: {config['sample_max']:,}")
    logger.info(f"  Max calls: {config['maxcall']:,}")
    logger.info(f"  Live points: {config['nlive']:,}")
    
    # Determine models to run
    if args.models:
        models_to_run = args.models
    elif args.priority:
        models_to_run = PRIORITY_MODELS
    else:
        # Default to priority models
        models_to_run = PRIORITY_MODELS
        logger.info("No models specified, running priority models")
    
    # Handle continuation
    if args.continue_from and args.continue_from in models_to_run:
        skip_idx = models_to_run.index(args.continue_from)
        models_to_run = models_to_run[skip_idx:]
        logger.info(f"Continuing from {args.continue_from}")
    
    logger.info(f"\nModels to run: {models_to_run}")
    
    # Estimate runtime
    estimate_runtime(config, len(models_to_run))
    
    # Confirm before starting
    response = input("\nProceed with production run? (y/n): ")
    if response.lower() != 'y':
        logger.info("Production run cancelled")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / f"config_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'config': config,
            'models': models_to_run,
            'start_time': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Configuration saved to {config_file}")
    
    # Run models
    results = []
    total_start = time.time()
    
    for i, model_name in enumerate(models_to_run, 1):
        logger.info(f"\n[{i}/{len(models_to_run)}] Processing {model_name}")
        
        try:
            result = run_production_fit(model_name, config, str(output_dir))
            results.append(result)
            
            # Save intermediate results
            with open(output_dir / f"results_intermediate_{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
                json.dump(results, f, indent=2)
                
        except KeyboardInterrupt:
            logger.warning("\nProduction run interrupted by user")
            break
        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            results.append({'model': model_name, 'success': False, 'error': str(e)})
    
    # Final summary
    total_elapsed = time.time() - total_start
    successful = [r for r in results if r.get('success', False)]
    
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION RUN COMPLETE")
    logger.info("="*80)
    logger.info(f"Total runtime: {total_elapsed/3600:.2f} hours")
    logger.info(f"Successful models: {len(successful)}/{len(results)}")
    
    if successful:
        logger.info("\nSuccessful models:")
        for r in successful:
            logger.info(f"  ✓ {r['model']} ({r['runtime']/60:.1f} min)")
    
    # Save final results
    final_results = {
        'config': config,
        'models': models_to_run,
        'results': results,
        'total_runtime': total_elapsed,
        'start_time': datetime.now().isoformat(),
        'successful_count': len(successful)
    }
    
    results_file = output_dir / f"final_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\nFinal results saved to {results_file}")

if __name__ == '__main__':
    main()
