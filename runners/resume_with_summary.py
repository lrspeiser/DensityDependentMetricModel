#!/usr/bin/env python3
"""
resume_with_summary.py - Resume a Dynesty run with enhanced summaries

This script properly handles resuming from NPZ checkpoints and provides
enhanced summary output during the run.
"""

import sys
import os
import numpy as np
import pickle
import json
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_summary import DynestyRunSummary

def analyze_checkpoint(checkpoint_file):
    """Analyze a checkpoint file and provide detailed summary."""
    
    print("\n" + "="*80)
    print("CHECKPOINT ANALYSIS")
    print("="*80)
    
    # Load checkpoint with allow_pickle for object arrays
    data = np.load(checkpoint_file, allow_pickle=True)
    
    print(f"\nCheckpoint file: {checkpoint_file}")
    print(f"File size: {os.path.getsize(checkpoint_file) / 1024 / 1024:.2f} MB")
    
    # Basic info
    print(f"\nCheckpoint contents:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
        else:
            print(f"  {key}: {data[key]}")
    
    # Extract key metrics
    samples = data['samples']
    logz = data['logz']
    logl = data['logl']
    param_names = data['param_names']
    xi_type = str(data['xi_type']) if 'xi_type' in data else 'unknown'
    
    n_samples = len(samples)
    n_params = samples.shape[1]
    
    print(f"\nSampling statistics:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Parameters: {n_params}")
    print(f"  Xi type: {xi_type}")
    
    # Evidence progression
    print(f"\nEvidence progression:")
    print(f"  Initial LogZ: {logz[0]:.2f}")
    print(f"  Current LogZ: {logz[-1]:.2f}")
    print(f"  Total improvement: {logz[-1] - logz[0]:.2f}")
    
    # Recent convergence
    if len(logz) > 100:
        recent_dlogz = np.diff(logz[-100:])
        print(f"\nRecent convergence (last 100 samples):")
        print(f"  Mean dLogZ: {np.mean(recent_dlogz):.4f}")
        print(f"  Std dLogZ: {np.std(recent_dlogz):.4f}")
        print(f"  Min dLogZ: {np.min(recent_dlogz):.4f}")
        print(f"  Max dLogZ: {np.max(recent_dlogz):.4f}")
        
        # Check if converging
        if np.mean(recent_dlogz) < 0.01:
            print(f"  Status: CONVERGED (dLogZ < 0.01)")
        elif np.mean(recent_dlogz) < 0.1:
            print(f"  Status: CONVERGING")
        else:
            print(f"  Status: EXPLORING")
    
    # Best fit parameters
    best_idx = np.argmax(logl)
    best_params = samples[best_idx]
    print(f"\nBest fit parameters (max likelihood):")
    print(f"  LogL: {logl[best_idx]:.2f}")
    for i, (name, val) in enumerate(zip(param_names, best_params)):
        name_str = name.decode() if isinstance(name, bytes) else str(name)
        if 'rho_c' in name_str:
            print(f"  {name_str}: {val:.2e} (log10: {np.log10(val):.2f})")
        else:
            print(f"  {name_str}: {val:.4f}")
    
    # Model comparison
    BASELINE_LOGZ_GR = -1490897.5250096943
    delta_logz = logz[-1] - BASELINE_LOGZ_GR
    print(f"\nModel comparison vs GR:")
    print(f"  GR baseline LogZ: {BASELINE_LOGZ_GR:.2f}")
    print(f"  Current LogZ: {logz[-1]:.2f}")
    print(f"  Delta LogZ: {delta_logz:+.2f}")
    
    if delta_logz < 0:
        print(f"  Interpretation: Negative evidence (GR preferred)")
    elif delta_logz < 1:
        print(f"  Interpretation: Barely worth mentioning")
    elif delta_logz < 2.5:
        print(f"  Interpretation: Substantial evidence for DDMM")
    elif delta_logz < 5:
        print(f"  Interpretation: Strong evidence for DDMM")
    elif delta_logz < 10:
        print(f"  Interpretation: Very strong evidence for DDMM")
    else:
        print(f"  Interpretation: Decisive evidence for DDMM")
    
    print("\n" + "="*80)
    
    return {
        'n_samples': n_samples,
        'current_logz': float(logz[-1]),
        'best_logl': float(logl[best_idx]),
        'converged': np.mean(np.diff(logz[-100:])) < 0.01 if len(logz) > 100 else False,
        'xi_type': xi_type
    }

def convert_npz_to_dynesty_results(npz_file):
    """Convert NPZ checkpoint to dynesty Results object format."""
    
    data = np.load(npz_file, allow_pickle=True)
    
    # Create a mock Results object that dynesty can use
    class Results:
        def __init__(self):
            self.samples = data['samples']
            self.logl = data['logl']
            self.logz = data['logz']
            self.logzerr = data['logzerr'] if 'logzerr' in data else np.zeros_like(data['logz'])
            self.logwt = data['logwt'] if 'logwt' in data else np.zeros(len(data['samples']))
            self.weights = data['weights'] if 'weights' in data else np.ones(len(data['samples'])) / len(data['samples'])
            self.ncall = data['n_calls'] if 'n_calls' in data else np.ones(len(data['samples'])) * 100
            self.blob = data['blob'] if 'blob' in data else None
    
    return Results()

def main():
    parser = argparse.ArgumentParser(description="Resume Dynesty run with enhanced summaries")
    parser.add_argument('run_dir', help='Directory containing the run to resume')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze checkpoint, don\'t resume')
    parser.add_argument('--summary_interval', type=int, default=30, help='Seconds between summary updates')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        sys.exit(1)
    
    # Find checkpoint file
    checkpoint_files = list(run_dir.glob("*checkpoint*latest*.npz"))
    if not checkpoint_files:
        checkpoint_files = list(run_dir.glob("*checkpoint*.npz"))
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found in {run_dir}")
        sys.exit(1)
    
    checkpoint_file = checkpoint_files[0]
    print(f"Using checkpoint: {checkpoint_file}")
    
    # Analyze checkpoint
    stats = analyze_checkpoint(checkpoint_file)
    
    if args.analyze_only:
        print("\nAnalysis complete (--analyze_only flag set)")
        return
    
    # Check if converged
    if stats['converged']:
        print("\n*** WARNING: Run appears to be already converged ***")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    print(f"\nPreparing to resume {stats['xi_type']} run with {stats['n_samples']} samples...")
    print(f"Current LogZ: {stats['current_logz']:.2f}")
    
    # Convert checkpoint for resume
    print("\nConverting checkpoint for dynesty resume...")
    results = convert_npz_to_dynesty_results(checkpoint_file)
    
    # Save as pickle for dynesty
    pickle_checkpoint = run_dir / "dynesty_checkpoint.pkl"
    with open(pickle_checkpoint, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle checkpoint: {pickle_checkpoint}")
    
    # Build resume command
    cli_file = run_dir / "cli_command.txt"
    if cli_file.exists():
        with open(cli_file, 'r') as f:
            lines = f.readlines()
            original_cmd = lines[-1].strip() if lines else ""
        print(f"\nOriginal command: {original_cmd}")
    else:
        original_cmd = ""
    
    # Extract xi_type from run directory name or checkpoint
    xi_type = stats['xi_type']
    if xi_type == 'unknown':
        # Try to extract from directory name
        dir_name = run_dir.name
        if 'tension_field' in dir_name:
            xi_type = 'tension_field'
        elif 'grav_color' in dir_name:
            xi_type = 'grav_color'
        elif 'power' in dir_name:
            xi_type = 'power'
        else:
            xi_type = input("Enter xi_type (e.g., tension_field, power, grav_color): ")
    
    # Build resume command
    resume_cmd = f"""python runners/run_dynesty_cupy.py \\
    --xi {xi_type} \\
    --nlive 400 \\
    --dlogz_target 0.001 \\
    --output_dir {run_dir} \\
    --maxcall 10000000 \\
    --max_sample_gaia 10000 \\
    --resume \\
    --summary_interval {args.summary_interval} \\
    --periodic_analysis \\
    --analysis_interval_min 5"""
    
    print(f"\nResume command:")
    print(resume_cmd)
    
    response = input("\nExecute resume command? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting resume...")
        os.system(resume_cmd.replace('\\', '').replace('\n', ' '))
    else:
        print("Resume command not executed. You can run it manually.")

if __name__ == "__main__":
    main()