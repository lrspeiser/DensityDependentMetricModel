#!/usr/bin/env python3
"""Analyze the grav_color_void_safe results."""

import numpy as np
import pickle
from pathlib import Path
import json

# Load the results
run_dir = Path("runs/grav_color_void_safe_20250806_125037")

# Load NPZ results
npz_file = run_dir / "posterior_samples.npz"
data = np.load(npz_file, allow_pickle=True)

print("=" * 60)
print("GRAV_COLOR_VOID_SAFE MODEL RESULTS")
print("=" * 60)

# Get basic info
samples = data['samples']
logl = data['logl']
logz = data['logz']
param_names = data['param_names']

print(f"\nFinal LogZ: {logz:.2f}")

# Compare to GR baseline
BASELINE_LOGZ_GR = -1490897.53
delta_logz = logz - BASELINE_LOGZ_GR
print(f"Delta LogZ vs GR: {delta_logz:+.2f}")

if delta_logz > 0:
    print(">>> DDMM model PREFERRED over GR! <<<")
else:
    print(">>> GR model preferred over DDMM <<<")

# Get best-fit parameters
best_idx = np.argmax(logl)
best_params = samples[best_idx]
best_logl = logl[best_idx]

print(f"\nBest LogL: {best_logl:.2f}")
print("\nBest-fit parameters:")
print("-" * 40)
for name, value in zip(param_names, best_params):
    if 'M_' in name or 'rho_c' in name:
        print(f"{name:25s}: {value:.3e}")
    else:
        print(f"{name:25s}: {value:.6f}")

# Load run stats for more info
stats_file = run_dir / "run_stats.json"
if stats_file.exists():
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    print("\nRun Statistics:")
    print("-" * 40)
    print(f"Total calls: {stats.get('ncall', 'N/A')}")
    eff = stats.get('efficiency', 'N/A')
    if eff != 'N/A':
        print(f"Efficiency: {eff:.2%}")
    else:
        print(f"Efficiency: {eff}")
    runtime = stats.get('runtime_minutes', 'N/A')
    if runtime != 'N/A':
        print(f"Runtime: {runtime:.1f} minutes")
    else:
        print(f"Runtime: {runtime}")

# Check summary file for convergence
summary_file = run_dir / "run_summary.json"
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    if 'convergence' in summary:
        print("\nConvergence Info:")
        print("-" * 40)
        conv = summary['convergence']
        print(f"dlogz: {conv.get('dlogz', 'N/A')}")
        print(f"dlogz_target: {conv.get('dlogz_target', 'N/A')}")
        print(f"Converged: {conv.get('converged', 'N/A')}")

# Parameter statistics
print("\nParameter Statistics (mean ± std):")
print("-" * 40)
for i, name in enumerate(param_names):
    param_samples = samples[:, i]
    mean = np.mean(param_samples)
    std = np.std(param_samples)
    if 'M_' in name or 'rho_c' in name:
        print(f"{name:25s}: {mean:.3e} ± {std:.3e}")
    else:
        print(f"{name:25s}: {mean:.4f} ± {std:.4f}")

# Special focus on grav_color_void_safe specific parameters
if 'gamma_exp' in param_names:
    gamma_idx = list(param_names).index('gamma_exp')
    gamma_best = best_params[gamma_idx]
    print(f"\nGamma exponent (best): {gamma_best:.4f}")
    
if 'lambda_g' in param_names:
    lambda_idx = list(param_names).index('lambda_g')
    lambda_best = best_params[lambda_idx]
    print(f"Lambda_g (best): {lambda_best:.4f}")