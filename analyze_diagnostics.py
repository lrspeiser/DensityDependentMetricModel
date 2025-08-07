#!/usr/bin/env python3
"""
Analyze diagnostic data to understand parameter space and failure modes.
"""

import numpy as np
import h5py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def analyze_diagnostic_run(diagnostic_file):
    """Analyze a single diagnostic HDF5 file."""
    
    print(f"\nAnalyzing: {diagnostic_file}")
    print("=" * 70)
    
    with h5py.File(diagnostic_file, 'r') as f:
        # Load data
        parameters = f['parameters'][:]
        logl = f['logl'][:]
        chi2 = f['chi2'][:] if 'chi2' in f else None
        failure_reasons = f['failure_reason'][:] if 'failure_reason' in f else None
        
        # Metadata
        xi_type = f.attrs.get('xi_type', 'unknown')
        n_calls = f.attrs.get('n_calls', len(logl))
        best_logl = f.attrs.get('best_logl', np.max(logl[np.isfinite(logl)]))
        n_data = f.attrs.get('n_data_points', 0)
    
    # Basic statistics
    valid_mask = np.isfinite(logl)
    n_valid = np.sum(valid_mask)
    n_failed = n_calls - n_valid
    
    print(f"Model: {xi_type}")
    print(f"Total calls: {n_calls}")
    print(f"Valid results: {n_valid} ({100*n_valid/n_calls:.1f}%)")
    print(f"Failed: {n_failed} ({100*n_failed/n_calls:.1f}%)")
    
    if n_valid > 0:
        valid_logl = logl[valid_mask]
        print(f"\nLog-Likelihood Statistics:")
        print(f"  Best: {best_logl:.2f}")
        print(f"  Median: {np.median(valid_logl):.2f}")
        print(f"  Worst finite: {np.min(valid_logl):.2f}")
        
        if chi2 is not None:
            valid_chi2 = chi2[valid_mask]
            chi2_per_dof = valid_chi2 / n_data if n_data > 0 else valid_chi2
            print(f"\nChi2/DOF Statistics:")
            print(f"  Best: {np.min(chi2_per_dof):.2f}")
            print(f"  Median: {np.median(chi2_per_dof):.2f}")
            print(f"  Worst: {np.max(chi2_per_dof):.2f}")
    
    # Analyze failure modes
    if failure_reasons is not None:
        failure_counts = {}
        for reason in failure_reasons:
            if reason and len(reason) > 0:
                reason_str = reason.decode() if isinstance(reason, bytes) else str(reason)
                if 'Cassini' in reason_str:
                    key = 'Cassini violation'
                elif 'NaN' in reason_str:
                    key = 'NaN in velocities'
                elif 'Unrealistic' in reason_str:
                    key = 'Unrealistic velocities'
                elif 'Non-positive' in reason_str:
                    key = 'Invalid parameter value'
                else:
                    key = 'Other'
                failure_counts[key] = failure_counts.get(key, 0) + 1
        
        if failure_counts:
            print("\nFailure Modes:")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count} ({100*count/n_calls:.1f}%)")
    
    # Analyze parameter distributions for successful runs
    if n_valid > 10:
        valid_params = parameters[valid_mask]
        
        print("\nParameter Ranges (successful evaluations):")
        n_params = valid_params.shape[1]
        
        # We don't have param names in HDF5, so use indices
        for i in range(min(n_params, 5)):  # Show first 5 parameters
            param_vals = valid_params[:, i]
            print(f"  Param {i}: [{np.min(param_vals):.3e}, {np.max(param_vals):.3e}]")
    
    # Find correlations between parameters and success
    if n_valid > 100:
        print("\nParameter Impact on LogL:")
        valid_params = parameters[valid_mask]
        valid_logl = logl[valid_mask]
        
        # Find parameters most correlated with good likelihood
        correlations = []
        for i in range(valid_params.shape[1]):
            corr, _ = stats.spearmanr(valid_params[:, i], valid_logl)
            if np.isfinite(corr):
                correlations.append((i, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("  Most impactful parameters (|correlation| with logL):")
        for i, (param_idx, corr) in enumerate(correlations[:5]):
            print(f"    Param {param_idx}: {corr:+.3f}")
    
    return {
        'xi_type': xi_type,
        'n_calls': n_calls,
        'n_valid': n_valid,
        'n_failed': n_failed,
        'best_logl': best_logl,
        'success_rate': n_valid / n_calls if n_calls > 0 else 0
    }

def compare_models(run_dir):
    """Compare diagnostic results across different models."""
    
    diagnostic_files = list(Path(run_dir).glob("*/diagnostics_*.h5"))
    
    if not diagnostic_files:
        print("No diagnostic files found!")
        return
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    results = []
    for diag_file in diagnostic_files:
        try:
            result = analyze_diagnostic_run(diag_file)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {diag_file}: {e}")
    
    if results:
        # Sort by best logL
        results.sort(key=lambda x: x['best_logl'], reverse=True)
        
        print("\nRanking by Best LogL:")
        print("-" * 50)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['xi_type']:20s}: LogL={r['best_logl']:12.2f}, Success={r['success_rate']:.1%}")
        
        # Find most reliable model
        results.sort(key=lambda x: x['success_rate'], reverse=True)
        print("\nRanking by Success Rate:")
        print("-" * 50)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['xi_type']:20s}: Success={r['success_rate']:.1%}, LogL={r['best_logl']:12.2f}")

def find_viable_parameter_regions(diagnostic_file, output_file=None):
    """Find regions of parameter space that produce viable results."""
    
    with h5py.File(diagnostic_file, 'r') as f:
        parameters = f['parameters'][:]
        logl = f['logl'][:]
        n_data = f.attrs.get('n_data_points', 144000)
    
    # Define "good" as within factor of 10 of expected good likelihood
    expected_good_logl = -n_data / 2  # Chi2/dof ~ 1
    threshold_logl = expected_good_logl - np.log(10) * n_data / 2  # Factor of 10 worse
    
    good_mask = (logl > threshold_logl) & np.isfinite(logl)
    
    if np.sum(good_mask) == 0:
        print("No viable parameter regions found!")
        return None
    
    good_params = parameters[good_mask]
    good_logl = logl[good_mask]
    
    # Find best region
    best_idx = np.argmax(good_logl)
    best_params = good_params[best_idx]
    
    # Compute statistics for viable region
    viable_region = {
        'n_good': np.sum(good_mask),
        'best_logl': good_logl[best_idx],
        'best_params': best_params.tolist(),
        'param_ranges': []
    }
    
    print(f"\nViable Parameter Regions ({np.sum(good_mask)} points):")
    print("-" * 50)
    
    for i in range(good_params.shape[1]):
        param_vals = good_params[:, i]
        p_min, p_25, p_50, p_75, p_max = np.percentile(param_vals, [0, 25, 50, 75, 100])
        
        viable_region['param_ranges'].append({
            'index': i,
            'min': float(p_min),
            'q25': float(p_25),
            'median': float(p_50),
            'q75': float(p_75),
            'max': float(p_max),
            'best': float(best_params[i])
        })
        
        print(f"Param {i:2d}: [{p_min:.3e}, {p_max:.3e}], best={best_params[i]:.3e}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(viable_region, f, indent=2)
        print(f"\nViable regions saved to: {output_file}")
    
    return viable_region

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze diagnostic runs")
    parser.add_argument('--run_dir', type=str, default='runs',
                       help='Directory containing diagnostic runs')
    parser.add_argument('--specific_file', type=str,
                       help='Analyze specific diagnostic file')
    parser.add_argument('--find_viable', action='store_true',
                       help='Find viable parameter regions')
    
    args = parser.parse_args()
    
    if args.specific_file:
        result = analyze_diagnostic_run(args.specific_file)
        
        if args.find_viable:
            output_file = Path(args.specific_file).parent / "viable_regions.json"
            find_viable_parameter_regions(args.specific_file, output_file)
    else:
        compare_models(args.run_dir)

if __name__ == "__main__":
    main()