#!/usr/bin/env python3
"""
Analyze GR baseline results and format for academic paper
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def analyze_gr_results():
    """Analyze the GR baseline results and format for academic paper"""
    
    # Load the posterior samples
    npz_file = Path("runs/gr_20250804_153029/posterior_samples.npz")
    data = np.load(npz_file, allow_pickle=True)
    
    print("Available keys in NPZ file:", list(data.keys()))
    
    samples = data['samples']
    logz = data['logz']
    logl = data['logl']
    weights = data['weights']
    
    print(f"Samples shape: {samples.shape}")
    print(f"LogZ shape: {logz.shape}, type: {type(logz)}")
    print(f"LogL shape: {logl.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Determine parameter names based on sample shape
    if samples.shape[1] == 2:
        # Legacy 2-parameter model
        param_names = ['M_disk_solar', 'R_d_kpc']
        model_type = "Legacy (2-parameter)"
    elif samples.shape[1] == 11:
        # New comprehensive 11-parameter model
        param_names = [
            'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
            'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
            'M_bulge_solar', 'R_bulge_kpc',
            'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc'
        ]
        model_type = "Comprehensive (11-parameter)"
    else:
        # Try to get from NPZ file
        param_names = data.get('param_names', [f'param_{i}' for i in range(samples.shape[1])])
        model_type = f"Unknown ({samples.shape[1]}-parameter)"
    
    print("="*80)
    print("GR BASELINE RESULTS - ACADEMIC PAPER FORMAT")
    print("="*80)
    
    # Basic run information
    print(f"\n1. SAMPLING SUMMARY")
    
    # Handle logz properly (could be scalar or array)
    if logz.ndim == 0:
        final_logz = float(logz)
    else:
        final_logz = float(logz[-1])
    
    print(f"   Final Log-Evidence (LogZ): {final_logz:.2f}")
    print(f"   Number of Posterior Samples: {len(samples):,}")
    print(f"   Maximum Log-Likelihood: {float(np.max(logl)):.2f}")
    print(f"   Mean Log-Likelihood: {float(np.mean(logl)):.2f}")
    print(f"   Model Type: {model_type}")
    
    # Parameter analysis
    print(f"\n2. PARAMETER ESTIMATES")
    print(f"   {'Parameter':<25} {'Median':<15} {'Std Dev':<15} {'16th %':<15} {'84th %':<15}")
    print(f"   {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    param_results = {}
    total_baryonic_mass = 0.0
    
    for i, name in enumerate(param_names):
        values = samples[:, i]
        
        # Calculate statistics
        median = np.median(values)
        mean = np.mean(values)
        std = np.std(values)
        p16 = np.percentile(values, 16)
        p84 = np.percentile(values, 84)
        p2_5 = np.percentile(values, 2.5)
        p97_5 = np.percentile(values, 97.5)
        
        # Store for later use
        param_results[name] = {
            'median': float(median),
            'mean': float(mean),
            'std': float(std),
            'p16': float(p16),
            'p84': float(p84),
            'p2_5': float(p2_5),
            'p97_5': float(p97_5),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        
        # Add to total baryonic mass if it's a mass parameter
        if 'M_' in name and 'solar' in name:
            total_baryonic_mass += median
        
        # Format output
        if median > 1e6:
            print(f"   {name:<25} {median:.2e}      {std:.2e}      {p16:.2e}      {p84:.2e}")
        else:
            print(f"   {name:<25} {median:.3f}      {std:.3f}      {p16:.3f}      {p84:.3f}")
    
    print(f"\n   Total Baryonic Mass: {total_baryonic_mass:.2e} M_☉")
    
    # Check for parameter bounds
    print(f"\n3. PARAMETER BOUND ANALYSIS")
    
    if samples.shape[1] == 11:
        # New comprehensive model bounds
        bounds_info = {
            'M_thin_disk_solar': {'low': 1e10, 'high': 1e11},
            'R_thin_disk_kpc': {'low': 2.0, 'high': 4.0},
            'hz_thin_disk_kpc': {'low': 0.2, 'high': 0.4},
            'M_thick_disk_solar': {'low': 1e9, 'high': 1e10},
            'R_thick_disk_kpc': {'low': 3.0, 'high': 5.0},
            'hz_thick_disk_kpc': {'low': 0.6, 'high': 1.0},
            'M_bulge_solar': {'low': 1e9, 'high': 1e10},
            'R_bulge_kpc': {'low': 0.5, 'high': 2.0},
            'M_gas_solar': {'low': 1e9, 'high': 1e10},
            'R_gas_kpc': {'low': 5.0, 'high': 10.0},
            'hz_gas_kpc': {'low': 0.1, 'high': 0.3}
        }
    else:
        # Legacy model bounds
        bounds_info = {
            'M_disk_solar': {'low': 1e9, 'high': 1e12},
            'R_d_kpc': {'low': 0.5, 'high': 15.0}
        }
    
    for name, bounds in bounds_info.items():
        if name in param_names:
            values = samples[:, param_names.index(name)]
            min_val = np.min(values)
            max_val = np.max(values)
            
            low_hit = min_val < bounds['low'] * 1.1
            high_hit = max_val > bounds['high'] * 0.9
            
            if low_hit or high_hit:
                print(f"   ⚠️  {name}: Hitting prior bounds")
                if low_hit:
                    print(f"      Lower bound: {bounds['low']:.2e}, Min value: {min_val:.2e}")
                if high_hit:
                    print(f"      Upper bound: {bounds['high']:.2e}, Max value: {max_val:.2e}")
            else:
                print(f"   ✓ {name}: Well within prior bounds")
    
    # Model comparison
    print(f"\n4. MODEL COMPARISON")
    gr_baseline_logz = -1490897.53  # From your logs
    current_logz = final_logz
    delta_logz = current_logz - gr_baseline_logz
    
    print(f"   GR Baseline LogZ: {gr_baseline_logz:.2f}")
    print(f"   This Run LogZ:    {current_logz:.2f}")
    print(f"   Δ LogZ:           {delta_logz:+.2f}")
    
    if delta_logz > 5:
        interpretation = "Decisive evidence"
    elif delta_logz > 3:
        interpretation = "Strong evidence"
    elif delta_logz > 1:
        interpretation = "Moderate evidence"
    else:
        interpretation = "Weak evidence"
    
    print(f"   Interpretation:   {interpretation}")
    
    # Data information
    print(f"\n5. DATA SUMMARY")
    print(f"   Model Type: GR (General Relativity, baryonic matter only)")
    print(f"   Parameters: {len(param_names)} ({', '.join(param_names)})")
    print(f"   Total Baryonic Mass: {total_baryonic_mass:.2e} M_☉")
    print(f"   No dark matter components included")
    
    # Hardware information
    print(f"\n6. COMPUTATIONAL DETAILS")
    hw_file = Path("runs/gr_20250804_153029/hardware_info.json")
    if hw_file.exists():
        with open(hw_file, 'r') as f:
            hw = json.load(f)
        
        print(f"   CPU: {hw['cpu']['cores']} cores")
        print(f"   GPU: {hw['gpu']['gpu_0']['name']}")
        print(f"   Memory: {hw['memory']['total_gb']:.1f} GB")
    
    # Save formatted results
    results_summary = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'GR_baseline',
            'parameters': param_names,
            'final_logz': final_logz,
            'max_logl': float(np.max(logl)),
            'n_samples': len(samples),
            'total_baryonic_mass': total_baryonic_mass
        },
        'parameter_results': param_results,
        'model_comparison': {
            'gr_baseline_logz': gr_baseline_logz,
            'current_logz': current_logz,
            'delta_logz': delta_logz,
            'interpretation': interpretation
        },
        'bounds_analysis': bounds_info
    }
    
    # Save to file
    output_file = Path("gr_baseline_paper_results.json")
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n7. OUTPUT FILES")
    print(f"   ✓ Detailed results saved to: {output_file}")
    print(f"   ✓ Raw samples in: {npz_file}")
    print(f"   ✓ Progress tracking in: runs/gr_20250804_153029/dynesty_progress.json")
    
    # LaTeX table format
    print(f"\n8. LATEX TABLE FORMAT")
    print(f"\\begin{{table}}[h]")
    print(f"\\centering")
    print(f"\\caption{{GR Baseline Parameter Estimates}}")
    print(f"\\begin{{tabular}}{{lccccc}}")
    print(f"\\hline")
    print(f"Parameter & Median & Std Dev & 16th \\% & 84th \\% & 95\\% CI \\\\")
    print(f"\\hline")
    
    for name in param_names:
        stats = param_results[name]
        if stats['median'] > 1e6:
            print(f"{name} & {stats['median']:.2e} & {stats['std']:.2e} & {stats['p16']:.2e} & {stats['p84']:.2e} & [{stats['p2_5']:.2e}, {stats['p97_5']:.2e}] \\\\")
        else:
            print(f"{name} & {stats['median']:.3f} & {stats['std']:.3f} & {stats['p16']:.3f} & {stats['p84']:.3f} & [{stats['p2_5']:.3f}, {stats['p97_5']:.3f}] \\\\")
    
    print(f"\\hline")
    print(f"\\end{{tabular}}")
    print(f"\\label{{tab:gr_baseline_params}}")
    print(f"\\end{{table}}")
    
    print(f"\n" + "="*80)
    print(f"ANALYSIS COMPLETE - Ready for academic paper")
    print(f"="*80)

if __name__ == "__main__":
    analyze_gr_results() 