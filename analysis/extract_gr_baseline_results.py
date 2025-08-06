#!/usr/bin/env python3
"""
extract_gr_baseline_results.py - Extract key results from GR baseline run for README
"""

import numpy as np
import json
import pickle
from pathlib import Path
import argparse

def extract_results_from_npz(npz_file):
    """Extract results from .npz file"""
    print(f"Loading results from: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract key data
    samples = data['samples']
    logl = data['logl']
    weights = data['weights']
    logz = data['logz']
    dlogz = data.get('dlogz', np.nan)
    param_names = data.get('param_names', [])
    
    # Convert param_names if it's a numpy array
    if isinstance(param_names, np.ndarray):
        param_names = param_names.tolist()
    
    # If no parameter names, create them based on samples shape
    if not param_names and len(samples.shape) > 1:
        n_params = samples.shape[1]
        # Use known parameter names for GR baseline
        if n_params == 2:
            param_names = ['M_disk_solar', 'R_d_kpc']
            print(f"✓ Using GR baseline parameter names: {param_names}")
        else:
            param_names = [f"param_{i}" for i in range(n_params)]
            print(f"✓ Created default parameter names: {param_names}")
    
    print(f"✓ Loaded {len(samples)} samples with {len(param_names)} parameters")
    print(f"✓ Final LogZ: {logz:.2f}")
    print(f"✓ Parameter names: {param_names}")
    
    # Calculate parameter statistics
    param_stats = {}
    for i, name in enumerate(param_names):
        param_values = samples[:, i]
        param_stats[name] = {
            'median': float(np.median(param_values)),
            'mean': float(np.mean(param_values)),
            'std': float(np.std(param_values)),
            'min': float(np.min(param_values)),
            'max': float(np.max(param_values)),
            '16th_percentile': float(np.percentile(param_values, 16)),
            '84th_percentile': float(np.percentile(param_values, 84))
        }
    
    return {
        'n_samples': len(samples),
        'n_parameters': len(param_names),
        'final_logz': float(logz),
        'final_dlogz': float(dlogz) if np.isfinite(dlogz) else None,
        'parameter_names': param_names,
        'parameter_statistics': param_stats,
        'max_logl': float(np.max(logl)),
        'min_logl': float(np.min(logl)),
        'mean_logl': float(np.mean(logl))
    }

def extract_results_from_progress_json(json_file):
    """Extract results from progress JSON file"""
    print(f"Loading progress data from: {json_file}")
    
    with open(json_file, 'r') as f:
        progress_data = json.load(f)
    
    # Extract key metrics
    results = {
        'iterations': progress_data.get('iterations', 0),
        'ncall': progress_data.get('ncall', 0),
        'efficiency_percent': progress_data.get('efficiency_percent', 0),
        'final_logz': progress_data.get('final_logz', 0),
        'final_logzerr': progress_data.get('final_logzerr', 0),
        'elapsed_hours': progress_data.get('elapsed_hours', 0)
    }
    
    print(f"✓ {results['iterations']} iterations completed")
    print(f"✓ {results['ncall']} function calls")
    print(f"✓ {results['efficiency_percent']:.1f}% efficiency")
    print(f"✓ {results['elapsed_hours']:.1f} hours elapsed")
    
    return results

def extract_results_from_pkl(pkl_file):
    """Extract results from pickle file"""
    print(f"Loading full results from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract key attributes
    extracted = {
        'has_samples': hasattr(results, 'samples'),
        'has_logz': hasattr(results, 'logz'),
        'has_logl': hasattr(results, 'logl'),
        'has_weights': hasattr(results, 'weights'),
        'n_samples': len(results.samples) if hasattr(results, 'samples') else 0,
        'final_logz': float(results.logz[-1]) if hasattr(results, 'logz') and len(results.logz) > 0 else None,
        'final_logzerr': float(results.logzerr[-1]) if hasattr(results, 'logzerr') and len(results.logzerr) > 0 else None,
        'ncall': int(np.sum(results.ncall)) if hasattr(results, 'ncall') else 0,
        'efficiency': float(results.efficiency) if hasattr(results, 'efficiency') else None
    }
    
    print(f"✓ Full dynesty results object loaded")
    print(f"✓ Final LogZ: {extracted['final_logz']:.2f}")
    print(f"✓ LogZ error: {extracted['final_logzerr']:.2f}")
    
    return extracted

def generate_readme_section(results_dict, xi_type="gr"):
    """Generate a README section with the results"""
    
    print("\n" + "="*60)
    print("README SECTION FOR GR BASELINE RESULTS")
    print("="*60)
    
    # Extract key values
    final_logz = results_dict.get('final_logz', 'N/A')
    n_samples = results_dict.get('n_samples', 'N/A')
    n_parameters = results_dict.get('n_parameters', 'N/A')
    efficiency = results_dict.get('efficiency_percent', 'N/A')
    elapsed_hours = results_dict.get('elapsed_hours', 'N/A')
    
    # Parameter results
    param_stats = results_dict.get('parameter_statistics', {})
    
    readme_text = f"""
## GR Baseline Results (xi = {xi_type.upper()})

### Sampling Summary
- **Final Log-Evidence (LogZ):** {final_logz:.2f}
- **Number of Samples:** {n_samples:,}
- **Number of Parameters:** {n_parameters}
- **Sampling Efficiency:** {efficiency:.1f}%
- **Runtime:** {elapsed_hours:.1f} hours

### Parameter Estimates
"""
    
    for param_name, stats in param_stats.items():
        median = stats['median']
        std = stats['std']
        p16 = stats['16th_percentile']
        p84 = stats['84th_percentile']
        
        # Format based on parameter magnitude
        if median > 1e6:
            readme_text += f"- **{param_name}:** {median:.2e} ± {std:.2e} ({p16:.2e} - {p84:.2e})\n"
        else:
            readme_text += f"- **{param_name}:** {median:.3f} ± {std:.3f} ({p16:.3f} - {p84:.3f})\n"
    
    readme_text += f"""
### Model Performance
- **Maximum Log-Likelihood:** {results_dict.get('max_logl', 'N/A'):.2f}
- **Mean Log-Likelihood:** {results_dict.get('mean_logl', 'N/A'):.2f}

### Technical Details
- **Sampling Method:** Dynamic Nested Sampling (CuPy-optimized)
- **Live Points:** 1000
- **Convergence Criterion:** dlogz < 0.01
- **Hardware:** NVIDIA RTX 5090 GPU + 24-core CPU
"""
    
    return readme_text

def save_results_summary(results_dict, output_file):
    """Save a clean summary of results"""
    summary = {
        'timestamp': '2025-08-04',
        'xi_type': 'gr',
        'model': 'GR Baseline',
        'results': results_dict
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract GR baseline results for README")
    parser.add_argument('--results_dir', type=str, default='cupy_results',
                       help='Directory containing results files')
    parser.add_argument('--output_file', type=str, default='gr_baseline_summary.json',
                       help='Output file for results summary')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found!")
        return
    
    # Load results from different file types
    npz_file = results_dir / "posterior_samples.npz"
    json_file = results_dir / "dynesty_progress.json"
    pkl_file = results_dir / "results.pkl"
    
    combined_results = {}
    
    # Extract from NPZ file (main results)
    if npz_file.exists():
        npz_results = extract_results_from_npz(npz_file)
        combined_results.update(npz_results)
    else:
        print(f"Warning: {npz_file} not found")
    
    # Extract from progress JSON
    if json_file.exists():
        json_results = extract_results_from_progress_json(json_file)
        combined_results.update(json_results)
    else:
        print(f"Warning: {json_file} not found")
    
    # Extract from pickle file
    if pkl_file.exists():
        pkl_results = extract_results_from_pkl(pkl_file)
        combined_results.update(pkl_results)
    else:
        print(f"Warning: {pkl_file} not found")
    
    # Generate README section
    readme_section = generate_readme_section(combined_results, "gr")
    
    # Save results summary
    save_results_summary(combined_results, args.output_file)
    
    # Print README section
    print(readme_section)
    
    # Also save README section to file
    readme_file = Path("gr_baseline_readme_section.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_section)
    
    print(f"\n✓ README section saved to: {readme_file}")
    print(f"✓ Results summary saved to: {args.output_file}")
    
    print("\n" + "="*60)
    print("KEY VALUES FOR README:")
    print("="*60)
    print(f"Final LogZ: {combined_results.get('final_logz', 'N/A'):.2f}")
    print(f"Number of samples: {combined_results.get('n_samples', 'N/A'):,}")
    print(f"Sampling efficiency: {combined_results.get('efficiency_percent', 'N/A'):.1f}%")
    print(f"Runtime: {combined_results.get('elapsed_hours', 'N/A'):.1f} hours")

if __name__ == "__main__":
    main() 