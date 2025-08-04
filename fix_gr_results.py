#!/usr/bin/env python3
"""
Fix GR baseline results - extract runtime info and fix parameter formatting
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

def extract_runtime_info():
    """Extract runtime information from various sources"""
    
    # Check resource usage file
    resource_file = Path("cupy_results/resource_usage.json")
    if resource_file.exists():
        with open(resource_file, 'r') as f:
            resource_data = json.load(f)
        
        if resource_data:
            # Get first and last timestamps
            first_time = resource_data[0].get('timestamp', '')
            last_time = resource_data[-1].get('timestamp', '')
            
            if first_time and last_time:
                try:
                    # Parse timestamps
                    start_dt = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                    elapsed = end_dt - start_dt
                    elapsed_hours = elapsed.total_seconds() / 3600
                    print(f"✓ Runtime from resource monitoring: {elapsed_hours:.2f} hours")
                    return elapsed_hours
                except:
                    pass
    
    # Check hardware info file
    hardware_file = Path("cupy_results/hardware_info.json")
    if hardware_file.exists():
        with open(hardware_file, 'r') as f:
            hardware_data = json.load(f)
        print(f"✓ Hardware info: {hardware_data}")
    
    return None

def fix_parameter_formatting():
    """Fix the parameter formatting in the results"""
    
    # Load the NPZ file
    npz_file = Path("cupy_results/posterior_samples.npz")
    data = np.load(npz_file, allow_pickle=True)
    
    samples = data['samples']
    param_names = ['M_disk_solar', 'R_d_kpc']
    
    print("\n=== PARAMETER ANALYSIS ===")
    print(f"Sample shape: {samples.shape}")
    print(f"Parameter names: {param_names}")
    
    # Analyze each parameter
    for i, name in enumerate(param_names):
        values = samples[:, i]
        
        # Calculate statistics
        median = np.median(values)
        mean = np.mean(values)
        std = np.std(values)
        p16 = np.percentile(values, 16)
        p84 = np.percentile(values, 84)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"\n{name}:")
        print(f"  Median: {median:.3e}")
        print(f"  Mean:   {mean:.3e}")
        print(f"  Std:    {std:.3e}")
        print(f"  16th %: {p16:.3e}")
        print(f"  84th %: {p84:.3e}")
        print(f"  Range:  {min_val:.3e} to {max_val:.3e}")
        
        # Check if hitting bounds
        if name == 'M_disk_solar':
            if max_val > 4e11:
                print(f"  ⚠️  WARNING: Hitting upper bound (5e11)")
            if min_val < 1.1e10:
                print(f"  ⚠️  WARNING: Hitting lower bound (1e10)")
        elif name == 'R_d_kpc':
            if max_val > 7.9:
                print(f"  ⚠️  WARNING: Hitting upper bound (8.0)")
            if min_val < 1.1:
                print(f"  ⚠️  WARNING: Hitting lower bound (1.0)")

def generate_corrected_readme():
    """Generate a corrected README section"""
    
    # Load results
    npz_file = Path("cupy_results/posterior_samples.npz")
    data = np.load(npz_file, allow_pickle=True)
    
    samples = data['samples']
    logz = data['logz']
    logl = data['logl']
    
    # Get runtime
    runtime_hours = extract_runtime_info()
    
    # Calculate parameter statistics
    param_names = ['M_disk_solar', 'R_d_kpc']
    param_stats = {}
    
    for i, name in enumerate(param_names):
        values = samples[:, i]
        param_stats[name] = {
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'p16': float(np.percentile(values, 16)),
            'p84': float(np.percentile(values, 84))
        }
    
    # Generate README text
    readme_text = f"""
## GR Baseline Results (xi = GR)

### Sampling Summary
- **Final Log-Evidence (LogZ):** {float(logz):.2f}
- **Number of Samples:** {len(samples):,}
- **Number of Parameters:** {len(param_names)}
- **Runtime:** {runtime_hours:.1f} hours (estimated)

### Parameter Estimates
"""
    
    for param_name, stats in param_stats.items():
        median = stats['median']
        std = stats['std']
        p16 = stats['p16']
        p84 = stats['p84']
        
        if median > 1e6:
            readme_text += f"- **{param_name}:** {median:.2e} ± {std:.2e} ({p16:.2e} - {p84:.2e})\n"
        else:
            readme_text += f"- **{param_name}:** {median:.3f} ± {std:.3f} ({p16:.3f} - {p84:.3f})\n"
    
    readme_text += f"""
### Model Performance
- **Maximum Log-Likelihood:** {float(np.max(logl)):.2f}
- **Mean Log-Likelihood:** {float(np.mean(logl)):.2f}

### Technical Details
- **Sampling Method:** Dynamic Nested Sampling (CuPy-optimized)
- **Live Points:** 1000
- **Convergence Criterion:** dlogz < 0.01
- **Hardware:** NVIDIA RTX 5090 GPU + 24-core CPU
- **Data:** Gaia DR3 Milky Way rotation curve
"""
    
    return readme_text

def main():
    print("=== GR BASELINE RESULTS ANALYSIS ===")
    
    # Fix parameter formatting
    fix_parameter_formatting()
    
    # Generate corrected README
    readme_text = generate_corrected_readme()
    
    # Save corrected README
    with open("gr_baseline_readme_corrected.md", 'w', encoding='utf-8') as f:
        f.write(readme_text)
    
    print("\n" + "="*60)
    print("CORRECTED README SECTION:")
    print("="*60)
    print(readme_text)
    
    print(f"\n✓ Corrected README saved to: gr_baseline_readme_corrected.md")

if __name__ == "__main__":
    main() 