#!/usr/bin/env python3
"""Check what parameters the model is exploring."""

import pickle
import numpy as np
from pathlib import Path

# Parameter names for elastic_strain
param_names = [
    'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
    'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc', 
    'M_bulge_solar', 'R_bulge_kpc',
    'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
    'relaxation_scale', 'strain_critical', 'k_elastic'
]

checkpoint = Path("runs/elastic_strain_20250806_224453/dynesty_checkpoint.pkl")

with open(checkpoint, 'rb') as f:
    results = pickle.load(f)

if hasattr(results, 'samples') and hasattr(results, 'logl'):
    # Get best fit
    best_idx = np.argmax(results.logl)
    best_params = results.samples[best_idx]
    
    print("Best-fit parameters:")
    print("-" * 50)
    for name, value in zip(param_names, best_params):
        if 'solar' in name:
            print(f"{name:25s}: {value:.3e}")
        else:
            print(f"{name:25s}: {value:.6f}")
    
    print(f"\nBest LogL: {results.logl[best_idx]:.2f}")
    
    # Check k_elastic distribution
    k_elastic_idx = param_names.index('k_elastic')
    k_elastic_values = results.samples[:, k_elastic_idx]
    
    print(f"\nk_elastic statistics:")
    print(f"  Min: {np.min(k_elastic_values):.6f}")
    print(f"  Max: {np.max(k_elastic_values):.6f}")
    print(f"  Mean: {np.mean(k_elastic_values):.6f}")
    print(f"  Best: {best_params[k_elastic_idx]:.6f}")
    
    # Check mass parameters
    M_thin_idx = param_names.index('M_thin_disk_solar')
    M_thin_values = results.samples[:, M_thin_idx]
    print(f"\nM_thin_disk_solar statistics:")
    print(f"  Min: {np.min(M_thin_values):.3e}")
    print(f"  Max: {np.max(M_thin_values):.3e}")
    print(f"  Best: {best_params[M_thin_idx]:.3e}")