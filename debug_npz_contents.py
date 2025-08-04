#!/usr/bin/env python3
"""
Debug script to inspect NPZ file contents
"""

import numpy as np
from pathlib import Path

def inspect_npz(npz_file):
    """Inspect contents of NPZ file"""
    print(f"Inspecting: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    
    print("\nAvailable keys:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\nData shapes and types:")
    for key in data.keys():
        value = data[key]
        print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'scalar'}")
        
        if key == 'samples':
            print(f"    Samples shape: {value.shape}")
            print(f"    Number of parameters: {value.shape[1] if len(value.shape) > 1 else 0}")
        
        if key == 'param_names':
            print(f"    Parameter names: {value}")
            if isinstance(value, np.ndarray):
                print(f"    As list: {value.tolist()}")
    
    # Try to infer parameter names if missing
    if 'samples' in data and 'param_names' not in data:
        samples = data['samples']
        n_params = samples.shape[1] if len(samples.shape) > 1 else 0
        print(f"\nInferred {n_params} parameters from samples shape")
        
        # Create default parameter names
        default_names = [f"param_{i}" for i in range(n_params)]
        print(f"Default parameter names: {default_names}")

if __name__ == "__main__":
    npz_file = Path("cupy_results/posterior_samples.npz")
    if npz_file.exists():
        inspect_npz(npz_file)
    else:
        print(f"File not found: {npz_file}") 