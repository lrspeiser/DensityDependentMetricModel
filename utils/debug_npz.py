#!/usr/bin/env python3
"""
Debug script to inspect NPZ file structure
"""

import numpy as np
from pathlib import Path

def debug_npz():
    """Debug the NPZ file structure"""
    
    npz_file = Path("runs/gr_20250804_153029/posterior_samples.npz")
    
    print("="*60)
    print("DEBUGGING NPZ FILE STRUCTURE")
    print("="*60)
    
    # Load the file
    data = np.load(npz_file, allow_pickle=True)
    
    print(f"File: {npz_file}")
    print(f"Available keys: {list(data.keys())}")
    print()
    
    # Inspect each key
    for key in data.keys():
        arr = data[key]
        print(f"Key: {key}")
        print(f"  Type: {type(arr)}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Size: {arr.size}")
        
        if arr.size <= 10:
            print(f"  Values: {arr}")
        else:
            print(f"  First 5 values: {arr.flat[:5]}")
            print(f"  Last 5 values: {arr.flat[-5:]}")
        
        print()

if __name__ == "__main__":
    debug_npz() 