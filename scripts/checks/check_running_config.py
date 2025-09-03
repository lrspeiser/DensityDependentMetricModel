#!/usr/bin/env python3
"""
Check configuration of currently running sampling.
"""

import json
import os
import pickle
from datetime import datetime

# Check the latest run directory
run_dir = "runs/balanced_screening_single_20250807_123401"

print("=" * 70)
print("CHECKING RUNNING SAMPLER CONFIGURATION")
print("=" * 70)

# Check config file
config_file = os.path.join(run_dir, "config.json")
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"\nRun started: {config.get('start_time', 'Unknown')}")
    print(f"Xi type: {config.get('xi_type', 'Unknown')}")
    print(f"Data points: {config.get('n_data', 'Unknown')}")
    
    if 'param_bounds' in config:
        bounds = config['param_bounds']
        print("\nParameter bounds:")
        for i, (low, high) in enumerate(zip(bounds['low'], bounds['high'])):
            param_names = ['M_thin_disk', 'R_thin_disk', 'hz_thin_disk',
                          'M_thick_disk', 'R_thick_disk', 'hz_thick_disk', 
                          'M_bulge', 'R_bulge',
                          'M_gas', 'R_gas', 'hz_gas',
                          'rho_c', 'R_screen', 'n_exp', 'A_max']
            if i < len(param_names):
                print(f"  {param_names[i]:15s}: [{low:.2e}, {high:.2e}]")
else:
    print(f"\nNo config file found at {config_file}")

# Check when the gravitational constant was fixed
fix_time = datetime(2025, 8, 7, 13, 30)  # Approximate time of fix
print(f"\nGravitational constant fixed at: {fix_time}")

# Parse start time from directory name
dir_time_str = "20250807_123401"
dir_time = datetime.strptime(dir_time_str, "%Y%m%d_%H%M%S")
print(f"Run started at: {dir_time}")

if dir_time < fix_time:
    print("\n⚠️  WARNING: This run started BEFORE the gravitational constant fix!")
    print("    The model is using G = 4.302e-3 (wrong) instead of 4.302e-6")
    print("    This explains the poor likelihood values.")
    print("\n    RECOMMENDATION: Stop this run and start a new one.")
else:
    print("\n✓ This run started AFTER the gravitational constant fix.")
    print("  Should be using the correct physics.")

print("\n" + "=" * 70)