#!/usr/bin/env python3
"""
run_smart_dynesty_batch.py - Run multiple strategies in parallel
"""

import subprocess
import json
from pathlib import Path

strategies = [
    {
        'name': 'tight_physical',
        'args': {
            '--use_previous_best': True,
            '--tighten_bounds_factor': 0.05,  # Very tight
            '--sample_method': 'rwalk',  # More local
            '--enlarge_factor': 1.2,  # Small enlargement
            '--bound_method': 'single'  # Single ellipsoid
        }
    },
    {
        'name': 'constrained_sampling',
        'args': {
            '--use_previous_best': True,
            '--enforce_physical_constraints': True,
            '--constraint_penalty': 100,
            '--sample_method': 'rslice'
        }
    },
    {
        'name': 'mode_locked',
        'args': {
            '--use_mode_locking': True,
            '--lock_strength': 0.8,
            '--mode_file': 'physical_mode.npz'
        }
    }
]

def run_strategy(strategy):
    """Run a single strategy."""
    output_dir = f"chains_strategy_{strategy['name']}"
    cmd = ['python', 'run_dynesty.py', '--output_dir', output_dir]
    
    for arg, value in strategy['args'].items():
        if isinstance(value, bool):
            if value:
                cmd.append(arg)
        else:
            cmd.extend([arg, str(value)])
            
    logger.info(f"Running strategy: {strategy['name']}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd)