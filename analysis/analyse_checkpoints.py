#!/usr/bin/env python3
# save as: analyze_checkpoints.py
import numpy as np
from pathlib import Path
import json

def analyze_run_history(output_dir):
    """Analyze all runs to find patterns in where they stop."""
    
    output_dir = Path(output_dir)
    
    # Find all progress files
    progress_files = sorted(output_dir.glob("**/dynesty_progress.json"))
    
    print(f"Found {len(progress_files)} progress files")
    
    for pf in progress_files:
        try:
            with open(pf) as f:
                data = json.load(f)
            
            print(f"\n=== {pf} ===")
            print(f"Phase: {data.get('phase', 'unknown')}")
            print(f"Samples: {data.get('n_samples', 0)}")
            print(f"Efficiency: {data.get('efficiency_percent', 0):.2f}%")
            print(f"Elapsed: {data.get('elapsed_hours', 0):.2f} hours")
            print(f"dlogZ: {data.get('dlogz', 'N/A')}")
            
            # Check if stuck
            hb = data.get('heartbeat', {})
            if hb.get('is_stuck'):
                print("⚠️  SAMPLER WAS STUCK!")
                
        except Exception as e:
            print(f"Error reading {pf}: {e}")

if __name__ == "__main__":
    import sys
    analyze_run_history(sys.argv[1] if len(sys.argv) > 1 else "chains_dynesty")