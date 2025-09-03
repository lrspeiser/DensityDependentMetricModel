#!/usr/bin/env python3
"""Quick script to check Dynesty run progress from checkpoint."""

import pickle
import numpy as np
from pathlib import Path
import sys

def check_checkpoint(checkpoint_file):
    """Load and analyze a dynesty checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        results = pickle.load(f)
    
    if hasattr(results, 'logz'):
        n_iter = len(results.logz)
        current_logz = results.logz[-1] if n_iter > 0 else np.nan
        
        print(f"\nProgress Summary:")
        print(f"  Iterations: {n_iter}")
        print(f"  Current LogZ: {current_logz:.2f}")
        
        if n_iter > 10:
            recent_dlogz = np.diff(results.logz[-10:])
            print(f"  Recent dLogZ (avg): {np.mean(recent_dlogz):.4f}")
            print(f"  Converging: {'Yes' if np.mean(recent_dlogz) < 0.01 else 'No'}")
        
        if hasattr(results, 'samples') and hasattr(results, 'logl'):
            n_samples = len(results.samples)
            best_idx = np.argmax(results.logl)
            best_logl = results.logl[best_idx]
            print(f"  Samples: {n_samples}")
            print(f"  Best LogL: {best_logl:.2f}")
            
            # Model comparison
            BASELINE_LOGZ_GR = -1490897.53
            delta_logz = current_logz - BASELINE_LOGZ_GR
            print(f"\nModel Comparison:")
            print(f"  vs GR: ΔLogZ = {delta_logz:+.2f}")
            if delta_logz > 0:
                print(f"  Status: DDMM preferred")
            else:
                print(f"  Status: GR preferred")
    else:
        print("No logz data in checkpoint")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint = Path(sys.argv[1])
    else:
        # Find latest elastic_strain run
        runs_dir = Path("runs")
        elastic_runs = sorted(runs_dir.glob("elastic_strain_*"))
        if elastic_runs:
            checkpoint = elastic_runs[-1] / "dynesty_checkpoint.pkl"
        else:
            print("No elastic_strain runs found")
            sys.exit(1)
    
    if checkpoint.exists():
        check_checkpoint(checkpoint)
    else:
        print(f"Checkpoint not found: {checkpoint}")