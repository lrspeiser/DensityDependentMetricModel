#\!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cupy as cp
from pathlib import Path
import json
from datetime import datetime

# Import the models
from core.density_metric_cupy import xi_power_law_cupy, xi_hybrid_safe_cupy

def test_xi_models():
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"debug_xi_test_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating output directory: {output_dir}")
    
    # Test parameters
    test_params = {
        "power": {"rho_c": 1e-29, "n_exp": 1.5, "A": 5.0},
        "hybrid_safe": {"rho_c": 1e-29, "n_exp": 1.5, "A": 5.0}
    }
    
    # Test density range
    log_rho = np.linspace(-35, -20, 100)
    rho_test = 10**log_rho
    
    print("Testing Xi Models:")
    
    for model_name in ["power", "hybrid_safe"]:
        print(f"{model_name} model...")
        
        params = test_params[model_name]
        rho_c = cp.array(params["rho_c"], dtype=cp.float32)
        n_exp = cp.array(params["n_exp"], dtype=cp.float32)
        A = cp.array(params["A"], dtype=cp.float32)
        rho_cp = cp.array(rho_test, dtype=cp.float32)
        
        try:
            if model_name == "power":
                xi = xi_power_law_cupy(rho_cp, rho_c, n_exp, A)
            else:
                xi = xi_hybrid_safe_cupy(rho_cp, rho_c, n_exp, A)
            
            xi_np = cp.asnumpy(xi)
            
            n_finite = np.sum(np.isfinite(xi_np))
            if n_finite > 0:
                print(f"  Min: {np.min(xi_np[np.isfinite(xi_np)]):.6f}")
                print(f"  Max: {np.max(xi_np[np.isfinite(xi_np)]):.6f}")
            print(f"  Finite: {n_finite}/{len(xi_np)}")
            
            np.save(str(output_dir / f"xi_{model_name}.npy"), xi_np)
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    test_xi_models()
