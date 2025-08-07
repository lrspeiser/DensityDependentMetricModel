import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cupy as cp
from pathlib import Path
from datetime import datetime
from core.density_metric_cupy import xi_power_law_cupy, xi_hybrid_safe_cupy

print("Testing Xi Models...")
test_params = {"rho_c": 1e-29, "n_exp": 1.5, "A": 5.0}
log_rho = np.linspace(-35, -20, 100)
rho_test = 10**log_rho

rho_c = cp.array(test_params["rho_c"], dtype=cp.float32)
n_exp = cp.array(test_params["n_exp"], dtype=cp.float32)
A = cp.array(test_params["A"], dtype=cp.float32)
rho_cp = cp.array(rho_test, dtype=cp.float32)

print("Power law model:")
xi = xi_power_law_cupy(rho_cp, rho_c, n_exp, A)
xi_np = cp.asnumpy(xi)
print(f"  Min: {np.min(xi_np):.6f}, Max: {np.max(xi_np):.6f}")

print("Hybrid safe model:")
xi = xi_hybrid_safe_cupy(rho_cp, rho_c, n_exp, A)
xi_np = cp.asnumpy(xi)
print(f"  Min: {np.min(xi_np):.6f}, Max: {np.max(xi_np):.6f}")
