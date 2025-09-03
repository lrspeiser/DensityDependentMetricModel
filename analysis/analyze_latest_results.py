#!/usr/bin/env python3
"""Analyze the latest grav_color_void_safe_fixed results."""

import numpy as np
import pickle
from pathlib import Path

# Load the latest results
run_dir = Path("runs/grav_color_void_safe_fixed_20250806_235701")

print("=" * 70)
print("LATEST GRAV_COLOR_VOID_SAFE_FIXED MODEL RESULTS")
print("=" * 70)

# Load NPZ results
npz_file = run_dir / "posterior_samples.npz"
if npz_file.exists():
    data = np.load(npz_file, allow_pickle=True)
    
    samples = data['samples']
    logl = data['logl']
    logz = data['logz']
    param_names = data['param_names']
    xi_type = str(data['xi_type'])
    
    print(f"\nModel Type: {xi_type}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of parameters: {len(param_names)}")
    
    print(f"\nFinal LogZ: {logz:.2f}")
    
    # Compare to GR baseline
    BASELINE_LOGZ_GR = -1490897.53
    delta_logz = logz - BASELINE_LOGZ_GR
    print(f"Delta LogZ vs GR: {delta_logz:+.2f}")
    
    if delta_logz > 10:
        print(">>> STRONG EVIDENCE: DDMM model STRONGLY PREFERRED over GR! <<<")
    elif delta_logz > 5:
        print(">>> SUBSTANTIAL EVIDENCE: DDMM model preferred over GR! <<<")
    elif delta_logz > 0:
        print(">>> WEAK EVIDENCE: DDMM model slightly preferred over GR <<<")
    else:
        print(">>> GR model preferred over DDMM <<<")
    
    # Get best-fit parameters
    best_idx = np.argmax(logl)
    best_params = samples[best_idx]
    best_logl = logl[best_idx]
    
    print(f"\nBest LogL: {best_logl:.2f}")
    
    # Check if this is a reasonable likelihood
    # For 144,000 data points, a chi2/dof ~ 1 would give logL ~ -72,000
    expected_good_logl = -144000 / 2  # Assuming chi2 ~ N_data
    print(f"Expected good LogL (chi2/dof~1): {expected_good_logl:.0f}")
    
    if best_logl < -1e9:
        print("WARNING: Extremely poor likelihood - model may be failing!")
    elif best_logl < -1e6:
        print("WARNING: Poor likelihood - parameters may need adjustment")
    
    print("\nBest-fit parameters:")
    print("-" * 50)
    for name, value in zip(param_names, best_params):
        if 'M_' in name or 'rho_c' in name:
            print(f"{name:25s}: {value:.3e}")
        else:
            print(f"{name:25s}: {value:.6f}")
    
    # Parameter statistics
    print("\nParameter Statistics (16%, 50%, 84% quantiles):")
    print("-" * 50)
    for i, name in enumerate(param_names):
        param_samples = samples[:, i]
        q16, q50, q84 = np.percentile(param_samples, [16, 50, 84])
        if 'M_' in name or 'rho_c' in name:
            print(f"{name:25s}: {q50:.3e} (+{q84-q50:.3e} / -{q50-q16:.3e})")
        else:
            print(f"{name:25s}: {q50:.4f} (+{q84-q50:.4f} / -{q50-q16:.4f})")
    
    # Special focus on grav_color_void_safe specific parameters
    if 'gamma_exp' in param_names:
        gamma_idx = list(param_names).index('gamma_exp')
        gamma_best = best_params[gamma_idx]
        gamma_samples = samples[:, gamma_idx]
        print(f"\n*** Gamma exponent (controls density dependence) ***")
        print(f"  Best: {gamma_best:.4f}")
        print(f"  Median: {np.median(gamma_samples):.4f}")
        print(f"  Range: [{np.min(gamma_samples):.4f}, {np.max(gamma_samples):.4f}]")
        
    if 'lambda_g' in param_names:
        lambda_idx = list(param_names).index('lambda_g')
        lambda_best = best_params[lambda_idx]
        lambda_samples = samples[:, lambda_idx]
        print(f"\n*** Lambda_g (enhancement strength) ***")
        print(f"  Best: {lambda_best:.4f}")
        print(f"  Median: {np.median(lambda_samples):.4f}")
        print(f"  Range: [{np.min(lambda_samples):.4f}, {np.max(lambda_samples):.4f}]")
    
    # Check log-likelihood evolution
    print(f"\nLog-Likelihood Evolution:")
    print(f"  Initial: {logl[0]:.2f}")
    print(f"  Final: {logl[-1]:.2f}")
    print(f"  Best: {best_logl:.2f}")
    print(f"  Improvement: {best_logl - logl[0]:.2f}")

# Also load pickle for more detailed results
pkl_file = run_dir / "results.pkl"
if pkl_file.exists():
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    
    if hasattr(results, 'logzerr'):
        print(f"\nLogZ Error: ± {results.logzerr[-1]:.2f}")
    
    if hasattr(results, 'eff'):
        print(f"Sampling Efficiency: {results.eff:.2%}")
    
    if hasattr(results, 'niter'):
        print(f"Number of iterations: {results.niter}")
    
    if hasattr(results, 'ncall'):
        print(f"Total likelihood calls: {results.ncall}")

print("\n" + "=" * 70)