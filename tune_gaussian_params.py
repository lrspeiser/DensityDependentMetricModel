#!/usr/bin/env python3
"""
Tune Gaussian xi parameters to satisfy all constraints
"""
import numpy as np
from density_metric2 import xi_gaussian_enhancement

def test_params(rho_peak, sigma_log, lambda_max):
    """Test a parameter set and return scores"""
    # Key densities
    rho_halo = 0.01
    rho_galaxy = 0.5
    rho_solar = 100.0
    rho_stellar = 1e6
    
    # Calculate xi values
    xi_halo = xi_gaussian_enhancement(rho_halo, rho_peak, sigma_log, lambda_max)[0]
    xi_galaxy = xi_gaussian_enhancement(rho_galaxy, rho_peak, sigma_log, lambda_max)[0]
    xi_solar = xi_gaussian_enhancement(rho_solar, rho_peak, sigma_log, lambda_max)[0]
    xi_stellar = xi_gaussian_enhancement(rho_stellar, rho_peak, sigma_log, lambda_max)[0]
    
    # Score based on requirements
    scores = {
        'solar_deviation': abs(xi_solar - 1.0),  # Should be < 0.05
        'galaxy_enhancement': xi_galaxy,         # Should be > 2.0
        'halo_enhancement': xi_halo,             # Should be > 1.5
        'stellar_normal': abs(xi_stellar - 1.0), # Should be < 0.01
    }
    
    # Overall pass/fail
    passes = (
        scores['solar_deviation'] < 0.05 and
        scores['galaxy_enhancement'] > 2.0 and
        scores['halo_enhancement'] > 1.5 and
        scores['stellar_normal'] < 0.01
    )
    
    return scores, passes, {
        'halo': xi_halo,
        'galaxy': xi_galaxy,
        'solar': xi_solar,
        'stellar': xi_stellar
    }

print("Testing Gaussian Xi Parameter Combinations")
print("="*80)

# Test different parameter combinations
test_configs = [
    # (rho_peak, sigma_log, lambda_max, description)
    (0.5, 1.0, 2.0, "Original"),
    (0.3, 0.8, 2.5, "Shifted & narrower"),
    (0.1, 0.8, 3.0, "Lower peak, stronger"),
    (0.2, 0.7, 3.0, "Optimized 1"),
    (0.15, 0.75, 3.5, "Optimized 2"),
    (0.1, 0.6, 4.0, "Narrow strong"),
    (0.05, 0.5, 5.0, "Very low peak"),
]

best_config = None
best_score = float('inf')

for rho_peak, sigma_log, lambda_max, desc in test_configs:
    scores, passes, xi_vals = test_params(rho_peak, sigma_log, lambda_max)
    
    print(f"\n{desc}: ρ_peak={rho_peak}, σ={sigma_log}, λ={lambda_max}")
    print("-"*60)
    print(f"ξ(0.01) = {xi_vals['halo']:.3f}  [halo, want >1.5]")
    print(f"ξ(0.5)  = {xi_vals['galaxy']:.3f}  [galaxy, want >2.0]")
    print(f"ξ(100)  = {xi_vals['solar']:.3f}  [Solar System, want ≈1.0]")
    print(f"ξ(1e6)  = {xi_vals['stellar']:.3f}  [stellar, want ≈1.0]")
    
    # Scoring
    total_score = scores['solar_deviation'] + (1.0 / scores['galaxy_enhancement'])
    
    if passes:
        print("✓ PASSES ALL CONSTRAINTS!")
        if total_score < best_score:
            best_score = total_score
            best_config = (rho_peak, sigma_log, lambda_max, desc)
    else:
        print("✗ Fails:", end=" ")
        if scores['solar_deviation'] >= 0.05:
            print(f"Solar({scores['solar_deviation']:.3f})", end=" ")
        if scores['galaxy_enhancement'] <= 2.0:
            print(f"Galaxy({scores['galaxy_enhancement']:.3f})", end=" ")
        print()

# Fine-tune around promising values
print("\n" + "="*80)
print("FINE-TUNING SEARCH")
print("="*80)

# Grid search around promising region
rho_peaks = np.logspace(-2, -0.5, 10)  # 0.01 to 0.3
sigma_logs = np.linspace(0.5, 1.0, 6)   # 0.5 to 1.0
lambda_maxs = np.linspace(2.0, 5.0, 7)  # 2.0 to 5.0

valid_configs = []

for rho_p in rho_peaks:
    for sigma in sigma_logs:
        for lam in lambda_maxs:
            scores, passes, xi_vals = test_params(rho_p, sigma, lam)
            if passes:
                total_score = scores['solar_deviation'] + (1.0 / scores['galaxy_enhancement'])
                valid_configs.append((total_score, rho_p, sigma, lam, xi_vals))

# Sort by score
valid_configs.sort(key=lambda x: x[0])

print(f"\nFound {len(valid_configs)} valid configurations!")
print("\nTop 5 configurations:")
print("-"*80)

for i, (score, rho_p, sigma, lam, xi_vals) in enumerate(valid_configs[:5]):
    print(f"\n#{i+1}: ρ_peak={rho_p:.3f}, σ={sigma:.2f}, λ={lam:.1f}")
    print(f"   ξ(halo)={xi_vals['halo']:.3f}, ξ(galaxy)={xi_vals['galaxy']:.3f}, "
          f"ξ(solar)={xi_vals['solar']:.3f}")

if valid_configs:
    # Best config
    _, rho_best, sigma_best, lam_best, _ = valid_configs[0]
    print("\n" + "="*80)
    print("RECOMMENDED PARAMETERS:")
    print(f"  rho_c = {rho_best:.3f}")
    print(f"  n_exp = {sigma_best:.2f}")
    print(f"  A = {lam_best:.1f}")
    print("\nUse these in your parameter file or command line:")
    print(f"  --rho_c_fixed {rho_best:.3f} --n_exp_fixed {sigma_best:.2f} --A_fixed {lam_best:.1f}")
elif best_config:
    print("\n" + "="*80)
    print("BEST PARAMETERS FROM INITIAL TESTS:")
    print(f"  rho_c = {best_config[0]}")
    print(f"  n_exp = {best_config[1]}")
    print(f"  A = {best_config[2]}")
else:
    print("\nNo valid configurations found! You may need to expand the search range.")