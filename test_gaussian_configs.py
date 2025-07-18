#!/usr/bin/env python3
import numpy as np
from density_metric2 import xi_gaussian_enhancement

# Test each configuration
configs = [
    ("conservative", 0.5, 0.8, 1.5),
    ("standard", 0.5, 1.0, 2.0),
    ("broad", 0.3, 1.5, 2.0),
    ("strong", 0.5, 1.0, 3.0),
    ("shifted", 0.1, 1.2, 2.5)
]

# Key density checkpoints
test_points = [
    (0.01, "Halo edge"),
    (0.5, "Galaxy disk"),
    (10, "Galaxy center"),
    (100, "Solar System"),
    (1e6, "Stellar")
]

print("\nGaussian Xi Function Tests")
print("="*80)

for name, rho_c, n_exp, A in configs:
    print(f"\n{name.upper()}: rho_c={rho_c}, sigma={n_exp}, lambda={A}")
    print("-"*60)
    print("Density      | Location      | xi    | v_factor | Status")
    print("-"*60)
    
    for rho, loc in test_points:
        xi = xi_gaussian_enhancement(rho, rho_c, n_exp, A)[0]
        v_factor = np.sqrt(xi)
        
        # Check constraints
        if loc == "Solar System":
            status = "✓" if abs(xi - 1.0) < 0.1 else "✗ FAIL"
        elif loc in ["Galaxy disk", "Halo edge"]:
            status = "✓" if xi > 1.5 else "weak"
        else:
            status = "ok"
            
        print(f"{rho:12.2e} | {loc:13s} | {xi:5.3f} | {v_factor:8.3f} | {status}")

print("\n" + "="*80)
print("Best configs should have:")
print("- Solar System: xi ≈ 1.0 (✓)")
print("- Galaxy disk: xi > 1.5 (✓)")
print("- Smooth transitions between regimes")
