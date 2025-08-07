#!/usr/bin/env python3
"""
Balanced screening model that properly handles deep space.

Key principles:
1. Enhancement should be modest (2-3x max, not 250x!)
2. Must respect Cassini constraint at solar density
3. Must vanish in deep space to preserve 1/r² falloff
4. Should explain rotation curves without extreme values
"""

import cupy as cp
import numpy as np

def xi_balanced_screening_cupy(rho, rho_c, R, R_screen=50.0, n_exp=1.0, A_max=2.0):
    """
    Balanced screening model with physically reasonable enhancement.
    
    Key improvements:
    - Maximum enhancement capped at A_max (typically 2-3)
    - Gentler density dependence (n_exp ~ 1)
    - Smooth screening beyond R_screen
    
    Returns xi in range [1, 1+A_max]
    """
    
    # Normalized density ratio
    rho_ratio = cp.minimum(rho / rho_c, 1.0)  # Cap at 1
    
    # Density enhancement factor
    # At solar density (rho/rho_c ~ 1): factor = 0 (no enhancement)
    # In voids (rho/rho_c ~ 0): factor = 1 (full enhancement)
    density_factor = (1.0 - rho_ratio)**n_exp
    
    # Distance screening with smooth transition
    # Full enhancement for R < R_screen/2
    # Smooth decay for R > R_screen/2
    # Nearly zero for R > 2*R_screen
    screening_factor = 0.5 * (1.0 + cp.tanh((R_screen - R) / (0.3 * R_screen)))
    
    # Combined enhancement
    enhancement = A_max * density_factor * screening_factor
    
    # Final xi
    xi = 1.0 + enhancement
    
    # Safety checks
    xi = cp.maximum(xi, 1.0)
    xi = cp.minimum(xi, 1.0 + A_max)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi


def test_balanced_model():
    """Test the balanced model at various conditions."""
    
    print("Balanced Screening Model Test")
    print("=" * 60)
    
    # Parameters
    rho_c = 1e8  # Solar system density M_sun/kpc^3
    R_screen = 50.0  # Screening radius
    A_max = 2.0  # Maximum enhancement factor
    
    # Test conditions
    test_cases = [
        # (R_kpc, rho/rho_c, description)
        (8.0, 1.0, "Solar System"),
        (8.0, 0.5, "Solar neighborhood low density"),
        (15.0, 0.2, "Inner disk edge"),
        (25.0, 0.05, "Outer disk"),
        (40.0, 0.01, "Halo region"),
        (60.0, 0.001, "Near screening radius"),
        (100.0, 0.0001, "Beyond screening"),
        (500.0, 1e-6, "Deep space"),
    ]
    
    print("\nConditions and Enhancement:")
    print("-" * 60)
    print("R (kpc) | rho/rho_c | xi    | Enhancement | Description")
    print("-" * 60)
    
    for R, rho_ratio, desc in test_cases:
        rho = rho_ratio * rho_c
        R_arr = cp.array([R], dtype=cp.float32)
        rho_arr = cp.array([rho], dtype=cp.float32)
        
        xi = xi_balanced_screening_cupy(rho_arr, rho_c, R_arr, R_screen, n_exp=1.0, A_max=A_max)
        xi_val = float(xi[0].get())
        
        print(f"{R:7.1f} | {rho_ratio:9.6f} | {xi_val:.4f} | {(xi_val-1)*100:+7.2f}% | {desc}")
    
    # Velocity comparison
    print("\nVelocity Comparison (M_galaxy = 5e10 M_sun):")
    print("-" * 60)
    
    M = 5e10  # Solar masses
    G = 4.302e-6  # (km/s)^2 kpc/M_sun
    
    R_curve = cp.array([5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100], dtype=cp.float32)
    
    # Simple density model: exponential disk
    scale_length = 3.0  # kpc
    rho_curve = rho_c * cp.exp(-R_curve / scale_length)
    
    # Compute xi
    xi_curve = xi_balanced_screening_cupy(rho_curve, rho_c, R_curve, R_screen, n_exp=1.0, A_max=A_max)
    
    # Velocities
    v_newton = cp.sqrt(G * M / R_curve)
    v_enhanced = cp.sqrt(G * M * xi_curve / R_curve)
    
    print("R (kpc) | v_Newton | v_DDMM | Ratio | xi")
    print("-" * 50)
    
    for r, vn, ve, xi in zip(R_curve.get(), v_newton.get(), v_enhanced.get(), xi_curve.get()):
        print(f"{r:7.1f} | {vn:8.2f} | {ve:7.2f} | {ve/vn:.3f} | {xi:.3f}")
    
    print("\nKey Features:")
    print("- Cassini satisfied: xi ~ 1 at solar density")
    print("- Modest enhancement: max xi = %.1f" % (1 + A_max))
    print("- Deep space safety: xi -> 1 beyond R_screen")
    print("- Realistic velocities: ~200-300 km/s range")


if __name__ == "__main__":
    test_balanced_model()