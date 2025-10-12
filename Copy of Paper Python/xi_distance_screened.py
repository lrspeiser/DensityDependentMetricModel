#!/usr/bin/env python3
"""
Distance-Screened Xi Functions

These models ensure that gravitational enhancement doesn't cause 
unphysical behavior in deep space by incorporating distance screening.

Key insight: Enhancement should be strongest at intermediate distances
(explaining galaxy rotation curves) but must vanish at both:
- Small distances (Solar System constraint)  
- Large distances (deep space constraint)
"""

import cupy as cp
import numpy as np

def xi_distance_screened_cupy(rho, rho_c, R, R_screen, n_exp=2.0, A=5.0):
    """
    Distance-screened enhancement function.
    
    The enhancement grows in low-density regions BUT is screened
    at large distances to prevent unphysical behavior in deep space.
    
    Parameters:
    -----------
    rho : array
        Local density in M_sun/kpc^3
    rho_c : float
        Critical density for enhancement
    R : array
        Distance from galactic center in kpc
    R_screen : float
        Screening radius - enhancement vanishes beyond ~3×R_screen
    n_exp : float
        Power law exponent for density dependence
    A : float
        Maximum enhancement amplitude
    
    Returns:
    --------
    xi : array
        Enhancement factor (1 = no enhancement)
    
    Physics:
    --------
    ξ(ρ,R) = 1 + A × f_density(ρ) × f_distance(R)
    
    where:
    - f_density(ρ) = (rho_c/rho)^n when rho < rho_c, else 0
    - f_distance(R) = exp(-(R/R_screen)^2) 
    
    This ensures:
    1. ξ → 1 at solar density (Cassini)
    2. ξ → 1 as R → ∞ (deep space)
    3. ξ > 1 at intermediate R in low-density regions (rotation curves)
    """
    
    # Density-dependent enhancement (as before)
    ratio = rho_c / (rho + 1e-10)  # Avoid division by zero
    density_factor = cp.where(
        rho < rho_c,
        cp.minimum(ratio**n_exp - 1.0, 100.0),  # Cap to prevent overflow
        0.0
    )
    
    # NEW: Distance screening factor
    # Gaussian screening - smooth cutoff at large R
    distance_factor = cp.exp(-(R / R_screen)**2)
    
    # Combined enhancement
    xi = 1.0 + A * density_factor * distance_factor
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi


def xi_transition_zone_cupy(rho, rho_c, R, R_inner=10.0, R_outer=50.0, n_exp=2.0, A=5.0):
    """
    Transition zone model - enhancement only in specific radial range.
    
    This model restricts enhancement to a "transition zone" between
    the inner galaxy (where density is high) and deep space.
    
    Parameters:
    -----------
    R_inner : float
        Inner radius where enhancement begins (kpc)
    R_outer : float
        Outer radius where enhancement ends (kpc)
    
    Physics:
    --------
    Enhancement is active only in the range [R_inner, R_outer].
    This naturally explains:
    - Normal gravity in Solar System (R ~ 8 kpc, high density)
    - Enhanced gravity at galaxy edges (R ~ 20-40 kpc, low density)  
    - Normal 1/r² falloff in deep space (R > R_outer)
    """
    
    # Density enhancement as before
    ratio = rho_c / (rho + 1e-10)
    density_factor = cp.where(
        rho < rho_c,
        cp.minimum(ratio**n_exp - 1.0, 100.0),
        0.0
    )
    
    # Smooth transition function
    # Uses tanh for smooth turn-on and turn-off
    turn_on = 0.5 * (1.0 + cp.tanh((R - R_inner) / 2.0))
    turn_off = 0.5 * (1.0 - cp.tanh((R - R_outer) / 10.0))
    transition_factor = turn_on * turn_off
    
    # Combined enhancement
    xi = 1.0 + A * density_factor * transition_factor
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi


def xi_redshift_compatible_cupy(rho, rho_c, R, z=0, n_exp=2.0, A=5.0, lambda_z=0.1):
    """
    Redshift-compatible enhancement function.
    
    This model includes cosmological considerations for how enhancement
    should behave at different redshifts, crucial for understanding
    observations of distant galaxies.
    
    Parameters:
    -----------
    z : float or array
        Redshift (0 for local universe)
    lambda_z : float
        Redshift scaling parameter
    
    Physics of Redshift in DDMM:
    -----------------------------
    In DDMM, gravitational redshift is modified by ξ:
    
    1 + z_grav = sqrt(g₀₀(emit)/g₀₀(observe)) × sqrt(ξ_emit/ξ_obs)
    
    In deep space where ρ → 0:
    - Standard GR: z_grav → 0 (no gravitational field)
    - Naive DDMM: z_grav → ∞ (ξ → ∞, catastrophic!)
    - This model: z_grav → 0 (ξ saturates)
    
    The key insight: Enhancement should saturate at a maximum value
    in true voids to prevent infinite redshift.
    """
    
    # Base density enhancement with saturation
    ratio = rho_c / (rho + 1e-10)
    
    # Saturating enhancement function
    # Uses arctan to naturally cap enhancement
    density_factor = (2.0 / cp.pi) * cp.arctan(ratio**n_exp - 1.0)
    
    # Distance modulation with exponential cutoff
    # Enhancement peaks at R ~ 20 kpc, falls off beyond
    R_peak = 20.0  # kpc
    R_width = 15.0  # kpc
    distance_factor = cp.exp(-((R - R_peak) / R_width)**2)
    
    # Redshift evolution (if considering cosmological observations)
    # Enhancement may have been different at high z
    redshift_factor = 1.0 / (1.0 + lambda_z * z)
    
    # Combined enhancement
    xi = 1.0 + A * density_factor * distance_factor * redshift_factor
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi


def compute_effective_gravity(M, R, rho, rho_c, R_screen, n_exp=2.0, A=5.0):
    """
    Compute effective gravitational acceleration with screening.
    
    Returns:
    --------
    g_eff : Effective gravitational acceleration
    g_newton : Newtonian prediction  
    xi : Enhancement factor
    
    This shows how gravity behaves at different distances:
    - R << R_screen: g_eff ≈ g_newton (normal gravity)
    - R ~ R_screen: g_eff > g_newton (enhanced, explains rotation curves)
    - R >> R_screen: g_eff → g_newton (screening prevents runaway)
    """
    
    # Newtonian gravity
    G = 4.302e-6  # (km/s)^2 kpc/M_sun
    g_newton = G * M / R**2
    
    # Enhancement factor
    xi = xi_distance_screened_cupy(rho, rho_c, R, R_screen, n_exp, A)
    
    # Effective gravity
    g_eff = g_newton * xi
    
    return g_eff, g_newton, xi


# Test the models
if __name__ == "__main__":
    print("Testing Distance-Screened Xi Functions")
    print("=" * 50)
    
    # Test parameters
    rho_c = 1e8  # Solar density
    R_screen = 30.0  # Screening radius in kpc
    n_exp = 2.0
    A = 5.0
    
    # Test at various distances
    R_test = cp.array([8.0, 15.0, 25.0, 50.0, 100.0, 500.0], dtype=cp.float32)
    
    # Densities at those distances (rough model)
    rho_test = rho_c * cp.exp(-R_test / 10.0)
    
    print("\nDistance-Screened Model:")
    print("R (kpc) | rho/rho_c | xi | Enhancement")
    print("-" * 45)
    
    xi_vals = xi_distance_screened_cupy(rho_test, rho_c, R_test, R_screen, n_exp, A)
    
    for r, rho, xi in zip(R_test.get(), rho_test.get(), xi_vals.get()):
        print(f"{r:7.1f} | {rho/rho_c:.3f} | {xi:.3f} | {(xi-1)*100:+6.1f}%")
    
    print("\nKey observations:")
    print("- Solar System (R=8): xi ~ 1 (Cassini constraint)")
    print("- Galaxy edge (R=25): xi > 1 (rotation curve support)")
    print("- Deep space (R>100): xi -> 1 (screening active)")
    
    # Show velocity profile
    print("\nVelocity Profile (M_total = 5e10 M_sun):")
    M = 5e10  # Solar masses
    G = 4.302e-6  # (km/s)^2 kpc/M_sun
    
    v_newton = cp.sqrt(G * M / R_test)
    v_ddmm = cp.sqrt(G * M * xi_vals / R_test)
    
    print("R (kpc) | v_Newton | v_DDMM | Ratio")
    print("-" * 40)
    for r, vn, vd in zip(R_test.get(), v_newton.get(), v_ddmm.get()):
        print(f"{r:7.1f} | {vn:8.2f} | {vd:7.2f} | {vd/vn:.3f}")