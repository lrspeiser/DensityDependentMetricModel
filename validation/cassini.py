#!/usr/bin/env python3
"""
Check if xi models pass the Cassini test at Saturn
"""
import numpy as np

# Import xi functions
from density_metric2 import (
    xi_power_law, xi_enhanced_bounded, xi_gravitational_color,
    xi_mass_threshold, xi_mond_like, xi_gaussian_enhancement
)

def cassini_test_check():
    """
    Check xi behavior at Solar System densities.
    Cassini measured gravity at Saturn to ~10^-5 precision.
    So we need |ξ - 1| < 10^-5 at Saturn.
    """
    
    print("="*80)
    print("CASSINI TEST COMPATIBILITY CHECK")
    print("="*80)
    
    # Physical scales
    print("\nRELEVANT DENSITY SCALES:")
    print("-"*40)
    
    # Saturn system parameters
    M_saturn_msun = 2.86e-4  # Saturn mass in solar masses
    R_saturn_km = 58232      # Saturn radius in km
    R_cassini_km = 200000    # Typical Cassini orbit radius
    R_cassini_kpc = R_cassini_km / 3.086e16  # Convert to kpc
    
    # Density estimates (M☉/kpc³)
    rho_saturn_surface = M_saturn_msun / (4/3 * np.pi * (R_saturn_km/3.086e16)**3)
    rho_saturn_orbit = M_saturn_msun / (4/3 * np.pi * R_cassini_kpc**3)
    
    # For comparison - other environments
    rho_sun_core = 1.5e5 * (2e33/2e30) * (3.086e18)**3  # 150 g/cm³ to M☉/kpc³
    rho_earth_surface = 5.5 * (2e33/2e30) * (3.086e18)**3  # 5.5 g/cm³
    rho_galaxy_disk = 1e8    # Typical galaxy disk
    rho_galaxy_halo = 1e6    # Typical galaxy halo
    
    print(f"Saturn surface density: {rho_saturn_surface:.2e} M☉/kpc³")
    print(f"Saturn orbit density:   {rho_saturn_orbit:.2e} M☉/kpc³")
    print(f"Sun core density:       {rho_sun_core:.2e} M☉/kpc³")
    print(f"Earth surface density:  {rho_earth_surface:.2e} M☉/kpc³")
    print(f"Galaxy disk density:    {rho_galaxy_disk:.2e} M☉/kpc³")
    print(f"Galaxy halo density:    {rho_galaxy_halo:.2e} M☉/kpc³")
    
    # Test densities spanning the range
    test_densities = np.logspace(6, 25, 20)  # 10^6 to 10^25 M☉/kpc³
    
    # Cassini precision requirement
    cassini_precision = 1e-5
    
    print(f"\nCASSINI REQUIREMENT: |ξ - 1| < {cassini_precision} at ρ > 10^20 M☉/kpc³")
    print("="*80)
    
    # 1. Test density-based models
    print("\n1. DENSITY-BASED XI MODELS")
    print("-"*60)
    
    # Parameters that might work for galaxies
    galaxy_params = {
        'rho_c': 1e8,     # Transition density
        'n_exp': 1.5,     # Power law index
        'A': 1.0,         # Enhancement factor
        'gamma': 2.7,     # For grav color
        'lambda_g': 1.5   # For grav color
    }
    
    models = [
        ('power_law', lambda rho: xi_power_law(rho, galaxy_params['rho_c'], galaxy_params['n_exp'])),
        ('enhanced_bounded', lambda rho: xi_enhanced_bounded(rho, galaxy_params['rho_c'], galaxy_params['n_exp'], galaxy_params['A'])),
        ('gravitational_color', lambda rho: xi_gravitational_color(rho, galaxy_params['rho_c'], galaxy_params['gamma'], galaxy_params['lambda_g'])),
        ('mond_like', lambda rho: xi_mond_like(rho, galaxy_params['rho_c'], galaxy_params['n_exp'])),
    ]
    
    print(f"Using galaxy parameters: ρ_c = {galaxy_params['rho_c']:.1e} M☉/kpc³")
    print("\nModel            | ξ(galaxy) | ξ(Saturn) | Cassini test")
    print("-"*60)
    
    for name, xi_func in models:
        xi_galaxy = xi_func(rho_galaxy_disk)[0]
        xi_saturn = xi_func(rho_saturn_orbit)[0]
        
        passes_cassini = abs(xi_saturn - 1.0) < cassini_precision
        status = "✓ PASS" if passes_cassini else "✗ FAIL"
        
        print(f"{name:<16} | {xi_galaxy:9.3f} | {xi_saturn:9.6f} | {status}")
    
    # 2. Test mass threshold model
    print("\n2. MASS THRESHOLD MODEL (SPECIAL CASE)")
    print("-"*60)
    
    print("The mass threshold model depends on ENCLOSED MASS, not density.")
    print(f"Saturn's mass: {M_saturn_msun:.2e} M☉")
    print("Galaxy enclosed masses: 10^9 - 10^11 M☉")
    
    # Test different M_crit values
    test_M_crit = [1e-10, 1e-5, 1e0, 1e5, 1e10]
    xi_boost = 2.0
    width = 0.3
    
    print(f"\nWith xi_boost = {xi_boost}, width = {width}:")
    print("M_crit (M☉) | ξ(Saturn) | ξ(galaxy R=10kpc) | Cassini test")
    print("-"*65)
    
    for M_crit in test_M_crit:
        # At Saturn
        xi_saturn_mt = xi_mass_threshold(
            rho=None,
            rho_c=M_crit,
            n_exp=width,
            A=xi_boost-1,
            r_kpc=R_cassini_kpc,
            params={'M_enclosed_msun': M_saturn_msun}
        )
        
        # At galaxy (assume M_enc ~ 5e10 at R=10 kpc)
        xi_galaxy_mt = xi_mass_threshold(
            rho=None,
            rho_c=M_crit,
            n_exp=width,
            A=xi_boost-1,
            r_kpc=10.0,
            params={'M_enclosed_msun': 5e10}
            )
        
        passes = abs(xi_saturn_mt - 1.0) < cassini_precision
        status = "✓ PASS" if passes else "✗ FAIL"
        
        print(f"{M_crit:11.1e} | {xi_saturn_mt:9.6f} | {xi_galaxy_mt:17.3f} | {status}")
    
    print("\n⚠️  CRITICAL ISSUE WITH MASS THRESHOLD MODEL:")
    print("   - To pass Cassini: need M_crit << 10^-4 M☉ (smaller than Saturn)")
    print("   - For galaxy fits: need M_crit ~ 10^10 M☉")
    print("   - These requirements are incompatible by 14 orders of magnitude!")
    
    # 3. Show what parameters would work
    print("\n3. PARAMETER REQUIREMENTS FOR CASSINI + GALAXY COMPATIBILITY")
    print("-"*70)
    
    # For density-based models
    print("\nFor density-based models (power, enhanced, grav_color):")
    print("Need ρ_c such that:")
    print(f"  - ρ_galaxy << ρ_c << ρ_saturn")
    print(f"  - {rho_galaxy_disk:.1e} << ρ_c << {rho_saturn_orbit:.1e}")
    print(f"  - Good choice: ρ_c ~ 10^12 - 10^15 M☉/kpc³")
    
    # Test better parameters
    better_rho_c = 1e13  # Between galaxy and Saturn
    
    print(f"\nTesting with ρ_c = {better_rho_c:.1e} M☉/kpc³:")
    print("Model            | ξ(galaxy) | ξ(Saturn) | Cassini test")
    print("-"*60)
    
    for name, _ in models[:3]:  # Test first 3 models
        if name == 'power_law':
            xi_galaxy = xi_power_law(rho_galaxy_disk, better_rho_c, 1.5)[0]
            xi_saturn = xi_power_law(rho_saturn_orbit, better_rho_c, 1.5)[0]
        elif name == 'enhanced_bounded':
            xi_galaxy = xi_enhanced_bounded(rho_galaxy_disk, better_rho_c, 1.5, 1.0)[0]
            xi_saturn = xi_enhanced_bounded(rho_saturn_orbit, better_rho_c, 1.5, 1.0)[0]
        elif name == 'gravitational_color':
            xi_galaxy = xi_gravitational_color(rho_galaxy_disk, better_rho_c, 2.7, 8.0)[0]
            xi_saturn = xi_gravitational_color(rho_saturn_orbit, better_rho_c, 2.7, 8.0)[0]
        
        passes = abs(xi_saturn - 1.0) < cassini_precision
        status = "✓ PASS" if passes else "✗ FAIL"
        
        print(f"{name:<16} | {xi_galaxy:9.3f} | {xi_saturn:9.6f} | {status}")
    
    print("\n" + "="*80)
    print("CONCLUSIONS:")
    print("="*80)
    print("\n1. The mass_threshold model CANNOT pass both Cassini and galaxy tests")
    print("   because Saturn's mass and galaxy masses differ by ~14 orders of magnitude.")
    print("\n2. Density-based models CAN work with ρ_c ~ 10^12 - 10^15 M☉/kpc³")
    print("\n3. RECOMMENDED: Use enhanced, power, or grav_color models with:")
    print("   - ρ_c ~ 10^13 M☉/kpc³")
    print("   - n ~ 1-2")
    print("   - Moderate enhancement factors")


if __name__ == "__main__":
    cassini_test_check()