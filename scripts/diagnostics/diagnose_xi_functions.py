#!/usr/bin/env python3
"""
Diagnostic script to test xi functions and understand why some models are failing.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.density_metric_cupy import (
    xi_elastic_strain_cupy,
    xi_hookean_potential_cupy,
    xi_tension_field_cupy,
    xi_power_law_cupy,
    rho_baryon_total_midplane_solar_kpc3
)

def test_xi_function(xi_func, params, name, R_values=None):
    """Test a xi function across a range of densities."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Test density range from intergalactic to solar system
    rho_test = cp.logspace(-5, 12, 100)  # From 1e-5 to 1e12 M_sun/kpc^3
    
    if R_values is not None:
        R_test = cp.asarray(R_values)
    else:
        R_test = cp.ones_like(rho_test) * 8.0  # Default to solar radius
    
    try:
        if 'tension' in name.lower() or 'hookean' in name.lower():
            # These functions need R as second argument
            xi_values = xi_func(rho_test, R_test, params)
        elif 'elastic' in name.lower():
            # Elastic strain needs rho_c as second argument
            rho_c = 1e8  # Default critical density
            xi_values = xi_func(rho_test, rho_c, params)
        else:
            # Power law and enhanced functions
            if 'power' in name.lower():
                rho_c = params.get('rho_c', 1e8)
                n = params.get('n', 1.0)
                xi_values = xi_func(rho_test, rho_c, n)
            elif 'enhanced' in name.lower():
                rho_c = params.get('rho_c', 1e8)
                n = params.get('n', 1.0)
                A = params.get('A', 1.0)
                xi_values = xi_func(rho_test, rho_c, n, A)
        
        # Convert to numpy for analysis
        rho_np = cp.asnumpy(rho_test)
        xi_np = cp.asnumpy(xi_values - 1.0)  # Show enhancement (xi - 1)
        
        # Find key values
        solar_density = 1e9  # Approximate solar system density
        galaxy_density = 1e7  # Typical galaxy disk density
        void_density = 1e2   # Intergalactic void density
        
        idx_solar = np.argmin(np.abs(rho_np - solar_density))
        idx_galaxy = np.argmin(np.abs(rho_np - galaxy_density))
        idx_void = np.argmin(np.abs(rho_np - void_density))
        
        print(f"\nEnhancement (xi - 1) at key densities:")
        print(f"  Solar System (ρ = 1e9 M☉/kpc³): {xi_np[idx_solar]:.2e}")
        print(f"  Galaxy Disk  (ρ = 1e7 M☉/kpc³): {xi_np[idx_galaxy]:.2e}")
        print(f"  Void        (ρ = 1e2 M☉/kpc³): {xi_np[idx_void]:.2e}")
        
        # Check for problematic values
        print(f"\nValue ranges:")
        print(f"  Min xi - 1: {np.min(xi_np):.2e}")
        print(f"  Max xi - 1: {np.max(xi_np):.2e}")
        print(f"  Mean xi - 1: {np.mean(xi_np):.2e}")
        
        # Check for NaN or Inf
        n_nan = np.sum(np.isnan(xi_np))
        n_inf = np.sum(np.isinf(xi_np))
        n_negative = np.sum(xi_np < -1)  # xi should be >= 0, so xi-1 >= -1
        
        if n_nan > 0:
            print(f"  WARNING: {n_nan} NaN values!")
        if n_inf > 0:
            print(f"  WARNING: {n_inf} Inf values!")
        if n_negative > 0:
            print(f"  WARNING: {n_negative} unphysical negative values (xi < 0)!")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.loglog(rho_np, np.abs(xi_np) + 1e-20, label=f'{name} |xi - 1|')
        plt.axhline(y=1e-5, color='r', linestyle='--', alpha=0.5, label='Cassini bound')
        plt.axvline(x=solar_density, color='orange', linestyle=':', alpha=0.5, label='Solar density')
        plt.axvline(x=galaxy_density, color='blue', linestyle=':', alpha=0.5, label='Galaxy density')
        plt.axvline(x=void_density, color='green', linestyle=':', alpha=0.5, label='Void density')
        plt.xlabel('Density (M☉/kpc³)')
        plt.ylabel('|Enhancement| = |xi - 1|')
        plt.title(f'{name} Xi Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([1e-20, 1e2])
        
        # Save plot
        plot_dir = Path('diagnostic_plots')
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(plot_dir / f'{name.lower().replace(" ", "_")}_xi.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baryon_density():
    """Test the baryon density calculation."""
    print(f"\n{'='*60}")
    print(f"Testing Baryon Density Calculation")
    print(f"{'='*60}")
    
    # Typical galaxy parameters
    R_test = np.linspace(0.1, 30, 100)  # 0.1 to 30 kpc
    
    # Reasonable galaxy parameters
    params = {
        'M_thin_disk_solar': 5e10,
        'R_thin_disk_kpc': 3.0,
        'hz_thin_disk_kpc': 0.3,
        'M_thick_disk_solar': 5e9,
        'R_thick_disk_kpc': 4.0,
        'hz_thick_disk_kpc': 0.8,
        'M_bulge_solar': 2e10,
        'R_bulge_kpc': 1.0,
        'M_gas_solar': 5e9,
        'R_gas_kpc': 7.0,
        'hz_gas_kpc': 0.2
    }
    
    # Calculate density
    R_cp = cp.asarray(R_test)
    rho_values = rho_baryon_total_midplane_solar_kpc3(R_cp, params)
    
    # Convert to numpy
    rho_np = cp.asnumpy(rho_values)
    
    print(f"\nDensity at key radii:")
    print(f"  R = 0.1 kpc: {rho_np[0]:.2e} M☉/kpc³")
    print(f"  R = 1.0 kpc: {rho_np[9]:.2e} M☉/kpc³")
    print(f"  R = 8.0 kpc (Solar): {rho_np[np.argmin(np.abs(R_test - 8.0))]:.2e} M☉/kpc³")
    print(f"  R = 20 kpc: {rho_np[np.argmin(np.abs(R_test - 20))]:.2e} M☉/kpc³")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(R_test, rho_np, 'b-', linewidth=2)
    plt.axvline(x=8.122, color='orange', linestyle='--', label='Solar radius')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (M☉/kpc³)')
    plt.title('Baryon Density Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_dir = Path('diagnostic_plots')
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'baryon_density.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("XI FUNCTION DIAGNOSTIC TESTS")
    print("="*60)
    
    # First test baryon density calculation
    test_baryon_density()
    
    # Test successful models for comparison
    print("\n" + "="*60)
    print("TESTING KNOWN WORKING MODELS")
    print("="*60)
    
    # Power law (known to work)
    power_params = {'rho_c': 1e15, 'n': 1.2}
    test_xi_function(xi_power_law_cupy, power_params, "Power Law")
    
    # Enhanced power law - manually compute since function doesn't exist
    # Skip for now
    
    print("\n" + "="*60)
    print("TESTING PROBLEMATIC MODELS")
    print("="*60)
    
    # Elastic strain
    elastic_params = {
        'relaxation_scale': 1.0,
        'strain_critical': 10.0,
        'k_elastic': 0.5,
        'rho_solar': 1e9
    }
    test_xi_function(xi_elastic_strain_cupy, elastic_params, "Elastic Strain")
    
    # Try with different parameters
    elastic_params2 = {
        'relaxation_scale': 2.0,
        'strain_critical': 15.0,
        'k_elastic': 1.0,
        'rho_solar': 1e9
    }
    test_xi_function(xi_elastic_strain_cupy, elastic_params2, "Elastic Strain (alt params)")
    
    # Hookean
    R_values = cp.linspace(0.1, 30, 100)
    hookean_params = {
        'k_spacetime': 0.1,
        'rho_equilibrium': 1e9,
        'stress_break': 100.0
    }
    test_xi_function(xi_hookean_potential_cupy, hookean_params, "Hookean", R_values)
    
    # Try with different parameters
    hookean_params2 = {
        'k_spacetime': 0.5,
        'rho_equilibrium': 1e8,
        'stress_break': 50.0
    }
    test_xi_function(xi_hookean_potential_cupy, hookean_params2, "Hookean (alt params)", R_values)
    
    # Tension field
    tension_params = {
        'rho_relaxation': 1e8,
        'tension_max': 5.0,
        'R_snap': 25.0
    }
    test_xi_function(xi_tension_field_cupy, tension_params, "Tension Field", R_values)
    
    # Try with different parameters
    tension_params2 = {
        'rho_relaxation': 1e9,
        'tension_max': 2.0,
        'R_snap': 15.0
    }
    test_xi_function(xi_tension_field_cupy, tension_params2, "Tension Field (alt params)", R_values)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nPlots saved to diagnostic_plots/")
    print("\nKey findings:")
    print("1. Check if xi functions produce NaN/Inf values")
    print("2. Verify Solar System constraint (|xi - 1| < 1e-5 at solar density)")
    print("3. Ensure reasonable enhancement in galaxy/void regions")

if __name__ == "__main__":
    main()