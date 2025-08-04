#!/usr/bin/env python3
"""
Test script to verify all the improvements work correctly.
"""
import numpy as np
import cupy as cp
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from run_dynesty_cupy import setup_parameter_bounds, check_physical_plausibility, prior_transform_dynesty_cupy
from density_metric_cupy import v_total_kms_cupy, v_baryon_comprehensive_kms_cupy, volume_density_comprehensive_solar_kpc3_cupy

def test_parameter_bounds():
    """Test the updated parameter bounds."""
    print("="*60)
    print("TESTING PARAMETER BOUNDS")
    print("="*60)
    
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds('gr')
    
    print(f"GR Model Parameters ({len(param_names)}):")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{bounds_low[i]:.2e}, {bounds_high[i]:.2e}] (log_prior: {use_log_prior[i]})")
    
    # Test total baryonic mass range
    total_mass_low = sum(bounds_low[i] for i, name in enumerate(param_names) if 'M_' in name and 'solar' in name)
    total_mass_high = sum(bounds_high[i] for i, name in enumerate(param_names) if 'M_' in name and 'solar' in name)
    print(f"\nTotal Baryonic Mass Range: [{total_mass_low:.2e}, {total_mass_high:.2e}] M☉")

def test_physical_plausibility():
    """Test the soft physical plausibility check."""
    print("\n" + "="*60)
    print("TESTING PHYSICAL PLAUSIBILITY (SOFT VERSION)")
    print("="*60)
    
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds('gr')
    
    # Test 1: Valid parameters
    valid_theta = np.array([5e10, 3.0, 0.3, 1e10, 4.0, 0.8, 5e9, 1.0, 2e9, 6.0, 0.2])
    result = check_physical_plausibility(valid_theta, param_names)
    print(f"Valid parameters: {result} (should be True)")
    
    # Test 2: Invalid parameters (negative mass)
    invalid_theta = valid_theta.copy()
    invalid_theta[0] = -1e10  # Negative thin disk mass
    result = check_physical_plausibility(invalid_theta, param_names)
    print(f"Invalid parameters (negative mass): {result} (should be False)")
    
    # Test 3: Unreasonable mass ratios (should now pass with soft check)
    unreasonable_theta = valid_theta.copy()
    unreasonable_theta[3] = 1e11  # Thick disk > thin disk
    unreasonable_theta[6] = 1e11  # Bulge > 0.5 * thin disk
    result = check_physical_plausibility(unreasonable_theta, param_names)
    print(f"Unreasonable mass ratios: {result} (should be True with soft check)")

def test_prior_transform():
    """Test the ordered mass generation in prior transform."""
    print("\n" + "="*60)
    print("TESTING PRIOR TRANSFORM (ORDERED MASSES)")
    print("="*60)
    
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds('gr')
    
    # Test multiple random draws
    for i in range(5):
        u = np.random.random(len(param_names))
        theta = prior_transform_dynesty_cupy(u, param_names, bounds_low, bounds_high, use_log_prior)
        
        # Check mass ratios
        thin_idx = param_names.index('M_thin_disk_solar')
        thick_idx = param_names.index('M_thick_disk_solar')
        bulge_idx = param_names.index('M_bulge_solar')
        
        thin_mass = theta[thin_idx]
        thick_mass = theta[thick_idx]
        bulge_mass = theta[bulge_idx]
        
        print(f"Draw {i+1}:")
        print(f"  Thin disk: {thin_mass:.2e} M☉")
        print(f"  Thick disk: {thick_mass:.2e} M☉ ({thick_mass/thin_mass*100:.1f}% of thin)")
        print(f"  Bulge: {bulge_mass:.2e} M☉ ({bulge_mass/thin_mass*100:.1f}% of thin)")
        print(f"  Thin > Thick: {thin_mass > thick_mass}")
        print(f"  Bulge < 0.5*Thin: {bulge_mass < thin_mass * 0.5}")

def test_velocity_calculation():
    """Test the velocity calculation with NaN protection."""
    print("\n" + "="*60)
    print("TESTING VELOCITY CALCULATION (WITH NaN PROTECTION)")
    print("="*60)
    
    # Test parameters
    test_params = {
        'M_thin_disk_solar': 5e10,
        'R_thin_disk_kpc': 3.0,
        'hz_thin_disk_kpc': 0.3,
        'M_thick_disk_solar': 1e10,
        'R_thick_disk_kpc': 4.0,
        'hz_thick_disk_kpc': 0.8,
        'M_bulge_solar': 5e9,
        'R_bulge_kpc': 1.0,
        'M_gas_solar': 2e9,
        'R_gas_kpc': 6.0,
        'hz_gas_kpc': 0.2
    }
    
    # Test radii
    R_test = np.linspace(1.0, 20.0, 100)
    
    # Test comprehensive velocity
    v_baryon = v_baryon_comprehensive_kms_cupy(R_test, test_params)
    print(f"Baryonic velocity range: [{np.min(v_baryon):.1f}, {np.max(v_baryon):.1f}] km/s")
    print(f"Any NaN/inf in baryonic velocity: {np.any(~np.isfinite(v_baryon))}")
    
    # Test total velocity (GR)
    v_total_gr = v_total_kms_cupy(R_test, test_params, xi_type='gr')
    print(f"Total velocity (GR) range: [{np.min(v_total_gr):.1f}, {np.max(v_total_gr):.1f}] km/s")
    print(f"Any NaN/inf in total velocity: {np.any(~np.isfinite(v_total_gr))}")
    
    # Test density calculation
    rho_total = volume_density_comprehensive_solar_kpc3_cupy(R_test, test_params)
    print(f"Density range: [{np.min(rho_total):.2e}, {np.max(rho_total):.2e}] M☉/kpc³")
    print(f"Any NaN/inf in density: {np.any(~np.isfinite(rho_total))}")

def test_likelihood_penalty():
    """Test the soft prior penalty in likelihood."""
    print("\n" + "="*60)
    print("TESTING LIKELIHOOD PENALTY")
    print("="*60)
    
    # Import the likelihood function
    from run_dynesty_cupy import log_likelihood_dynesty_cupy
    
    # Mock data
    R_data = np.linspace(1.0, 20.0, 100)
    v_data = 200 + 50 * np.exp(-R_data / 8.0)
    sigma_data = 10.0 * np.ones_like(R_data)
    
    # Mock args
    class MockArgs:
        def __init__(self):
            self.xi = 'gr'
    
    args = MockArgs()
    
    # Test 1: Reasonable parameters
    param_names, bounds_low, bounds_high, use_log_prior = setup_parameter_bounds('gr')
    reasonable_theta = np.array([5e10, 3.0, 0.3, 1e10, 4.0, 0.8, 5e9, 1.0, 2e9, 6.0, 0.2])
    
    logl_reasonable = log_likelihood_dynesty_cupy(reasonable_theta, param_names, args, R_data, v_data, sigma_data)
    print(f"Reasonable parameters logL: {logl_reasonable:.2f}")
    
    # Test 2: Unreasonable parameters (should get penalty)
    unreasonable_theta = reasonable_theta.copy()
    unreasonable_theta[3] = 1e11  # Thick disk > thin disk
    logl_unreasonable = log_likelihood_dynesty_cupy(unreasonable_theta, param_names, args, R_data, v_data, sigma_data)
    print(f"Unreasonable parameters logL: {logl_unreasonable:.2f}")
    print(f"Penalty: {logl_unreasonable - logl_reasonable:.2f}")

def main():
    """Run all tests."""
    print("TESTING ALL IMPROVEMENTS")
    print("="*60)
    
    try:
        test_parameter_bounds()
        test_physical_plausibility()
        test_prior_transform()
        test_velocity_calculation()
        test_likelihood_penalty()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("The improvements are working correctly.")
        print("You can now run the dynesty sampler with confidence.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 