#!/usr/bin/env python3
"""
test_sparc_validation.py - Test DDMM validation with SPARC data from Rotmod_LTG
"""

import numpy as np
import sys
from pathlib import Path

# Add the parent directory to path if needed
sys.path.append('.')

from validate_ddmm import DDMMValidator
from sparc_data_loader import SPARCDataLoader

def run_sparc_validation():
    """Run validation using SPARC data in Rotmod_LTG folder"""
    
    # Your SPARC data location
    sparc_data_path = "Rotmod_LTG"
    
    # Load your best-fit parameters from the Milky Way fit
    params_file = "chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz"
    
    print(f"Loading parameters from: {params_file}")
    data = np.load(params_file)
    samples = data['samples']
    weights = data['weights']
    
    # Extract parameters (adjust indices based on your actual fit)
    param_names = ['rho_c_solar_kpc3', 'n_exp']
    median_params = np.average(samples[:, :2], weights=weights, axis=0)  # First 2 params
    
    model_params = {
        'rho_c_solar_kpc3': median_params[0],
        'n_exp': median_params[1],
        'xi_type': 'power'
    }
    
    print(f"\nUsing DDMM parameters:")
    print(f"  ρ_c = {model_params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
    print(f"  n = {model_params['n_exp']:.2f}")
    
    # Test data loading first
    print(f"\nTesting SPARC data loader on {sparc_data_path}...")
    loader = SPARCDataLoader(sparc_data_path)
    galaxies = loader.load_all_galaxies()
    
    if not galaxies:
        print("ERROR: No galaxies loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(galaxies)} galaxies")
    print("Sample galaxies:", list(galaxies.keys())[:5])
    
    # Initialize validator
    validator = DDMMValidator(model_params, output_dir='sparc_validation_results')
    
    # Run SPARC test with progressively more galaxies
    for n_gal in [5, 20, 50, 100]:
        if n_gal <= len(galaxies):
            print(f"\n{'='*60}")
            print(f"Testing with {n_gal} galaxies...")
            result = validator.test_sparc_galaxies(sparc_data_path, n_galaxies=n_gal)
            
            print(f"\nResults for {n_gal} galaxies:")
            print(f"  Passed: {result.passed}")
            print(f"  Score: {result.score:.2f}")
            print(f"  Mean RMS: {result.details['statistics']['mean_rms']:.1f} km/s")
            
    # Generate report
    validator.generate_report('sparc_validation_report.json')
    print(f"\nFull report saved to: sparc_validation_report.json")

if __name__ == "__main__":
    run_sparc_validation()