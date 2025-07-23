#!/usr/bin/env python3 create_json_file_ddmm_results.py
"""Extract DDMM parameters from dynesty results and create JSON file"""

import numpy as np
import json
import pickle
import gzip
import sys

def extract_ddmm_parameters(results_file, output_json='ddmm_params.json'):
    """Extract best-fit DDMM parameters from dynesty results"""
    
    # Load results
    if results_file.endswith('.pkl.gz'):
        with gzip.open(results_file, 'rb') as f:
            results = pickle.load(f)
    elif results_file.endswith('.pkl'):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    elif results_file.endswith('.npz'):
        data = np.load(results_file, allow_pickle=True)
        # Convert to results-like object
        class Results:
            pass
        results = Results()
        results.samples = data['samples']
        results.logwt = data.get('logwt', data.get('log_wt'))
        results.logz = data.get('logz', data.get('log_z'))
    else:
        raise ValueError("Unknown file format")
    
    # Calculate weights
    weights = np.exp(results.logwt - results.logz[-1])
    
    # Get parameter names - these should match your MCMC setup
    # Option 1: If you fitted gravity parameters
    gravity_param_names = ['rho_c_solar_kpc3', 'n_exp', 'A']
    
    # Option 2: If you fitted baryon parameters
    baryon_param_names = [
        'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
        'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
        'M_bulge_solar', 'a_bulge_kpc',
        'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
    ]
    
    # Check which parameters were fitted
    n_params = results.samples.shape[1]
    if n_params == len(gravity_param_names):
        param_names = gravity_param_names
        print("Detected gravity parameter fit")
    elif n_params == len(baryon_param_names):
        param_names = baryon_param_names
        print("Detected baryon parameter fit")
    else:
        print(f"Warning: {n_params} parameters found, expected {len(gravity_param_names)} or {len(baryon_param_names)}")
        # Try to get from results object
        param_names = getattr(results, 'param_names', None)
        if param_names is None:
            raise ValueError("Cannot determine parameter names")
    
    # Calculate best-fit values (weighted median)
    best_params = {}
    for i, name in enumerate(param_names):
        # Weighted percentile (50th = median)
        sorted_idx = np.argsort(results.samples[:, i])
        sorted_samples = results.samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        
        # Find median
        idx = np.searchsorted(cumsum, 0.5)
        best_params[name] = float(sorted_samples[idx])
        
        # Also get uncertainties (16th and 84th percentiles)
        idx_low = np.searchsorted(cumsum, 0.16)
        idx_high = np.searchsorted(cumsum, 0.84)
        err_low = sorted_samples[idx] - sorted_samples[idx_low]
        err_high = sorted_samples[idx_high] - sorted_samples[idx]
        
        print(f"{name} = {best_params[name]:.3e} +{err_high:.3e} -{err_low:.3e}")
    
    # Create full parameter dictionary
    ddmm_params = {}
    
    # If gravity parameters were fitted, use them
    if 'rho_c_solar_kpc3' in best_params:
        ddmm_params.update(best_params)
    else:
        # Use standard DDMM gravity parameters from literature
        print("\nUsing standard DDMM gravity parameters:")
        ddmm_params['rho_c_solar_kpc3'] = 1.03e9  # From your papers
        ddmm_params['n_exp'] = 1.0
        ddmm_params['A'] = 0.89
        print(f"rho_c = {ddmm_params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
        print(f"n = {ddmm_params['n_exp']}")
        print(f"A = {ddmm_params['A']}")
    
    # If baryon parameters were fitted, use them
    if 'M_disk_thin_solar' in best_params:
        ddmm_params.update(best_params)
    else:
        # Use standard Milky Way parameters
        print("\nUsing standard Milky Way baryon parameters")
        ddmm_params.update({
            'M_disk_thin_solar': 4.3e10,
            'R_d_thin_kpc': 2.6,
            'h_z_thin_kpc': 0.3,
            'M_disk_thick_solar': 1.0e10,
            'R_d_thick_kpc': 3.6,
            'h_z_thick_kpc': 0.9,
            'M_bulge_solar': 1.9e10,
            'a_bulge_kpc': 0.9,
            'M_gas_solar': 1.0e10,
            'R_d_gas_kpc': 7.0,
            'h_z_gas_kpc': 0.08
        })
    
    # Add other required parameters
    ddmm_params['xi_type'] = 'power'
    ddmm_params['include_disk_thin'] = True
    ddmm_params['include_disk_thick'] = True
    ddmm_params['include_bulge'] = True
    ddmm_params['include_gas'] = True
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(ddmm_params, f, indent=2)
    
    print(f"\nParameters saved to {output_json}")
    return ddmm_params

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ddmm_params.py <dynesty_results.pkl.gz> [output.json]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'ddmm_params.json'
    
    extract_ddmm_parameters(results_file, output_file)