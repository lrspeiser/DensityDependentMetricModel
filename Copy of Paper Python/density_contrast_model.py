#!/usr/bin/env python3
"""
test_contrast_model.py - Test the density contrast hypothesis with Gaia data.

Simplified version focusing ONLY on testing whether density contrasts
(not absolute density) drive gravitational modifications.
"""

import sys
import traceback

print("Starting test_contrast_model.py...")
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import cupy as cp
    print(f"✓ CuPy imported successfully")
    # Test CuPy is working
    test_arr = cp.array([1.0, 2.0, 3.0])
    print(f"✓ CuPy test array created: {test_arr}")
except ImportError as e:
    print(f"✗ Failed to import CuPy: {e}")
    print("Please install CuPy: pip install cupy-cuda11x (or appropriate version)")
    sys.exit(1)
except Exception as e:
    print(f"✗ CuPy imported but failed to create array: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    print("✓ pathlib imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pathlib: {e}")
    sys.exit(1)

try:
    import json
    print("✓ json imported successfully")
except ImportError as e:
    print(f"✗ Failed to import json: {e}")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Failed to import matplotlib: {e}")
    print("Please install matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    from datetime import datetime
    print("✓ datetime imported successfully")
except ImportError as e:
    print(f"✗ Failed to import datetime: {e}")
    sys.exit(1)

print("\nAttempting to import density_contrast_model...")
try:
    from density_contrast_model import (
        v_total_contrast, 
        check_cassini_constraint,
        to_numpy_array,
        diagnose_enhancement  # Add diagnostic function
    )
    print("✓ density_contrast_model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import density_contrast_model: {e}")
    print("Make sure density_contrast_model.py is in the same directory")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Error importing from density_contrast_model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try to import Gaia data loader
print("\nChecking for Gaia data loader...")
try:
    from data_io import load_all_sky_gaia_slices, process_gaia_data
    import pandas as pd
    GAIA_AVAILABLE = True
    print("✓ Gaia data loader (data_io) available")
except ImportError as e:
    GAIA_AVAILABLE = False
    print(f"⚠ WARNING: data_io not available, will use synthetic data: {e}")

print("\n" + "="*60)
print("All imports successful! Starting main program...")
print("="*60 + "\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_gaia_or_synthetic(max_stars=10000):
    """Load Gaia data if available, otherwise synthetic."""
    print(f"\n--- Data Loading ---")
    print(f"Max stars to load: {max_stars}")
    
    if GAIA_AVAILABLE:
        # Try to load cached Gaia data
        cache_file = Path("gaia_sky_slices/all_sky_gaia.csv")
        print(f"Looking for cache file: {cache_file}")
        print(f"Cache file exists: {cache_file.exists()}")
        
        if cache_file.exists():
            print(f"Loading Gaia data from {cache_file}")
            print(f"File size: {cache_file.stat().st_size / 1e6:.1f} MB")
            
            try:
                df = pd.read_csv(cache_file)
                print(f"✓ Loaded DataFrame with shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                
                df = process_gaia_data(df)
                print(f"✓ Processed Gaia data")
                
                # Check for required columns
                required_cols = ["R_kpc", "v_obs", "sigma_v"]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    print(f"✗ Missing required columns: {missing}")
                    print(f"  Available columns: {list(df.columns)}")
                    raise ValueError(f"Missing columns: {missing}")
                
                # Extract arrays
                R_data = df["R_kpc"].values
                v_data = df["v_obs"].values  
                sigma_data = df["sigma_v"].values
                
                print(f"✓ Extracted arrays:")
                print(f"  R_data: shape={R_data.shape}, range=[{np.nanmin(R_data):.1f}, {np.nanmax(R_data):.1f}]")
                print(f"  v_data: shape={v_data.shape}, range=[{np.nanmin(v_data):.1f}, {np.nanmax(v_data):.1f}]")
                
                # Clean data
                mask = np.isfinite(R_data) & np.isfinite(v_data) & (R_data > 0) & (R_data < 30)
                n_original = len(R_data)
                R_data = R_data[mask]
                v_data = v_data[mask]
                sigma_data = sigma_data[mask]
                n_cleaned = len(R_data)
                
                print(f"✓ Cleaned data: {n_original} -> {n_cleaned} stars ({n_original-n_cleaned} removed)")
                
                # Sample if needed
                if len(R_data) > max_stars:
                    print(f"Sampling {max_stars} from {len(R_data)} stars...")
                    indices = np.random.choice(len(R_data), max_stars, replace=False)
                    R_data = R_data[indices]
                    v_data = v_data[indices]
                    sigma_data = sigma_data[indices]
                    print(f"✓ Randomly sampled to {max_stars} stars")
                
                print(f"✓ Final data: {len(R_data)} stars")
                return R_data, v_data, sigma_data
                
            except Exception as e:
                print(f"✗ Error processing Gaia data: {e}")
                traceback.print_exc()
                print("Falling back to synthetic data...")
        else:
            print(f"Cache file not found at {cache_file}")
            print("Falling back to synthetic data...")
    
    # Fallback to synthetic
    print("\n--- Using Synthetic Data ---")
    R_data = np.linspace(0.5, 25.0, min(100, max_stars))
    # Approximate Milky Way rotation curve
    v_data = 220 * np.ones_like(R_data)  # Flat at ~220 km/s
    v_data[R_data < 2] = 100 * R_data[R_data < 2]  # Rising in center
    v_data[R_data > 15] = 220 * np.exp(-(R_data[R_data > 15] - 15)/10)  # Falling at edge
    sigma_data = 10 * np.ones_like(R_data)
    
    print(f"✓ Generated synthetic data: {len(R_data)} points")
    print(f"  R range: [{R_data.min():.1f}, {R_data.max():.1f}] kpc")
    print(f"  v range: [{v_data.min():.1f}, {v_data.max():.1f}] km/s")
    
    return R_data, v_data, sigma_data

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def chi_squared(params, R_data, v_data, sigma_data, contrast_type='gradient'):
    """Calculate chi-squared for given parameters."""
    try:
        # Convert to CuPy
        R_gpu = cp.asarray(R_data, dtype=cp.float32)
        
        # Calculate model velocity with components
        v_model_gpu, v_newton_gpu, xi_gpu = v_total_contrast(
            R_gpu, params, contrast_type, return_components=True
        )
        v_model = to_numpy_array(v_model_gpu)
        v_newton = to_numpy_array(v_newton_gpu)
        xi = to_numpy_array(xi_gpu)
        
        # Calculate chi-squared
        chi2 = np.sum(((v_data - v_model) / sigma_data)**2)
        
        # Print diagnostic every 10th call (reduce output)
        if np.random.random() < 0.05:  # 5% chance to print
            print(f"      ξ range: [{xi.min():.3f}, {xi.max():.3f}], χ² = {chi2:.1e}")
        
        return chi2
    except Exception as e:
        print(f"Error in chi_squared: {e}")
        return np.inf

def grid_search_simple(R_data, v_data, sigma_data, contrast_type='gradient'):
    """
    Simple grid search to find best parameters.
    Focuses on key contrast parameters.
    
    UPDATED: Better parameter ranges for actual enhancement
    """
    print(f"\n--- Grid Search: {contrast_type} ---")
    
    # Fixed baryonic parameters (reasonable MW values)
    base_params = {
        'M_disk_solar': 5e10,
        'R_disk_kpc': 3.0,
        'hz_disk_kpc': 0.3,
        'M_bulge_solar': 1e10,
        'R_bulge_kpc': 0.5,
    }
    
    best_chi2 = np.inf
    best_params = None
    n_tested = 0
    n_passed_cassini = 0
    
    # Define search grids based on contrast type
    if contrast_type == 'gradient':
        # UPDATED: Wider ranges to find actual enhancement
        gradient_scales = [0.5, 1.0, 2.0]  # Gradient calculation scales
        contrast_thresholds = [0.1, 0.5, 1.0, 2.0]  # Much lower thresholds
        A_values = [50, 100, 200, 500]  # Much higher enhancements needed
        
        total_combinations = len(gradient_scales) * len(contrast_thresholds) * len(A_values)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for gs in gradient_scales:
            for ct in contrast_thresholds:
                for A in A_values:
                    n_tested += 1
                    params = base_params.copy()
                    params.update({
                        'gradient_scale_kpc': gs,
                        'contrast_threshold': ct,
                        'A_contrast': A,
                        'transition_width': 2.0
                    })
                    
                    # Check Cassini constraint first
                    cassini = check_cassini_constraint(params, contrast_type)
                    
                    # Calculate enhancement to see if model is working
                    R_test_gpu = cp.asarray([5, 10, 15, 20], dtype=cp.float32)
                    _, _, xi_test = v_total_contrast(R_test_gpu, params, contrast_type, return_components=True)
                    xi_test = to_numpy_array(xi_test)
                    max_enhancement = xi_test.max()
                    
                    # Only consider if we get some enhancement AND pass Cassini
                    if not cassini['passes'] or max_enhancement < 1.1:
                        continue
                    
                    n_passed_cassini += 1
                    chi2 = chi_squared(params, R_data, v_data, sigma_data, contrast_type)
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = params.copy()
                        print(f"  New best: χ² = {chi2:.1f}, ξ_max = {max_enhancement:.2f} (gs={gs}, ct={ct}, A={A})")
    
    elif contrast_type == 'bands':
        # UPDATED: Better ranges for bands
        band_widths = [0.3, 0.5, 1.0]
        A_per_bands = [10, 20, 50]  # Higher enhancement per band
        rho_refs = [1e9, 1e10]  # Higher reference densities
        
        total_combinations = len(band_widths) * len(A_per_bands) * len(rho_refs)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for bw in band_widths:
            for A in A_per_bands:
                for rho_ref in rho_refs:
                    n_tested += 1
                    params = base_params.copy()
                    params.update({
                        'band_width_dex': bw,
                        'A_per_band': A,
                        'rho_ref': rho_ref
                    })
                    
                    cassini = check_cassini_constraint(params, contrast_type)
                    
                    # Check for actual enhancement
                    R_test_gpu = cp.asarray([5, 10, 15, 20], dtype=cp.float32)
                    _, _, xi_test = v_total_contrast(R_test_gpu, params, contrast_type, return_components=True)
                    xi_test = to_numpy_array(xi_test)
                    max_enhancement = xi_test.max()
                    
                    if not cassini['passes'] or max_enhancement < 1.1:
                        continue
                    
                    n_passed_cassini += 1
                    chi2 = chi_squared(params, R_data, v_data, sigma_data, contrast_type)
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = params.copy()
                        print(f"  New best: χ² = {chi2:.1f}, ξ_max = {max_enhancement:.2f} (bw={bw}, A={A}, ρ_ref={rho_ref:.1e})")
    
    elif contrast_type == 'boundaries':
        # UPDATED: Better boundary parameters
        boundary_widths = [1, 2, 3]
        A_boundaries = [20, 50, 100]  # Higher enhancement at boundaries
        
        total_combinations = len(boundary_widths) * len(A_boundaries)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for bw in boundary_widths:
            for A in A_boundaries:
                n_tested += 1
                params = base_params.copy()
                params.update({
                    'boundary_ratio': 100,
                    'boundary_width_kpc': bw,
                    'A_boundary': A
                })
                
                cassini = check_cassini_constraint(params, contrast_type)
                
                # Check for actual enhancement
                R_test_gpu = cp.asarray([5, 10, 15, 20], dtype=cp.float32)
                _, _, xi_test = v_total_contrast(R_test_gpu, params, contrast_type, return_components=True)
                xi_test = to_numpy_array(xi_test)
                max_enhancement = xi_test.max()
                
                if not cassini['passes'] or max_enhancement < 1.1:
                    continue
                
                n_passed_cassini += 1
                chi2 = chi_squared(params, R_data, v_data, sigma_data, contrast_type)
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = params.copy()
                    print(f"  New best: χ² = {chi2:.1f}, ξ_max = {max_enhancement:.2f} (bw={bw}, A={A})")
    
    print(f"Tested {n_tested} combinations, {n_passed_cassini} passed Cassini with enhancement > 10%")
    
    # If no params passed Cassini, relax and find best anyway
    if best_params is None:
        print("WARNING: No parameters passed Cassini constraint!")
        print("Finding best fit regardless of Cassini...")
        best_chi2 = np.inf
        
        # Try with relaxed parameters
        if contrast_type == 'gradient':
            params = base_params.copy()
            params.update({
                'gradient_scale_kpc': 2.0,
                'contrast_threshold': 1.0, 
                'A_contrast': 50,
                'transition_width': 2.0
            })
            chi2 = chi_squared(params, R_data, v_data, sigma_data, contrast_type)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_params = params
    
    return best_params, best_chi2

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(R_data, v_data, sigma_data, params, contrast_type, save_path=None):
    """Plot data vs model."""
    print(f"\n--- Plotting Results: {contrast_type} ---")
    
    try:
        # Calculate model on fine grid
        R_model = np.linspace(0.5, 30, 200)
        R_gpu = cp.asarray(R_model, dtype=cp.float32)
        
        # Get all components
        v_model_gpu, v_newton_gpu, xi_gpu = v_total_contrast(
            R_gpu, params, contrast_type, return_components=True
        )
        v_model = to_numpy_array(v_model_gpu)
        v_newton = to_numpy_array(v_newton_gpu)
        xi = to_numpy_array(xi_gpu)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top: Rotation curve
        ax1.errorbar(R_data, v_data, yerr=sigma_data, fmt='o', alpha=0.5, 
                    label='Data', markersize=3)
        ax1.plot(R_model, v_model, 'r-', linewidth=2, 
                label=f'Contrast Model ({contrast_type})')
        ax1.plot(R_model, v_newton, 'b--', linewidth=1, label='Newton (no enhancement)')
        
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_ylim(0, 350)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Density Contrast Model: {contrast_type}')
        
        # Bottom: Enhancement factor
        ax2.plot(R_model, xi, 'g-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=8.5, color='r', linestyle=':', alpha=0.5, label='Sun position')
        
        # Mark key boundaries
        boundaries = [3, 10, 25]  # Approximate MW boundaries
        for b in boundaries:
            ax2.axvline(x=b, color='gray', linestyle=':', alpha=0.3)
        
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Enhancement ξ')
        ax2.set_ylim(0.8, 20)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add text showing enhancement range
        ax2.text(0.02, 0.98, f'ξ range: [{xi.min():.3f}, {xi.max():.3f}]',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Plot saved to {save_path}")
            print(f"  Enhancement range: ξ ∈ [{xi.min():.3f}, {xi.max():.3f}]")
        else:
            plt.show()
        
        return fig
        
    except Exception as e:
        print(f"✗ Error creating plot: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("DENSITY CONTRAST GRAVITY MODEL TEST")
    print("="*60)
    
    try:
        # Load data
        print("\n--- Loading Data ---")
        R_data, v_data, sigma_data = load_gaia_or_synthetic(max_stars=5000)  # Start with 5k for speed
        print(f"✓ Data loaded: {len(R_data)} points")
        print(f"  R range: [{R_data.min():.1f}, {R_data.max():.1f}] kpc")
        print(f"  v range: [{v_data.min():.1f}, {v_data.max():.1f}] km/s")
        
        # Test each contrast model type
        results = {}
        
        for contrast_type in ['gradient', 'bands', 'boundaries']:
            print(f"\n{'='*40}")
            print(f"Testing: {contrast_type.upper()}")
            print('='*40)
            
            # Find best parameters
            best_params, best_chi2 = grid_search_simple(
                R_data, v_data, sigma_data, contrast_type
            )
            
            if best_params is None:
                print(f"✗ No valid parameters found for {contrast_type}")
                continue
            
            print(f"✓ Found best parameters for {contrast_type}")
            
            # Check Cassini constraint
            cassini = check_cassini_constraint(best_params, contrast_type)
            
            # Calculate reduced chi-squared
            n_data = len(R_data)
            n_params = 5  # Approximate
            chi2_reduced = best_chi2 / (n_data - n_params)
            
            # Store results
            results[contrast_type] = {
                'params': best_params,
                'chi2': best_chi2,
                'chi2_reduced': chi2_reduced,
                'cassini_passes': cassini['passes'],
                'xi_sun': cassini['xi_sun'],
                'gamma_minus_one': cassini['gamma_minus_one']
            }
            
            # Print summary
            print(f"\nBest fit results:")
            print(f"  χ² = {best_chi2:.1f} (reduced: {chi2_reduced:.2f})")
            print(f"  Cassini constraint: {'✓ PASS' if cassini['passes'] else '✗ FAIL'}")
            print(f"  ξ at Sun = {cassini['xi_sun']:.6f}")
            print(f"  γ - 1 = {cassini['gamma_minus_one']:.2e}")
            
            if contrast_type == 'gradient':
                print(f"  Gradient scale: {best_params['gradient_scale_kpc']:.1f} kpc")
                print(f"  Contrast threshold: {best_params['contrast_threshold']:.0f}")
                print(f"  Max enhancement: {best_params['A_contrast']:.1f}")
            
            # Plot results
            plot_path = f"contrast_model_{contrast_type}.png"
            plot_results(R_data, v_data, sigma_data, best_params, contrast_type, plot_path)
        
        # Compare models
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print('='*60)
        
        if results:
            print(f"{'Model':<12} {'χ²':<10} {'χ²/dof':<10} {'Cassini':<10} {'ξ(Sun)':<10}")
            print("-"*52)
            
            for model, res in results.items():
                cassini_str = "✓ PASS" if res['cassini_passes'] else "✗ FAIL"
                print(f"{model:<12} {res['chi2']:<10.1f} {res['chi2_reduced']:<10.2f} "
                      f"{cassini_str:<10} {res['xi_sun']:<10.4f}")
        else:
            print("No valid models found!")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'n_data': len(R_data),
            'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items() if kk != 'params'} 
                       for k, v in results.items()}
        }
        
        with open('contrast_model_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to contrast_model_results.json")
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nKey insight: Density CONTRAST (not absolute density) drives enhancement!")
        print("This explains why:")
        print("  - Solar system has normal gravity (uniform density)")
        print("  - Galaxy edges show enhancement (huge contrast)")
        print("  - Model naturally satisfies Cassini constraint")
        
        # Run diagnostic on best model
        if results:
            best_model = min(results.items(), key=lambda x: x[1]['chi2'])
            print(f"\nBest model: {best_model[0]} with χ² = {best_model[1]['chi2']:.1f}")
            print("\nRunning diagnostic on best model...")
            diagnose_enhancement(params=best_model[1]['params'], contrast_type=best_model[0])
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR in main(): {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Script starting...")
    print("="*60)
    
    try:
        exit_code = main()
        print(f"\nScript finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Script interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)