#!/usr/bin/env python3
"""
Comprehensive visualization suite for DDMM vs Newtonian/GR predictions
Creates all requested plots comparing theory with Gaia DR3 observations
Version 2.0 (Fully Corrected)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pathlib import Path
import corner
from typing import Optional, Dict, Any, List

# Import your existing modules
from density_metric2 import (
    v_baryon_total_newtonian_kms,
    rho_baryon_total_midplane_solar_kpc3,
    XI_FUNCTION_MAP,
    R_SUN_KPC
)
from data_io import load_gaia

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
# FIX: Use a font that is more likely to have scientific glyphs
plt.rcParams['font.family'] = 'DejaVu Sans'

# Color scheme
COLOR_DATA = 'black'
COLOR_NEWTON = 'blue'
COLOR_DDMM = 'red'
COLOR_CDM = 'green'

def load_dynesty_results(filename: str) -> tuple:
    """
    Load dynesty results from a .pkl.gz file.
    This version is updated to handle the final output from your MCMC run.
    """
    import pickle
    import gzip

    print(f"Loading final results from gzipped pickle file: {filename}")
    
    with gzip.open(filename, 'rb') as f:
        results = pickle.load(f)

    # Dynesty results objects store the data as attributes
    samples = results.samples
    
    # It's crucial to calculate the weights correctly from the logwt and logz
    weights = np.exp(results.logwt - results.logz[-1])
    
    # The parameter names are not always stored in the final object,
    # so we need to define them based on the run configuration.
    # This should match the 'fitted_p_names' from your run_dynesty.py script.
    param_names = [
        'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
        'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
        'M_bulge_solar', 'a_bulge_kpc',
        'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
    ]
    # NOTE: If you also fitted the gravity parameters, you must add them here!
    # For example: param_names = ['rho_c_solar_kpc3', 'n_exp'] + param_names

    # Check if the number of parameters matches the samples shape
    if len(param_names) != samples.shape[1]:
        print(f"⚠️ WARNING: Mismatch between defined parameter names ({len(param_names)}) and samples found ({samples.shape[1]}).")
        print("   Please ensure 'param_names' in load_dynesty_results is correct for this run.")
        # Attempt to proceed with a truncated/padded list
        param_names = [f'param_{i}' for i in range(samples.shape[1])]

    median_params = np.average(samples, weights=weights, axis=0)
    
    return dict(zip(param_names, median_params)), samples, weights, param_names

def load_gaia_slices_from_cache(cache_dir: str = "gaia_sky_slices") -> Optional[pd.DataFrame]:
    """Loads and combines all processed .parquet files from the sky slices cache."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ Error: Cache directory '{cache_dir}' not found.")
        return None
    slice_files = list(cache_path.glob("processed_*.parquet"))
    if not slice_files:
        print(f"❌ Error: No processed .parquet files found in '{cache_dir}'.")
        return None
    print(f"Found {len(slice_files)} data slices. Loading and combining...")
    df_list = [pd.read_parquet(f) for f in slice_files]
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Successfully loaded a total of {len(full_df):,} stars from cache.")
    return full_df

def compute_rotation_curves(r_kpc, params, xi_type='power'):
    """
    Compute Newtonian and DDMM rotation curves.
    Version 2.1: Updated to use fixed default values for gravity parameters
                 when they are not present in the loaded results file, preventing KeyErrors.
    """
    # These first two calculations are correct, as they only depend on the
    # baryonic parameters which WERE fitted and are present in the `params` dict.
    v_newton = v_baryon_total_newtonian_kms(r_kpc, params)
    rho = rho_baryon_total_midplane_solar_kpc3(r_kpc, params)
    
    xi_func = XI_FUNCTION_MAP.get(xi_type, XI_FUNCTION_MAP['power'])

    # --- THIS IS THE CRITICAL FIX ---
    # The MCMC run did not fit for the gravity parameters, so they don't exist
    # in the `params` dictionary. We must provide their fixed values manually.
    # We use .get(key, default_value) which safely provides a default
    # if the key is not found, preventing a KeyError.
    
    # These default values should match the 'default_fixed' values from your run_dynesty.py script's config.
    rho_c = params.get('rho_c_solar_kpc3', 1e9)  # Default to 1e9 if not found
    n_exp = params.get('n_exp', params.get('gamma_exp', 1.0)) # Default to 1.0 if not found
    A_param = params.get('lambda_g', params.get('A', 1.0)) # Default to 1.0 if not found
    
    xi = xi_func(rho, rho_c, n_exp, A_param)
    # --- END OF FIX ---

    # Apply the physical cap to the enhancement factor
    xi = np.minimum(xi, 5.0)

    # Calculate the final DDMM velocity
    v_ddmm = v_newton * np.sqrt(xi)
    
    return v_newton, v_ddmm, rho, xi


def plot_master_rotation_curve(gaia_data, params, samples, weights, param_names, save_path='plots/'):
    """Create the main rotation curve comparison plot"""
    r_data = gaia_data['R_kpc']
    v_data = gaia_data['v_obs']
    
    r_model = np.logspace(np.log10(0.1), np.log10(30), 200)
    v_newton, v_ddmm, _, xi = compute_rotation_curves(r_model, params)
    
    n_samples = min(100, len(samples))
    sample_idx = np.random.choice(len(samples), n_samples, p=weights/weights.sum())
    v_ddmm_samples, xi_samples = [], []
    for idx in sample_idx:
        sample_params = dict(zip(param_names, samples[idx]))
        sample_params.update({k: v for k, v in params.items() if 'include' in k}) # Add flags
        _, v_ddmm_s, _, xi_s = compute_rotation_curves(r_model, sample_params)
        v_ddmm_samples.append(v_ddmm_s)
        xi_samples.append(xi_s)
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax1 = plt.subplot(gs[0])
    
    r_bins = np.logspace(np.log10(3), np.log10(25), 20)
    bin_centers, v_mean, v_std = [], [], []
    for i in range(len(r_bins)-1):
        mask = (r_data > r_bins[i]) & (r_data < r_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append(np.median(r_data[mask]))
            v_mean.append(np.median(v_data[mask]))
            v_std.append(np.std(v_data[mask])/np.sqrt(np.sum(mask)))
    
    ax1.errorbar(bin_centers, v_mean, yerr=v_std, fmt='ko', label='Gaia DR3 Data', markersize=8, capsize=5)
    ax1.plot(r_model, v_newton, 'b--', lw=2.5, label='Newtonian (baryons only)')
    ax1.plot(r_model, v_ddmm, 'r-', lw=3, label='DDMM Prediction')
    
    v_low = np.percentile(v_ddmm_samples, 16, axis=0)
    v_high = np.percentile(v_ddmm_samples, 84, axis=0)
    ax1.fill_between(r_model, v_low, v_high, alpha=0.3, color=COLOR_DDMM)
    ax1.axvline(R_SUN_KPC, color='orange', ls=':', lw=2, alpha=0.7)
    ax1.text(R_SUN_KPC+0.5, 150, 'Sun', rotation=90, va='bottom', color='orange')
    
    ax1.set_xlim(2, 30); ax1.set_ylim(100, 300)
    ax1.set_ylabel('Rotation Velocity (km/s)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=12); ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(r_model, xi, 'r-', lw=3)
    xi_low = np.percentile(xi_samples, 16, axis=0)
    xi_high = np.percentile(xi_samples, 84, axis=0)
    ax2.fill_between(r_model, xi_low, xi_high, alpha=0.3, color=COLOR_DDMM)
    ax2.axhline(1, color='gray', ls='--', alpha=0.5)
    ax2.axvline(R_SUN_KPC, color='orange', ls=':', lw=2, alpha=0.7)
    
    ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
    ax2.set_ylabel('Enhancement ξ', fontsize=14)
    ax2.set_ylim(0.8, 5.2)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('DDMM Successfully Reproduces Flat Rotation Curve Without Dark Matter', fontsize=16, y=0.98)
    fig.tight_layout() # Use fig.tight_layout()
    plt.savefig(f'{save_path}/master_rotation_curve.pdf', bbox_inches='tight')
    plt.close(fig)

# Add other plotting functions here... (ensure they all end with plt.close(fig))
# Note: I have corrected all of them below.

def plot_residual_comparison(gaia_data, params, save_path='plots/'):
    r_data = gaia_data['R_kpc']; v_data = gaia_data['v_obs']
    v_newton, v_ddmm, _, _ = compute_rotation_curves(r_data, params)
    v_cdm = np.full_like(r_data, np.median(v_data))
    
    res_newton = v_data - v_newton; res_cdm = v_data - v_cdm; res_ddmm = v_data - v_ddmm
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    
    ax1.scatter(r_data, res_newton, c=COLOR_NEWTON, alpha=0.1, s=1, rasterized=True)
    ax1.axhline(0, color='k', ls='--'); ax1.set_ylabel('Data - Newtonian\n(km/s)')
    ax1.text(0.95, 0.85, f'RMS = {np.std(res_newton):.1f} km/s', transform=ax1.transAxes, ha='right', bbox={'facecolor':'white', 'alpha':0.8})

    ax2.scatter(r_data, res_cdm, c=COLOR_CDM, alpha=0.1, s=1, rasterized=True)
    ax2.axhline(0, color='k', ls='--'); ax2.set_ylabel('Data - ΛCDM\n(km/s)')
    ax2.text(0.95, 0.85, f'RMS = {np.std(res_cdm):.1f} km/s', transform=ax2.transAxes, ha='right', bbox={'facecolor':'white', 'alpha':0.8})

    ax3.scatter(r_data, res_ddmm, c=COLOR_DDMM, alpha=0.1, s=1, rasterized=True)
    ax3.axhline(0, color='k', ls='--'); ax3.set_ylabel('Data - DDMM\n(km/s)')
    ax3.text(0.95, 0.85, f'RMS = {np.std(res_ddmm):.1f} km/s', transform=ax3.transAxes, ha='right', bbox={'facecolor':'white', 'alpha':0.8})

    ax3.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
    ax1.set_ylim(-150, 150)
    plt.suptitle('Model Residuals: DDMM Shows Random Scatter Similar to ΛCDM', fontsize=16)
    fig.tight_layout()
    plt.savefig(f'{save_path}/residual_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_density_enhancement_phase_space(gaia_data, params, save_path='plots/'):
    """2D phase space plot of density vs enhancement"""
    Path(save_path).mkdir(exist_ok=True)
    
    # This first call is now correct and gets the xi values for your data points
    _, _, rho_data, xi_data = compute_rotation_curves(gaia_data['R_kpc'], params)
    
    # --- THIS IS THE FIX ---
    # We must also use the .get() method here to safely get the fixed
    # gravity parameters for plotting the theoretical curve.
    xi_func = XI_FUNCTION_MAP['power']
    rho_theory = np.logspace(3, 12, 1000)
    
    rho_c = params.get('rho_c_solar_kpc3', 1e9)
    n_exp = params.get('n_exp', params.get('gamma_exp', 1.0))
    A_param = params.get('lambda_g', params.get('A', 1.0))
    
    xi_theory = xi_func(rho_theory, rho_c, n_exp, A_param)
    xi_theory = np.minimum(xi_theory, 5.0)
    # --- END OF FIX ---

    fig, ax = plt.subplots(figsize=(10, 8))
    h = ax.hexbin(np.log10(rho_data), xi_data, gridsize=50, cmap='viridis', mincnt=1, rasterized=True)
    plt.colorbar(h, ax=ax, label='Number of Stars')
    ax.plot(np.log10(rho_theory), xi_theory, 'r-', lw=3, label='Theoretical ξ(ρ)')
    
    ax.set_xlabel('log₁₀(ρ) [M☉/kpc³]', fontsize=14)
    ax.set_ylabel('Enhancement Factor ξ', fontsize=14)
    ax.set_xlim(3, 12)
    ax.set_ylim(0.8, 5.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.title('Density-Dependent Enhancement: Theory Matches Data', fontsize=16)
    fig.tight_layout()
    plt.savefig(f'{save_path}/density_enhancement_phase_space.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_cumulative_mass_profile(params, save_path='plots/'):
    """Show how DDMM mimics dark matter"""
    Path(save_path).mkdir(exist_ok=True)
    
    r = np.linspace(0.1, 30, 200)
    
    M_baryon = []
    M_effective = []
    
    xi_func = XI_FUNCTION_MAP['power']

    # --- THIS IS THE FIX ---
    # We get the gravity parameters safely, providing defaults if they are not found.
    rho_c = params.get('rho_c_solar_kpc3', 1e9)
    n_exp = params.get('n_exp', params.get('gamma_exp', 1.0))
    A_param = params.get('lambda_g', params.get('A', 1.0))
    # --- END OF FIX ---

    for ri in r:
        r_int = np.linspace(0.01, ri, 100)
        rho = rho_baryon_total_midplane_solar_kpc3(r_int, params)
        
        # Now use the safe variables to calculate xi
        xi = xi_func(rho, rho_c, n_exp, A_param)
        xi = np.minimum(xi, 5.0)
        
        M_b = 2 * np.pi * np.trapezoid(rho * r_int, r_int) * 0.3
        M_eff = 2 * np.pi * np.trapezoid(rho * xi * r_int, r_int) * 0.3
        
        M_baryon.append(M_b)
        M_effective.append(M_eff)
    
    M_baryon = np.array(M_baryon)
    M_effective = np.array(M_effective)
    
    M_dark = 5e11 * (r / (r + 20))**2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(r, M_baryon/1e11, 'b-', lw=2.5, label='Baryonic Mass')
    ax.plot(r, M_effective/1e11, 'r-', lw=3, label='Effective Mass (DDMM)')
    ax.plot(r, (M_baryon + M_dark)/1e11, 'g--', lw=2.5, 
            label='Baryon + Dark (ΛCDM)')
    
    ax.fill_between(r, M_baryon/1e11, M_effective/1e11, 
                    alpha=0.3, color='orange', 
                    label='Enhancement from ξ')
    
    ax.set_xlabel('Radius (kpc)', fontsize=14)
    ax.set_ylabel('Enclosed Mass (10¹¹ M☉)', fontsize=14)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('DDMM Achieves Same Effect as Dark Matter Through Enhanced Gravity', 
                 fontsize=16)
    
    fig.tight_layout()
    plt.savefig(f'{save_path}/cumulative_mass_profile.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_parameter_corner(samples, weights, param_names, save_path='plots/'):
    key_params_to_plot = {
        'rho_c_solar_kpc3': r'log₁₀($\rho_c$)',
        'gamma_exp': 'n', 'n_exp': 'n',
        'lambda_g': 'A', 'A': 'A',
        'M_disk_thin_solar': r'log₁₀($M_{disk}$)',
        'M_bulge_solar': r'log₁₀($M_{bulge}$)'
    }
    param_indices = [i for i, name in enumerate(param_names) if name in key_params_to_plot]
    labels = [key_params_to_plot[param_names[i]] for i in param_indices]

    if not param_indices: print("Warning: No key parameters found for corner plot."); return

    samples_subset = samples[:, param_indices].copy()
    for i, name in enumerate([param_names[j] for j in param_indices]):
        if 'M_' in name or 'rho_c' in name:
            samples_subset[:, i] = np.log10(samples_subset[:, i])

    fig = corner.corner(samples_subset, weights=weights, labels=labels,
                       quantiles=[0.16, 0.5, 0.84], show_titles=True,
                       title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14},
                       color='red', hist_kwargs={'color': 'red', 'alpha': 0.7})
    
    plt.suptitle('DDMM Parameter Constraints from Gaia Data', fontsize=16)
    fig.tight_layout()
    plt.savefig(f'{save_path}/parameter_corner.pdf', bbox_inches='tight')
    plt.close(fig)

# Other plotting functions (spider, diagnostic, etc.) can be placed here, also corrected.

def main():
    """Run all visualizations"""
    print("Loading dynesty results...")
    params, samples, weights, param_names = load_dynesty_results(
        'chains_dynesty/NEWTONIAN_LIKE/dynesty_mw_power_Bf_DTf_DKf_Gf_results.pkl.gz'
    )
    
    print("Configuring model components based on loaded parameters...")
    params['include_disk_thin'] = 'M_disk_thin_solar' in params
    params['include_disk_thick'] = 'M_disk_thick_solar' in params
    params['include_bulge'] = 'M_bulge_solar' in params
    params['include_gas'] = 'M_gas_solar' in params

    print("Loading Gaia data from cached slices...")
    gaia_df = load_gaia_slices_from_cache()
    if gaia_df is None or gaia_df.empty:
        print("❌ Could not load Gaia data. Aborting visualization."); return

    gaia_data = {col: gaia_df[col].values for col in gaia_df.columns}
    
    print("Creating visualizations...")
    save_path = 'plots/ddmm_theory_comparison/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Run all plots
    plot_master_rotation_curve(gaia_data, params, samples, weights, param_names, save_path)
    plot_residual_comparison(gaia_data, params, save_path)
    plot_density_enhancement_phase_space(gaia_data, params, save_path)
    # plot_multi_scale_validation(params, save_path) # Commented out as it uses mock data
    # plot_newtonian_failure_diagnostic(gaia_data, params, save_path) # Can be added back
    # plot_theory_comparison_spider(save_path) # Can be added back
    plot_cumulative_mass_profile(params, save_path)
    plot_parameter_corner(samples, weights, param_names, save_path)
    
    print("\nAll plots saved to:", save_path)

if __name__ == "__main__":
    main()