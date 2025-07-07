#!/usr/bin/env python3
"""
test_fitted_params.py - Test how well the dynesty fitted parameters work
"""
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_gaia
from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP

print("TESTING DYNESTY FITTED PARAMETERS")
print("="*50)

# Load data
gaia_data = load_gaia(sample_max=20000)  # Subset for speed
R_data = gaia_data['R_kpc']
v_data = gaia_data['v_obs']
sigma_data = gaia_data['sigma_v']

print(f"Testing with {len(R_data):,} stars")
print(f"R range: {np.min(R_data):.2f} - {np.max(R_data):.2f} kpc")

# Fitted parameters from dynesty
fitted_params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'M_disk_thin_solar': 1.269e11,  # From dynesty
    'R_d_thin_kpc': 4.138,          # From dynesty  
    'h_z_thin_kpc': 0.5953,         # From dynesty
    'rho_c_solar_kpc3': 1.642e9,    # From dynesty
    'n_exp': 1.560                   # From dynesty
}

# Your original optimized parameters (for comparison)
optimized_params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'M_disk_thin_solar': 9.6e10,    # Your optimization
    'R_d_thin_kpc': 2.8,            # Your optimization
    'h_z_thin_kpc': 0.3,            # Your optimization  
    'rho_c_solar_kpc3': 8.0e8,      # Your optimization
    'n_exp': 1.5                     # Your optimization
}

print(f"\n1. TESTING FITTED PARAMETERS:")
v_newton_fit = v_baryon_total_newtonian_kms(R_data, fitted_params)
rho_fit = rho_baryon_total_midplane_solar_kpc3(R_data, fitted_params)
xi_fit = XI_FUNCTION_MAP['power'](rho_fit, fitted_params['rho_c_solar_kpc3'], fitted_params['n_exp'])
v_model_fit = v_newton_fit * np.sqrt(np.maximum(xi_fit, 0.0))

residuals_fit = v_data - v_model_fit
chi2_fit = np.sum((residuals_fit / sigma_data)**2)
rms_fit = np.sqrt(np.mean(residuals_fit**2))

print(f"  Model v range: {np.min(v_model_fit):.1f} - {np.max(v_model_fit):.1f} km/s")
print(f"  RMS residual: {rms_fit:.1f} km/s")
print(f"  χ² total: {chi2_fit:.1f}")
print(f"  χ² per star: {chi2_fit/len(R_data):.1f}")

print(f"\n2. TESTING YOUR OPTIMIZED PARAMETERS:")
v_newton_opt = v_baryon_total_newtonian_kms(R_data, optimized_params)
rho_opt = rho_baryon_total_midplane_solar_kpc3(R_data, optimized_params)
xi_opt = XI_FUNCTION_MAP['power'](rho_opt, optimized_params['rho_c_solar_kpc3'], optimized_params['n_exp'])
v_model_opt = v_newton_opt * np.sqrt(np.maximum(xi_opt, 0.0))

residuals_opt = v_data - v_model_opt
chi2_opt = np.sum((residuals_opt / sigma_data)**2)
rms_opt = np.sqrt(np.mean(residuals_opt**2))

print(f"  Model v range: {np.min(v_model_opt):.1f} - {np.max(v_model_opt):.1f} km/s")
print(f"  RMS residual: {rms_opt:.1f} km/s")
print(f"  χ² total: {chi2_opt:.1f}")
print(f"  χ² per star: {chi2_opt/len(R_data):.1f}")

# Test at different radial bins
print(f"\n3. RADIAL PERFORMANCE COMPARISON:")
R_bins = [4, 6, 8, 10, 12, 15]
for R_center in R_bins:
    mask = np.abs(R_data - R_center) < 0.5
    if np.sum(mask) > 10:
        v_obs_bin = v_data[mask]
        v_fit_bin = v_model_fit[mask]
        v_opt_bin = v_model_opt[mask]
        
        rms_fit_bin = np.sqrt(np.mean((v_obs_bin - v_fit_bin)**2))
        rms_opt_bin = np.sqrt(np.mean((v_obs_bin - v_opt_bin)**2))
        
        print(f"  R ≈ {R_center} kpc ({np.sum(mask)} stars):")
        print(f"    Observed: {np.mean(v_obs_bin):.1f} ± {np.std(v_obs_bin):.1f} km/s")
        print(f"    Fitted:   {np.mean(v_fit_bin):.1f} ± {np.std(v_fit_bin):.1f} km/s (RMS: {rms_fit_bin:.1f})")
        print(f"    Optimized:{np.mean(v_opt_bin):.1f} ± {np.std(v_opt_bin):.1f} km/s (RMS: {rms_opt_bin:.1f})")

# Create diagnostic plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Rotation curves
R_plot = np.linspace(0.5, 20, 100)
v_fit_plot = v_baryon_total_newtonian_kms(R_plot, fitted_params) * np.sqrt(XI_FUNCTION_MAP['power'](rho_baryon_total_midplane_solar_kpc3(R_plot, fitted_params), fitted_params['rho_c_solar_kpc3'], fitted_params['n_exp']))
v_opt_plot = v_baryon_total_newtonian_kms(R_plot, optimized_params) * np.sqrt(XI_FUNCTION_MAP['power'](rho_baryon_total_midplane_solar_kpc3(R_plot, optimized_params), optimized_params['rho_c_solar_kpc3'], optimized_params['n_exp']))

ax1.scatter(R_data, v_data, alpha=0.3, s=1, color='gray', label='Data')
ax1.plot(R_plot, v_fit_plot, 'r-', lw=2, label='Dynesty Fit')
ax1.plot(R_plot, v_opt_plot, 'b--', lw=2, label='Your Optimization')
ax1.axvline(8.0, color='orange', linestyle=':', alpha=0.7, label='Solar Radius')
ax1.set_xlabel('R (kpc)')
ax1.set_ylabel('v (km/s)')
ax1.set_title('Rotation Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs R (fitted params)
ax2.scatter(R_data, residuals_fit, alpha=0.3, s=1)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('R (kpc)')
ax2.set_ylabel('Residuals (km/s)')
ax2.set_title(f'Fitted Params Residuals (RMS={rms_fit:.1f})')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals vs R (optimized params)
ax3.scatter(R_data, residuals_opt, alpha=0.3, s=1)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_xlabel('R (kpc)')
ax3.set_ylabel('Residuals (km/s)')
ax3.set_title(f'Optimized Params Residuals (RMS={rms_opt:.1f})')
ax3.grid(True, alpha=0.3)

# Plot 4: Xi comparison
xi_fit_plot = XI_FUNCTION_MAP['power'](rho_baryon_total_midplane_solar_kpc3(R_plot, fitted_params), fitted_params['rho_c_solar_kpc3'], fitted_params['n_exp'])
xi_opt_plot = XI_FUNCTION_MAP['power'](rho_baryon_total_midplane_solar_kpc3(R_plot, optimized_params), optimized_params['rho_c_solar_kpc3'], optimized_params['n_exp'])

ax4.plot(R_plot, xi_fit_plot, 'r-', lw=2, label='Dynesty Fit')
ax4.plot(R_plot, xi_opt_plot, 'b--', lw=2, label='Your Optimization')
ax4.axvline(8.0, color='orange', linestyle=':', alpha=0.7)
ax4.set_xlabel('R (kpc)')
ax4.set_ylabel('ξ(ρ)')
ax4.set_title('Density-Dependent Modification')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_comparison.png', dpi=150)
print(f"\n4. DIAGNOSTIC PLOT SAVED: parameter_comparison.png")

print(f"\n" + "="*50)
print("SUMMARY:")
if rms_fit < rms_opt:
    print(f"✅ Dynesty fitted parameters perform BETTER")
    print(f"   (RMS: {rms_fit:.1f} vs {rms_opt:.1f} km/s)")
else:
    print(f"⚠️  Your optimized parameters perform BETTER")
    print(f"   (RMS: {rms_opt:.1f} vs {rms_fit:.1f} km/s)")
    print(f"   This suggests optimization for R=8kpc doesn't work globally")

print(f"\nKey differences:")
print(f"  Disk mass: {fitted_params['M_disk_thin_solar']:.2e} vs {optimized_params['M_disk_thin_solar']:.2e}")
print(f"  Scale length: {fitted_params['R_d_thin_kpc']:.1f} vs {optimized_params['R_d_thin_kpc']:.1f} kpc")
print(f"  Critical density: {fitted_params['rho_c_solar_kpc3']:.2e} vs {optimized_params['rho_c_solar_kpc3']:.2e}")