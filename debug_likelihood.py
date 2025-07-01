#!/usr/bin/env python3
"""
debug_likelihood.py - Debug why likelihood is extremely negative
"""
import numpy as np
import matplotlib.pyplot as plt
from density_metric2 import v_baryon_total_newtonian_kms, rho_baryon_total_midplane_solar_kpc3, XI_FUNCTION_MAP
from data_io import load_gaia

print("DEBUG: Likelihood Calculation Issues")
print("="*50)

# Load Gaia data
print("Loading Gaia data...")
gaia_data = load_gaia(sample_max=10000)
if gaia_data is None:
    print("ERROR: Could not load Gaia data")
    exit(1)

R_data = gaia_data['R_kpc']
v_data = gaia_data['v_obs']
sigma_data = gaia_data['sigma_v']

print(f"Loaded {len(R_data)} data points")
print(f"R range: {np.min(R_data):.2f} - {np.max(R_data):.2f} kpc")
print(f"v range: {np.min(v_data):.1f} - {np.max(v_data):.1f} km/s")
print(f"σ range: {np.min(sigma_data):.1f} - {np.max(sigma_data):.1f} km/s")

# Test parameters from your optimization
params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'M_disk_thin_solar': 9.6e10,
    'R_d_thin_kpc': 2.8,
    'h_z_thin_kpc': 0.3,  # Test different values
    'rho_c_solar_kpc3': 8.0e08,
    'n_exp': 1.5
}

print("\nTesting model with optimal parameters...")

# Calculate model predictions
v_newton = v_baryon_total_newtonian_kms(R_data, params)
rho_mid = rho_baryon_total_midplane_solar_kpc3(R_data, params)

# Get xi values
xi_func = XI_FUNCTION_MAP['power']
xi_values = xi_func(rho_mid, params['rho_c_solar_kpc3'], params['n_exp'])

# Handle potential array outputs
if hasattr(xi_values, '__len__'):
    xi_values = xi_values
else:
    xi_values = np.full_like(R_data, xi_values)

v_model = v_newton * np.sqrt(np.maximum(xi_values, 0.0))

print(f"Model v range: {np.min(v_model):.1f} - {np.max(v_model):.1f} km/s")
print(f"Model v at R=8 kpc: {np.interp(8.0, R_data, v_model):.1f} km/s")

# Calculate residuals and likelihood components
residuals = v_data - v_model
sigma_safe = np.maximum(sigma_data, 1e-9)
chi2_terms = (residuals / sigma_safe)**2
log_terms = chi2_terms + np.log(2 * np.pi * sigma_safe**2)

print(f"\nDiagnostics:")
print(f"Residuals: mean={np.mean(residuals):.1f}, std={np.std(residuals):.1f} km/s")
print(f"Chi2 terms: min={np.min(chi2_terms):.1f}, max={np.max(chi2_terms):.1f}")
print(f"Log terms: min={np.min(log_terms):.1f}, max={np.max(log_terms):.1f}")

# Calculate likelihood
log_L = -0.5 * np.sum(log_terms)
print(f"\nLog-likelihood: {log_L:.2f}")
print(f"Per data point: {log_L/len(R_data):.2f}")

# Check for problematic values
n_bad_v = np.sum(~np.isfinite(v_model))
n_bad_xi = np.sum(~np.isfinite(xi_values))
n_bad_residuals = np.sum(~np.isfinite(residuals))

print(f"\nQuality checks:")
print(f"Bad v_model values: {n_bad_v}")
print(f"Bad xi values: {n_bad_xi}")
print(f"Bad residuals: {n_bad_residuals}")
print(f"Negative xi values: {np.sum(xi_values < 0)}")
print(f"Xi range: {np.min(xi_values):.3f} - {np.max(xi_values):.3f}")

# Test different h_z values
print(f"\nTesting different h_z values:")
h_z_test_values = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for h_z in h_z_test_values:
    test_params = params.copy()
    test_params['h_z_thin_kpc'] = h_z
    
    rho_test = rho_baryon_total_midplane_solar_kpc3(R_data, test_params)
    xi_test = xi_func(rho_test, params['rho_c_solar_kpc3'], params['n_exp'])
    
    if hasattr(xi_test, '__len__'):
        xi_test = xi_test
    else:
        xi_test = np.full_like(R_data, xi_test)
        
    v_test = v_newton * np.sqrt(np.maximum(xi_test, 0.0))
    residuals_test = v_data - v_test
    chi2_test = np.sum((residuals_test / sigma_safe)**2)
    log_L_test = -0.5 * np.sum((residuals_test / sigma_safe)**2 + np.log(2 * np.pi * sigma_safe**2))
    
    print(f"  h_z = {h_z:.1f}: logL = {log_L_test:.1f}, χ² = {chi2_test:.1f}, RMS = {np.sqrt(np.mean(residuals_test**2)):.1f}")

# Create diagnostic plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Rotation curve
ax1.scatter(R_data, v_data, alpha=0.3, s=1, color='gray', label='Data')
ax1.plot(R_data, v_model, 'r-', alpha=0.7, label='Model')
ax1.set_xlabel('R (kpc)')
ax1.set_ylabel('v (km/s)')
ax1.set_title('Rotation Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2.scatter(R_data, residuals, alpha=0.3, s=1)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('R (kpc)')
ax2.set_ylabel('Residuals (km/s)')
ax2.set_title(f'Residuals (RMS={np.sqrt(np.mean(residuals**2)):.1f})')
ax2.grid(True, alpha=0.3)

# Plot 3: Xi values
ax3.scatter(R_data, xi_values, alpha=0.3, s=1)
ax3.set_xlabel('R (kpc)')
ax3.set_ylabel('ξ(ρ)')
ax3.set_title('Density-Dependent Modification')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_likelihood.png', dpi=150)
print(f"\nDiagnostic plot saved as 'debug_likelihood.png'")

# Summary
print(f"\nSUMMARY:")
if log_L < -1e6:
    print("❌ LIKELIHOOD IS EXTREMELY NEGATIVE")
    print("Possible causes:")
    print("1. Very large residuals compared to uncertainties")
    print("2. Model completely wrong for the data")  
    print("3. Units mismatch")
    print("4. Wrong data/parameter combination")
elif log_L < -1e4:
    print("⚠️  LIKELIHOOD IS VERY NEGATIVE") 
    print("Model fits poorly but calculation seems reasonable")
else:
    print("✅ LIKELIHOOD SEEMS REASONABLE")
    print("Should work fine for MCMC/dynesty")

# Data units check
print(f"\nUnits check:")
print(f"R_data type: {type(R_data[0])}, example: {R_data[0]}")
print(f"v_data type: {type(v_data[0])}, example: {v_data[0]}")
print(f"sigma_data type: {type(sigma_data[0])}, example: {sigma_data[0]}")