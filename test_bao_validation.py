#!/usr/bin/env python3
# test_bao_validation.py

from bao_data_loader import BAODataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load your BAO data
loader = BAODataLoader('bao')
measurements = loader.load_all_measurements()
distances = loader.get_distance_measurements()

# Your DDMM parameters
rho_c = 1.64e9  # From your MW fit
n_exp = 1.56

# Plot BAO distance measurements
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Extract data for plotting
z_list = [d['z'] for d in distances if 'DV_over_rd' in d]
dv_list = [d['DV_over_rd'] for d in distances if 'DV_over_rd' in d]
dv_err = [d['DV_over_rd_err'] for d in distances if 'DV_over_rd' in d]

# Upper panel: Distance measurements
ax1.errorbar(z_list, dv_list, yerr=dv_err, fmt='o', label='SDSS BAO')
ax1.set_ylabel('DV/rd')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Lower panel: Growth rate
z_growth = [d['z'] for d in distances if 'fs8' in d]
fs8_list = [d['fs8'] for d in distances if 'fs8' in d]
fs8_err = [d['fs8_err'] for d in distances if 'fs8' in d]

ax2.errorbar(z_growth, fs8_list, yerr=fs8_err, fmt='s', color='red', label='SDSS fs8')
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('f(z)σ8(z)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('SDSS BAO Measurements')
plt.tight_layout()
plt.savefig('bao_measurements.png')
print("Saved plot to bao_measurements.png")

# Test DDMM predictions
print("\nDDMM Predictions:")
print(f"ρ_c = {rho_c:.2e} M☉/kpc³, n = {n_exp:.2f}")

for d in distances[:3]:  # First 3 measurements
    z = d['z']
    rho_z = 1e6 * (1 + z)**3
    xi_z = 1.0 / (1.0 + (rho_z / rho_c)**n_exp)
    print(f"\nz = {z:.2f}:")
    print(f"  ρ(z) ≈ {rho_z:.2e} M☉/kpc³")
    print(f"  ξ(z) = {xi_z:.4f}")
    print(f"  Distance scaling ∝ √ξ = {np.sqrt(xi_z):.3f}")