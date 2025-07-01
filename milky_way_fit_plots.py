#!/usr/bin/env python3
"""
milky_way_fit_plots.py - Generate publication-quality plots showing 
density-dependent model performance for Milky Way rotation curve
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from data_io import load_gaia
from density_metric2 import (v_baryon_total_newtonian_kms, 
                            rho_baryon_total_midplane_solar_kpc3, 
                            XI_FUNCTION_MAP)

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'figure.dpi': 150
})

print("GENERATING MILKY WAY ROTATION CURVE ANALYSIS PLOTS")
print("="*60)

# Load Gaia data
print("Loading Gaia data...")
gaia_data = load_gaia(sample_max=80000)
if gaia_data is None:
    print("ERROR: Could not load Gaia data")
    exit(1)

R_data = gaia_data['R_kpc']
v_data = gaia_data['v_obs']
sigma_data = gaia_data['sigma_v']

print(f"Loaded {len(R_data):,} stars")
print(f"R range: {np.min(R_data):.2f} - {np.max(R_data):.2f} kpc")
print(f"v range: {np.min(v_data):.1f} - {np.max(v_data):.1f} km/s")

# Fitted parameters from dynesty (your successful results)
fitted_params = {
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False,
    'include_bulge_density': False,
    'M_disk_thin_solar': 1.269e11,  # From your dynesty fit
    'R_d_thin_kpc': 4.138,          # From your dynesty fit
    'h_z_thin_kpc': 0.595,          # From your dynesty fit
    'rho_c_solar_kpc3': 1.642e9,    # From your dynesty fit
    'n_exp': 1.560                   # From your dynesty fit
}

print(f"\nUsing fitted parameters:")
print(f"  M_disk = {fitted_params['M_disk_thin_solar']:.2e} M☉")
print(f"  R_d = {fitted_params['R_d_thin_kpc']:.2f} kpc")
print(f"  h_z = {fitted_params['h_z_thin_kpc']:.3f} kpc")
print(f"  ρ_c = {fitted_params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
print(f"  n = {fitted_params['n_exp']:.2f}")

# Calculate model predictions
print("\nCalculating model predictions...")
v_newton = v_baryon_total_newtonian_kms(R_data, fitted_params)
rho_mid = rho_baryon_total_midplane_solar_kpc3(R_data, fitted_params)
xi_values = XI_FUNCTION_MAP['power'](rho_mid, fitted_params['rho_c_solar_kpc3'], fitted_params['n_exp'])
v_model = v_newton * np.sqrt(np.maximum(xi_values, 0.0))

# Calculate residuals and statistics
residuals = v_data - v_model
rms_total = np.sqrt(np.mean(residuals**2))
chi2_total = np.sum((residuals / sigma_data)**2)

print(f"Model performance:")
print(f"  RMS residual: {rms_total:.1f} km/s")
print(f"  χ² total: {chi2_total:.1f}")
print(f"  χ² per star: {chi2_total/len(R_data):.1f}")

# Create smooth curves for plotting
R_smooth = np.linspace(0.1, 25, 500)
v_newton_smooth = v_baryon_total_newtonian_kms(R_smooth, fitted_params)
rho_smooth = rho_baryon_total_midplane_solar_kpc3(R_smooth, fitted_params)
xi_smooth = XI_FUNCTION_MAP['power'](rho_smooth, fitted_params['rho_c_solar_kpc3'], fitted_params['n_exp'])
v_model_smooth = v_newton_smooth * np.sqrt(np.maximum(xi_smooth, 0.0))

print("\nGenerating plots...")

# Create the main figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3, height_ratios=[2, 1, 1])

# Plot 1: Main rotation curve
ax1 = fig.add_subplot(gs[0, :])

# Sample data for cleaner visualization (every 20th point)
sample_mask = np.arange(0, len(R_data), 20)
R_sample = R_data[sample_mask]
v_sample = v_data[sample_mask]
sigma_sample = sigma_data[sample_mask]

# Plot data with error bars (subset for clarity)
ax1.errorbar(R_sample, v_sample, yerr=sigma_sample, 
             fmt='o', color='lightgray', alpha=0.6, markersize=2, 
             elinewidth=0.5, capsize=0, label='Gaia DR3 Data')

# Plot all data as scatter (very transparent)
ax1.scatter(R_data, v_data, c='gray', alpha=0.1, s=0.5, rasterized=True)

# Plot model curves
ax1.plot(R_smooth, v_model_smooth, 'r-', linewidth=3, 
         label='Density-Dependent Model', zorder=5)
ax1.plot(R_smooth, v_newton_smooth, 'g--', linewidth=2, 
         label='Pure Newtonian (Baryons)', alpha=0.8, zorder=4)

# Highlight solar radius
ax1.axvline(8.122, color='orange', linestyle=':', linewidth=2, 
            alpha=0.7, label='Solar Radius')

# Add performance text
ax1.text(0.05, 0.95, f'RMS = {rms_total:.1f} km/s\nN = {len(R_data):,} stars', 
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel('Galactocentric Radius (kpc)')
ax1.set_ylabel('Circular Velocity (km/s)')
ax1.set_title('Milky Way Rotation Curve: Density-Dependent Model Fit', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 25)
ax1.set_ylim(50, 350)

# Plot 2: Residuals vs R
ax2 = fig.add_subplot(gs[1, 0])

# Calculate binned residuals for cleaner visualization
R_bins = np.linspace(0, 22, 45)
bin_centers = []
bin_residuals = []
bin_errors = []

for i in range(len(R_bins)-1):
    mask = (R_data >= R_bins[i]) & (R_data < R_bins[i+1])
    if np.sum(mask) > 10:  # At least 10 stars in bin
        bin_centers.append((R_bins[i] + R_bins[i+1]) / 2)
        bin_residuals.append(np.mean(residuals[mask]))
        bin_errors.append(np.std(residuals[mask]) / np.sqrt(np.sum(mask)))

ax2.scatter(R_data, residuals, alpha=0.1, s=0.3, color='gray', rasterized=True)
ax2.errorbar(bin_centers, bin_residuals, yerr=bin_errors, 
             fmt='ro', markersize=4, capsize=2, label='Binned Mean')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(rms_total, color='red', linestyle='--', alpha=0.7, label=f'±{rms_total:.1f} km/s')
ax2.axhline(-rms_total, color='red', linestyle='--', alpha=0.7)

ax2.set_xlabel('Galactocentric Radius (kpc)')
ax2.set_ylabel('Residuals (km/s)')
ax2.set_title('Model Residuals vs. Radius')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 22)

# Plot 3: Xi(ρ) modification profile
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(R_smooth, xi_smooth, 'b-', linewidth=3, label='ξ(ρ(R))')
ax3.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Newtonian (ξ=1)')
ax3.axvline(8.122, color='orange', linestyle=':', alpha=0.7)

ax3.set_xlabel('Galactocentric Radius (kpc)')
ax3.set_ylabel('Gravitational Modification ξ')
ax3.set_title('Density-Dependent Modification')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 25)
ax3.set_ylim(0, 1.1)

# Plot 4: Radial performance comparison
ax4 = fig.add_subplot(gs[2, :])

# Calculate performance in radial bins
R_centers = [4, 6, 8, 10, 12, 15, 18]
radial_performance = []

for R_center in R_centers:
    mask = np.abs(R_data - R_center) < 1.0  # ±1 kpc window
    if np.sum(mask) > 50:
        v_obs_bin = v_data[mask]
        v_mod_bin = v_model[mask]
        n_stars = np.sum(mask)
        rms_bin = np.sqrt(np.mean((v_obs_bin - v_mod_bin)**2))
        mean_obs = np.mean(v_obs_bin)
        mean_mod = np.mean(v_mod_bin)
        
        radial_performance.append({
            'R': R_center,
            'n_stars': n_stars,
            'rms': rms_bin,
            'mean_obs': mean_obs,
            'mean_mod': mean_mod,
            'bias': mean_mod - mean_obs
        })

if radial_performance:
    R_perf = [p['R'] for p in radial_performance]
    rms_perf = [p['rms'] for p in radial_performance]
    n_stars_perf = [p['n_stars'] for p in radial_performance]
    
    # Bar plot of RMS performance
    bars = ax4.bar(R_perf, rms_perf, width=1.5, alpha=0.7, 
                   color='skyblue', edgecolor='navy', linewidth=1)
    
    # Add star count labels on bars
    for i, (bar, n_stars) in enumerate(zip(bars, n_stars_perf)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{n_stars:,}', ha='center', va='bottom', fontsize=10)
    
    # Add overall RMS line
    ax4.axhline(rms_total, color='red', linestyle='-', linewidth=2, 
                label=f'Overall RMS = {rms_total:.1f} km/s')

ax4.set_xlabel('Galactocentric Radius (kpc)')
ax4.set_ylabel('RMS Residual (km/s)')
ax4.set_title('Radial Performance (numbers show star counts in each bin)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, max(rms_perf) * 1.2 if radial_performance else 50)

plt.suptitle('Milky Way Density-Dependent Gravity Model: Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Save the plot
output_filename = 'milky_way_density_model_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved comprehensive analysis plot: {output_filename}")

# Create a second figure focused on the rotation curve comparison
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Sample data for cleaner visualization
sample_step = max(1, len(R_data) // 2000)  # Show ~2000 points max
R_sample = R_data[::sample_step]
v_sample = v_data[::sample_step]
sigma_sample = sigma_data[::sample_step]

# Plot data
ax.scatter(R_data, v_data, c='lightgray', alpha=0.3, s=0.5, 
           label=f'Gaia DR3 Data ({len(R_data):,} stars)', rasterized=True)

# Plot model curves with confidence band
v_upper = v_model_smooth * 1.05  # Simple 5% uncertainty band
v_lower = v_model_smooth * 0.95
ax.fill_between(R_smooth, v_lower, v_upper, color='red', alpha=0.2, 
                label='Model Uncertainty')

ax.plot(R_smooth, v_model_smooth, 'r-', linewidth=3, 
        label='Density-Dependent Model', zorder=5)
ax.plot(R_smooth, v_newton_smooth, 'g--', linewidth=2, 
        label='Newtonian (Baryons Only)', alpha=0.8)

# Add classic flat rotation curve reference
flat_v = 220  # km/s
ax.axhline(flat_v, color='purple', linestyle=':', linewidth=2, 
           alpha=0.7, label='Flat Curve (220 km/s)')

# Highlight key regions
ax.axvspan(7, 9, alpha=0.1, color='orange', label='Solar Neighborhood')

# Add annotations
ax.annotate('Inner Galaxy\n(High ρ, ξ < 1)', xy=(3, 180), xytext=(5, 120),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

ax.annotate('Outer Galaxy\n(Low ρ, ξ ≈ 1)', xy=(18, 210), xytext=(20, 280),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Performance text box
textstr = f'''Model Performance:
• RMS Residual: {rms_total:.1f} km/s
• Data Points: {len(R_data):,} stars
• Radius Range: {np.min(R_data):.1f}-{np.max(R_data):.1f} kpc
• Disk Mass: {fitted_params['M_disk_thin_solar']:.2e} M☉
• Critical Density: {fitted_params['rho_c_solar_kpc3']:.1e} M☉/kpc³'''

ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
ax.set_ylabel('Circular Velocity (km/s)', fontsize=14)
ax.set_title('Milky Way Rotation Curve: Density-Dependent vs. Newtonian Gravity', 
             fontsize=16, fontweight='bold')
ax.legend(loc='center right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)
ax.set_ylim(50, 350)

# Save rotation curve plot
output_filename2 = 'milky_way_rotation_curve_comparison.png'
plt.savefig(output_filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved rotation curve comparison: {output_filename2}")

# Print summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"="*40)
print(f"Overall Performance:")
print(f"  RMS residual: {rms_total:.1f} km/s")
print(f"  Mean |residual|: {np.mean(np.abs(residuals)):.1f} km/s")
print(f"  Max |residual|: {np.max(np.abs(residuals)):.1f} km/s")
print(f"  χ² per star: {chi2_total/len(R_data):.1f}")

# Radial performance summary
if radial_performance:
    print(f"\nRadial Performance:")
    for p in radial_performance:
        print(f"  R ≈ {p['R']:2d} kpc: RMS = {p['rms']:5.1f} km/s "
              f"(N = {p['n_stars']:4d}, bias = {p['bias']:+5.1f})")

print(f"\nModel Parameters:")
print(f"  Disk mass: {fitted_params['M_disk_thin_solar']:.2e} M☉")
print(f"  Scale length: {fitted_params['R_d_thin_kpc']:.2f} kpc")
print(f"  Scale height: {fitted_params['h_z_thin_kpc']:.3f} kpc")
print(f"  Critical density: {fitted_params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
print(f"  Power index: {fitted_params['n_exp']:.2f}")

print(f"\nξ(ρ) range across galaxy:")
print(f"  Inner (R~2 kpc): ξ ≈ {xi_smooth[np.argmin(np.abs(R_smooth - 2))]:.3f}")
print(f"  Solar (R~8 kpc): ξ ≈ {xi_smooth[np.argmin(np.abs(R_smooth - 8))]:.3f}")
print(f"  Outer (R~20 kpc): ξ ≈ {xi_smooth[np.argmin(np.abs(R_smooth - 20))]:.3f}")

print(f"\nPlots generated successfully!")
print(f"Files created:")
print(f"  - {output_filename}")
print(f"  - {output_filename2}")