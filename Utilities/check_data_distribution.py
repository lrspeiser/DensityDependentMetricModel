#!/usr/bin/env python3
"""
check_data_distribution.py - Check what data we're actually using
"""
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_gaia

# Load data
print("Loading Gaia data...")
gaia_data = load_gaia(sample_max=1000)

R_data = gaia_data["R_kpc"]
v_data = gaia_data["v_obs"]
sigma_data = gaia_data["sigma_v"]

print("\nData Statistics:")
print(f"Number of stars: {len(R_data)}")
print(f"R range: {R_data.min():.2f} - {R_data.max():.2f} kpc")
print(f"R mean ± std: {R_data.mean():.2f} ± {R_data.std():.2f} kpc")
print(f"v range: {v_data.min():.1f} - {v_data.max():.1f} km/s")
print(f"v mean ± std: {v_data.mean():.1f} ± {v_data.std():.1f} km/s")
print(f"σ_v mean: {sigma_data.mean():.1f} km/s")

# Check R distribution
print(f"\nR distribution:")
print(f"Stars with R < 6 kpc: {np.sum(R_data < 6)}")
print(f"Stars with 6 < R < 10 kpc: {np.sum((R_data > 6) & (R_data < 10))}")
print(f"Stars with R > 10 kpc: {np.sum(R_data > 10)}")

# Plot distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# R histogram
ax = axes[0, 0]
ax.hist(R_data, bins=50, alpha=0.7)
ax.set_xlabel('R (kpc)')
ax.set_ylabel('Count')
ax.set_title('Radial Distribution')

# v histogram
ax = axes[0, 1]
ax.hist(v_data, bins=50, alpha=0.7)
ax.set_xlabel('v (km/s)')
ax.set_ylabel('Count')
ax.set_title('Velocity Distribution')

# R vs v scatter
ax = axes[1, 0]
ax.scatter(R_data, v_data, alpha=0.5, s=10)
ax.set_xlabel('R (kpc)')
ax.set_ylabel('v (km/s)')
ax.set_title('Rotation Curve')
ax.grid(True, alpha=0.3)

# v vs sigma_v
ax = axes[1, 1]
ax.scatter(v_data, sigma_data, alpha=0.5, s=10)
ax.set_xlabel('v (km/s)')
ax.set_ylabel('σ_v (km/s)')
ax.set_title('Velocity vs Uncertainty')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=150)
plt.show()

# If data is too concentrated, suggest using synthetic data
if R_data.std() < 1.0:
    print("\n⚠️  WARNING: Data appears to be concentrated at a single radius!")
    print("This might be the simplified test data from fix_gaia_timeout.py")
    print("Consider using create_test_gaia_data.py for more realistic synthetic data.")