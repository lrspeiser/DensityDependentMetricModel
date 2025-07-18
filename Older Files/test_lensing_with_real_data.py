#!/usr/bin/env python3
# test_lensing_with_real_data.py

import numpy as np
import matplotlib.pyplot as plt
from frontier_lensing_loader import FrontierFieldsLoader
from des_y3_loader import DESY3Loader
from kids_loader import KiDSLoader

# Your DDMM parameters
rho_c = 1.64e9  # M_sun/kpc^3
n_exp = 1.56

print("Testing DDMM with real lensing data...\n")

# 1. Test Frontier Fields (MACS0416)
print("="*60)
print("1. Frontier Fields MACS0416 Cluster Lensing")
print("="*60)

ff_loader = FrontierFieldsLoader('hlsp_frontier')
kappa = ff_loader.load_convergence_map()
ff_loader.convert_to_physical_units(z_lens=0.396)

# Plot convergence map
plt.figure(figsize=(10, 8))
plt.imshow(kappa['data'], origin='lower', cmap='hot')
plt.colorbar(label='Convergence κ')
plt.title('MACS0416 Convergence Map')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.savefig('macs0416_convergence.png')
print("Saved convergence map to macs0416_convergence.png")

# For DDMM: κ should trace ρ×ξ(ρ) instead of just ρ
# High density regions might have suppressed lensing

# 2. Test DES Y3
print("\n" + "="*60)
print("2. DES Y3 Cosmic Shear")
print("="*60)

des_loader = DESY3Loader('DES_Y3')
des_data = des_loader.load_2pt_data()
shear_data = des_loader.get_cosmic_shear_data()

if shear_data:
    print(f"\nFound cosmic shear data:")
    print(f"  ξ₊: {shear_data['xi_plus'] is not None}")
    print(f"  ξ₋: {shear_data['xi_minus'] is not None}")
    print(f"  Has covariance: {shear_data['has_covariance']}")

# 3. Test KiDS
print("\n" + "="*60)
print("3. KiDS Weak Lensing")
print("="*60)

kids_loader = KiDSLoader('Kids')
# Don't load full catalog in test (it's huge), just show it works
try:
    # Get catalog info without loading everything
    from astropy.io import fits
    with fits.open('Kids/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits') as hdul:
        n_galaxies = hdul[1].header['NAXIS2']
        print(f"KiDS catalog contains {n_galaxies:,} galaxies")
except:
    pass

print("\n" + "="*60)
print("DDMM Lensing Predictions")
print("="*60)

# Example: How DDMM modifies lensing
densities = np.logspace(6, 10, 100)  # M_sun/kpc^3
xi_values = 1.0 / (1.0 + (densities / rho_c)**n_exp)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.loglog(densities, xi_values)
plt.axhline(1.0, color='k', ls='--', alpha=0.5)
plt.xlabel('ρ (M☉/kpc³)')
plt.ylabel('ξ(ρ)')
plt.title('DDMM Modification Function')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.loglog(densities, densities * xi_values, label='ρ×ξ (DDMM lensing)')
plt.loglog(densities, densities, '--', label='ρ (standard lensing)')
plt.xlabel('ρ (M☉/kpc³)')
plt.ylabel('Effective lensing density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ddmm_lensing_modification.png')
print("\nSaved DDMM lensing predictions to ddmm_lensing_modification.png")