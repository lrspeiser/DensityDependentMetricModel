import numpy as np
from density_metric2 import *

def calculate_Kz_penalty(params, z_test=1.1, Kz_obs=2.3e-3, Kz_err=0.5e-3):
    """Calculate chi2 penalty for K_z constraint."""
    # Get disk parameters
    M_disk = params.get('M_disk_thin_solar', 0)
    R_d = params.get('R_d_thin_kpc', 3.0)
    h_z = params.get('h_z_thin_kpc', 0.3)
    
    # Surface density at solar radius
    Sigma_solar = (M_disk / (2 * np.pi * R_d**2)) * np.exp(-8.122/R_d)
    
    # K_z at height z
    Kz_model_kpc = 2 * np.pi * G_ASTRO_UNITS * Sigma_solar * (1 - np.exp(-z_test/h_z))
    Kz_model = Kz_model_kpc / 1000  # Convert to (km/s)²/pc
    
    # Chi-squared
    chi2 = ((Kz_model - Kz_obs) / Kz_err)**2
    
    return chi2, Kz_model

# Test with your fitted parameters
params = {
    'M_disk_thin_solar': 1.269e11,
    'R_d_thin_kpc': 4.138,
    'h_z_thin_kpc': 0.595
}

chi2, Kz_model = calculate_Kz_penalty(params)
print(f"Current model: K_z = {Kz_model:.3f} (km/s)²/pc")
print(f"Chi-squared penalty: {chi2:.1f}")
print(f"This is {np.sqrt(chi2):.0f} sigma away from observations!")

# What mass would give correct K_z?
target_mass = params['M_disk_thin_solar'] * (2.3e-3 / Kz_model)
print(f"\nTo get correct K_z, need M_disk ≈ {target_mass:.2e} M_sun")
print(f"This is {params['M_disk_thin_solar']/target_mass:.0f}× smaller than fitted value")
