#!/usr/bin/env python3
"""
generate_cluster_profiles.py

Generate synthetic cluster gas density and temperature profiles based on 
typical profiles from literature (beta-model for gas, polytropic for temperature).

This creates the missing cluster data files needed for lensing analysis.
Based on typical profiles from Vikhlinin+ 2006, Arnaud+ 2010, etc.

README pointing to how to get real cluster data from literature when available.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Physical constants
kpc_to_m = 3.0857e19
M_sun = 1.98847e30
m_p = 1.673e-27  # proton mass in kg

# Typical cluster parameters from literature
CLUSTER_PARAMS = {
    'ABELL_1689': {
        'z': 0.184,
        'M200_1e14': 9.0,  # 10^14 M_sun
        'r200_kpc': 1800,
        'rc_kpc': 150,     # core radius for beta model
        'beta': 0.65,      # beta model slope
        'n0_cm3': 0.01,    # central density in cm^-3
        'T0_keV': 9.0,     # central temperature
        'source': 'Limousin+ 2007, Kawaharada+ 2010'
    },
    'ABELL_2029': {
        'z': 0.0767,
        'M200_1e14': 5.0,
        'r200_kpc': 1500,
        'rc_kpc': 100,
        'beta': 0.60,
        'n0_cm3': 0.015,
        'T0_keV': 7.5,
        'source': 'Lewis+ 2003, Clarke+ 2004'
    },
    'A478': {
        'z': 0.0881,
        'M200_1e14': 5.5,
        'r200_kpc': 1550,
        'rc_kpc': 120,
        'beta': 0.63,
        'n0_cm3': 0.012,
        'T0_keV': 6.5,
        'source': 'Schmidt & Allen 2007'
    },
    'A1795': {
        'z': 0.0625,
        'M200_1e14': 4.0,
        'r200_kpc': 1400,
        'rc_kpc': 80,
        'beta': 0.58,
        'n0_cm3': 0.018,
        'T0_keV': 5.8,
        'source': 'Ettori+ 2002'
    },
    'A2029': {  # Alternative name for ABELL_2029
        'z': 0.0767,
        'M200_1e14': 5.0,
        'r200_kpc': 1500,
        'rc_kpc': 100,
        'beta': 0.60,
        'n0_cm3': 0.015,
        'T0_keV': 7.5,
        'source': 'Lewis+ 2003'
    },
    'ABELL_0426': {  # Perseus cluster
        'z': 0.0179,
        'M200_1e14': 6.0,
        'r200_kpc': 1600,
        'rc_kpc': 60,
        'beta': 0.55,
        'n0_cm3': 0.05,
        'T0_keV': 6.0,
        'source': 'Simionescu+ 2011'
    }
}

def beta_model_density(r_kpc, n0_cm3, rc_kpc, beta):
    """
    Classic beta-model for cluster gas density.
    n(r) = n0 * (1 + (r/rc)^2)^(-3*beta/2)
    
    Returns density in kg/m^3
    """
    r = np.atleast_1d(r_kpc)
    n_cm3 = n0_cm3 * (1 + (r/rc_kpc)**2)**(-1.5*beta)
    
    # Convert from cm^-3 to kg/m^3
    # Assume mean molecular weight mu = 0.6 for ionized gas
    mu = 0.6
    rho_kg_m3 = n_cm3 * 1e6 * mu * m_p
    
    return rho_kg_m3

def polytropic_temperature(r_kpc, T0_keV, rc_kpc, gamma=1.15):
    """
    Polytropic temperature profile.
    T(r) = T0 * (n(r)/n0)^(gamma-1)
    
    For simplicity, use a phenomenological form:
    T(r) = T0 * (1 + (r/rc)^2)^(-alpha)
    with alpha ~ 0.3-0.5
    
    Returns temperature in K
    """
    r = np.atleast_1d(r_kpc)
    alpha = 0.4  # typical value
    T_keV = T0_keV * (1 + (r/rc_kpc)**2)**(-alpha)
    
    # Convert keV to K (1 keV = 1.16e7 K)
    T_K = T_keV * 1.16e7
    
    return T_K

def integrate_mass(r_kpc, rho_kg_m3):
    """
    Integrate mass profile M(<r) = 4π ∫ρ(r')r'^2 dr'
    """
    r = np.atleast_1d(r_kpc)
    rho = np.atleast_1d(rho_kg_m3)
    
    # Numerical integration using trapezoidal rule
    mass_integrated = np.zeros_like(r)
    
    for i in range(len(r)):
        if i == 0:
            mass_integrated[i] = 0
        else:
            # Use finer grid for integration
            r_fine = np.linspace(0, r[i], 1000)
            rho_fine = np.interp(r_fine, r[:i+1], rho[:i+1], left=rho[0])
            
            # 4π ∫ρ r^2 dr in SI units
            integrand = 4 * np.pi * rho_fine * (r_fine * kpc_to_m)**2
            mass_integrated[i] = np.trapz(integrand, r_fine * kpc_to_m) / M_sun
    
    return mass_integrated

def generate_cluster_profile(cluster_name, output_dir='data'):
    """
    Generate gas density and temperature profiles for a cluster.
    """
    if cluster_name not in CLUSTER_PARAMS:
        print(f"Unknown cluster: {cluster_name}")
        return None
    
    params = CLUSTER_PARAMS[cluster_name]
    
    # Create radial grid (log-spaced from 1 to r200)
    r_kpc = np.logspace(0, np.log10(params['r200_kpc']), 100)
    
    # Generate density profile
    rho_kg_m3 = beta_model_density(
        r_kpc, 
        params['n0_cm3'],
        params['rc_kpc'],
        params['beta']
    )
    
    # Generate temperature profile
    T_K = polytropic_temperature(
        r_kpc,
        params['T0_keV'],
        params['rc_kpc']
    )
    
    # Integrate mass profile
    mass_integrated = integrate_mass(r_kpc, rho_kg_m3)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save gas profile
    gas_df = pd.DataFrame({
        'radius_kpc': r_kpc,
        'density_kg_m3': rho_kg_m3,
        'mass_integrated': mass_integrated
    })
    gas_file = output_path / f"{cluster_name}_gas_profile.csv"
    gas_df.to_csv(gas_file, index=False)
    print(f"Created: {gas_file}")
    
    # Save temperature profile
    temp_df = pd.DataFrame({
        'radius_kpc': r_kpc,
        'temperature_K': T_K
    })
    temp_file = output_path / f"{cluster_name}_temperature_profile.csv"
    temp_df.to_csv(temp_file, index=False)
    print(f"Created: {temp_file}")
    
    # Save clumping profile (assume smooth, no clumping)
    clump_df = pd.DataFrame({
        'radius_kpc': r_kpc,
        'clumping_factor': np.ones_like(r_kpc)  # C=1 everywhere
    })
    clump_file = output_path / f"{cluster_name}_clumping_profile.csv"
    clump_df.to_csv(clump_file, index=False)
    print(f"Created: {clump_file}")
    
    # Print summary
    print(f"\nCluster {cluster_name}:")
    print(f"  z = {params['z']}")
    print(f"  M200 = {params['M200_1e14']:.1f} × 10^14 M_sun")
    print(f"  r200 = {params['r200_kpc']:.0f} kpc")
    print(f"  rc = {params['rc_kpc']:.0f} kpc")
    print(f"  β = {params['beta']:.2f}")
    print(f"  Central density = {params['n0_cm3']:.3f} cm^-3")
    print(f"  Central temperature = {params['T0_keV']:.1f} keV")
    print(f"  Total gas mass = {mass_integrated[-1]:.2e} M_sun")
    print(f"  Source: {params['source']}")
    
    return gas_df, temp_df

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic cluster gas profiles for lensing analysis'
    )
    parser.add_argument(
        '--clusters',
        nargs='*',
        default=None,
        help='List of cluster names to generate (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data',
        help='Output directory for profiles'
    )
    args = parser.parse_args()
    
    # Determine which clusters to process
    if args.clusters:
        clusters = args.clusters
    else:
        clusters = list(CLUSTER_PARAMS.keys())
    
    print("="*60)
    print("GENERATING SYNTHETIC CLUSTER PROFILES")
    print("="*60)
    print("\nNOTE: These are synthetic profiles based on typical")
    print("parameters from literature. For accurate analysis,")
    print("obtain real observational data from:")
    print("- Chandra X-ray Observatory archive")
    print("- XMM-Newton archive")
    print("- Published catalogs (ACCEPT, HIFLUGCS)")
    print("="*60)
    
    # Generate profiles
    for cluster_name in clusters:
        print(f"\nProcessing {cluster_name}...")
        generate_cluster_profile(cluster_name, args.output_dir)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nProfiles saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Run cluster_lensing_analysis.py to test lensing")
    print("2. Replace with real observational data when available")
    print("3. Validate against known cluster lensing observations")

if __name__ == '__main__':
    main()