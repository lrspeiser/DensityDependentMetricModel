"""
Cluster Lensing Analysis (Fixed Version)
=========================================
Correctly calculates Einstein radii for galaxy clusters.
Tests our geometric enhancement model against observed lensing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458    # m/s
M_sun = 1.98847e30  # kg
kpc_to_m = 3.0857e19  # m
Mpc_to_m = 3.0857e22  # m

# Cosmology (simplified flat Lambda-CDM)
H0 = 70  # km/s/Mpc
Omega_m = 0.3
Omega_L = 0.7

def angular_diameter_distance(z):
    """Angular diameter distance in Mpc for flat Lambda-CDM"""
    # Simple approximation for low z
    if z < 0.1:
        return c * z / (H0 * (1 + z))  # Mpc
    
    # More accurate for higher z
    from scipy.integrate import quad
    
    def E(z_prime):
        return np.sqrt(Omega_m * (1 + z_prime)**3 + Omega_L)
    
    integral, _ = quad(lambda zp: 1/E(zp), 0, z)
    D_c = c / H0 * integral  # Mpc
    D_a = D_c / (1 + z)
    return D_a

def load_cluster_profile(cluster_name):
    """Load gas density profile for a cluster"""
    filename = f"data/{cluster_name}_gas_profile.csv"
    if not Path(filename).exists():
        print(f"Profile not found: {filename}")
        return None
    
    df = pd.read_csv(filename)
    return df

def calculate_mass_within_r(profile_df, r_target_kpc):
    """Calculate total gas mass within radius r"""
    r_kpc = profile_df['r_kpc'].values
    rho_gas = profile_df['rho_gas'].values  # Msun/kpc^3
    
    if r_target_kpc < r_kpc[0]:
        return 0
    
    if r_target_kpc > r_kpc[-1]:
        r_target_kpc = r_kpc[-1]
    
    # Integrate mass using spherical shells
    M_total = 0
    for i in range(len(r_kpc)-1):
        if r_kpc[i+1] > r_target_kpc:
            break
        r_inner = r_kpc[i]
        r_outer = min(r_kpc[i+1], r_target_kpc)
        rho_avg = 0.5 * (rho_gas[i] + rho_gas[i+1])
        
        # Volume of spherical shell
        V_shell = 4/3 * np.pi * (r_outer**3 - r_inner**3)
        M_shell = rho_avg * V_shell
        M_total += M_shell
    
    return M_total  # Msun

def einstein_radius_from_mass(M_lens_Msun, z_lens):
    """
    Calculate Einstein radius in arcsec given lens mass and redshift.
    Assumes source at infinity (strong lensing regime).
    """
    # Convert mass to kg
    M_lens_kg = M_lens_Msun * M_sun
    
    # Angular diameter distance to lens
    D_l = angular_diameter_distance(z_lens) * Mpc_to_m  # meters
    
    # For source at high redshift (D_s >> D_l), Einstein radius is:
    # theta_E = sqrt(4GM/c^2 * D_ls/D_l/D_s) ≈ sqrt(4GM/c^2/D_l) for D_s >> D_l
    
    # This is the Einstein radius in radians
    theta_E_rad = np.sqrt(4 * G * M_lens_kg / (c**2 * D_l))
    
    # Convert to arcsec
    theta_E_arcsec = theta_E_rad * 206265
    
    return theta_E_arcsec

def analyze_cluster_lensing():
    """Analyze lensing for known clusters"""
    
    # Observed data (from baseline.md)
    clusters = {
        'abell_1689': {
            'name': 'Abell 1689',
            'z': 0.184,
            'M_gas_obs': 1.2e14,  # Msun
            'M_lens_obs': 5.8e14,  # Msun (from lensing)
            'R_E_obs': 47.0  # arcsec
        },
        'abell_2029': {
            'name': 'Abell 2029',
            'z': 0.077,
            'M_gas_obs': 0.8e14,
            'M_lens_obs': 3.2e14,
            'R_E_obs': 28.0
        },
        'a478': {
            'name': 'A478',
            'z': 0.088,
            'M_gas_obs': 0.9e14,
            'M_lens_obs': 3.5e14,
            'R_E_obs': 31.0
        },
        'macs_j0416': {
            'name': 'MACS J0416',
            'z': 0.396,
            'M_gas_obs': 1.5e14,
            'M_lens_obs': 8.2e14,
            'R_E_obs': 35.0
        },
        'bullet': {
            'name': 'Bullet',
            'z': 0.296,
            'M_gas_obs': 2.0e14,
            'M_lens_obs': 15.0e14,
            'R_E_obs': 55.0
        }
    }
    
    print("="*70)
    print("CLUSTER LENSING ANALYSIS - FIXED VERSION")
    print("="*70)
    
    results = []
    
    for cluster_id, info in clusters.items():
        print(f"\n{info['name']}:")
        print(f"  z = {info['z']:.3f}")
        print(f"  Observed M_gas = {info['M_gas_obs']:.1e} Msun")
        print(f"  Observed M_lens = {info['M_lens_obs']:.1e} Msun")
        print(f"  Observed R_E = {info['R_E_obs']:.1f} arcsec")
        
        # Calculate Einstein radius from observed lensing mass
        R_E_from_lens_mass = einstein_radius_from_mass(info['M_lens_obs'], info['z'])
        print(f"  R_E from M_lens = {R_E_from_lens_mass:.1f} arcsec")
        
        # Calculate Einstein radius from gas mass only
        R_E_from_gas = einstein_radius_from_mass(info['M_gas_obs'], info['z'])
        print(f"  R_E from M_gas = {R_E_from_gas:.1f} arcsec")
        
        # Mass discrepancy factor
        mass_factor = info['M_lens_obs'] / info['M_gas_obs']
        print(f"  Mass discrepancy = {mass_factor:.1f}x")
        
        # Enhancement needed
        enhancement = info['R_E_obs'] / R_E_from_gas
        print(f"  Lensing enhancement needed = {enhancement:.1f}x")
        
        results.append({
            'name': info['name'],
            'z': info['z'],
            'M_gas_obs': info['M_gas_obs'],
            'M_lens_obs': info['M_lens_obs'],
            'R_E_obs': info['R_E_obs'],
            'R_E_from_lens': R_E_from_lens_mass,
            'R_E_from_gas': R_E_from_gas,
            'mass_factor': mass_factor,
            'enhancement': enhancement
        })
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Einstein radii comparison
    ax = axes[0, 0]
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    
    R_E_obs = [r['R_E_obs'] for r in results]
    R_E_gas = [r['R_E_from_gas'] for r in results]
    R_E_lens = [r['R_E_from_lens'] for r in results]
    
    width = 0.25
    ax.bar(x - width, R_E_obs, width, label='Observed', color='black')
    ax.bar(x, R_E_gas, width, label='From Gas Only', color='red', alpha=0.7)
    ax.bar(x + width, R_E_lens, width, label='From Lens Mass', color='blue', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Einstein Radius [arcsec]')
    ax.set_title('Einstein Radius: Gas vs Lensing Mass')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Mass discrepancy
    ax = axes[0, 1]
    mass_factors = [r['mass_factor'] for r in results]
    
    ax.bar(x, mass_factors, color='purple', alpha=0.7)
    ax.axhline(1, color='red', linestyle='--', label='No discrepancy')
    ax.axhline(5, color='orange', linestyle=':', label='Typical MOND/DM factor')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('M_lens / M_gas')
    ax.set_title('Mass Discrepancy Factor')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Enhancement vs redshift
    ax = axes[1, 0]
    redshifts = [r['z'] for r in results]
    enhancements = [r['enhancement'] for r in results]
    
    ax.scatter(redshifts, enhancements, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name, (redshifts[i], enhancements[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Enhancement Factor (R_E_obs / R_E_gas)')
    ax.set_title('Required Lensing Enhancement vs Redshift')
    ax.grid(alpha=0.3)
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    avg_mass_factor = np.mean(mass_factors)
    avg_enhancement = np.mean(enhancements)
    
    summary = f"""
SUMMARY OF CLUSTER LENSING PROBLEM
===================================

Average mass discrepancy: {avg_mass_factor:.1f}x
(Lensing sees ~{avg_mass_factor:.0f}× more mass than gas)

Average enhancement needed: {avg_enhancement:.1f}x
(Einstein radius ~{avg_enhancement:.1f}× larger than 
expected from gas mass alone)

KEY INSIGHTS:
• MOND fails here - doesn't modify light paths
• Dark matter invokes {avg_mass_factor:.0f}× invisible mass
• Our model needs photon-matter coupling 
  difference to explain discrepancy

GEOMETRIC ENHANCEMENT MODEL:
• Environmental scalar field φ_env
• Different coupling for photons vs matter
• φ_env ~ ρ_gas^α × T^β
• Enhancement factor ~ 1 + 2φ_env
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('results/lensing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'cluster_lensing_fixed.png', dpi=150)
    plt.show()
    
    # Save data
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'cluster_lensing_fixed.csv', index=False)
    print(f"\nResults saved to {output_dir}")
    
    return results

def test_geometric_enhancement():
    """Test how geometric enhancement could explain lensing"""
    
    print("\n" + "="*70)
    print("TESTING GEOMETRIC ENHANCEMENT MODEL")
    print("="*70)
    
    # For a simple test, use Abell 1689
    z = 0.184
    M_gas = 1.2e14  # Msun
    M_lens_obs = 5.8e14  # Msun
    
    print(f"\nTest case: Abell 1689")
    print(f"  Gas mass: {M_gas:.1e} Msun")
    print(f"  Lensing mass: {M_lens_obs:.1e} Msun")
    
    # In our model, the effective lensing mass is enhanced by environmental scalar
    # M_eff = M_gas * (1 + enhancement_factor)
    
    enhancement_needed = M_lens_obs / M_gas
    print(f"  Enhancement needed: {enhancement_needed:.1f}x")
    
    # Model the enhancement as coming from density and temperature
    # φ_env = a * (ρ/ρ_crit)^α * (T/T_ref)^β
    
    # Typical values for Abell 1689
    rho_central = 0.01  # cm^-3
    rho_crit = 1e-6  # cm^-3 (cosmological critical density)
    T_central = 9.0  # keV
    T_ref = 1.0  # keV
    
    # Try different coupling parameters
    print("\nTrying different coupling parameters:")
    
    for alpha in [0.5, 1.0, 1.5]:
        for beta in [0.5, 1.0, 1.5]:
            phi_env = (rho_central/rho_crit)**alpha * (T_central/T_ref)**beta
            
            # In our model, photon deflection enhanced by factor (1 + a_photon * phi_env)
            # while matter feels (1 + a_matter * phi_env)
            # The difference gives the lensing discrepancy
            
            # If matter coupling is ~1 and photon coupling is enhanced:
            a_photon = (enhancement_needed - 1) / phi_env
            
            print(f"  α={alpha:.1f}, β={beta:.1f}: φ_env={phi_env:.2e}, a_photon={a_photon:.3f}")

if __name__ == '__main__':
    results = analyze_cluster_lensing()
    test_geometric_enhancement()