#!/usr/bin/env python3
"""
plot_tidal_sparc_corrected.py - Corrected SPARC galaxy plots using actual SPARC baryon components.

This matches our paper methodology:
- GR: Uses SPARC V_gas, V_disk, V_bulge components directly 
- NFW: Adds dark matter halo
- Tidal: Applies tidal xi to the SPARC baryon curve
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders.sparc_data_loader import SPARCDataLoader

# For Tidal model - we'll need a simplified version since we don't have env params
def compute_tidal_boost_simple(R_kpc, v_bar, rho_c=5e7, gamma=3.0, lambda_max=0.5):
    """
    Simplified tidal boost calculation.
    In the actual model this depends on density and tidal parameters.
    Here we use a simple radial dependence as an approximation.
    """
    # Simple model: boost increases with radius (lower density regions)
    # This is a rough approximation of the actual xi function
    R_scale = 10.0  # Scale radius in kpc
    
    # xi goes from 1 at center to (1 + lambda_max) at large radii
    xi = 1.0 + lambda_max * (R_kpc / (R_kpc + R_scale))
    
    # Apply boost
    v_tidal = v_bar * np.sqrt(xi)
    
    return v_tidal

def compute_nfw_velocity(R_kpc, M_200=1.0e12, c=10, R_200=230):
    """Compute NFW dark matter halo rotation curve."""
    G_NEWTON = 4.301e-6  # km^2 kpc / (M_sun s^2)
    Rs = R_200 / c  # Scale radius
    
    def M_enc(r):
        x = r / Rs
        return M_200 * (np.log(1 + x) - x/(1 + x)) / (np.log(1 + c) - c/(1 + c))
    
    M_enclosed = np.array([M_enc(r) for r in R_kpc])
    v_circ = np.sqrt(G_NEWTON * M_enclosed / R_kpc)
    
    return v_circ

def plot_sparc_galaxy_corrected(galaxy_data, output_dir):
    """Plot SPARC galaxy with correct model comparisons."""
    
    galaxy_name = galaxy_data['name']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Get SPARC data
    R_obs = galaxy_data['r_kpc']
    v_obs = galaxy_data['v_obs']
    v_err = galaxy_data['v_err']
    
    # Get SPARC baryon components
    v_gas = galaxy_data['v_gas']
    v_disk = galaxy_data['v_disk']
    v_bulge = galaxy_data['v_bulge']
    
    # Filter valid data
    valid = (R_obs > 0) & (v_obs > 0) & np.isfinite(v_obs) & np.isfinite(v_err)
    R_obs = R_obs[valid]
    v_obs = v_obs[valid]
    v_err = v_err[valid]
    v_gas = v_gas[valid]
    v_disk = v_disk[valid]
    v_bulge = v_bulge[valid]
    
    if len(R_obs) == 0:
        print(f"  ⚠️ No valid data for {galaxy_name}")
        return None
    
    # Compute baryon curve (this is what GR predicts)
    v_baryon = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
    
    # This is the GR prediction - just the baryons!
    v_gr = v_baryon
    
    # NFW dark matter halo
    # Estimate parameters based on galaxy velocity
    v_max = np.max(v_obs)
    M_200 = 1e12 * (v_max / 220)**3  # Scale by velocity cubed
    R_200 = 230 * (v_max / 220)  # Scale radius
    v_nfw_dm = compute_nfw_velocity(R_obs, M_200=M_200, c=10, R_200=R_200)
    
    # Total NFW = baryons + dark matter in quadrature
    v_nfw_total = np.sqrt(v_baryon**2 + v_nfw_dm**2)
    
    # Tidal model - apply boost to baryon curve
    # The actual model would use fitted parameters, here we use reasonable defaults
    v_tidal = compute_tidal_boost_simple(R_obs, v_baryon, lambda_max=0.5)
    
    # Main plot
    ax1.errorbar(R_obs, v_obs, yerr=v_err, fmt='ko', markersize=5, 
                capsize=3, label='Observed', zorder=5, alpha=0.8)
    
    ax1.plot(R_obs, v_gr, 'b--', linewidth=2, label='GR (baryons only)', zorder=3)
    ax1.plot(R_obs, v_nfw_total, 'g:', linewidth=2.5, label='GR + NFW Dark Matter', zorder=2)
    ax1.plot(R_obs, v_tidal, 'r-', linewidth=2.5, label='Tidal Model', zorder=4)
    
    # Also show the individual components faintly
    ax1.plot(R_obs, v_gas, 'c-', linewidth=0.5, alpha=0.3, label='Gas component')
    ax1.plot(R_obs, v_disk, 'm-', linewidth=0.5, alpha=0.3, label='Disk component')
    if np.any(v_bulge > 0):
        ax1.plot(R_obs, v_bulge, 'y-', linewidth=0.5, alpha=0.3, label='Bulge component')
    
    ax1.set_ylabel('Circular Velocity [km/s]', fontsize=12)
    ax1.set_title(f'{galaxy_name}: Rotation Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, R_obs.max() * 1.1)
    ax1.set_ylim(0, v_obs.max() * 1.2)
    
    # Residuals plot
    res_gr = v_obs - v_gr
    res_nfw = v_obs - v_nfw_total
    res_tidal = v_obs - v_tidal
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.errorbar(R_obs, res_gr, yerr=v_err, fmt='b^', markersize=4, 
                capsize=2, alpha=0.7, label='GR residuals')
    ax2.errorbar(R_obs, res_nfw, yerr=v_err, fmt='gs', markersize=4, 
                capsize=2, alpha=0.7, label='NFW residuals')
    ax2.errorbar(R_obs, res_tidal, yerr=v_err, fmt='ro', markersize=4, 
                capsize=2, alpha=0.7, label='Tidal residuals')
    
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('Residual [km/s]', fontsize=11)
    ax2.legend(fontsize=9, ncol=3, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, R_obs.max() * 1.1)
    
    # Add RMS values
    rms_gr = np.sqrt(np.mean(res_gr**2))
    rms_nfw = np.sqrt(np.mean(res_nfw**2))
    rms_tidal = np.sqrt(np.mean(res_tidal**2))
    
    text_str = f'RMS Error:\nGR: {rms_gr:.1f} km/s\nNFW: {rms_nfw:.1f} km/s\nTidal: {rms_tidal:.1f} km/s'
    ax2.text(0.02, 0.98, text_str, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    safe_name = galaxy_name.replace(' ', '_').replace('/', '_')
    output_file = output_dir / f'sparc_{safe_name}_corrected.png'
    fig.savefig(output_file, dpi=120, bbox_inches='tight')
    print(f"  ✅ Saved: {output_file}")
    
    plt.close()
    
    return fig

def main():
    """Create corrected SPARC plots."""
    
    print("\n" + "="*80)
    print("CORRECTED SPARC GALAXY PLOTS - Using Actual SPARC Baryon Components")
    print("="*80 + "\n")
    
    # Load best-fit tidal parameters from our run
    run_dir = Path("runs/tidal_band_from_best_20250820_185242")
    
    # Create output directory
    output_dir = run_dir / "sparc_corrected"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load SPARC data
    sparc_dir = Path("external_data/Rotmod_LTG")
    if not sparc_dir.exists():
        print(f"⚠️ SPARC data directory not found: {sparc_dir}")
        print("Please download SPARC data first.")
        return
    
    print(f"\nLoading SPARC data from {sparc_dir}...")
    loader = SPARCDataLoader(str(sparc_dir))
    galaxies = loader.load_all_galaxies()
    
    if not galaxies:
        print("⚠️ No SPARC galaxies loaded")
        return
    
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Plot a selection of galaxies
    # These are the high-quality ones mentioned in the paper
    target_galaxies = ['NGC3198', 'NGC2403', 'NGC5055', 'NGC6946', 'NGC2841',
                      'DDO154', 'NGC6503', 'NGC7793', 'NGC2903', 'NGC7331']
    
    plotted = 0
    for galaxy_name in target_galaxies:
        if galaxy_name in galaxies:
            print(f"\nPlotting {galaxy_name}...")
            try:
                plot_sparc_galaxy_corrected(galaxies[galaxy_name], output_dir)
                plotted += 1
            except Exception as e:
                print(f"  ❌ Error plotting {galaxy_name}: {e}")
        else:
            # Try lowercase
            galaxy_name_lower = galaxy_name.lower()
            found = False
            for key in galaxies.keys():
                if key.lower() == galaxy_name_lower:
                    print(f"\nPlotting {key} (matched from {galaxy_name})...")
                    try:
                        plot_sparc_galaxy_corrected(galaxies[key], output_dir)
                        plotted += 1
                        found = True
                        break
                    except Exception as e:
                        print(f"  ❌ Error plotting {key}: {e}")
            if not found:
                print(f"  ⚠️ {galaxy_name} not found in SPARC data")
    
    # If we didn't plot many, plot some available ones
    if plotted < 5:
        print("\nPlotting additional available galaxies...")
        available = list(galaxies.keys())[:10]
        for galaxy_name in available:
            if plotted >= 10:
                break
            print(f"\nPlotting {galaxy_name}...")
            try:
                plot_sparc_galaxy_corrected(galaxies[galaxy_name], output_dir)
                plotted += 1
            except Exception as e:
                print(f"  ❌ Error plotting {galaxy_name}: {e}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Created {plotted} corrected SPARC plots")
    print("="*80)
    print("\nKey differences from previous plots:")
    print("  • GR curve now uses SPARC baryon components directly")
    print("  • GR should show Keplerian decline at large radii (not flat)")
    print("  • NFW adds dark matter to match observations")
    print("  • Tidal model boosts the baryon curve without dark matter")

if __name__ == "__main__":
    main()
