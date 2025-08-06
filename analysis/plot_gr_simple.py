#!/usr/bin/env python3
"""
Create simple plots for GR baseline results
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from density_metric2 import (
    v_baryon_total_newtonian_kms,
    rho_baryon_total_midplane_solar_kpc3,
    R_SUN_KPC
)

def to_scalar(value):
    """Convert JAX array or numpy array to Python scalar"""
    if hasattr(value, 'item'):
        return value.item()
    elif hasattr(value, '__len__') and len(value) == 1:
        return float(value[0])
    else:
        return float(value)

def create_gr_plots():
    """Create key plots for GR baseline"""
    
    # Load summary
    with open('chains_GR_reparameterized/gr_baseline_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Set up parameters
    params = {
        'M_disk_thin_solar': summary['derived_masses']['M_disk_thin_solar'],
        'M_disk_thick_solar': summary['derived_masses']['M_disk_thick_solar'],
        'M_bulge_solar': summary['best_fit_params']['M_bulge_solar'],
        'a_bulge_kpc': summary['best_fit_params']['a_bulge_kpc'],
        'M_gas_solar': summary['best_fit_params']['M_gas_solar'],
        'R_d_gas_kpc': summary['best_fit_params']['R_d_gas_kpc'],
        'h_z_gas_kpc': summary['best_fit_params']['h_z_gas_kpc'],
        'R_d_thin_kpc': 2.6,
        'h_z_thin_kpc': 0.3,
        'R_d_thick_kpc': 4.5,
        'h_z_thick_kpc': 0.9,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True
    }
    
    output_dir = Path('chains_GR_reparameterized/plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Main rotation curve plot
    print("Creating rotation curve plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    r = np.logspace(np.log10(0.5), np.log10(30), 200)
    v_gr = v_baryon_total_newtonian_kms(r, params)
    
    # Plot curve
    ax.plot(r, v_gr, 'b-', lw=3, label='GR (Newtonian)', zorder=5)
    
    # Add observational expectation
    ax.axhline(220, color='red', ls='--', lw=2, alpha=0.7, 
               label='Observed MW (flat ~220 km/s)')
    ax.fill_between(r, 200, 240, alpha=0.2, color='red', 
                    label='Typical MW range')
    
    # Mark solar position
    v_solar = to_scalar(v_baryon_total_newtonian_kms(R_SUN_KPC, params))
    ax.plot(R_SUN_KPC, v_solar, 'o', color='orange', markersize=10, 
            label=f'Solar position ({v_solar:.0f} km/s)', zorder=10)
    ax.axvline(R_SUN_KPC, color='orange', ls=':', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
    ax.set_ylabel('Circular Velocity (km/s)', fontsize=14)
    ax.set_xlim(0.5, 30)
    ax.set_ylim(0, 300)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '30'])
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Title with key info
    ax.set_title(f'GR Baseline: M_total = {summary["derived_masses"]["M_total_baryons"]/1e11:.0f}×10¹¹ M☉, log(Z) = {summary["logZ"]:.0f}', 
                 fontsize=16)
    
    # Add text annotations
    ax.text(15, 100, 'Keplerian\ndecline', fontsize=14, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.annotate('', xy=(20, 80), xytext=(10, 140),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gr_rotation_curve_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Component breakdown
    print("Creating component breakdown...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Individual components
    components = [
        ('disk_thin', 'Thin Disk', 'blue', '-'),
        ('disk_thick', 'Thick Disk', 'green', '--'),
        ('bulge', 'Bulge', 'red', '-.'),
        ('gas', 'Gas', 'cyan', ':')
    ]
    
    for comp, label, color, style in components:
        params_comp = params.copy()
        # Turn off all components
        for c in ['disk_thin', 'disk_thick', 'bulge', 'gas']:
            params_comp[f'include_{c}'] = False
        # Turn on just this component
        params_comp[f'include_{comp}'] = True
        
        v_comp = v_baryon_total_newtonian_kms(r, params_comp)
        mass = params[f'M_{comp}_solar'] if f'M_{comp}_solar' in params else \
               (params['M_disk_thin_solar'] if comp == 'disk_thin' else params['M_disk_thick_solar'])
        ax.plot(r, v_comp, color=color, ls=style, lw=2.5,
                label=f'{label} ({mass/1e10:.1f}×10¹⁰ M☉)')
    
    # Total
    ax.plot(r, v_gr, 'k-', lw=3, label='Total', alpha=0.8)
    ax.axhline(220, color='gray', ls='--', alpha=0.5)
    
    ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
    ax.set_ylabel('Velocity Contribution (km/s)', fontsize=14)
    ax.set_xlim(0.5, 30)
    ax.set_ylim(0, 250)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '30'])
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('GR Baseline: Velocity Contributions by Component', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gr_component_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary comparison plot
    print("Creating summary plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mass comparison
    masses = {
        'Thin Disk': summary['derived_masses']['M_disk_thin_solar']/1e10,
        'Thick Disk': summary['derived_masses']['M_disk_thick_solar']/1e10,
        'Bulge': summary['best_fit_params']['M_bulge_solar']/1e10,
        'Gas': summary['best_fit_params']['M_gas_solar']/1e10
    }
    
    colors = ['skyblue', 'lightgreen', 'salmon', 'cyan']
    bars = ax1.bar(masses.keys(), masses.values(), color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, (name, value) in zip(bars, masses.items()):
        height = bar.get_height()
        note = ' (MAX!)' if (name == 'Bulge' and value > 24.9) or (name == 'Gas' and value > 59.9) else ''
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}{note}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Mass (10¹⁰ M☉)', fontsize=12)
    ax1.set_ylim(0, 70)
    ax1.set_title('Baryon Masses (GR Baseline)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Key metrics
    metrics = {
        'v(R☉)': f"{to_scalar(v_baryon_total_newtonian_kms(R_SUN_KPC, params)):.0f} km/s",
        'v(obs)': "220 km/s",
        'Missing': f"{(220/to_scalar(v_baryon_total_newtonian_kms(R_SUN_KPC, params)))**2:.1f}×",
        'log(Z)': f"{summary['logZ']:.0f}",
        'M_total': f"{summary['derived_masses']['M_total_baryons']/1e11:.0f}×10¹¹"
    }
    
    y_pos = np.arange(len(metrics))
    for i, (key, value) in enumerate(metrics.items()):
        ax2.text(0.1, 0.9 - i*0.18, f'{key}:', fontsize=14, weight='bold', 
                transform=ax2.transAxes)
        ax2.text(0.5, 0.9 - i*0.18, value, fontsize=14, 
                transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Key Metrics', fontsize=14)
    
    plt.suptitle('GR Baseline Summary: Newtonian Gravity Requires Dark Matter', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'gr_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All plots saved to {output_dir}/")
    print("   - gr_rotation_curve_main.png")
    print("   - gr_component_breakdown.png") 
    print("   - gr_summary.png")

if __name__ == "__main__":
    create_gr_plots()