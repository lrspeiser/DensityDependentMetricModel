#!/usr/bin/env python3
"""
Generate comprehensive comparison plots from model results.
Creates rotation curves at different radii and validates against academic models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append('..')

# Import physics modules
from core.density_metric_cupy import (
    v_baryon_total_newtonian_kms_cupy,
    volume_density_total_midplane_solar_kpc3_cupy,
    v_baryon_comprehensive_kms_cupy,
    volume_density_comprehensive_solar_kpc3_cupy,
    v_total_kms_cupy as v_total_core,
    xi_power_law_cupy,
    xi_exponential_cupy,
    xi_gravitational_color_cupy,
    xi_logistic_law_cupy,
    xi_gaussian_enhancement_cupy,
    xi_mond_like_cupy,
    DEFAULT_DTYPE
)
import cupy as cp

# Physical constants
G = 4.30091e-6  # kpc (km/s)^2 / Msun

# Fixed baryon parameters (from literature)
BARYON_PARAMS = {
    'M_disk_thin_solar': 4.0e10,  # M_sun
    'M_disk_thick_solar': 1.5e10,
    'M_bulge_solar': 1.2e10,
    'M_gas_solar': 3.0e10,
    'R_d_thin_kpc': 2.6,
    'R_d_thick_kpc': 4.5,
    'R_d_gas_kpc': 7.0,
    'a_bulge_kpc': 0.7,
    'h_z_thin_kpc': 0.3,
    'h_z_thick_kpc': 0.9,
    'h_z_gas_kpc': 0.15
}

def load_model_results(analysis_dir):
    """Load all model results from analysis directory."""
    analysis_path = Path(analysis_dir)
    
    # Load combined results
    results_file = analysis_path / 'all_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    
    # Otherwise load individual results
    results = {}
    for model_dir in analysis_path.iterdir():
        if model_dir.is_dir():
            summary_file = model_dir / 'results_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    results[model_dir.name] = json.load(f)
    
    return results

def calculate_rotation_curve(R_kpc, model_name, params):
    """Calculate rotation curve for a given model using comprehensive baryons and
    fitter-consistent formulas (no mock data)."""

    R_gpu = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)

    # Comprehensive baryonic setup (matches fitter defaults)
    p_baryons = {
        'M_thin_disk_solar': BARYON_PARAMS['M_disk_thin_solar'],
        'R_thin_disk_kpc': BARYON_PARAMS['R_d_thin_kpc'],
        'hz_thin_disk_kpc': BARYON_PARAMS['h_z_thin_kpc'],
        'M_thick_disk_solar': BARYON_PARAMS['M_disk_thick_solar'],
        'R_thick_disk_kpc': BARYON_PARAMS['R_d_thick_kpc'],
        'hz_thick_disk_kpc': BARYON_PARAMS['h_z_thick_kpc'],
        'M_bulge_solar': BARYON_PARAMS['M_bulge_solar'],
        'R_bulge_kpc': BARYON_PARAMS['a_bulge_kpc'],
        'M_gas_solar': BARYON_PARAMS['M_gas_solar'],
        'R_gas_kpc': BARYON_PARAMS['R_d_gas_kpc'],
        'hz_gas_kpc': BARYON_PARAMS['h_z_gas_kpc'],
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }

    # Newtonian baryon-only baseline
    v_baryon = v_baryon_comprehensive_kms_cupy(R_gpu, p_baryons)

    if model_name == 'gr':
        # Pure GR - baryons only
        return cp.asnumpy(v_baryon)

    if model_name == 'nfw':
        # NFW halo consistent with fitter (M_vir, c_vir)
        M_vir = cp.asarray(params.get('M_vir', 1.0e12), dtype=DEFAULT_DTYPE)
        c_vir = cp.asarray(params.get('c_vir', 12.0), dtype=DEFAULT_DTYPE)

        rho_crit = cp.asarray(100.0, dtype=DEFAULT_DTYPE)  # Msun/kpc^3 (approx)
        R_vir = cp.power(M_vir / (200.0 * rho_crit * (4.0 * cp.pi / 3.0)), 1.0/3.0)
        r_s = R_vir / cp.maximum(c_vir, cp.asarray(1e-6, dtype=DEFAULT_DTYPE))
        x = cp.clip(R_gpu / cp.maximum(r_s, cp.asarray(1e-6, dtype=DEFAULT_DTYPE)), 1e-8, cp.inf)

        g_x = cp.log1p(x) - x / (1.0 + x)
        g_c = cp.log1p(c_vir) - c_vir / (1.0 + c_vir)
        g_c = cp.maximum(g_c, cp.asarray(1e-12, dtype=DEFAULT_DTYPE))
        M_enc = M_vir * g_x / g_c

        v_nfw_sq = G * M_enc / cp.maximum(R_gpu, cp.asarray(1e-6, dtype=DEFAULT_DTYPE))
        v_total = cp.sqrt(cp.maximum(v_baryon**2 + v_nfw_sq, 0.0))
        return cp.asnumpy(v_total)

    # Experimental and registry-backed models: use the core v_total for consistency
    registry_models = {
        'grav_color', 'grav_color_void_safe', 'balanced_screening',
        'tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor',
        'rar_gate', 'rar_blend'
    }

    if model_name in registry_models:
        p_xi = dict(p_baryons)
        # Pass through fitted xi hyperparameters
        p_xi.update(params)
        # Allow experimental xi where needed
        p_xi['allow_experimental'] = True
        v_total = v_total_core(R_gpu, p_xi, xi_type=model_name)
        return cp.asnumpy(v_total)

    # Simple xi(ρ) family computed against comprehensive baryons
    rho = volume_density_comprehensive_solar_kpc3_cupy(R_gpu, p_baryons)
    rho_c = params.get('rho_c_solar_kpc3', params.get('rho_c', 1e8))

    if model_name == 'power':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_power_law_cupy(rho, rho_c, n_exp, A)
    elif model_name == 'exponential':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_exponential_cupy(rho, rho_c, n_exp, A)
    elif model_name == 'grav_color':  # kept for completeness; registry path handles preferred logic
        gamma = params.get('gamma_exp', params.get('gamma', 2.7))
        lambda_g = params.get('lambda_g', 8.0)
        xi = xi_gravitational_color_cupy(rho, rho_c, gamma, lambda_g)
    elif model_name == 'logistic':
        n_exp = params.get('n_exp', 2.0)
        A = params.get('A', 1.0)
        xi = xi_logistic_law_cupy(rho, rho_c, n_exp, A)
    elif model_name == 'gaussian':
        sigma_log = params.get('sigma_log', 1.0)
        A = params.get('A', 1.0)
        xi = xi_gaussian_enhancement_cupy(rho, rho_c, sigma_log, A)
    elif model_name == 'mond':
        n_exp = params.get('n_exp', 2.0)
        xi = xi_mond_like_cupy(rho, rho_c, n_exp)
    else:
        xi = cp.ones_like(R_gpu)

    v_total = v_baryon * cp.sqrt(cp.maximum(xi, 0.0))
    return cp.asnumpy(v_total)

def generate_comparison_plots(analysis_dir, output_dir=None):
    """Generate comprehensive comparison plots."""
    
    print("Loading model results...")
    results = load_model_results(analysis_dir)
    
    if output_dir is None:
        output_dir = Path(analysis_dir) / 'comparison_plots'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define radial ranges for plotting
    R_full = np.logspace(np.log10(0.5), np.log10(30), 200)  # 0.5 to 30 kpc
    
    # Specific distances to highlight
    highlight_radii = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25]  # kpc
    
    # Create main comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('Model Comparison: Galactic Rotation Curves', fontsize=16, fontweight='bold')
    
    # Color map for models
    colors = {
        'gr': 'black',
        'nfw': 'darkblue',
        'power': 'red',
        'exponential': 'orange',
        'grav_color': 'green',
        'logistic': 'purple',
        'gaussian': 'pink',
        'mond': 'brown'
    }
    
    # Plot 1: Full rotation curves
    ax = axes[0, 0]
    for model_name, model_result in results.items():
        if model_result.get('success') and model_result.get('best_params'):
            v_model = calculate_rotation_curve(R_full, model_name, model_result['best_params'])
            label = f"{model_name} (RMSE={model_result.get('rmse', 0):.1f})"
            ax.plot(R_full, v_model, color=colors.get(model_name, 'gray'), 
                   label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('v (km/s)', fontsize=12)
    ax.set_title('Full Rotation Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 350)
    
    # Plot 2: Inner galaxy (0-10 kpc)
    ax = axes[0, 1]
    R_inner = np.linspace(0.5, 10, 100)
    for model_name, model_result in results.items():
        if model_result.get('success') and model_result.get('best_params'):
            v_model = calculate_rotation_curve(R_inner, model_name, model_result['best_params'])
            ax.plot(R_inner, v_model, color=colors.get(model_name, 'gray'), 
                   label=model_name, linewidth=2, alpha=0.8)
    
    # Mark solar radius
    ax.axvline(8.0, color='gold', linestyle='--', alpha=0.5, label='Solar radius')
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('v (km/s)', fontsize=12)
    ax.set_title('Inner Galaxy Detail', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # Plot 3: Residuals from NFW
    ax = axes[1, 0]
    if 'nfw' in results and results['nfw'].get('success'):
        v_nfw = calculate_rotation_curve(R_full, 'nfw', results['nfw']['best_params'])
        
        for model_name, model_result in results.items():
            if model_result.get('success') and model_result.get('best_params') and model_name != 'nfw':
                v_model = calculate_rotation_curve(R_full, model_name, model_result['best_params'])
                residual = v_model - v_nfw
                ax.plot(R_full, residual, color=colors.get(model_name, 'gray'), 
                       label=model_name, linewidth=2, alpha=0.8)
    
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('Δv from NFW (km/s)', fontsize=12)
    ax.set_title('Residuals from NFW Model', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    # Plot 4: Xi factor comparison
    ax = axes[1, 1]
    R_xi = np.linspace(1, 20, 100)
    
    # Calculate density for xi using comprehensive baryons
    p_baryons = {
        'M_thin_disk_solar': BARYON_PARAMS['M_disk_thin_solar'],
        'R_thin_disk_kpc': BARYON_PARAMS['R_d_thin_kpc'],
        'hz_thin_disk_kpc': BARYON_PARAMS['h_z_thin_kpc'],
        'M_thick_disk_solar': BARYON_PARAMS['M_disk_thick_solar'],
        'R_thick_disk_kpc': BARYON_PARAMS['R_d_thick_kpc'],
        'hz_thick_disk_kpc': BARYON_PARAMS['h_z_thick_kpc'],
        'M_bulge_solar': BARYON_PARAMS['M_bulge_solar'],
        'R_bulge_kpc': BARYON_PARAMS['a_bulge_kpc'],
        'M_gas_solar': BARYON_PARAMS['M_gas_solar'],
        'R_gas_kpc': BARYON_PARAMS['R_d_gas_kpc'],
        'hz_gas_kpc': BARYON_PARAMS['h_z_gas_kpc'],
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }
    rho = volume_density_comprehensive_solar_kpc3_cupy(cp.asarray(R_xi, dtype=DEFAULT_DTYPE), p_baryons)
    
    for model_name in ['power', 'exponential', 'grav_color']:
        if model_name in results and results[model_name].get('success'):
            params = results[model_name]['best_params']
            rho_c = params.get('rho_c_solar_kpc3', params.get('rho_c', 1e8))
            
            if model_name == 'power':
                xi = xi_power_law_cupy(rho, rho_c, params.get('n_exp', 2.0), params.get('A', 1.0))
            elif model_name == 'exponential':
                xi = xi_exponential_cupy(rho, rho_c, params.get('n_exp', 2.0), params.get('A', 1.0))
            elif model_name == 'grav_color':
                xi = xi_gravitational_color_cupy(rho, rho_c, 
                                                params.get('gamma_exp', params.get('gamma', 2.7)),
                                                params.get('lambda_g', 8.0))
            
            ax.plot(R_xi, cp.asnumpy(xi), color=colors.get(model_name, 'gray'),
                   label=model_name, linewidth=2, alpha=0.8)
    
    ax.axhline(1, color='black', linestyle='--', alpha=0.3, label='GR (ξ=1)')
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('ξ factor', fontsize=12)
    ax.set_title('Metric Modification Factor', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 20)
    
    # Plot 5: Velocity at specific radii
    ax = axes[2, 0]
    bar_width = 0.8 / len(results)
    x_pos = np.arange(len(highlight_radii))
    
    for i, (model_name, model_result) in enumerate(results.items()):
        if model_result.get('success') and model_result.get('best_params'):
            velocities = []
            for r in highlight_radii:
                v = calculate_rotation_curve(np.array([r]), model_name, model_result['best_params'])[0]
                velocities.append(v)
            
            ax.bar(x_pos + i * bar_width, velocities, bar_width, 
                  label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('v (km/s)', fontsize=12)
    ax.set_title('Velocities at Specific Radii', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width * (len(results) - 1) / 2)
    ax.set_xticklabels(highlight_radii)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Model performance metrics
    ax = axes[2, 1]
    
    # Extract metrics
    model_names = []
    rmse_values = []
    chi2_values = []
    
    for name, result in results.items():
        if result.get('success') and result.get('rmse'):
            model_names.append(name)
            rmse_values.append(result['rmse'])
            chi2 = result.get('chi2')
            if chi2 is not None:
                chi2_values.append(chi2 / 1000)  # Scale for display
            else:
                chi2_values.append(0)  # Default value if chi2 not available
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE (km/s)', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, chi2_values, width, label='χ²/1000', color='coral', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RMSE (km/s)', fontsize=12, color='steelblue')
    ax2.set_ylabel('χ²/1000', fontsize=12, color='coral')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Add value labels on bars
    for bar, val in zip(bars1, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save main comparison plot
    plot_file = output_dir / 'model_comparison_full.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_file}")
    
    # Create academic validation plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Academic Model Validation', fontsize=16, fontweight='bold')
    
    # GR vs NFW comparison
    ax = axes[0]
    if 'gr' in results and 'nfw' in results:
        v_gr = calculate_rotation_curve(R_full, 'gr', results['gr'].get('best_params', {}))
        v_nfw = calculate_rotation_curve(R_full, 'nfw', results['nfw'].get('best_params', {}))
        
        ax.plot(R_full, v_gr, 'k-', label='GR (Baryons only)', linewidth=2)
        ax.plot(R_full, v_nfw, 'b-', label='NFW (Baryons + DM)', linewidth=2)
        
        # Show the dark matter contribution
        v_dm = np.sqrt(np.maximum(v_nfw**2 - v_gr**2, 0))
        ax.plot(R_full, v_dm, 'b--', label='DM contribution', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('R (kpc)', fontsize=12)
        ax.set_ylabel('v (km/s)', fontsize=12)
        ax.set_title('Standard Model Components', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 300)
    
    # Best alternative vs NFW
    ax = axes[1]
    if 'nfw' in results and results['nfw'].get('success'):
        v_nfw = calculate_rotation_curve(R_full, 'nfw', results['nfw']['best_params'])
        ax.plot(R_full, v_nfw, 'b-', label='NFW (Standard)', linewidth=2)
        
        # Find best alternative
        alt_models = [(n, r) for n, r in results.items() 
                     if r.get('success') and r.get('rmse') and n not in ['gr', 'nfw']]
        if alt_models:
            best_alt = min(alt_models, key=lambda x: x[1]['rmse'])
            v_alt = calculate_rotation_curve(R_full, best_alt[0], best_alt[1]['best_params'])
            ax.plot(R_full, v_alt, 'r-', 
                   label=f'{best_alt[0]} (RMSE={best_alt[1]["rmse"]:.1f})', linewidth=2)
            
            # Show difference
            diff = v_alt - v_nfw
            ax2 = ax.twinx()
            ax2.plot(R_full, diff, 'g--', alpha=0.5, linewidth=1)
            ax2.set_ylabel('Difference (km/s)', color='g', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.axhline(0, color='g', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('R (kpc)', fontsize=12)
    ax.set_ylabel('v (km/s)', fontsize=12)
    ax.set_title('Best Alternative vs Standard Model', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 300)
    
    plt.tight_layout()
    
    # Save validation plot
    validation_file = output_dir / 'academic_validation.png'
    plt.savefig(validation_file, dpi=150, bbox_inches='tight')
    print(f"Saved validation plot to {validation_file}")
    
    # Generate data table for paper
    generate_results_table(results, output_dir)
    
    plt.show()
    
    return results

def generate_results_table(results, output_dir):
    """Generate LaTeX and CSV tables of results."""
    
    # Prepare data for table
    table_data = []
    for model_name, result in results.items():
        if result.get('success'):
            row = {
                'Model': model_name,
                'RMSE (km/s)': result.get('rmse', np.nan),
                'χ²': result.get('chi2') if result.get('chi2') is not None else np.nan,
                'log Z': result.get('logz', np.nan)
            }
            
            # Add key parameters
            if result.get('best_params'):
                params = result['best_params']
                if 'rho_c_solar_kpc3' in params or 'rho_c' in params:
                    row['ρ_c (M☉/kpc³)'] = params.get('rho_c_solar_kpc3', params.get('rho_c'))
                if 'n_exp' in params:
                    row['n'] = params['n_exp']
                if 'A' in params:
                    row['A'] = params['A']
                if 'M_vir' in params:
                    row['M_vir (M☉)'] = params['M_vir']
            
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    df = df.sort_values('RMSE (km/s)')
    
    # Save CSV
    csv_file = output_dir / 'model_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved results table to {csv_file}")
    
    # Generate LaTeX table
    latex_file = output_dir / 'model_results.tex'
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Comparison Results on 144,000 Gaia DR3 Stars}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & RMSE (km/s) & $\\chi^2$ & $\\log Z$ \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            model = row['Model']
            rmse = row['RMSE (km/s)']
            chi2 = row['χ²']
            logz = row['log Z']
            
            f.write(f"{model} & {rmse:.2f} & {chi2:.0f} & {logz:.1f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to {latex_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        analysis_dir = sys.argv[1]
    else:
        # Find most recent analysis directory
        from pathlib import Path
        dirs = sorted(Path('.').glob('full_analysis_*'))
        if dirs:
            analysis_dir = dirs[-1]
        else:
            print("No analysis directory found!")
            sys.exit(1)
    
    print(f"Generating plots for {analysis_dir}")
    results = generate_comparison_plots(analysis_dir)
