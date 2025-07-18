#!/usr/bin/env python3
"""
dynesty_monitor.py - Monitor dynesty progress and parameter evolution
Safe to run while sampling is ongoing
"""
import numpy as np
import glob
import os
from pathlib import Path
import time
from datetime import datetime

def find_latest_dynesty_run():
    """Find the most recent dynesty output files"""
    chains_dir = Path("chains_dynesty")
    if not chains_dir.exists():
        return None, None
    
    # Look for sample files
    sample_files = list(chains_dir.glob("dynesty_mw_*_samples.npz"))
    if not sample_files:
        return None, None
    
    # Get the most recent one
    latest_file = max(sample_files, key=os.path.getmtime)
    
    # Infer the base name
    base_name = str(latest_file).replace("_samples.npz", "")
    
    return latest_file, base_name

def safe_load_samples(filepath):
    """Safely load samples file even if being written to"""
    try:
        # Try to load the file
        data = np.load(filepath)
        samples = data.get('samples', None)
        weights = data.get('weights', None)
        logl = data.get('logl', None)
        
        if samples is None:
            return None, None, None
            
        # Basic sanity checks
        if len(samples) == 0:
            return None, None, None
            
        return samples, weights, logl
        
    except (OSError, IOError, ValueError) as e:
        # File might be being written to
        print(f"⚠️  Could not read file (likely being written): {e}")
        return None, None, None

def get_parameter_names():
    """Get parameter names based on what components are being fitted"""
    # This is the order used in run_dynesty.py for multi-component MW model
    possible_params = [
        'rho_c_solar_kpc3',
        'n_exp', 
        'M_bulge_solar',
        'a_bulge_kpc',
        'M_disk_thin_solar',
        'R_d_thin_kpc', 
        'h_z_thin_kpc',
        'M_disk_thick_solar',
        'R_d_thick_kpc',
        'h_z_thick_kpc',
        'M_gas_solar',
        'R_d_gas_kpc',
        'h_z_gas_kpc'
    ]
    return possible_params

def format_parameter_value(value, param_name):
    """Format parameter values appropriately"""
    if 'M_' in param_name and 'solar' in param_name:
        return f"{value:.2e} M☉"
    elif 'rho_c' in param_name:
        return f"{value:.2e} M☉/kpc³"
    elif 'kpc' in param_name:
        return f"{value:.3f} kpc"
    elif 'n_exp' in param_name:
        return f"{value:.3f}"
    else:
        return f"{value:.3e}"

def monitor_progress():
    """Main monitoring function"""
    print("="*60)
    print(f"DYNESTY PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Find the latest run
    sample_file, base_name = find_latest_dynesty_run()
    
    if sample_file is None:
        print("❌ No dynesty output files found in chains_dynesty/")
        print("   Make sure your dynesty run has started and --output_dir is chains_dynesty")
        return
    
    print(f"📁 Monitoring: {sample_file.name}")
    print(f"📊 Base name: {Path(base_name).name}")
    
    # Check file modification time
    mod_time = datetime.fromtimestamp(os.path.getmtime(sample_file))
    time_since_mod = datetime.now() - mod_time
    print(f"🕒 Last modified: {mod_time.strftime('%H:%M:%S')} ({time_since_mod.total_seconds():.0f}s ago)")
    
    if time_since_mod.total_seconds() > 300:  # 5 minutes
        print("⚠️  File hasn't been updated recently - sampling might be stuck or finished")
    
    # Try to load the samples
    samples, weights, logl = safe_load_samples(sample_file)
    
    if samples is None:
        print("❌ Could not load samples (file might be corrupted or being written)")
        return
    
    n_samples, n_params = samples.shape
    print(f"📈 Current samples: {n_samples:,} × {n_params} parameters")
    
    if n_samples < 50:
        print("⚠️  Very few samples yet - check back later")
        return
    
    # Get parameter names
    param_names = get_parameter_names()[:n_params]
    
    print(f"\n📊 CURRENT PARAMETER ESTIMATES (median ± MAD):")
    print("─" * 60)
    
    # Calculate statistics for each parameter
    for i, param_name in enumerate(param_names):
        values = samples[:, i]
        
        # Use last 1000 samples for more current estimate
        recent_values = values[-min(1000, len(values)):]
        
        median_val = np.median(recent_values)
        mad = np.median(np.abs(recent_values - median_val))  # Median Absolute Deviation
        
        # Format nicely
        param_display = param_name.replace('_solar', '').replace('_kpc3', '').replace('_kpc', '')
        formatted_val = format_parameter_value(median_val, param_name)
        formatted_mad = format_parameter_value(mad, param_name)
        
        print(f"  {param_display:<20}: {formatted_val:<15} ± {formatted_mad}")
    
    # Check for key indicators
    print(f"\n🎯 KEY INDICATORS:")
    print("─" * 30)
    
    # Find xi parameters
    rho_c_idx = next((i for i, name in enumerate(param_names) if 'rho_c' in name), None)
    n_exp_idx = next((i for i, name in enumerate(param_names) if 'n_exp' in name), None)
    
    if rho_c_idx is not None and n_exp_idx is not None:
        recent_samples = samples[-min(1000, len(samples)):]
        rho_c_vals = recent_samples[:, rho_c_idx]
        n_exp_vals = recent_samples[:, n_exp_idx]
        
        print(f"  Critical density: {np.median(rho_c_vals):.2e} M☉/kpc³")
        print(f"  Power index: {np.median(n_exp_vals):.3f}")
        
        # Estimate xi range (approximate)
        rho_inner = 1e9  # Typical inner galaxy density
        rho_outer = 1e6  # Typical outer galaxy density
        
        median_rho_c = np.median(rho_c_vals)
        median_n = np.median(n_exp_vals)
        
        xi_inner = 1 / (1 + (rho_inner / median_rho_c)**median_n)
        xi_outer = 1 / (1 + (rho_outer / median_rho_c)**median_n)
        
        print(f"  ξ inner (~R=2kpc): ~{xi_inner:.3f}")
        print(f"  ξ outer (~R=20kpc): ~{xi_outer:.3f}")
        
        if xi_inner > 0.8:
            print("  ⚠️  ξ close to 1 in inner regions - weak density dependence")
        elif xi_inner < 0.3:
            print("  ✅ Strong density dependence in inner regions")
        else:
            print("  ✅ Moderate density dependence")
    
    # Check baryonic masses
    mass_params = [(i, name) for i, name in enumerate(param_names) if 'M_' in name and 'solar' in name]
    
    if mass_params:
        print(f"\n💫 BARYONIC MASS COMPONENTS:")
        total_mass = 0
        recent_samples = samples[-min(1000, len(samples)):]
        
        for i, name in mass_params:
            mass_vals = recent_samples[:, i]
            median_mass = np.median(mass_vals)
            total_mass += median_mass
            
            component = name.replace('M_', '').replace('_solar', '')
            print(f"  {component:<12}: {median_mass:.2e} M☉")
        
        print(f"  {'Total':<12}: {total_mass:.2e} M☉")
        
        # Check if masses are realistic
        if total_mass > 2e11:
            print("  ⚠️  High total mass - may be compensating for weak ξ")
        elif total_mass < 5e10:
            print("  ⚠️  Low total mass - insufficient baryonic matter")
        else:
            print("  ✅ Reasonable total baryonic mass")
    
    # Sampling efficiency
    if logl is not None and len(logl) > 100:
        recent_logl = logl[-min(1000, len(logl)):]
        logl_range = np.max(recent_logl) - np.min(recent_logl)
        print(f"\n📈 SAMPLING DIAGNOSTICS:")
        print(f"  Log-likelihood range: {logl_range:.2f}")
        print(f"  Best log-L: {np.max(recent_logl):.2f}")
        
        if logl_range < 1:
            print("  ⚠️  Small likelihood range - may be converged or stuck")
        elif logl_range > 100:
            print("  ⚠️  Large likelihood range - still exploring")
        else:
            print("  ✅ Reasonable likelihood exploration")
    
    # Convergence check
    if n_samples > 2000:
        # Check parameter stability over recent samples
        recent_frac = 0.3  # Last 30% of samples
        split_point = int(n_samples * (1 - recent_frac))
        
        early_samples = samples[split_point//2:split_point]
        late_samples = samples[split_point:]
        
        if len(early_samples) > 100 and len(late_samples) > 100:
            print(f"\n🎯 CONVERGENCE CHECK:")
            stable_params = 0
            
            for i, param_name in enumerate(param_names):
                early_median = np.median(early_samples[:, i])
                late_median = np.median(late_samples[:, i])
                
                if early_median != 0:
                    rel_change = abs(late_median - early_median) / abs(early_median)
                    if rel_change < 0.1:  # Less than 10% change
                        stable_params += 1
            
            stability = stable_params / n_params
            print(f"  Parameter stability: {stability:.1%} ({stable_params}/{n_params})")
            
            if stability > 0.8:
                print("  ✅ Parameters appear to be converging")
            elif stability > 0.5:
                print("  ⚠️  Partial convergence - needs more time")
            else:
                print("  ❌ Parameters still changing significantly")
    
    print(f"\n🔄 Next check: Run this script again in 30-60 minutes")
    print("="*60)

if __name__ == "__main__":
    monitor_progress()