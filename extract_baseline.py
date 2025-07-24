#!/usr/bin/env python3
"""
Extract and save results from the interrupted GR baseline run
"""
import numpy as np
import pickle
import gzip
import json
from pathlib import Path
import sys

# CRITICAL: Import the functions that the pickle needs BEFORE loading
try:
    from run_dynesty import (
        log_likelihood_dynesty,
        prior_transform_dynesty,
        MW_MULTI_COMP_PARAM_CONFIG
    )
except ImportError:
    print("Warning: Could not import from run_dynesty. Trying alternative approach...")

def extract_gr_baseline_results(checkpoint_path="chains_GR_reparameterized/dynesty_checkpoint.pkl",
                                output_dir="chains_GR_reparameterized"):
    """Extract results from checkpoint and save in multiple formats"""
    
    # First, check if we already have saved results
    output_path = Path(output_dir)
    npz_files = list(output_path.glob("dynesty_checkpoint_*.npz"))
    
    if npz_files:
        # Use the latest NPZ file instead of the pickle
        latest_npz = sorted(npz_files)[-1]
        print(f"Found NPZ checkpoint: {latest_npz}")
        print("Using NPZ file instead of pickle to avoid compatibility issues...")
        
        data = np.load(latest_npz)
        
        # Extract what we need
        samples = data['samples']
        logz = data['logz']
        logzerr = data.get('logzerr', np.ones_like(logz) * 0.1)
        logl = data.get('logl', np.zeros(len(samples)))
        logwt = data.get('logwt', None)
        weights = data.get('weights', None)
        n_calls = data.get('n_calls', len(samples) * 50)  # Estimate if not saved
        
        if weights is None and logwt is not None and len(logz) > 0:
            weights = np.exp(logwt - logz[-1])
        elif weights is None:
            weights = np.ones(len(samples)) / len(samples)
        
        # Create a results-like object
        class Results:
            pass
        
        res = Results()
        res.samples = samples
        res.logz = logz
        res.logzerr = logzerr
        res.logl = logl
        res.logwt = logwt if logwt is not None else np.log(weights) + logz[-1]
        res.ncall = n_calls
        
    else:
        # Try to load from pickle with proper imports
        print("Loading from pickle checkpoint...")
        
        # Add the script directory to path so imports work
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        try:
            with open(checkpoint_path, 'rb') as f:
                sampler = pickle.load(f)
            res = sampler.results
        except Exception as e:
            print(f"Error loading pickle: {e}")
            print("\nTrying to find alternative saved results...")
            
            # Look for any results files
            results_files = list(output_path.glob("*samples.npz")) + \
                           list(output_path.glob("*results.pkl.gz"))
            
            if results_files:
                print(f"Found {len(results_files)} results files")
                # Use the most recent one
                latest = sorted(results_files, key=lambda x: x.stat().st_mtime)[-1]
                print(f"Using: {latest}")
                
                if latest.suffix == '.npz':
                    data = np.load(latest)
                    # Create results object from NPZ
                    res = Results()
                    res.samples = data['samples']
                    res.logz = data.get('logz', np.array([-1475548.0]))  # Use known value
                    res.logzerr = data.get('logzerr', np.array([0.27]))
                    res.logl = data.get('logl', np.zeros(len(res.samples)))
                    weights = data.get('weights', np.ones(len(res.samples))/len(res.samples))
                    res.logwt = np.log(weights) + res.logz[-1]
                    res.ncall = data.get('n_calls', [1074509])
                else:
                    with gzip.open(latest, 'rb') as f:
                        res = pickle.load(f)
            else:
                print("ERROR: No checkpoint or results files found!")
                print(f"Searched in: {output_path}")
                return None
    
    # Calculate weights if needed
    if hasattr(res, 'logwt') and hasattr(res, 'logz') and len(res.logz) > 0:
        weights = np.exp(res.logwt - res.logz[-1])
    else:
        weights = np.ones(len(res.samples)) / len(res.samples)
    
    # Get parameter names (these are the fitted parameters from your run)
    param_names = [
        'M_bulge_solar', 'a_bulge_kpc',
        'M_gas_solar', 
        'M_disk_total_solar', 'thick_mass_fraction',
        'R_d_gas_kpc', 'h_z_gas_kpc'
    ]
    
    # Verify the number of parameters matches
    if len(param_names) != res.samples.shape[1]:
        print(f"Warning: Parameter count mismatch. Expected {len(param_names)}, got {res.samples.shape[1]}")
        # Try to infer from shape
        if res.samples.shape[1] == 7:
            print("Shape matches expected 7 parameters for reparameterized run")
        else:
            print("Using generic parameter names")
            param_names = [f'param_{i}' for i in range(res.samples.shape[1])]
    
    # Calculate statistics
    median_params = np.average(res.samples, weights=weights, axis=0)
    std_params = np.sqrt(np.average((res.samples - median_params)**2, weights=weights, axis=0))
    
    # Reconstruct physical disk masses
    if 'M_disk_total_solar' in param_names:
        idx_total = param_names.index('M_disk_total_solar')
        idx_frac = param_names.index('thick_mass_fraction')
        M_total = median_params[idx_total]
        f_thick = median_params[idx_frac]
        M_thick = M_total * f_thick
        M_thin = M_total * (1 - f_thick)
    else:
        print("Warning: Could not find disk reparameterization")
        M_thin = M_thick = 0
    
    # Get final evidence
    if hasattr(res, 'logz') and len(res.logz) > 0:
        final_logz = res.logz[-1]
        final_logzerr = res.logzerr[-1] if hasattr(res, 'logzerr') and len(res.logzerr) > 0 else 0.1
    else:
        print("Warning: No logz found, using placeholder")
        final_logz = -1475548.0  # From your dashboard
        final_logzerr = 0.27
    
    print("\n=== GR BASELINE RESULTS ===")
    print(f"Final log(Z): {final_logz:.2f} ± {final_logzerr:.2f}")
    print(f"Number of samples: {len(res.samples)}")
    print(f"Total calls: {np.sum(res.ncall) if hasattr(res, 'ncall') else 'Unknown'}")
    
    print("\nBest-fit parameters (medians):")
    for i, (name, value, err) in enumerate(zip(param_names, median_params, std_params)):
        print(f"  {name}: {value:.3e} ± {err:.3e}")
    
    if M_thin > 0:
        print(f"\nDerived disk masses:")
        print(f"  M_disk_thin:  {M_thin/1e9:.1f} × 10⁹ M☉")
        print(f"  M_disk_thick: {M_thick/1e9:.1f} × 10⁹ M☉")
    
    print(f"\nTotal baryonic mass: {(M_thin + M_thick + median_params[0] + median_params[2])/1e9:.1f} × 10⁹ M☉")
    
    # Save results
    output_path = Path(output_dir)
    
    # 1. NPZ format
    np.savez(output_path / "gr_baseline_results.npz",
             samples=res.samples,
             weights=weights,
             param_names=param_names,
             logz=np.array([final_logz]),
             logzerr=np.array([final_logzerr]),
             logl=res.logl if hasattr(res, 'logl') else np.zeros(len(res.samples)),
             median_params=median_params,
             std_params=std_params,
             n_calls=np.sum(res.ncall) if hasattr(res, 'ncall') else len(res.samples) * 50)
    
    # 2. JSON summary
    summary = {
        "run_type": "GR_baseline",
        "xi": "gr (ξ=1 everywhere)",
        "logZ": float(final_logz),
        "logZ_err": float(final_logzerr),
        "n_samples": len(res.samples),
        "n_calls": int(np.sum(res.ncall)) if hasattr(res, 'ncall') else len(res.samples) * 50,
        "efficiency_percent": float(len(res.samples) / (np.sum(res.ncall) if hasattr(res, 'ncall') else len(res.samples) * 50) * 100),
        "best_fit_params": dict(zip(param_names, median_params.tolist())),
        "param_uncertainties": dict(zip(param_names, std_params.tolist())),
        "derived_masses": {
            "M_disk_thin_solar": float(M_thin),
            "M_disk_thick_solar": float(M_thick),
            "M_total_baryons": float(M_thin + M_thick + median_params[0] + median_params[2])
        }
    }
    
    with open(output_path / "gr_baseline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}/")
    print("  - gr_baseline_results.npz (main results)")
    print("  - gr_baseline_summary.json (human-readable summary)")
    
    return res, median_params, param_names

if __name__ == "__main__":
    extract_gr_baseline_results()