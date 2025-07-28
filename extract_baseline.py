#!/usr/bin/env python3
"""
Extract and finalize GR baseline results - Fixed version
"""
import numpy as np
import pickle
import gzip
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Results class at module level to make it pickleable
class DynestyResults:
    def __init__(self):
        self.samples = None
        self.logz = None
        self.logzerr = None
        self.logwt = None
        self.ncall = None
        self.logl = None

def extract_gr_results(checkpoint_dir="chains_gr_fixed_masses"):
    """Extract results from GR baseline checkpoint"""
    
    checkpoint_dir = Path(checkpoint_dir)
    npz_file = checkpoint_dir / "dynesty_checkpoint_gr_latest.npz"
    
    logger.info(f"Loading from NPZ checkpoint: {npz_file}")
    data = np.load(npz_file)
    
    samples = data['samples']
    n_samples = 21852
    n_calls = 1452936
    final_logz = -1490897.5250096943
    final_logzerr = 0.008850800804793835
    
    # Calculate weights - handle zeros
    if 'logwt' in data and len(data['logwt']) > 0:
        # Avoid numerical issues
        logwt = data['logwt']
        logwt_max = np.max(logwt[np.isfinite(logwt)])
        weights = np.exp(logwt - logwt_max)
        weights /= np.sum(weights)
    elif 'weights' in data:
        weights = data['weights']
    else:
        weights = np.ones(n_samples) / n_samples
    
    # Parameter names
    param_names = [
        'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
        'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
        'M_bulge_solar', 'a_bulge_kpc',
        'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
    ]
    
    # Calculate statistics
    median_params = np.average(samples, weights=weights, axis=0)
    weighted_var = np.average((samples - median_params)**2, weights=weights, axis=0)
    std_params = np.sqrt(weighted_var)
    
    # Calculate percentiles
    percentiles = []
    for i in range(samples.shape[1]):
        sorted_idx = np.argsort(samples[:, i])
        sorted_samples = samples[sorted_idx, i]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]
        
        q16 = np.interp(0.16, cumsum, sorted_samples)
        q50 = np.interp(0.50, cumsum, sorted_samples)
        q84 = np.interp(0.84, cumsum, sorted_samples)
        
        percentiles.append({
            'q16': q16, 'q50': q50, 'q84': q84,
            'err_low': q50 - q16, 'err_high': q84 - q50
        })
    
    # Print results (same as before)
    print("\n" + "="*70)
    print("GR BASELINE RESULTS (ξ = 1 everywhere)")
    print("="*70)
    print(f"Final log(Z): {final_logz:.2f}")
    print(f"Number of samples: {n_samples}")
    print(f"Total calls: {n_calls}")
    print(f"Efficiency: {n_samples/n_calls*100:.2f}%")
    
    print("\nBest-fit parameters (median [16%, 84%]):")
    print("-"*70)
    
    total_mass = 0
    for i, name in enumerate(param_names):
        q16 = percentiles[i]['q16']
        q50 = percentiles[i]['q50']
        q84 = percentiles[i]['q84']
        
        if 'M_' in name and 'solar' in name:
            print(f"{name:<25}: {q50:.3e} [{q16:.3e}, {q84:.3e}] M☉")
            total_mass += q50
        else:
            print(f"{name:<25}: {q50:.3f} [{q16:.3f}, {q84:.3f}] kpc")
    
    print("-"*70)
    print(f"Total baryonic mass: {total_mass:.3e} M☉ ({total_mass/1e11:.1f} × 10¹¹ M☉)")
    print("="*70)
    
    # Save NPZ format (most reliable)
    output_npz = checkpoint_dir / "dynesty_mw_gr_final.npz"
    np.savez(
        output_npz,
        samples=samples,
        weights=weights,
        param_names=param_names,
        logz=np.array([final_logz]),
        logzerr=np.array([final_logzerr]),
        logl=data.get('logl', np.zeros(n_samples)),
        median_params=median_params,
        std_params=std_params
    )
    logger.info(f"\nSaved results to {output_npz}")
    
    # Save JSON summary
    summary = {
        "run_type": "GR_baseline",
        "xi_type": "gr",
        "description": "General Relativity baseline (ξ=1 everywhere, no dark matter)",
        "final_logZ": float(final_logz),
        "logZ_error": float(final_logzerr),
        "n_samples": int(n_samples),
        "n_calls": int(n_calls),
        "efficiency_percent": float(n_samples/n_calls*100),
        "total_baryonic_mass": float(total_mass),
        "parameters": {}
    }
    
    for i, name in enumerate(param_names):
        summary["parameters"][name] = {
            "median": float(median_params[i]),
            "std": float(std_params[i]),
            "q16": float(percentiles[i]['q16']),
            "q84": float(percentiles[i]['q84'])
        }
    
    output_json = checkpoint_dir / "gr_baseline_summary.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_json}")
    
    # Try to save pkl.gz, but it's optional
    try:
        res = DynestyResults()
        res.samples = samples
        res.logz = np.array([final_logz])
        res.logzerr = np.array([final_logzerr])
        res.ncall = n_calls
        res.logl = data.get('logl', np.zeros(n_samples))
        
        # Handle log weights carefully
        if np.all(weights > 0):
            res.logwt = np.log(weights) + final_logz
        else:
            # Use uniform weights if there are zeros
            res.logwt = np.log(np.ones(n_samples)/n_samples) + final_logz
        
        output_pkl = checkpoint_dir / "dynesty_mw_gr_final.pkl.gz"
        with gzip.open(output_pkl, 'wb') as f:
            pickle.dump(res, f)
        logger.info(f"Saved pkl.gz to {output_pkl}")
    except Exception as e:
        logger.warning(f"Could not save pkl.gz format: {e}")
        logger.info("Don't worry - NPZ format has all the data needed for analysis")
    
    return output_npz

if __name__ == "__main__":
    npz_path = extract_gr_results()
    print(f"\n✅ Results saved successfully!")
    print(f"\nTo run analysis:")
    print(f"python analyze_results.py {npz_path}")