#!/usr/bin/env python3
"""
Extract and finalize results - CLI version
Accepts checkpoint directory as argument and extracts all data from files.
"""
import numpy as np
import pickle
import gzip
import json
from pathlib import Path
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_gr_results(checkpoint_path):
    """Extract results from GR baseline checkpoint"""
    
    checkpoint_path = Path(checkpoint_path)
    
    # Handle if path is a directory or file
    if checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
        # Find the latest checkpoint file
        npz_files = sorted(checkpoint_dir.glob("dynesty_checkpoint*.npz"))
        pkl_files = sorted(checkpoint_dir.glob("dynesty_checkpoint*.pkl"))
        
        if npz_files:
            checkpoint_file = npz_files[-1]
        elif pkl_files:
            checkpoint_file = pkl_files[-1]
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    else:
        checkpoint_file = checkpoint_path
        checkpoint_dir = checkpoint_path.parent
    
    logger.info(f"Loading checkpoint: {checkpoint_file}")
    
    # Load checkpoint based on file type
    if checkpoint_file.suffix == '.npz':
        data = np.load(checkpoint_file)
        samples = data['samples']
        
        # Extract all available data
        logz = data['logz'] if 'logz' in data else None
        logzerr = data['logzerr'] if 'logzerr' in data else None
        logwt = data['logwt'] if 'logwt' in data else None
        logl = data['logl'] if 'logl' in data else None
        weights = data['weights'] if 'weights' in data else None
        param_names = data['param_names'] if 'param_names' in data else None
        if 'n_calls' in data:
            nc = data['n_calls']
            n_calls = int(nc) if np.isscalar(nc) else int(np.sum(nc))
        else:
            n_calls = None
        
    elif checkpoint_file.suffix == '.pkl':
        with open(checkpoint_file, 'rb') as f:
            res = pickle.load(f)
        samples = res.samples
        logz = res.logz if hasattr(res, 'logz') else None
        logzerr = res.logzerr if hasattr(res, 'logzerr') else None
        logwt = res.logwt if hasattr(res, 'logwt') else None
        logl = res.logl if hasattr(res, 'logl') else None
        n_calls = None
        if hasattr(res, 'ncall'):
            nc = res.ncall
            n_calls = int(nc) if np.isscalar(nc) else int(np.sum(nc))
        weights = None
        param_names = None
    else:
        raise ValueError(f"Unknown file type: {checkpoint_file.suffix}")
    
    # Extract values dynamically
    n_samples = len(samples)
    
    # Get final logz value
    if logz is not None:
        if np.isscalar(logz):
            final_logz = float(logz)
        elif len(logz) > 0:
            final_logz = float(logz[-1])
        else:
            logger.warning("Empty logz array")
            final_logz = np.nan
    else:
        logger.warning("No logz found in checkpoint")
        final_logz = np.nan
    
    # Get final logz error
    if logzerr is not None:
        if np.isscalar(logzerr):
            final_logzerr = float(logzerr)
        elif len(logzerr) > 0:
            final_logzerr = float(logzerr[-1])
        else:
            final_logzerr = 0.0
    else:
        final_logzerr = 0.0
    
    # Calculate weights if not provided
    if weights is None:
        if logwt is not None and len(logwt) > 0:
            # Convert log weights to weights
            logwt_max = np.max(logwt[np.isfinite(logwt)])
            weights = np.exp(logwt - logwt_max)
            weights /= np.sum(weights)
        else:
            # Uniform weights as fallback
            weights = np.ones(n_samples) / n_samples
    
    # Try to get parameter names from various sources
    if param_names is None:
        # Check if there's a config file
        config_files = [
            checkpoint_dir / "run_config_enhanced.json",
            checkpoint_dir / "run_config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'explicitly_provided_flags' in config:
                        # Extract fitted parameters based on flags
                        fitted_params = []
                        for flag in config['explicitly_provided_flags']:
                            if flag.startswith('fit_') and config['all_parameters'].get(flag, False):
                                # Map flags to parameter names
                                if flag == 'fit_disk_thin':
                                    fitted_params.extend(['M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc'])
                                elif flag == 'fit_disk_thick':
                                    fitted_params.extend(['M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc'])
                                elif flag == 'fit_bulge':
                                    fitted_params.extend(['M_bulge_solar', 'a_bulge_kpc'])
                                elif flag == 'fit_gas':
                                    fitted_params.extend(['M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'])
                        if fitted_params:
                            param_names = fitted_params
                            break
    
    # Default parameter names if still not found
    if param_names is None:
        n_params = samples.shape[1]
        if n_params == 11:  # Standard MW model
            param_names = [
                'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
                'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
                'M_bulge_solar', 'a_bulge_kpc',
                'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
            ]
        else:
            param_names = [f'param_{i}' for i in range(n_params)]
            logger.warning(f"Using generic parameter names for {n_params} parameters")
    
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
    
    # Print results
    print("\n" + "="*70)
    print(f"RESULTS FROM: {checkpoint_file}")
    print("="*70)
    print(f"Final log(Z): {final_logz:.2f} ± {final_logzerr:.2f}")
    print(f"Number of samples: {n_samples}")
    if n_calls:
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
            unit = 'kpc' if 'kpc' in name else ''
            print(f"{name:<25}: {q50:.3f} [{q16:.3f}, {q84:.3f}] {unit}")
    
    if total_mass > 0:
        print("-"*70)
        print(f"Total baryonic mass: {total_mass:.3e} M☉ ({total_mass/1e11:.1f} × 10¹¹ M☉)")
    print("="*70)
    
    # Save NPZ format
    output_npz = checkpoint_dir / f"extracted_{checkpoint_file.stem}.npz"
    np.savez(
        output_npz,
        samples=samples,
        weights=weights,
        param_names=param_names,
        logz=np.array([final_logz]),
        logzerr=np.array([final_logzerr]),
        logl=logl if logl is not None else np.zeros(n_samples),
        median_params=median_params,
        std_params=std_params,
        n_calls=n_calls if n_calls else 0
    )
    logger.info(f"\nSaved results to {output_npz}")
    
    # Save JSON summary
    summary = {
        "source_file": str(checkpoint_file),
        "final_logZ": float(final_logz),
        "logZ_error": float(final_logzerr),
        "n_samples": int(n_samples),
        "n_calls": int(n_calls) if n_calls else None,
        "efficiency_percent": float(n_samples/n_calls*100) if n_calls else None,
        "total_baryonic_mass": float(total_mass) if total_mass > 0 else None,
        "parameters": {}
    }
    
    for i, name in enumerate(param_names):
        summary["parameters"][name] = {
            "median": float(median_params[i]),
            "std": float(std_params[i]),
            "q16": float(percentiles[i]['q16']),
            "q84": float(percentiles[i]['q84'])
        }
    
    output_json = checkpoint_dir / f"extracted_{checkpoint_file.stem}_summary.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_json}")
    
    return output_npz, output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract results from dynesty checkpoint files")
    parser.add_argument("checkpoint", help="Path to checkpoint file or directory")
    args = parser.parse_args()
    
    try:
        npz_path, json_path = extract_gr_results(args.checkpoint)
        print(f"\n✅ Results extracted successfully!")
        print(f"\nOutput files:")
        print(f"  - {npz_path}")
        print(f"  - {json_path}")
    except Exception as e:
        logger.error(f"Failed to extract results: {e}")
        sys.exit(1)