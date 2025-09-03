#!/usr/bin/env python3
"""
Debug why RAR Gate has good evidence but poor visual fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.density_metric_cupy import v_total_kms_cupy
import cupy as cp

def main():
    # Load binned Gaia data
    binned_df = pd.read_csv("data/mw_binned_velocities.csv")
    
    # Load RAR Gate parameters
    with open("runs/rar_gate_from_best_20250820_185422/run_summary_enhanced.json", 'r') as f:
        rar_data = json.load(f)
    rar_params = rar_data['parameter_estimates']['best_fit']
    
    # Load GR parameters  
    with open("runs/gr_20250812_113949/run_summary_enhanced.json", 'r') as f:
        gr_data = json.load(f)
    gr_params = gr_data['parameter_estimates']['best_fit']
    
    print("Parameter comparison:")
    print("-" * 60)
    print(f"{'Parameter':<25} {'GR':<15} {'RAR Gate':<15}")
    print("-" * 60)
    
    # Compare baryon masses
    gr_total = gr_params['M_thin_disk_solar'] + gr_params['M_thick_disk_solar'] + gr_params['M_bulge_solar'] + gr_params['M_gas_solar']
    rar_total = rar_params['M_thin_disk_solar'] + rar_params['M_thick_disk_solar'] + rar_params['M_bulge_solar'] + rar_params['M_gas_solar']
    
    print(f"{'Total baryons (M_sun)':<25} {gr_total:.2e} {rar_total:.2e}")
    print(f"{'M_thin_disk':<25} {gr_params['M_thin_disk_solar']:.2e} {rar_params['M_thin_disk_solar']:.2e}")
    print(f"{'R_thin_disk (kpc)':<25} {gr_params['R_thin_disk_kpc']:.2f} {rar_params['R_thin_disk_kpc']:.2f}")
    
    # RAR-specific parameters
    print("\nRAR Gate specific parameters:")
    print(f"  a0_m_s2: {rar_params['a0_m_s2']:.3e}")
    print(f"  gamma_exp: {rar_params['gamma_exp']:.3f}")
    print(f"  lambda_max: {rar_params['lambda_max']:.3f}")
    print(f"  T0: {rar_params['T0']:.1f}")
    print(f"  sigma_lnT: {rar_params['sigma_lnT']:.3f}")
    print(f"  wmin: {rar_params['wmin']:.3f}")
    
    # Compute rotation curves
    R_test = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
    
    print("\nVelocity comparison at key radii:")
    print("-" * 60)
    print(f"{'R (kpc)':<10} {'Gaia':<15} {'GR':<15} {'RAR Gate':<15}")
    print("-" * 60)
    
    for R in R_test:
        # Find closest Gaia point
        idx = np.argmin(np.abs(binned_df['R_kpc'] - R))
        v_gaia = binned_df.iloc[idx]['v_mean']
        
        # Compute GR velocity
        R_cp = cp.array([R], dtype=cp.float32)
        v_gr = v_total_kms_cupy(R_cp, gr_params, xi_type='gr')
        v_gr = cp.asnumpy(v_gr)[0]
        
        # Compute RAR Gate velocity
        rar_params_with_flag = dict(rar_params)
        rar_params_with_flag['allow_experimental'] = True
        v_rar = v_total_kms_cupy(R_cp, rar_params_with_flag, xi_type='rar_gate')
        v_rar = cp.asnumpy(v_rar)[0]
        
        print(f"{R:<10.1f} {v_gaia:<15.1f} {v_gr:<15.1f} {v_rar:<15.1f}")
    
    # Calculate chi-squared
    print("\nChi-squared calculation:")
    print("-" * 60)
    
    # Compute at all data points
    R_data = binned_df['R_kpc'].values
    v_data = binned_df['v_mean'].values
    v_err = binned_df['v_err'].values
    
    # GR chi-squared
    R_cp = cp.asarray(R_data, dtype=cp.float32)
    v_gr_all = v_total_kms_cupy(R_cp, gr_params, xi_type='gr')
    v_gr_all = cp.asnumpy(v_gr_all)
    chi2_gr = np.sum((v_data - v_gr_all)**2 / v_err**2)
    
    # RAR Gate chi-squared
    v_rar_all = v_total_kms_cupy(R_cp, rar_params_with_flag, xi_type='rar_gate')
    v_rar_all = cp.asnumpy(v_rar_all)
    chi2_rar = np.sum((v_data - v_rar_all)**2 / v_err**2)
    
    n_data = len(R_data)
    n_params_gr = 11  # Approximate
    n_params_rar = 17  # Approximate
    
    print(f"GR:       χ² = {chi2_gr:.1f}, χ²/dof = {chi2_gr/(n_data-n_params_gr):.1f}")
    print(f"RAR Gate: χ² = {chi2_rar:.1f}, χ²/dof = {chi2_rar/(n_data-n_params_rar):.1f}")
    
    print("\nEvidence comparison:")
    print("-" * 60)
    print(f"GR log(Z):       {gr_data['evidence_metrics']['logz']:.1f}")
    print(f"RAR Gate log(Z): {rar_data['evidence_metrics']['logz']:.1f}")
    print(f"Δlog(Z):         {rar_data['evidence_metrics']['logz'] - gr_data['evidence_metrics']['logz']:.1f}")
    
    print("\nConclusion:")
    print("-" * 60)
    if chi2_rar > chi2_gr:
        print("WARNING: RAR Gate has WORSE chi-squared but BETTER evidence!")
        print("This suggests:")
        print("  1. The evidence calculation might be incorrect")
        print("  2. Or the prior volume effects dominate")
        print("  3. Or there's an error in the likelihood calculation")
    else:
        print("RAR Gate has better chi-squared AND better evidence (consistent)")

if __name__ == "__main__":
    main()
