#!/usr/bin/env python3
"""
combine_results.py - Combine and analyze results from multiple dynesty runs
"""

import numpy as np
import logging
from pathlib import Path
import json
import pandas as pd
from analyze_bimodal_results import BimodalAnalyzer  # Import from script #1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_final_results(all_runs_dir):
    """Combine results from multiple runs."""
    # ... (the full function from #5)

def compare_runs_table(results_dict):
    """Create comparison table of all runs."""
    df_data = []
    
    for run_name, run_info in results_dict.items():
        row = {'run': run_name}
        row['physical_fraction'] = run_info['physical_fraction']
        row['n_samples'] = run_info['n_samples']
        
        # Add key parameters
        for param in ['rho_c_solar_kpc3', 'M_disk_thin_solar', 'R_d_thick_kpc']:
            if param in run_info['medians']:
                row[f'{param}_median'] = run_info['medians'][param]
                row[f'{param}_mad'] = run_info['uncertainties'][param]
                
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    return df

def export_best_for_paper(best_params, output_file='best_fit_params.json'):
    """Export best-fit parameters in paper-ready format."""
    paper_format = {}
    
    for param, value in best_params.items():
        if 'M_' in param and 'solar' in param:
            # Convert to scientific notation
            paper_format[param] = {
                'value': value,
                'latex': f"${value:.2e}$".replace('e+', r' \times 10^{').replace('e-', r' \times 10^{-') + '}'
            }
        else:
            paper_format[param] = {
                'value': value,
                'latex': f"${value:.3f}$"
            }
            
    with open(output_file, 'w') as f:
        json.dump(paper_format, f, indent=2)
        
    logger.info(f"Exported paper-ready parameters to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine dynesty results")
    parser.add_argument('--runs_dir', default='.', 
                       help='Directory containing chains_* subdirectories')
    parser.add_argument('--output_csv', default='all_runs_comparison.csv',
                       help='Output CSV with run comparisons')
    parser.add_argument('--plot_comparison', action='store_true',
                       help='Create comparison plots')
    
    args = parser.parse_args()
    
    # Find best results
    best_params = create_final_results(args.runs_dir)
    
    # Export for paper
    export_best_for_paper(best_params)
    
    # Create comparison table
    results = analyze_all_runs(args.runs_dir)  # You'd implement this
    df = compare_runs_table(results)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved comparison table to {args.output_csv}")