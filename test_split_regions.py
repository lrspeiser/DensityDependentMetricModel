#!/usr/bin/env python3
"""
Test script for split-region analysis to find optimal transition radius.

This script helps identify where GR behavior changes by:
1. Loading Gaia data and analyzing velocity vs radius trends
2. Testing different transition radii
3. Identifying where stars start traveling "too fast" for expected GR gravity
4. Suggesting optimal transition radius for split-region analysis

Usage:
    py test_split_regions.py --analyze_trends
    py test_split_regions.py --find_transition --max_radius 25.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from load_existing_gaia import load_existing_gaia_data
    DATA_IO_AVAILABLE = True
except ImportError:
    DATA_IO_AVAILABLE = False
    print("Warning: load_existing_gaia not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_velocity_trends(gaia_data, max_radius=25.0):
    """Analyze velocity trends vs radius to identify GR breakdown."""
    logger.info("Analyzing velocity trends vs radius...")
    
    R_kpc = np.array(gaia_data['R_kpc'])
    v_obs = np.array(gaia_data['v_obs'])
    sigma_v = np.array(gaia_data['sigma_v'])
    
    # Filter to reasonable radius range
    mask = (R_kpc >= 3.0) & (R_kpc <= max_radius)
    R_filtered = R_kpc[mask]
    v_filtered = v_obs[mask]
    sigma_filtered = sigma_v[mask]
    
    logger.info(f"Analyzing {len(R_filtered)} stars in radius range 3.0 - {max_radius} kpc")
    
    # Create radial bins for analysis
    n_bins = 20
    r_bins = np.linspace(3.0, max_radius, n_bins + 1)
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Calculate statistics in each bin
    v_medians = []
    v_means = []
    v_stds = []
    n_stars = []
    
    for i in range(len(r_bins) - 1):
        bin_mask = (R_filtered >= r_bins[i]) & (R_filtered < r_bins[i + 1])
        if np.sum(bin_mask) > 5:  # Require at least 5 stars per bin
            v_bin = v_filtered[bin_mask]
            v_medians.append(np.median(v_bin))
            v_means.append(np.mean(v_bin))
            v_stds.append(np.std(v_bin))
            n_stars.append(len(v_bin))
        else:
            v_medians.append(np.nan)
            v_means.append(np.nan)
            v_stds.append(np.nan)
            n_stars.append(0)
    
    # Convert to arrays
    v_medians = np.array(v_medians)
    v_means = np.array(v_means)
    v_stds = np.array(v_stds)
    n_stars = np.array(n_stars)
    
    # Find where velocity starts to increase (GR breakdown indicator)
    # Look for significant increase in velocity beyond what's expected from GR
    velocity_gradient = np.gradient(v_medians[~np.isnan(v_medians)])
    bin_centers_valid = bin_centers[~np.isnan(v_medians)]
    
    # Find transition point where velocity gradient becomes positive and significant
    transition_candidates = []
    for i in range(1, len(velocity_gradient)):
        if velocity_gradient[i] > 5.0:  # km/s per kpc increase
            transition_candidates.append(bin_centers_valid[i])
    
    # Calculate expected GR velocity (assuming flat rotation curve)
    # In GR, velocity should be roughly constant or slightly decreasing
    expected_gr_velocity = np.median(v_medians[:5])  # Use inner region as baseline
    
    # Find where observed velocity significantly exceeds GR expectation
    gr_breakdown_candidates = []
    for i, (r, v_med) in enumerate(zip(bin_centers, v_medians)):
        if not np.isnan(v_med) and v_med > expected_gr_velocity * 1.1:  # 10% excess
            gr_breakdown_candidates.append(r)
    
    # Print analysis results
    logger.info("VELOCITY TREND ANALYSIS:")
    logger.info(f"Expected GR velocity (inner region): {expected_gr_velocity:.1f} km/s")
    logger.info(f"Velocity gradient candidates: {transition_candidates}")
    logger.info(f"GR breakdown candidates (>10% excess): {gr_breakdown_candidates}")
    
    # Suggest optimal transition radius
    if gr_breakdown_candidates:
        suggested_transition = gr_breakdown_candidates[0]
        logger.info(f"Suggested transition radius: {suggested_transition:.1f} kpc")
    elif transition_candidates:
        suggested_transition = transition_candidates[0]
        logger.info(f"Suggested transition radius: {suggested_transition:.1f} kpc")
    else:
        suggested_transition = 12.0  # Default
        logger.info(f"Using default transition radius: {suggested_transition:.1f} kpc")
    
    return {
        'bin_centers': bin_centers,
        'v_medians': v_medians,
        'v_means': v_means,
        'v_stds': v_stds,
        'n_stars': n_stars,
        'expected_gr_velocity': expected_gr_velocity,
        'transition_candidates': transition_candidates,
        'gr_breakdown_candidates': gr_breakdown_candidates,
        'suggested_transition': suggested_transition
    }

def plot_velocity_trends(analysis_results, save_path='plots/'):
    """Plot velocity trends to visualize GR breakdown."""
    logger.info("Creating velocity trend plots...")
    
    Path(save_path).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Velocity vs radius
    bin_centers = analysis_results['bin_centers']
    v_medians = analysis_results['v_medians']
    v_stds = analysis_results['v_stds']
    n_stars = analysis_results['n_stars']
    expected_gr = analysis_results['expected_gr_velocity']
    suggested_transition = analysis_results['suggested_transition']
    
    # Plot median velocity with error bars
    valid_mask = ~np.isnan(v_medians)
    ax1.errorbar(bin_centers[valid_mask], v_medians[valid_mask], 
                yerr=v_stds[valid_mask], fmt='ko', capsize=5, label='Median velocity')
    
    # Plot expected GR velocity
    ax1.axhline(expected_gr, color='blue', linestyle='--', alpha=0.7, 
                label=f'Expected GR velocity ({expected_gr:.1f} km/s)')
    
    # Plot transition radius
    ax1.axvline(suggested_transition, color='red', linestyle=':', alpha=0.8,
                label=f'Suggested transition ({suggested_transition:.1f} kpc)')
    
    # Plot solar position
    ax1.axvline(8.122, color='orange', linestyle='-', alpha=0.6, label='Solar position')
    
    ax1.set_xlabel('Galactocentric Radius (kpc)')
    ax1.set_ylabel('Circular Velocity (km/s)')
    ax1.set_title('Milky Way Rotation Curve: Velocity vs Radius')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(3, 25)
    
    # Plot 2: Number of stars per bin
    ax2.bar(bin_centers[valid_mask], n_stars[valid_mask], alpha=0.7, color='green')
    ax2.axvline(suggested_transition, color='red', linestyle=':', alpha=0.8)
    ax2.axvline(8.122, color='orange', linestyle='-', alpha=0.6)
    
    ax2.set_xlabel('Galactocentric Radius (kpc)')
    ax2.set_ylabel('Number of Stars')
    ax2.set_title('Data Distribution: Stars per Radial Bin')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3, 25)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/velocity_trends_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"Velocity trends plot saved to {save_path}/velocity_trends_analysis.png")
    
    return fig

def test_transition_radii(gaia_data, max_radius=25.0):
    """Test different transition radii to find optimal split."""
    logger.info("Testing different transition radii...")
    
    R_kpc = np.array(gaia_data['R_kpc'])
    v_obs = np.array(gaia_data['v_obs'])
    
    # Test transition radii from 8 to 20 kpc
    transition_radii = np.arange(8.0, 21.0, 1.0)
    results = []
    
    for transition_radius in transition_radii:
        # Split data
        inner_mask = R_kpc <= transition_radius
        outer_mask = R_kpc > transition_radius
        
        n_inner = np.sum(inner_mask)
        n_outer = np.sum(outer_mask)
        
        if n_inner < 100 or n_outer < 100:
            continue  # Skip if too few stars in either region
        
        # Calculate statistics
        v_inner = v_obs[inner_mask]
        v_outer = v_obs[outer_mask]
        
        inner_stats = {
            'mean': np.mean(v_inner),
            'std': np.std(v_inner),
            'median': np.median(v_inner)
        }
        
        outer_stats = {
            'mean': np.mean(v_outer),
            'std': np.std(v_outer),
            'median': np.median(v_outer)
        }
        
        # Calculate velocity difference
        velocity_difference = outer_stats['median'] - inner_stats['median']
        
        results.append({
            'transition_radius': transition_radius,
            'n_inner': n_inner,
            'n_outer': n_outer,
            'inner_median': inner_stats['median'],
            'outer_median': outer_stats['median'],
            'velocity_difference': velocity_difference,
            'inner_std': inner_stats['std'],
            'outer_std': outer_stats['std']
        })
    
    # Find optimal transition radius
    if results:
        # Sort by velocity difference (larger difference = more distinct regions)
        results.sort(key=lambda x: abs(x['velocity_difference']), reverse=True)
        
        logger.info("TRANSITION RADIUS ANALYSIS:")
        logger.info("Top 5 transition radii by velocity difference:")
        for i, result in enumerate(results[:5]):
            logger.info(f"  {i+1}. {result['transition_radius']:.1f} kpc: "
                       f"Δv = {result['velocity_difference']:.1f} km/s "
                       f"(inner: {result['n_inner']} stars, outer: {result['n_outer']} stars)")
        
        optimal_radius = results[0]['transition_radius']
        logger.info(f"Optimal transition radius: {optimal_radius:.1f} kpc")
        
        return results, optimal_radius
    else:
        logger.warning("No valid transition radii found")
        return [], 12.0

def generate_cli_commands(optimal_transition_radius):
    """Generate CLI commands for split-region analysis."""
    logger.info("GENERATED CLI COMMANDS:")
    logger.info("=" * 60)
    
    commands = [
        f"# Test split-region analysis with optimal transition radius",
        f"py run_dynesty_split_regions.py --xi gr --transition_radius {optimal_transition_radius:.1f} --analyze_regions --compare_regions",
        "",
        f"# Aggressive run for i9 + RTX 5090",
        f"py run_dynesty_split_regions.py --xi gr --transition_radius {optimal_transition_radius:.1f} --nlive 2000 --maxcall 2000000 --num_threads 16 --analyze_regions --compare_regions",
        "",
        f"# Test different xi models",
        f"py run_dynesty_split_regions.py --xi enhanced --transition_radius {optimal_transition_radius:.1f} --analyze_regions --compare_regions",
        f"py run_dynesty_split_regions.py --xi power --transition_radius {optimal_transition_radius:.1f} --analyze_regions --compare_regions",
        f"py run_dynesty_split_regions.py --xi grav_color --transition_radius {optimal_transition_radius:.1f} --analyze_regions --compare_regions",
        "",
        f"# Quick test with fewer stars",
        f"py run_dynesty_split_regions.py --xi gr --transition_radius {optimal_transition_radius:.1f} --max_sample_gaia 10000 --nlive 500 --maxcall 100000 --analyze_regions"
    ]
    
    for cmd in commands:
        logger.info(cmd)
    
    logger.info("=" * 60)

def main():
    """Main function for testing split-region approach."""
    parser = argparse.ArgumentParser(description='Test split-region analysis approach')
    parser.add_argument('--analyze_trends', action='store_true',
                       help='Analyze velocity trends vs radius')
    parser.add_argument('--find_transition', action='store_true',
                       help='Find optimal transition radius')
    parser.add_argument('--max_radius', type=float, default=25.0,
                       help='Maximum radius to analyze (kpc)')
    parser.add_argument('--max_sample_gaia', type=int, default=50000,
                       help='Maximum number of Gaia stars to use')
    
    args = parser.parse_args()
    
    if not DATA_IO_AVAILABLE:
        logger.error("data_io not available")
        return
    
    logger.info("Loading existing Gaia data for split-region analysis...")
    gaia_data = load_existing_gaia_data(sample_max=args.max_sample_gaia)
    
    if gaia_data is None:
        logger.error("Failed to load Gaia data")
        return
    
    optimal_transition_radius = 12.0  # Default
    
    if args.analyze_trends:
        logger.info("Analyzing velocity trends...")
        analysis_results = analyze_velocity_trends(gaia_data, args.max_radius)
        plot_velocity_trends(analysis_results)
        optimal_transition_radius = analysis_results['suggested_transition']
    
    if args.find_transition:
        logger.info("Finding optimal transition radius...")
        results, optimal_transition_radius = test_transition_radii(gaia_data, args.max_radius)
    
    # Generate CLI commands
    generate_cli_commands(optimal_transition_radius)
    
    logger.info("Split-region analysis test complete!")

if __name__ == "__main__":
    main() 