#!/usr/bin/env python3
"""
analyze_results.py - Comprehensive analysis of Dynesty sampling results
for the density-dependent metric modification model.

Features:
- Load and analyze Dynesty output files
- Generate parameter statistics and corner plots
- Plot rotation curves with uncertainties
- Visualize xi enhancement factor
- Physical plausibility checks
- LaTeX table generation
- Comparison with observations

Author: Analysis Tools
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless environments
import corner
import argparse
import os
from pathlib import Path
import json
import pandas as pd
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import local modules
try:
    from density_metric2 import (
        v_baryon_total_newtonian_kms,
        rho_baryon_total_midplane_solar_kpc3,
        XI_FUNCTION_MAP,
        xi_gravitational_color,
        R_SUN_KPC
    )
    from data_io import load_gaia
    PHYSICS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import physics modules: {e}")
    print("Some functionality will be limited.")
    PHYSICS_AVAILABLE = False
    R_SUN_KPC = 8.122  # Default value

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DynestyAnalyzer:
    """Main class for analyzing Dynesty results."""
    
    def __init__(self, results_file, output_dir=None):
        """
        Initialize analyzer with results file.
        
        Parameters
        ----------
        results_file : str
            Path to .npz file with Dynesty results
        output_dir : str, optional
            Directory for output plots (default: same as results file)
        """
        self.results_file = Path(results_file)
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        self.output_dir = Path(output_dir) if output_dir else self.results_file.parent
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.load_results()
        
        # Detect model configuration from filename
        self.detect_model_config()
        
    def load_results(self):
        """Load Dynesty results from .npz file."""
        logger.info(f"Loading results from {self.results_file}")
        
        data = np.load(self.results_file)
        if 'param_names' in data:
            self.param_names = data['param_names'].tolist()
        else:
            logger.warning("No param_names found in .npz file — inferring from sample shape")
            n_params = data['samples'].shape[1]

            # Guess based on known order for grav_color
            if n_params == 6:
                self.param_names = [
                    'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g',
                    'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc'
                ]
            elif n_params == 8:
                self.param_names = [
                    'rho_c_solar_kpc3', 'gamma_exp', 'lambda_g',
                    'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
                    'M_bulge_solar', 'a_bulge_kpc'
                ]
            else:
                raise ValueError(f"Unknown parameter set with {n_params} parameters. Please update analyzer.")


        self.samples = data['samples']
        self.weights = data['weights'] if 'weights' in data else np.ones(len(self.samples)) / len(self.samples)
        self.logl = data['logl'] if 'logl' in data else None
        self.logz = data['logz'] if 'logz' in data else None
        self.logzerr = data['logzerr'] if 'logzerr' in data else None
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        logger.info(f"Loaded {len(self.samples)} samples with {self.samples.shape[1]} parameters")
        
        if 'blob' in data and data['blob'] is not None:
            self.blob = data['blob']
            self.rmse_values = self.blob.flatten() if self.blob.ndim > 1 else self.blob
        else:
            self.rmse_values = None
    
    def detect_model_config(self):
        """Detect model configuration from filename."""
        if self.param_names is not None:
            logger.info("⚙️ Skipping model config detection — using param_names from loaded file or shape inference")
            return
        filename = self.results_file.stem
        
        # Detect xi type
        if 'grav_color' in filename:
            self.xi_type = 'grav_color'
        elif 'enhanced' in filename:
            self.xi_type = 'enhanced'
        elif 'logistic' in filename:
            self.xi_type = 'logistic'
        else:
            self.xi_type = 'power'  # default
        
        # Detect components
        self.has_bulge = 'B' in filename and filename[filename.index('B')+1] in ['f', 'x']
        self.has_thin = 'DT' in filename
        self.has_thick = 'DK' in filename
        self.has_gas = 'G' in filename and filename[filename.index('G')+1] in ['f', 'x']
        
        logger.info(f"Detected configuration: xi={self.xi_type}, "
                   f"bulge={self.has_bulge}, thin={self.has_thin}, "
                   f"thick={self.has_thick}, gas={self.has_gas}")
        
        # Define parameter names based on detected config
        self.param_names = []
        self.param_labels = []
        
        # Xi parameters
        if self.xi_type == 'grav_color':
            self.param_names.extend(['rho_c_solar_kpc3', 'gamma_exp', 'lambda_g'])
            self.param_labels.extend([r'$\rho_c$ (M$_\odot$/kpc$^3$)', r'$\gamma$', r'$\lambda_g$'])
        else:
            self.param_names.extend(['rho_c_solar_kpc3', 'n_exp'])
            self.param_labels.extend([r'$\rho_c$ (M$_\odot$/kpc$^3$)', r'$n$'])
        
        # Component parameters
        if self.has_thin:
            self.param_names.extend(['M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc'])
            self.param_labels.extend([r'$M_{\rm thin}$ (M$_\odot$)', r'$R_{d,\rm thin}$ (kpc)', r'$h_{z,\rm thin}$ (kpc)'])
        
        if self.has_thick:
            self.param_names.extend(['M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc'])
            self.param_labels.extend([r'$M_{\rm thick}$ (M$_\odot$)', r'$R_{d,\rm thick}$ (kpc)', r'$h_{z,\rm thick}$ (kpc)'])
        
        if self.has_bulge:
            self.param_names.extend(['M_bulge_solar', 'a_bulge_kpc'])
            self.param_labels.extend([r'$M_{\rm bulge}$ (M$_\odot$)', r'$a_{\rm bulge}$ (kpc)'])
        
        if self.has_gas:
            self.param_names.extend(['M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'])
            self.param_labels.extend([r'$M_{\rm gas}$ (M$_\odot$)', r'$R_{d,\rm gas}$ (kpc)', r'$h_{z,\rm gas}$ (kpc)'])
    
    def get_parameter_stats(self):
        """Calculate parameter statistics."""
        stats_dict = {}
        
        for i, (name, label) in enumerate(zip(self.param_names, self.param_labels)):
            if i >= self.samples.shape[1]:
                logger.warning(f"Parameter {name} beyond sample dimensions")
                continue
                
            # Weighted statistics
            values = self.samples[:, i]
            median = self.weighted_quantile(values, 0.5, self.weights)
            q16 = self.weighted_quantile(values, 0.16, self.weights)
            q84 = self.weighted_quantile(values, 0.84, self.weights)
            mean = np.average(values, weights=self.weights)
            std = np.sqrt(np.average((values - mean)**2, weights=self.weights))
            
            stats_dict[name] = {
                'median': median,
                'mean': mean,
                'std': std,
                'q16': q16,
                'q84': q84,
                'err_low': median - q16,
                'err_high': q84 - median,
                'label': label
            }
        
        return stats_dict
    
    @staticmethod
    def weighted_quantile(values, quantile, weights):
        """Calculate weighted quantile."""
        indices = np.argsort(values)
        sorted_values = values[indices]
        sorted_weights = weights[indices]
        cumsum = np.cumsum(sorted_weights)
        return np.interp(quantile, cumsum, sorted_values)
    
    def print_summary(self, stats_dict=None):
        """Print parameter summary."""
        if stats_dict is None:
            stats_dict = self.get_parameter_stats()
        
        print("\n" + "="*80)
        print("PARAMETER SUMMARY")
        print("="*80)
        
        if self.logz is not None:
            print(f"\nModel Evidence: log(Z) = {self.logz[-1]:.3f}")
            if self.logzerr is not None:
                print(f"                        ± {self.logzerr[-1]:.3f}")
        
        if self.rmse_values is not None:
            rmse_median = np.median(self.rmse_values)
            print(f"\nMedian RMSE: {rmse_median:.2f} km/s")
        
        print(f"\nParameters ({self.xi_type} xi function):")
        print("-"*80)
        print(f"{'Parameter':<25} {'Median':<15} {'Mean ± Std':<20} {'68% Interval':<25}")
        print("-"*80)
        
        for name, stats in stats_dict.items():
            median_str = self.format_value(stats['median'], name)
            mean_std_str = f"{self.format_value(stats['mean'], name)} ± {self.format_value(stats['std'], name)}"
            interval_str = f"[{self.format_value(stats['q16'], name)}, {self.format_value(stats['q84'], name)}]"
            
            print(f"{stats['label']:<25} {median_str:<15} {mean_std_str:<20} {interval_str:<25}")
        
        # Calculate total mass if components present
        total_mass = 0
        mass_components = ['M_disk_thin_solar', 'M_disk_thick_solar', 'M_bulge_solar', 'M_gas_solar']
        for comp in mass_components:
            if comp in stats_dict:
                total_mass += stats_dict[comp]['median']
        
        if total_mass > 0:
            print("-"*80)
            print(f"Total baryonic mass: {total_mass:.2e} M☉")
        
        print("="*80)
    
    def format_value(self, value, param_name):
        """Format parameter value with appropriate precision."""
        if 'M_' in param_name and 'solar' in param_name:
            return f"{value:.2e}"
        elif 'rho_c' in param_name:
            return f"{value:.2e}"
        elif any(x in param_name for x in ['R_d', 'h_z', 'a_']):
            return f"{value:.3f}"
        elif param_name in ['n_exp', 'gamma_exp', 'lambda_g']:
            return f"{value:.3f}"
        else:
            return f"{value:.3g}"
    
    def plot_corner(self, save=True):
        """Generate corner plot of parameters."""
        logger.info("Generating corner plot...")
        
        # Prepare data
        n_params = min(len(self.param_labels), self.samples.shape[1])
        samples_to_plot = self.samples[:, :n_params]
        
        # Create corner plot
        fig = corner.corner(
            samples_to_plot,
            labels=self.param_labels[:n_params],
            weights=self.weights,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            smooth=1.0,
            smooth1d=1.0,
            plot_density=True,
            plot_datapoints=True,
            fill_contours=True,
            levels=(0.68, 0.95),
            color='blue',
            bins=30,
            hist_kwargs={}
        )
        
        fig.suptitle(f"Parameter Distributions - {self.xi_type} model", fontsize=16, y=0.98)
        
        if save:
            output_file = self.output_dir / f"corner_plot_{self.xi_type}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Corner plot saved to {output_file}")
        
        return fig
    
    def plot_rotation_curve(self, gaia_data=None, n_samples=100, save=True):
        """Plot rotation curve with uncertainties."""
        if not PHYSICS_AVAILABLE:
            logger.warning("Physics modules not available. Cannot plot rotation curve.")
            return None
        
        logger.info("Generating rotation curve plot...")
        
        # Radial grid
        R_grid = np.linspace(3, 30, 100)
        
        # Get random posterior samples
        indices = np.random.choice(len(self.samples), size=n_samples, p=self.weights)
        
        # Calculate rotation curves for each sample
        v_curves = []
        v_newton_curves = []
        
        for idx in indices:
            params_dict = self.create_params_dict(self.samples[idx])
            
            # Calculate Newtonian velocity
            v_newton = v_baryon_total_newtonian_kms(R_grid, params_dict)
            v_newton_curves.append(v_newton)
            
            # Calculate modified velocity
            v_mod = self.calculate_modified_velocity(R_grid, params_dict)
            v_curves.append(v_mod)
        
        v_curves = np.array(v_curves)
        v_newton_curves = np.array(v_newton_curves)
        
        # Calculate percentiles
        v_median = np.percentile(v_curves, 50, axis=0)
        v_16 = np.percentile(v_curves, 16, axis=0)
        v_84 = np.percentile(v_curves, 84, axis=0)
        
        v_newton_median = np.percentile(v_newton_curves, 50, axis=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot model
        ax.plot(R_grid, v_median, 'b-', linewidth=2, label=f'{self.xi_type} model')
        ax.fill_between(R_grid, v_16, v_84, alpha=0.3, color='blue', label='68% credible interval')
        
        # Plot Newtonian
        ax.plot(R_grid, v_newton_median, 'r--', linewidth=2, label='Newtonian (baryons only)')
        
        # Plot data if available
        if gaia_data is not None:
            ax.errorbar(gaia_data['R_kpc'], gaia_data['v_obs'], 
                       yerr=gaia_data['sigma_v'], fmt='k.', alpha=0.5,
                       markersize=3, label='Gaia DR3 data')
        
        # Solar position
        ax.axvline(R_SUN_KPC, color='orange', linestyle=':', linewidth=2, label=f'R☉ = {R_SUN_KPC} kpc')
        
        ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
        ax.set_ylabel('Circular Velocity (km/s)', fontsize=14)
        ax.set_title(f'Milky Way Rotation Curve - {self.xi_type} model', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(3, 30)
        ax.set_ylim(100, 350)
        
        if save:
            output_file = self.output_dir / f"rotation_curve_{self.xi_type}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Rotation curve saved to {output_file}")
        
        return fig
    
    def plot_xi_profile(self, save=True):
        """Plot xi enhancement factor as function of radius."""
        if not PHYSICS_AVAILABLE:
            logger.warning("Physics modules not available. Cannot plot xi profile.")
            return None
        
        logger.info("Generating xi profile plot...")
        
        # Get median parameters
        stats = self.get_parameter_stats()
        median_params = {name: stats[name]['median'] for name in stats}
        params_dict = self.create_params_dict(median_params)
        
        # Radial grid
        R_grid = np.linspace(3, 30, 100)
        
        # Calculate density and xi
        rho_values = rho_baryon_total_midplane_solar_kpc3(R_grid, params_dict)
        xi_values = self.calculate_xi(rho_values, params_dict)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Xi profile
        ax1.plot(R_grid, xi_values, 'b-', linewidth=2)
        ax1.axvline(R_SUN_KPC, color='orange', linestyle=':', linewidth=2, label=f'R☉')
        ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Newtonian (ξ=1)')
        ax1.set_ylabel('Enhancement Factor ξ', fontsize=14)
        ax1.set_title(f'Gravitational Enhancement Profile - {self.xi_type} model', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0.5, 5.5)
        
        # Density profile
        ax2.semilogy(R_grid, rho_values, 'g-', linewidth=2)
        ax2.axvline(R_SUN_KPC, color='orange', linestyle=':', linewidth=2)
        if 'rho_c_solar_kpc3' in median_params:
            ax2.axhline(median_params['rho_c_solar_kpc3'], color='red', linestyle='--', 
                       linewidth=2, label=f'ρ_c = {median_params["rho_c_solar_kpc3"]:.2e} M☉/kpc³')
        ax2.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
        ax2.set_ylabel('Midplane Density (M☉/kpc³)', fontsize=14)
        ax2.set_title('Baryonic Density Profile', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(3, 30)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / f"xi_profile_{self.xi_type}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Xi profile saved to {output_file}")
        
        return fig
    
    def create_params_dict(self, param_values):
        """Create parameter dictionary from array or dict."""
        if isinstance(param_values, dict):
            params_dict = param_values.copy()
        else:
            params_dict = {}
            for i, name in enumerate(self.param_names):
                if i < len(param_values):
                    params_dict[name] = param_values[i]
        
        # Add component flags
        params_dict['include_disk_thin'] = self.has_thin
        params_dict['include_disk_thick'] = self.has_thick
        params_dict['include_bulge'] = self.has_bulge
        params_dict['include_bulge_density'] = self.has_bulge
        params_dict['include_gas'] = self.has_gas
        
        return params_dict
    
    def calculate_xi(self, rho_values, params_dict):
        """Calculate xi values for given densities."""
        if self.xi_type == 'grav_color':
            xi_values = xi_gravitational_color(
                rho_values,
                params_dict.get('rho_c_solar_kpc3', 5e7),
                params_dict.get('gamma_exp', 2.0),
                params_dict.get('lambda_g', 1.5)
            )
        else:
            xi_func = XI_FUNCTION_MAP.get(self.xi_type, XI_FUNCTION_MAP['power'])
            xi_values = xi_func(
                rho_values,
                params_dict.get('rho_c_solar_kpc3', 5e8),
                params_dict.get('n_exp', 1.5)
            )
        
        return xi_values
    
    def calculate_modified_velocity(self, R_values, params_dict):
        """Calculate modified velocity including xi enhancement."""
        v_newton = v_baryon_total_newtonian_kms(R_values, params_dict)
        rho_values = rho_baryon_total_midplane_solar_kpc3(R_values, params_dict)
        xi_values = self.calculate_xi(rho_values, params_dict)
        return v_newton * np.sqrt(xi_values)
    
    def generate_latex_table(self, save=True):
        """Generate LaTeX table of results."""
        logger.info("Generating LaTeX table...")
        
        stats = self.get_parameter_stats()
        
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Best-fit parameters for " + self.xi_type.replace('_', ' ') + " model}")
        latex_lines.append("\\begin{tabular}{lcc}")
        latex_lines.append("\\hline")
        latex_lines.append("Parameter & Median & 68\\% Interval \\\\")
        latex_lines.append("\\hline")
        
        for name, stat in stats.items():
            label = stat['label'].replace('$', '')
            median = self.format_value(stat['median'], name)
            interval = f"[{self.format_value(stat['q16'], name)}, {self.format_value(stat['q84'], name)}]"
            latex_lines.append(f"{label} & {median} & {interval} \\\\")
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        latex_content = '\n'.join(latex_lines)
        
        if save:
            output_file = self.output_dir / f"results_table_{self.xi_type}.tex"
            with open(output_file, 'w') as f:
                f.write(latex_content)
            logger.info(f"LaTeX table saved to {output_file}")
        
        return latex_content
    
    def check_physical_plausibility(self):
        """Check if results are physically plausible."""
        logger.info("Checking physical plausibility...")
        
        stats = self.get_parameter_stats()
        issues = []
        
        # Check masses
        mass_components = {
            'M_disk_thin_solar': (3e10, 8e10),
            'M_disk_thick_solar': (5e9, 3e10),
            'M_bulge_solar': (0.5e10, 3e10),
            'M_gas_solar': (5e9, 5e10)
        }
        
        total_mass = 0
        for comp, (min_val, max_val) in mass_components.items():
            if comp in stats:
                mass = stats[comp]['median']
                total_mass += mass
                if mass < min_val or mass > max_val:
                    issues.append(f"{comp}: {mass:.2e} outside expected range [{min_val:.2e}, {max_val:.2e}]")
        
        if total_mass < 5e10 or total_mass > 2e11:
            issues.append(f"Total mass {total_mass:.2e} outside expected range [5e10, 2e11]")
        
        # Check scale lengths
        if 'R_d_thick_kpc' in stats and 'R_d_thin_kpc' in stats:
            if stats['R_d_thick_kpc']['median'] < stats['R_d_thin_kpc']['median']:
                issues.append("Thick disk scale length < thin disk scale length")
        
        # Check scale heights
        if 'h_z_thick_kpc' in stats and 'h_z_thin_kpc' in stats:
            if stats['h_z_thick_kpc']['median'] < 2 * stats['h_z_thin_kpc']['median']:
                issues.append("Thick disk not thick enough compared to thin disk")
        
        # Print results
        if issues:
            print("\n⚠️  PHYSICAL PLAUSIBILITY ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ All parameters pass physical plausibility checks")
        
        return len(issues) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze Dynesty sampling results for density-metric model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('results_file', type=str,
                       help='Path to .npz file with Dynesty results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results file)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--gaia_data', type=str, default=None,
                       help='Path to Gaia data for comparison')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of posterior samples for uncertainty bands')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DynestyAnalyzer(args.results_file, args.output_dir)
    
    # Print summary
    analyzer.print_summary()
    
    # Check physical plausibility
    analyzer.check_physical_plausibility()
    
    # Generate plots
    if not args.no_plots:
        # Corner plot
        analyzer.plot_corner()
        
        # Load Gaia data if requested
        gaia_data = None
        if args.gaia_data or PHYSICS_AVAILABLE:
            try:
                gaia_data = load_gaia(
                    processed_cache_filename="gaia_cache/gaia_query_cache_DR3_processed_for_fit.parquet"
                )
            except:
                logger.warning("Could not load Gaia data")
        
        # Rotation curve
        analyzer.plot_rotation_curve(gaia_data, n_samples=args.n_samples)
        
        # Xi profile
        analyzer.plot_xi_profile()
    
    # Generate LaTeX table
    analyzer.generate_latex_table()
    
    print(f"\n✨ Analysis complete! Results saved to {analyzer.output_dir}")


if __name__ == "__main__":
    main()