#!/usr/bin/env python3
"""
analyze_bimodal_results.py - Analyze and separate modes from bimodal sampling results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

class BimodalAnalyzer:
    """Analyze and separate bimodal distributions from dynesty results."""
    
    def __init__(self, samples_file):
        """Load samples and weights from dynesty output."""
        data = np.load(samples_file)
        self.samples = data['samples']
        self.weights = data['weights']
        self.param_names = [
            'rho_c_solar_kpc3', 'n_exp',
            'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
            'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
            'M_bulge_solar', 'a_bulge_kpc',
            'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
        ]
        self.n_samples, self.n_params = self.samples.shape
        logger.info(f"Loaded {self.n_samples} samples with {self.n_params} parameters")
        
    def find_modes_gmm(self, n_components=2, param_subset=None):
        """Use Gaussian Mixture Model to identify modes."""
        if param_subset is None:
            param_subset = list(range(self.n_params))
            
        X = self.samples[:, param_subset]
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, 
                            covariance_type='full',
                            n_init=10)
        
        # Weight the fit by sample weights
        gmm.fit(X, sample_weight=self.weights)
        
        # Get cluster assignments
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        
        # Calculate mode statistics
        modes = {}
        for i in range(n_components):
            mask = labels == i
            mode_weight = np.sum(self.weights[mask])
            modes[i] = {
                'weight': mode_weight / np.sum(self.weights),
                'n_samples': np.sum(mask),
                'center': gmm.means_[i],
                'covariance': gmm.covariances_[i],
                'samples_mask': mask
            }
            
        return modes, labels, probs
    
    def check_physical_validity(self, samples):
        """Check which samples pass physical constraints."""
        n_samples = len(samples)
        valid_mask = np.ones(n_samples, dtype=bool)
        
        # Extract parameters
        param_dict = {name: samples[:, i] for i, name in enumerate(self.param_names)}
        
        # Check thick disk > thin disk scale length
        if 'R_d_thick_kpc' in param_dict and 'R_d_thin_kpc' in param_dict:
            thick_idx = self.param_names.index('R_d_thick_kpc')
            thin_idx = self.param_names.index('R_d_thin_kpc')
            valid_mask &= samples[:, thick_idx] > samples[:, thin_idx] * 1.1
            
        # Check thick disk scale height > 2x thin disk
        if 'h_z_thick_kpc' in param_dict and 'h_z_thin_kpc' in param_dict:
            thick_h_idx = self.param_names.index('h_z_thick_kpc')
            thin_h_idx = self.param_names.index('h_z_thin_kpc')
            valid_mask &= samples[:, thick_h_idx] > samples[:, thin_h_idx] * 2.0
            
        # Check mass ratios
        if 'M_disk_thick_solar' in param_dict and 'M_disk_thin_solar' in param_dict:
            thick_m_idx = self.param_names.index('M_disk_thick_solar')
            thin_m_idx = self.param_names.index('M_disk_thin_solar')
            ratio = samples[:, thick_m_idx] / samples[:, thin_m_idx]
            valid_mask &= (ratio > 0.1) & (ratio < 0.7)
            
        return valid_mask
    
    def separate_physical_modes(self):
        """Separate samples into physically valid and invalid modes."""
        valid_mask = self.check_physical_validity(self.samples)
        
        physical_mode = {
            'samples': self.samples[valid_mask],
            'weights': self.weights[valid_mask],
            'fraction': np.sum(valid_mask) / len(valid_mask),
            'weight_fraction': np.sum(self.weights[valid_mask]) / np.sum(self.weights)
        }
        
        unphysical_mode = {
            'samples': self.samples[~valid_mask],
            'weights': self.weights[~valid_mask],
            'fraction': np.sum(~valid_mask) / len(valid_mask),
            'weight_fraction': np.sum(self.weights[~valid_mask]) / np.sum(self.weights)
        }
        
        logger.info(f"Physical mode: {physical_mode['fraction']:.1%} of samples, "
                   f"{physical_mode['weight_fraction']:.1%} of weight")
        
        return physical_mode, unphysical_mode
    
    def get_mode_parameters(self, mode_samples, mode_weights):
        """Get best-fit parameters for a specific mode."""
        # Weighted median
        medians = np.zeros(self.n_params)
        for i in range(self.n_params):
            sorted_idx = np.argsort(mode_samples[:, i])
            sorted_vals = mode_samples[sorted_idx, i]
            sorted_weights = mode_weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            medians[i] = sorted_vals[np.searchsorted(cumsum, 0.5 * cumsum[-1])]
            
        # MAD (median absolute deviation)
        mads = np.zeros(self.n_params)
        for i in range(self.n_params):
            deviations = np.abs(mode_samples[:, i] - medians[i])
            sorted_idx = np.argsort(deviations)
            sorted_devs = deviations[sorted_idx]
            sorted_weights = mode_weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            mads[i] = sorted_devs[np.searchsorted(cumsum, 0.5 * cumsum[-1])]
            
        return dict(zip(self.param_names, medians)), dict(zip(self.param_names, mads))
    
    def plot_mode_comparison(self, save_path=None):
        """Plot parameter distributions showing modes."""
        physical_mode, unphysical_mode = self.separate_physical_modes()
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        for i, param_name in enumerate(self.param_names):
            ax = axes[i]
            
            # Plot full distribution
            ax.hist(self.samples[:, i], bins=50, alpha=0.3, 
                   weights=self.weights, density=True, label='All')
            
            # Plot physical mode
            if len(physical_mode['samples']) > 0:
                ax.hist(physical_mode['samples'][:, i], bins=30, alpha=0.5,
                       weights=physical_mode['weights'], density=True, 
                       color='green', label='Physical')
                
            # Plot unphysical mode  
            if len(unphysical_mode['samples']) > 0:
                ax.hist(unphysical_mode['samples'][:, i], bins=30, alpha=0.5,
                       weights=unphysical_mode['weights'], density=True,
                       color='red', label='Unphysical')
                
            ax.set_xlabel(param_name.replace('_', ' '))
            ax.set_ylabel('Probability Density')
            if i == 0:
                ax.legend()
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def export_physical_mode(self, output_file):
        """Export only the physically valid mode for restart."""
        physical_mode, _ = self.separate_physical_modes()
        
        if len(physical_mode['samples']) == 0:
            logger.error("No physically valid samples found!")
            return
            
        medians, mads = self.get_mode_parameters(
            physical_mode['samples'], 
            physical_mode['weights']
        )
        
        # Save for restart
        np.savez(output_file,
                samples=physical_mode['samples'],
                weights=physical_mode['weights'],
                medians=list(medians.values()),
                mads=list(mads.values()),
                param_names=self.param_names)
        
        logger.info(f"Exported physical mode to {output_file}")
        logger.info("Median parameters:")
        for param, value in medians.items():
            logger.info(f"  {param}: {value:.3e} ± {mads[param]:.3e}")