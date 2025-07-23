#!/usr/bin/env python3
"""
enhanced_light_propagation.py - Enhanced DDMM light propagation module

This module enhances the existing validate_ddmm.py with:
1. Direct DDMM redshift formula: 1 + z = [(ρ_o + ρ_c)/(ρ_e + ρ_c)]^(α/2)
2. Full path integral implementation: ln(1 + z) = (α/2) ∫ (1/(ρ + ρ_c)) · (dρ/ds) ds
3. Realistic cosmic web density modeling
4. Multiple light path scenarios (through voids, clusters, filaments)
5. Comprehensive comparison with observations

Version: 3.0 (Enhanced light propagation)
"""

import numpy as np
from scipy.integrate import odeint, quad, simpson
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Physical constants
C_KMS = 299792.458  # Speed of light in km/s
H0 = 70.0  # Hubble constant in km/s/Mpc
OMEGA_M = 0.3  # Matter density parameter
OMEGA_LAMBDA = 0.7  # Dark energy density parameter
RHO_CRIT_0 = 2.775e11 * (H0/100)**2  # Critical density today in M☉/Mpc³

@dataclass
class CosmicWebStructure:
    """Represents cosmic web density field"""
    box_size: float  # Mpc
    resolution: int
    density_field: np.ndarray  # 3D density array in M☉/kpc³
    interpolator: RegularGridInterpolator


class EnhancedDDMMLightPropagation:
    """Enhanced DDMM light propagation with multiple models"""
    
    def __init__(self, rho_c: float, n_exp: float, A: float, xi_func=None):
        """
        Initialize enhanced light propagation calculator.
        
        Parameters:
        -----------
        rho_c : float
            Critical density scale in M☉/kpc³
        n_exp : float
            Power law exponent
        A : float
            Amplitude parameter
        xi_func : callable, optional
            Custom xi function
        """
        self.rho_c = rho_c
        self.n_exp = n_exp
        self.A = A
        self.xi_func = xi_func or self._default_xi_func
        
        # Initialize cosmic web
        self.cosmic_web = None
        
    def _default_xi_func(self, rho):
        """Default power-law xi function with cap"""
        xi_uncapped = self.A * (rho / self.rho_c)**(-self.n_exp)
        return np.minimum(xi_uncapped, 5.0)
    
    def calculate_xi(self, rho):
        """Calculate xi ensuring proper array handling"""
        xi = self.xi_func(rho)
        
        # Ensure output matches input shape
        if np.isscalar(rho) and hasattr(xi, '__len__'):
            return float(xi[0] if len(xi) > 0 else xi)
        return xi
    
    # ========================================================================
    # Direct DDMM Redshift Formula
    # ========================================================================
    
    def redshift_direct_formula(self, rho_observer: float, rho_emitter: float) -> float:
        """
        Calculate redshift using direct DDMM formula.
        
        1 + z = [(ρ_o + ρ_c)/(ρ_e + ρ_c)]^(α/2)
        
        where α relates to our parameters as: α = n_exp * A
        
        Parameters:
        -----------
        rho_observer : float
            Density at observer location in M☉/kpc³
        rho_emitter : float
            Density at emitter location in M☉/kpc³
            
        Returns:
        --------
        z : float
            Redshift
        """
        alpha = self.n_exp * self.A  # Effective coupling
        z_plus_1 = ((rho_observer + self.rho_c) / (rho_emitter + self.rho_c))**(alpha/2)
        return z_plus_1 - 1
    
    # ========================================================================
    # Path Integral Formulation
    # ========================================================================
    
    def redshift_path_integral(self, path_coords: np.ndarray, 
                              densities: Optional[np.ndarray] = None) -> float:
        """
        Calculate redshift via path integral through density field.
        
        ln(1 + z) = (α/2) ∫ (1/(ρ + ρ_c)) · (dρ/ds) ds
        
        Parameters:
        -----------
        path_coords : array, shape (N, 3)
            Coordinates along light path in Mpc
        densities : array, shape (N,), optional
            Density values along path in M☉/kpc³
            
        Returns:
        --------
        z : float
            Total accumulated redshift
        """
        if densities is None:
            if self.cosmic_web is None:
                raise ValueError("Either densities or cosmic_web must be provided")
            densities = self.get_density_along_path(path_coords)
        
        # Calculate path length elements
        ds = np.sqrt(np.sum(np.diff(path_coords, axis=0)**2, axis=1))
        ds = np.append(ds, ds[-1])  # Extend for same length as densities
        
        # Calculate density gradient using more sophisticated method
        if len(densities) > 3:
            # Use 5-point stencil for better gradient estimation
            drho_ds = np.gradient(densities, ds, edge_order=2)
        else:
            drho_ds = np.gradient(densities)
            if len(ds) > 1:
                drho_ds = drho_ds / np.mean(ds)
        
        # Integrand: (1/(ρ + ρ_c)) · (dρ/ds)
        # Add small epsilon to avoid division by zero
        integrand = drho_ds / (densities + self.rho_c + 1e-100)
        
        # Effective coupling
        alpha = self.n_exp * self.A
        
        # Numerical integration using Simpson's rule for better accuracy
        if len(integrand) > 2:
            integral = simpson(integrand * ds) if len(integrand) % 2 == 1 else np.trapz(integrand * ds)
        else:
            integral = np.sum(integrand * ds)
        
        ln_1_plus_z = (alpha / 2) * integral
        
        return np.exp(ln_1_plus_z) - 1
    
    # ========================================================================
    # Cosmic Web Modeling
    # ========================================================================
    
    def create_cosmic_web(self, box_size: float = 500, resolution: int = 256,
                         n_clusters: int = 50, n_voids: int = 100,
                         n_filaments: int = 200) -> CosmicWebStructure:
        """
        Create realistic cosmic web density field.
        
        Parameters:
        -----------
        box_size : float
            Size of simulation box in Mpc
        resolution : int
            Grid resolution per dimension
        n_clusters : int
            Number of galaxy clusters
        n_voids : int
            Number of cosmic voids
        n_filaments : int
            Number of filamentary structures
        """
        logger.info(f"Creating cosmic web: {box_size} Mpc, {resolution}³ grid")
        
        # Initialize with mean density
        mean_density = RHO_CRIT_0 * OMEGA_M / 1e9  # Convert to M☉/kpc³
        grid = np.ones((resolution, resolution, resolution)) * mean_density
        
        # Coordinate arrays
        x = np.linspace(0, box_size, resolution)
        y = np.linspace(0, box_size, resolution)
        z = np.linspace(0, box_size, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Add clusters (high density peaks)
        logger.info(f"  Adding {n_clusters} clusters...")
        for _ in range(n_clusters):
            center = np.random.uniform(0, box_size, 3)
            mass = 10**(np.random.uniform(14, 15.5))  # M☉
            r_200 = (mass / (200 * 4*np.pi/3 * mean_density * 1e9))**(1/3)  # Mpc
            
            # NFW profile
            r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            r_s = r_200 / 5  # Concentration c = 5
            rho_s = mass / (4 * np.pi * r_s**3 * (np.log(6) - 5/2))
            
            # Add to grid with cutoff at r_200
            mask = r < r_200
            grid[mask] += (rho_s / ((r[mask]/r_s) * (1 + r[mask]/r_s)**2)) / 1e9  # Convert to M☉/kpc³
        
        # Add voids (underdense regions)
        logger.info(f"  Adding {n_voids} voids...")
        for _ in range(n_voids):
            center = np.random.uniform(0, box_size, 3)
            radius = np.random.uniform(10, 50)  # Mpc
            
            r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            
            # Void profile (smooth transition)
            void_profile = 0.5 * (1 - np.tanh((radius - r) / 5))
            grid *= (0.1 + 0.9 * void_profile)  # 90% underdensity at center
        
        # Add filaments connecting clusters
        logger.info(f"  Adding {n_filaments} filaments...")
        cluster_positions = np.random.uniform(0, box_size, (n_clusters, 3))
        
        for _ in range(n_filaments):
            # Random pairs of clusters
            idx1, idx2 = np.random.choice(n_clusters, 2, replace=False)
            p1, p2 = cluster_positions[idx1], cluster_positions[idx2]
            
            # Cylindrical filament
            for t in np.linspace(0, 1, 50):
                center = p1 + t * (p2 - p1)
                r_perp = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + 
                               (Z - center[2])**2 - ((X - center[0])*(p2[0] - p1[0]) +
                                                    (Y - center[1])*(p2[1] - p1[1]) +
                                                    (Z - center[2])*(p2[2] - p1[2]))**2 / 
                                                    np.sum((p2 - p1)**2))
                
                fil_radius = 5  # Mpc
                mask = r_perp < fil_radius
                grid[mask] *= (1 + 2 * np.exp(-(r_perp[mask]/fil_radius)**2))
        
        # Smooth the field
        grid = gaussian_filter(grid, sigma=1.5)
        
        # Ensure positive densities
        grid = np.maximum(grid, mean_density * 0.01)
        
        # Create interpolator
        interpolator = RegularGridInterpolator((x, y, z), grid, 
                                             bounds_error=False, 
                                             fill_value=mean_density)
        
        self.cosmic_web = CosmicWebStructure(
            box_size=box_size,
            resolution=resolution,
            density_field=grid,
            interpolator=interpolator
        )
        
        logger.info(f"  Density range: [{grid.min():.2e}, {grid.max():.2e}] M☉/kpc³")
        logger.info(f"  Mean density: {grid.mean():.2e} M☉/kpc³")
        
        return self.cosmic_web
    
    def get_density_along_path(self, path_coords: np.ndarray) -> np.ndarray:
        """Get density values along a path through cosmic web"""
        if self.cosmic_web is None:
            raise ValueError("Cosmic web not initialized")
        
        # Ensure path is within box
        path_coords = np.array(path_coords)
        path_coords = np.clip(path_coords, 0, self.cosmic_web.box_size - 1e-6)
        
        densities = self.cosmic_web.interpolator(path_coords)
        return densities
    
    # ========================================================================
    # Advanced Light Propagation Models
    # ========================================================================
    
    def calculate_distances_pure_ddmm(self, z_targets: np.ndarray,
                                    n_paths: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate distances assuming ALL redshift from DDMM (no expansion).
        
        Returns dict with:
        - z_mean: mean redshift for each target
        - z_std: standard deviation from path variations
        - d_mean: mean distance
        - d_std: distance standard deviation
        """
        if self.cosmic_web is None:
            self.create_cosmic_web()
        
        results = {
            'z_mean': np.zeros_like(z_targets),
            'z_std': np.zeros_like(z_targets),
            'd_mean': np.zeros_like(z_targets),
            'd_std': np.zeros_like(z_targets)
        }
        
        for i, z_target in enumerate(z_targets):
            z_paths = []
            d_paths = []
            
            # Initial distance guess
            d_guess = C_KMS * z_target / H0
            
            for _ in range(n_paths):
                # Random direction
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2*np.pi)
                direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                # Observer at center of box
                observer_pos = np.array([self.cosmic_web.box_size/2] * 3)
                
                # Iterate to find distance giving correct redshift
                d_trial = d_guess
                for iteration in range(10):
                    # Create path
                    n_points = 200
                    t = np.linspace(0, d_trial, n_points)
                    path_coords = observer_pos[np.newaxis, :] + t[:, np.newaxis] * direction
                    
                    # Ensure within box (with periodic boundaries)
                    path_coords = path_coords % self.cosmic_web.box_size
                    
                    # Get densities
                    densities = self.get_density_along_path(path_coords)
                    
                    # Calculate redshift via path integral
                    z_calc = self.redshift_path_integral(path_coords, densities)
                    
                    # Also add endpoint contribution
                    rho_obs = densities[0]
                    rho_em = densities[-1]
                    z_endpoint = self.redshift_direct_formula(rho_obs, rho_em)
                    
                    # Combined redshift (geometric mean)
                    z_total = np.sqrt((1 + z_calc) * (1 + z_endpoint)) - 1
                    
                    # Update distance estimate
                    if abs(z_total - z_target) / (1 + z_target) < 0.01:
                        break
                    
                    # Newton-Raphson style update
                    d_trial *= (1 + z_target) / (1 + z_total)
                    d_trial = np.clip(d_trial, 0.1, self.cosmic_web.box_size)
                
                z_paths.append(z_total)
                d_paths.append(d_trial)
            
            results['z_mean'][i] = np.mean(z_paths)
            results['z_std'][i] = np.std(z_paths)
            results['d_mean'][i] = np.mean(d_paths)
            results['d_std'][i] = np.std(d_paths)
        
        return results
    
    def calculate_distances_hybrid_model(self, z_targets: np.ndarray,
                                       expansion_fraction: float = 0.7) -> np.ndarray:
        """
        Hybrid model: combines cosmic expansion with DDMM effects.
        
        Parameters:
        -----------
        z_targets : array
            Target redshifts
        expansion_fraction : float
            Fraction of redshift from expansion (rest from DDMM)
            
        Returns:
        --------
        distances : array
            Luminosity distances in Mpc
        """
        distances = []
        
        for z in z_targets:
            # Split redshift
            z_expansion = ((1 + z)**expansion_fraction - 1)
            z_ddmm = ((1 + z)**(1 - expansion_fraction) - 1)
            
            # Standard distance from expansion
            d_expansion = self._luminosity_distance_flrw(z_expansion)
            
            # DDMM correction factor
            # Average cosmic density at this epoch
            rho_avg = RHO_CRIT_0 * OMEGA_M * (1 + z_expansion)**3 / 1e9  # M☉/kpc³
            xi_avg = self.calculate_xi(rho_avg)
            
            # Modified distance
            d_total = d_expansion * xi_avg**(1/4) * (1 + z_ddmm)
            distances.append(d_total)
        
        return np.array(distances)
    
    def _luminosity_distance_flrw(self, z):
        """Standard FLRW luminosity distance"""
        from scipy.integrate import quad
        
        def E(z):
            return np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_LAMBDA)
        
        # Comoving distance
        integral, _ = quad(lambda zp: 1/E(zp), 0, z)
        d_c = C_KMS / H0 * integral
        
        return d_c * (1 + z)
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def plot_cosmic_web_slice(self, slice_index: Optional[int] = None,
                            save_path: Optional[str] = None):
        """Plot a slice through the cosmic web"""
        if self.cosmic_web is None:
            logger.warning("Cosmic web not initialized")
            return
        
        if slice_index is None:
            slice_index = self.cosmic_web.resolution // 2
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get slice
        density_slice = self.cosmic_web.density_field[:, :, slice_index]
        
        # Plot log density
        im = ax.imshow(np.log10(density_slice), 
                      extent=[0, self.cosmic_web.box_size, 0, self.cosmic_web.box_size],
                      cmap='viridis', origin='lower')
        
        ax.set_xlabel('x [Mpc]', fontsize=12)
        ax.set_ylabel('y [Mpc]', fontsize=12)
        ax.set_title(f'Cosmic Web Density (z-slice {slice_index})', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(ρ [M☉/kpc³])', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_path_dependent_redshift(self, z_target: float = 0.5,
                                   n_paths: int = 1000,
                                   save_path: Optional[str] = None):
        """Visualize path-dependent redshift effects"""
        if self.cosmic_web is None:
            self.create_cosmic_web()
        
        # Calculate redshifts along different paths
        void_redshifts = []
        cluster_redshifts = []
        random_redshifts = []
        
        observer_pos = np.array([self.cosmic_web.box_size/2] * 3)
        d_target = C_KMS * z_target / H0
        
        # Find void and cluster regions
        density_flat = self.cosmic_web.density_field.flatten()
        void_threshold = np.percentile(density_flat, 10)
        cluster_threshold = np.percentile(density_flat, 90)
        
        for _ in range(n_paths):
            # Random path
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            path_coords = np.array([observer_pos + t * direction * d_target 
                                  for t in np.linspace(0, 1, 100)])
            path_coords = path_coords % self.cosmic_web.box_size
            
            densities = self.get_density_along_path(path_coords)
            z = self.redshift_path_integral(path_coords, densities)
            
            # Classify path
            mean_density = np.mean(densities[10:-10])  # Exclude endpoints
            if mean_density < void_threshold:
                void_redshifts.append(z)
            elif mean_density > cluster_threshold:
                cluster_redshifts.append(z)
            else:
                random_redshifts.append(z)
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(min(void_redshifts + cluster_redshifts + random_redshifts),
                          max(void_redshifts + cluster_redshifts + random_redshifts), 50)
        
        ax.hist(void_redshifts, bins=bins, alpha=0.5, label=f'Through voids (n={len(void_redshifts)})', 
                color='blue', density=True)
        ax.hist(cluster_redshifts, bins=bins, alpha=0.5, label=f'Through clusters (n={len(cluster_redshifts)})', 
                color='red', density=True)
        ax.hist(random_redshifts, bins=bins, alpha=0.5, label=f'Random paths (n={len(random_redshifts)})', 
                color='green', density=True)
        
        ax.axvline(z_target, color='black', linestyle='--', label=f'Target z = {z_target}')
        ax.axvline(np.mean(void_redshifts), color='blue', linestyle=':',
                  label=f'Mean void z = {np.mean(void_redshifts):.3f}')
        ax.axvline(np.mean(cluster_redshifts), color='red', linestyle=':',
                  label=f'Mean cluster z = {np.mean(cluster_redshifts):.3f}')
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Path-Dependent Redshift in DDMM Theory', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = (f'Void excess: {(np.mean(void_redshifts)/z_target - 1)*100:.1f}%\n'
                     f'Cluster deficit: {(np.mean(cluster_redshifts)/z_target - 1)*100:.1f}%\n'
                     f'Scatter (void): {np.std(void_redshifts)/z_target*100:.1f}%')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def integrate_with_existing_validator(validator_instance, output_dir: Optional[Path] = None):
    """
    Integrate enhanced light propagation into existing validator.
    
    This function enhances the existing validate_ddmm.py test_supernovae method
    with improved light propagation calculations.
    """
    # Extract parameters from validator
    params = validator_instance.model_params
    
    # Create enhanced propagation calculator
    propagator = EnhancedDDMMLightPropagation(
        rho_c=params['rho_c_solar_kpc3'],
        n_exp=params['n_exp'],
        A=params['A']
    )
    
    # Create cosmic web
    logger.info("Creating realistic cosmic web for light propagation...")
    propagator.create_cosmic_web(box_size=500, resolution=128)
    
    # Save cosmic web visualization
    if output_dir:
        propagator.plot_cosmic_web_slice(save_path=output_dir / 'cosmic_web_slice.png')
        propagator.plot_path_dependent_redshift(save_path=output_dir / 'path_dependent_redshift.png')
    
    # Enhanced test function
def integrate_with_existing_validator(validator_instance, output_dir: Optional[Path] = None):
    """
    Integrate enhanced light propagation into existing validator.
    
    This function adds an enhanced supernova test method to the validator
    without replacing the existing dispatcher.
    """
    # Extract parameters from validator
    params = validator_instance.model_params
    
    # Create enhanced propagation calculator
    propagator = EnhancedDDMMLightPropagation(
        rho_c=params['rho_c_solar_kpc3'],
        n_exp=params['n_exp'],
        A=params['A']
    )
    
    # Create cosmic web
    logger.info("Creating realistic cosmic web for light propagation...")
    propagator.create_cosmic_web(box_size=500, resolution=128)
    
    # Save cosmic web visualization
    if output_dir:
        propagator.plot_cosmic_web_slice(save_path=output_dir / 'cosmic_web_slice.png')
        propagator.plot_path_dependent_redshift(save_path=output_dir / 'path_dependent_redshift.png')
    
    # Enhanced test function - note this is now test_supernovae_enhanced
    def test_supernovae_enhanced(self, pantheon_path: Optional[str] = None) -> 'TestResult':
        """Enhanced supernova test with improved DDMM light propagation"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: TYPE IA SUPERNOVAE (Enhanced DDMM Propagation)")
        logger.info("="*60)
        
        # Load data (same as original)
        sn_data = None
        if pantheon_path:
            sn_data = self._load_pantheon_data(pantheon_path)
        
        if sn_data is not None:
            z_obs = sn_data['z']
            mu_obs = sn_data['mu']
            mu_err = sn_data['mu_err']
        else:
            z_obs = np.logspace(-2, 0.3, 50)
            mu_obs = None
            mu_err = None
        
        logger.info("\nTesting enhanced light propagation models:")
        
        # Model 1: Standard ΛCDM
        d_L_lcdm = self._luminosity_distance_lcdm(z_obs)
        mu_lcdm = 5 * np.log10(d_L_lcdm) + 25
        
        # Model 2: Pure DDMM (no expansion) with path variations
        logger.info("\n  Model A: Pure DDMM with cosmic web path integration")
        ddmm_results = propagator.calculate_distances_pure_ddmm(z_obs, n_paths=50)
        mu_ddmm_pure = 5 * np.log10(ddmm_results['d_mean']) + 25
        mu_ddmm_pure_err = 5 * ddmm_results['d_std'] / (ddmm_results['d_mean'] * np.log(10))
        
        # Model 3: Hybrid (70% expansion, 30% DDMM)
        logger.info("\n  Model B: Hybrid (70% expansion + 30% DDMM)")
        d_L_hybrid = propagator.calculate_distances_hybrid_model(z_obs, expansion_fraction=0.7)
        mu_hybrid = 5 * np.log10(d_L_hybrid) + 25
        
        # Model 4: Direct formula comparison
        logger.info("\n  Model C: Direct DDMM formula (endpoint densities)")
        z_direct = []
        for z in z_obs:
            # Assume typical endpoint densities
            rho_obs = 1e8  # Galaxy
            rho_em = 1e8   # Galaxy  
            z_d = propagator.redshift_direct_formula(rho_obs, rho_em)
            z_direct.append(z_d)
        
        # Statistical comparison
        if mu_obs is not None:
            chi2_lcdm = np.sum(((mu_obs - mu_lcdm) / mu_err)**2)
            chi2_pure = np.sum(((mu_obs - mu_ddmm_pure) / mu_err)**2)
            chi2_hybrid = np.sum(((mu_obs - mu_hybrid) / mu_err)**2)
            
            # Calculate AIC/BIC
            n_data = len(mu_obs)
            aic_lcdm = chi2_lcdm + 2 * 2  # 2 parameters
            aic_pure = chi2_pure + 2 * 4   # 4 parameters
            aic_hybrid = chi2_hybrid + 2 * 5  # 5 parameters
            
            logger.info(f"\nStatistical comparison:")
            logger.info(f"  ΛCDM:      χ² = {chi2_lcdm:.1f}, AIC = {aic_lcdm:.1f}")
            logger.info(f"  Pure DDMM: χ² = {chi2_pure:.1f}, AIC = {aic_pure:.1f}")
            logger.info(f"  Hybrid:    χ² = {chi2_hybrid:.1f}, AIC = {aic_hybrid:.1f}")
            logger.info(f"\nPath variation in pure DDMM:")
            logger.info(f"  Mean scatter: {np.mean(mu_ddmm_pure_err):.3f} mag")
            logger.info(f"  Max scatter:  {np.max(mu_ddmm_pure_err):.3f} mag")
            
            best_aic = min(aic_lcdm, aic_pure, aic_hybrid)
            if best_aic == aic_pure:
                logger.info("\n✓ Pure DDMM provides best fit!")
            elif best_aic == aic_hybrid:
                logger.info("\n✓ Hybrid model provides best fit!")
            else:
                logger.info("\n✗ Standard ΛCDM still provides best fit")
            
            passed = (chi2_pure < 2 * chi2_lcdm) or (chi2_hybrid < 1.5 * chi2_lcdm)
            score = np.exp(-min(chi2_pure, chi2_hybrid) / chi2_lcdm)
        else:
            # Without data, check theoretical consistency
            max_dev_pure = np.max(np.abs(mu_ddmm_pure - mu_lcdm))
            max_dev_hybrid = np.max(np.abs(mu_hybrid - mu_lcdm))
            scatter = np.mean(mu_ddmm_pure_err)
            
            logger.info(f"\nTheoretical comparison:")
            logger.info(f"  Pure DDMM max deviation: {max_dev_pure:.3f} mag")
            logger.info(f"  Hybrid max deviation: {max_dev_hybrid:.3f} mag")
            logger.info(f"  Path-dependent scatter: {scatter:.3f} mag")
            
            passed = min(max_dev_pure, max_dev_hybrid) < 0.5
            score = 1.0 - min(max_dev_pure, max_dev_hybrid) / 0.5
        
        # Create enhanced visualization
        self._plot_enhanced_hubble_diagram(
            z_obs, mu_lcdm, mu_ddmm_pure, mu_ddmm_pure_err,
            mu_hybrid, mu_obs, mu_err
        )
        
        test_details = {
            'models': {
                'pure_ddmm': {
                    'chi2': float(chi2_pure) if mu_obs is not None else None,
                    'mean_scatter': float(np.mean(mu_ddmm_pure_err)),
                    'path_variation': float(np.mean(ddmm_results['z_std'] / ddmm_results['z_mean']))
                },
                'hybrid': {
                    'chi2': float(chi2_hybrid) if mu_obs is not None else None,
                    'expansion_fraction': 0.7
                }
            }
        }
        
        recommendations = []
        if not passed:
            recommendations.append("Consider environmental density variations")
            recommendations.append("Path-dependent effects are significant")
            if scatter > 0.1:
                recommendations.append(f"Large path scatter ({scatter:.2f} mag) challenges model")
        
        from validate_ddmm import TestResult  # Import from original module
        result = TestResult(
            test_name="Type Ia Supernovae (Enhanced)",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _plot_enhanced_hubble_diagram(self, z, mu_lcdm, mu_ddmm, mu_ddmm_err,
                                    mu_hybrid, mu_obs=None, mu_err=None):
        """Enhanced Hubble diagram with error bands"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.plot(z, mu_lcdm, 'b-', lw=2, label='ΛCDM', alpha=0.8)
        
        # Pure DDMM with error band
        ax1.plot(z, mu_ddmm, 'r-', lw=2, label='Pure DDMM', alpha=0.8)
        ax1.fill_between(z, mu_ddmm - mu_ddmm_err, mu_ddmm + mu_ddmm_err,
                        alpha=0.2, color='red', label='Path variations')
        
        ax1.plot(z, mu_hybrid, 'g--', lw=2, label='Hybrid (70/30)', alpha=0.8)
        
        if mu_obs is not None:
            ax1.errorbar(z, mu_obs, yerr=mu_err, fmt='ko', markersize=4,
                        alpha=0.5, label='Observed SNe Ia')
        
        ax1.set_ylabel('Distance Modulus μ', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_xlim(0.008, 2.5)
        ax1.set_ylim(32, 46)
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Enhanced DDMM Light Propagation Analysis', fontsize=16)
        
        # Residuals with error band
        ax2.axhline(0, color='gray', ls='--', alpha=0.5)
        ax2.plot(z, mu_ddmm - mu_lcdm, 'r-', lw=2, label='Pure DDMM - ΛCDM')
        ax2.fill_between(z, -mu_ddmm_err, mu_ddmm_err, alpha=0.2, color='red')
        ax2.plot(z, mu_hybrid - mu_lcdm, 'g--', lw=2, label='Hybrid - ΛCDM')
        
        ax2.set_xlabel('Redshift z', fontsize=14)
        ax2.set_ylabel('Δμ (mag)', fontsize=14)
        ax2.set_xscale('log')
        ax2.set_xlim(0.008, 2.5)
        ax2.set_ylim(-2, 2)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'enhanced_hubble_diagram.png', dpi=150)
        plt.close()
    
    # Add the enhanced method WITHOUT replacing test_supernovae
    import types
    validator_instance.test_supernovae_enhanced = types.MethodType(test_supernovae_enhanced, validator_instance)
    validator_instance._plot_enhanced_hubble_diagram = types.MethodType(_plot_enhanced_hubble_diagram, validator_instance)
    
    # Store the propagator for later use
    validator_instance._enhanced_propagator = propagator
    
    return propagator

# Example usage
if __name__ == "__main__":
    # Test the enhanced propagation module
    logger.info("Testing Enhanced DDMM Light Propagation")
    
    # Create propagator with example parameters
    propagator = EnhancedDDMMLightPropagation(
        rho_c=1e9,    # M☉/kpc³
        n_exp=1.0,
        A=1.0
    )
    
    # Create cosmic web
    cosmic_web = propagator.create_cosmic_web(box_size=300, resolution=128)
    
    # Visualize
    propagator.plot_cosmic_web_slice()
    propagator.plot_path_dependent_redshift(z_target=0.5, n_paths=1000)
    
    # Test different formulations
    logger.info("\nTesting redshift calculations:")
    
    # Direct formula
    z_direct = propagator.redshift_direct_formula(rho_observer=1e8, rho_emitter=1e6)
    logger.info(f"Direct formula: z = {z_direct:.4f}")
    
    # Path integral
    test_path = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100]])
    test_densities = np.array([1e8, 1e6, 1e7])
    z_path = propagator.redshift_path_integral(test_path, test_densities)
    logger.info(f"Path integral: z = {z_path:.4f}")
    
    # Full calculation with cosmic web
    z_targets = np.array([0.1, 0.5, 1.0])
    results = propagator.calculate_distances_pure_ddmm(z_targets, n_paths=20)
    
    logger.info("\nDistance calculations:")
    for i, z in enumerate(z_targets):
        logger.info(f"z = {z:.1f}: d = {results['d_mean'][i]:.1f} ± {results['d_std'][i]:.1f} Mpc")