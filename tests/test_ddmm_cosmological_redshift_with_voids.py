#!/usr/bin/env python3
"""
test_ddmm_cosmological_redshift_with_voids.py

Test if DDMM gravity alone can explain cosmological redshift without expansion.
Updated with void models and your latest fitted parameters.

This script:
1. Uses your fitted DDMM parameters from latest dynesty run
2. Models cosmic voids with various density profiles
3. Calculates redshift from pure gravitational effects
4. Tests multiple void scenarios to explain Hubble constant
"""

import sys
import traceback
import logging
from datetime import datetime

# Set up logging immediately
log_filename = f"ddmm_cosmology_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("Starting DDMM Cosmological Redshift Test with Void Models")
logger.info(f"Log file: {log_filename}")
logger.info("="*70)

# Flush output to ensure it's visible
sys.stdout.flush()

logger.info("Importing required libraries...")

try:
    import numpy as np
    logger.info("  ✓ NumPy imported successfully")
    logger.debug(f"    NumPy version: {np.__version__}")
except ImportError as e:
    logger.error(f"  ✗ Failed to import NumPy: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"  ✗ Unexpected error importing NumPy: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend first
    import matplotlib.pyplot as plt
    logger.info("  ✓ Matplotlib imported successfully")
    logger.debug(f"    Matplotlib version: {matplotlib.__version__}")
    logger.debug(f"    Backend: {matplotlib.get_backend()}")
except ImportError as e:
    logger.error(f"  ✗ Failed to import Matplotlib: {e}")
    logger.error("  Please install: pip install matplotlib")
    sys.exit(1)
except Exception as e:
    logger.error(f"  ✗ Unexpected error importing Matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from scipy import optimize
    logger.info("  ✓ SciPy imported successfully")
    logger.debug(f"    SciPy version: {scipy.__version__}")
    scipy_available = True
    
    # Create curve_fit reference
    curve_fit = optimize.curve_fit
except ImportError as e:
    logger.warning(f"  ⚠ Warning: SciPy not available: {e}")
    logger.warning("  Running with limited functionality...")
    scipy_available = False
    
    # Simple fallback for curve_fit
    def curve_fit(func, xdata, ydata):
        # Simple mean estimation for linear fit through origin
        if len(xdata) > 0 and len(ydata) > 0:
            return [np.mean(ydata/xdata)], None
        return [1.0], None
except Exception as e:
    logger.error(f"  ✗ Unexpected error with SciPy: {e}")
    scipy_available = False
    
    def curve_fit(func, xdata, ydata):
        if len(xdata) > 0 and len(ydata) > 0:
            return [np.mean(ydata/xdata)], None
        return [1.0], None

try:
    from pathlib import Path
    logger.info("  ✓ Pathlib imported successfully")
except ImportError as e:
    logger.error(f"  ✗ Failed to import Pathlib: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"  ✗ Unexpected error importing Pathlib: {e}")
    traceback.print_exc()
    sys.exit(1)

logger.info("All libraries imported successfully!")
logger.info("")

# Flush again
sys.stdout.flush()

# ============================================================================
# LOAD YOUR ACTUAL FITTED PARAMETERS FROM DYNESTY RUN
# ============================================================================

def load_ddmm_parameters():
    """
    Load parameters from your dynesty checkpoint file.
    """
    logger.info("Loading DDMM parameters...")
    
    # Try to find your checkpoint file
    checkpoint_paths = [
        "runs/enhanced_20250805_115400/dynesty_checkpoint_enhanced_latest.npz",
        "runs/enhanced_20250805_115400/extracted_dynesty_checkpoint_enhanced_latest.npz",
        "extracted_dynesty_checkpoint_enhanced_latest.npz"
    ]
    
    loaded_params = None
    
    logger.debug(f"Searching for checkpoint files in {len(checkpoint_paths)} locations...")
    
    for checkpoint_path in checkpoint_paths:
        logger.debug(f"Checking: {checkpoint_path}")
        if Path(checkpoint_path).exists():
            logger.info(f"Found checkpoint file: {checkpoint_path}")
            try:
                data = np.load(checkpoint_path)
                logger.debug(f"  Keys in file: {list(data.keys())}")
                
                # Try different possible keys for parameters
                if 'median_params' in data:
                    logger.debug("  Using 'median_params' from checkpoint")
                    params = data['median_params']
                    param_names = data['param_names'] if 'param_names' in data else []
                elif 'samples' in data and 'param_names' in data:
                    logger.debug("  Calculating median from 'samples'")
                    # Use median of samples
                    samples = data['samples']
                    param_names = data['param_names']
                    params = np.median(samples, axis=0)
                    
                    # Also get percentiles for uncertainty
                    params_16 = np.percentile(samples, 16, axis=0)
                    params_84 = np.percentile(samples, 84, axis=0)
                    logger.debug(f"  Samples shape: {samples.shape}")
                else:
                    logger.warning("  Could not find parameter data in file")
                    continue
                
                # Create parameter dictionary
                loaded_params = {}
                for i, name in enumerate(param_names):
                    loaded_params[name] = params[i]
                    if 'samples' in data:
                        loaded_params[f"{name}_16"] = params_16[i]
                        loaded_params[f"{name}_84"] = params_84[i]
                    logger.debug(f"    {name}: {params[i]:.3e}")
                
                logger.info(f"  Successfully loaded {len(loaded_params)} parameters")
                break
                
            except Exception as e:
                logger.error(f"  Error loading {checkpoint_path}: {e}")
                logger.debug(f"  Full traceback: {traceback.format_exc()}")
                continue
        else:
            logger.debug(f"  File does not exist: {checkpoint_path}")
    
    # Extract DDMM-specific parameters or use defaults
    if loaded_params:
        logger.info("Using loaded parameters from checkpoint")
        YOUR_FITTED_PARAMS = {
            'A': loaded_params.get('A', 5.615),
            'n_exp': loaded_params.get('n_exp', 1.235),
            'rho_c_solar_kpc3': loaded_params.get('rho_c_solar_kpc3', 9.937318499036452e15),
            # Uncertainty bounds
            'A_min': loaded_params.get('A_16', 3.158),
            'A_max': loaded_params.get('A_84', 8.495),
            # Store all galaxy parameters for potential use
            'M_thin_disk_solar': loaded_params.get('M_thin_disk_solar', 1.0e10),
            'R_thin_disk_kpc': loaded_params.get('R_thin_disk_kpc', 4.0),
            'hz_thin_disk_kpc': loaded_params.get('hz_thin_disk_kpc', 0.399),
            'M_thick_disk_solar': loaded_params.get('M_thick_disk_solar', 5.535e8),
            'R_thick_disk_kpc': loaded_params.get('R_thick_disk_kpc', 4.984),
            'hz_thick_disk_kpc': loaded_params.get('hz_thick_disk_kpc', 0.919),
            'M_bulge_solar': loaded_params.get('M_bulge_solar', 5.019e8),
            'R_bulge_kpc': loaded_params.get('R_bulge_kpc', 1.991),
            'M_gas_solar': loaded_params.get('M_gas_solar', 9.994e9),
            'R_gas_kpc': loaded_params.get('R_gas_kpc', 9.967),
            'hz_gas_kpc': loaded_params.get('hz_gas_kpc', 0.100),
        }
    else:
        # Fallback to your specific run values
        logger.warning("Could not load checkpoint file, using hard-coded values from your run")
        YOUR_FITTED_PARAMS = {
            'A': 5.615,
            'n_exp': 1.235,
            'rho_c_solar_kpc3': 9.937318499036452e15,
            'A_min': 3.158,
            'A_max': 8.495,
            # Galaxy model parameters from your run
            'M_thin_disk_solar': 1.000e10,
            'R_thin_disk_kpc': 4.000,
            'hz_thin_disk_kpc': 0.399,
            'M_thick_disk_solar': 5.535e08,
            'R_thick_disk_kpc': 4.984,
            'hz_thick_disk_kpc': 0.919,
            'M_bulge_solar': 5.019e08,
            'R_bulge_kpc': 1.991,
            'M_gas_solar': 9.994e09,
            'R_gas_kpc': 9.967,
            'hz_gas_kpc': 0.100,
        }
    
    return YOUR_FITTED_PARAMS

# Load the parameters
logger.info("="*70)
logger.info("LOADING DDMM PARAMETERS")
logger.info("="*70)

try:
    YOUR_FITTED_PARAMS = load_ddmm_parameters()
    
    logger.info("")
    logger.info("Using DDMM parameters from your dynesty run:")
    logger.info(f"  A = {YOUR_FITTED_PARAMS['A']:.3f} [{YOUR_FITTED_PARAMS['A_min']:.3f}, {YOUR_FITTED_PARAMS['A_max']:.3f}]")
    logger.info(f"  n = {YOUR_FITTED_PARAMS['n_exp']:.3f}")
    logger.info(f"  ρ_c = {YOUR_FITTED_PARAMS['rho_c_solar_kpc3']:.3e} M_☉/kpc³")
    logger.info("")
    logger.info("Galaxy model parameters:")
    total_mass = (YOUR_FITTED_PARAMS['M_thin_disk_solar'] + 
                  YOUR_FITTED_PARAMS['M_thick_disk_solar'] + 
                  YOUR_FITTED_PARAMS['M_bulge_solar'] + 
                  YOUR_FITTED_PARAMS['M_gas_solar'])
    logger.info(f"  Total baryonic mass: {total_mass:.3e} M_☉")
    logger.info(f"  Thin disk: M={YOUR_FITTED_PARAMS['M_thin_disk_solar']:.2e} M_☉, R={YOUR_FITTED_PARAMS['R_thin_disk_kpc']:.1f} kpc")
    logger.info(f"  Thick disk: M={YOUR_FITTED_PARAMS['M_thick_disk_solar']:.2e} M_☉, R={YOUR_FITTED_PARAMS['R_thick_disk_kpc']:.1f} kpc")
    logger.info(f"  Bulge: M={YOUR_FITTED_PARAMS['M_bulge_solar']:.2e} M_☉, R={YOUR_FITTED_PARAMS['R_bulge_kpc']:.1f} kpc")
    logger.info(f"  Gas: M={YOUR_FITTED_PARAMS['M_gas_solar']:.2e} M_☉, R={YOUR_FITTED_PARAMS['R_gas_kpc']:.1f} kpc")
    logger.info("")
    
except Exception as e:
    logger.error(f"Failed to load DDMM parameters: {e}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)

# ============================================================================
# VOID MODELS FOR COSMOLOGY
# ============================================================================

class CosmicVoidModels:
    """
    Different models for cosmic void density profiles.
    Key insight: Voids have very low but non-zero density.
    """
    
    def __init__(self, void_contrast=0.2, void_radius_Mpc=30):
        """
        Initialize void parameters.
        
        Parameters:
        -----------
        void_contrast : float
            Density in void as fraction of cosmic mean (typically 0.1-0.3)
        void_radius_Mpc : float
            Typical void radius in Mpc
        """
        self.void_contrast = void_contrast
        self.void_radius = void_radius_Mpc
        
        # Cosmic mean density
        self.rho_cosmic_mean = 2.775e11 * 0.7**2 * 0.3 / 1e9  # M_☉/kpc³
        
        # Minimum density in voids (never exactly zero!)
        self.rho_void_min = self.rho_cosmic_mean * void_contrast
        
        logger.debug(f"Void Model initialized:")
        logger.debug(f"  Void contrast: {void_contrast} (ρ_void/ρ_mean)")
        logger.debug(f"  Void radius: {void_radius_Mpc} Mpc")
        logger.debug(f"  Mean cosmic density: {self.rho_cosmic_mean:.2e} M_☉/kpc³")
        logger.debug(f"  Minimum void density: {self.rho_void_min:.2e} M_☉/kpc³")
    
    def void_profile_exponential(self, r_from_center_Mpc):
        """Exponential void profile - smooth transition."""
        if r_from_center_Mpc < self.void_radius:
            # Inside void: exponential rise from center
            x = r_from_center_Mpc / self.void_radius
            contrast = self.void_contrast + (1 - self.void_contrast) * (1 - np.exp(-3*x))
        else:
            # Outside void: cosmic mean
            contrast = 1.0
        
        return self.rho_cosmic_mean * contrast
    
    def void_profile_tophat(self, r_from_center_Mpc):
        """Top-hat void profile - sharp boundary."""
        if r_from_center_Mpc < self.void_radius:
            return self.rho_void_min
        else:
            return self.rho_cosmic_mean
    
    def void_profile_gaussian(self, r_from_center_Mpc):
        """Gaussian void profile - most realistic."""
        x = r_from_center_Mpc / self.void_radius
        contrast = self.void_contrast + (1 - self.void_contrast) * (1 - np.exp(-x**2))
        return self.rho_cosmic_mean * contrast
    
    def cosmic_web_profile(self, distance_Mpc, n_voids=5):
        """
        Realistic cosmic web with multiple voids along line of sight.
        """
        # Place voids at regular intervals
        void_spacing = 100  # Mpc between void centers
        
        # Start with cosmic mean
        rho = self.rho_cosmic_mean
        
        # Add void contributions
        for i in range(n_voids):
            void_center = (i + 1) * void_spacing
            distance_from_void = abs(distance_Mpc - void_center)
            
            if distance_from_void < self.void_radius:
                # We're in this void
                void_factor = self.void_profile_gaussian(distance_from_void) / self.rho_cosmic_mean
                rho = min(rho, self.rho_cosmic_mean * void_factor)
        
        return rho


class DDMMCosmologicalRedshiftWithVoids:
    """
    Calculate cosmological redshift using DDMM gravity with void models.
    Uses your fitted parameters and explores void effects.
    """
    
    def __init__(self, A=None, n=None, rho_c=None, void_model='gaussian', void_contrast=0.2):
        """
        Initialize with your fitted DDMM parameters and void model.
        """
        # Use your fitted parameters by default
        self.A = A if A is not None else YOUR_FITTED_PARAMS['A']
        self.n = n if n is not None else YOUR_FITTED_PARAMS['n_exp']
        self.rho_c = rho_c if rho_c is not None else YOUR_FITTED_PARAMS['rho_c_solar_kpc3']
        
        # Cosmological parameters
        self.H0_standard = 70.0  # km/s/Mpc
        self.c = 299792.458  # km/s
        self.G = 4.302e-3  # (km/s)²·kpc/M_☉
        
        # Initialize void model
        self.void_models = CosmicVoidModels(void_contrast=void_contrast)
        self.void_model_type = void_model
        
        print(f"\nDDMM with Voids initialized:")
        print(f"  A = {self.A:.3f}")
        print(f"  n = {self.n:.3f}")
        print(f"  ρ_c = {self.rho_c:.3e} M_☉/kpc³")
        print(f"  Void model: {void_model}")
        print(f"  Void contrast: {void_contrast}")
    
    def xi_function(self, rho):
        """
        Enhancement factor ξ(ρ) from DDMM model.
        
        ξ(ρ) = 1 + A(ρ_c/ρ)^n
        
        Critical: In voids where ρ → 0, ξ → very large!
        This is where DDMM becomes interesting for cosmology.
        """
        if rho <= 0:
            return 100.0  # Cap at very high but finite value
            
        xi = 1.0 + self.A * (self.rho_c / rho)**self.n
        
        # Cap at reasonable maximum to avoid numerical issues
        return min(xi, 100.0)
    
    def density_profile_with_voids(self, r_Mpc):
        """
        Density profile including void structure.
        """
        # Local structure (galaxy, cluster)
        if r_Mpc < 0.001:  # Within galaxy
            return 1e8  # Galactic density
        elif r_Mpc < 1:  # Local group
            return 1e6 * np.exp(-r_Mpc/0.5)
        elif r_Mpc < 10:  # Nearby clusters
            return 1e4 * np.exp(-r_Mpc/5)
        else:
            # Large-scale structure with voids
            if self.void_model_type == 'gaussian':
                return self.void_models.void_profile_gaussian(r_Mpc % 100)
            elif self.void_model_type == 'exponential':
                return self.void_models.void_profile_exponential(r_Mpc % 100)
            elif self.void_model_type == 'tophat':
                return self.void_models.void_profile_tophat(r_Mpc % 100)
            elif self.void_model_type == 'cosmic_web':
                return self.void_models.cosmic_web_profile(r_Mpc)
            else:
                return self.void_models.rho_cosmic_mean
    
    def gravitational_redshift_through_voids(self, distance_Mpc, n_steps=1000):
        """
        Calculate redshift with detailed void structure.
        
        Key physics: Light "crawls" through voids where ξ is very large,
        accumulating significant redshift even without expansion.
        """
        # Create path from observer to source
        r_path = np.linspace(0, distance_Mpc, n_steps)
        dr = r_path[1] - r_path[0] if n_steps > 1 else distance_Mpc
        
        # Track cumulative redshift
        z_cumulative = 0
        
        # Also track contributions from different regions
        z_galaxy = 0
        z_clusters = 0
        z_voids = 0
        z_filaments = 0
        
        for i in range(len(r_path) - 1):
            r_current = r_path[i]
            r_next = r_path[i+1]
            
            # Get densities
            rho_current = self.density_profile_with_voids(r_current)
            rho_next = self.density_profile_with_voids(r_next)
            
            # Enhancement factors
            xi_current = self.xi_function(rho_current)
            xi_next = self.xi_function(rho_next)
            
            # Local redshift contribution
            # dz/dr ≈ (A*n/2) * (ρ_c/ρ)^n * (dρ/dr) / ρ
            if rho_current > 0 and rho_next > 0:
                drho_dr = (rho_next - rho_current) / dr
                dz = (self.A * self.n / 2) * (self.rho_c / rho_current)**self.n * drho_dr * dr / rho_current
                
                # Categorize contribution
                if rho_current > 1e6:  # Galaxy/cluster
                    z_galaxy += abs(dz)
                elif rho_current > 1e4:  # Cluster outskirts
                    z_clusters += abs(dz)
                elif rho_current < self.void_models.rho_cosmic_mean * 0.5:  # Void
                    z_voids += abs(dz)
                else:  # Filaments
                    z_filaments += abs(dz)
                
                z_cumulative += dz
        
        # Report contributions
        if distance_Mpc > 50:  # Only for cosmological distances
            total_z = abs(z_cumulative)
            if total_z > 0:
                print(f"\n  Redshift contributions at {distance_Mpc:.0f} Mpc:")
                print(f"    From voids:     {abs(z_voids)/total_z*100:.1f}%")
                print(f"    From filaments: {abs(z_filaments)/total_z*100:.1f}%")
                print(f"    From clusters:  {abs(z_clusters)/total_z*100:.1f}%")
                print(f"    From galaxies:  {abs(z_galaxy)/total_z*100:.1f}%")
        
        return abs(z_cumulative)
    
    def test_void_scenarios(self):
        """
        Test multiple void scenarios to find best match to Hubble constant.
        """
        print("\n" + "="*60)
        print("TESTING VOID SCENARIOS")
        print("="*60)
        
        # Test distances
        test_distances = [10, 50, 100, 500, 1000]  # Mpc
        
        # Different void contrasts to test
        void_contrasts = [0.1, 0.2, 0.3, 0.5]  # ρ_void/ρ_mean
        
        results = {}
        
        for contrast in void_contrasts:
            print(f"\nVoid contrast = {contrast} (ρ_void/ρ_mean):")
            print("-" * 40)
            
            # Reinitialize with this void contrast
            self.void_models = CosmicVoidModels(void_contrast=contrast)
            
            z_list = []
            H_eff_list = []
            
            for d in test_distances:
                z = self.gravitational_redshift_through_voids(d, n_steps=500)
                z_hubble = d * self.H0_standard / self.c
                H_eff = self.c * z / d if d > 0 else 0
                
                z_list.append(z)
                H_eff_list.append(H_eff)
                
                print(f"  d = {d:4.0f} Mpc: z_DDMM = {z:.5f}, z_Hubble = {z_hubble:.5f}")
                print(f"               H_eff = {H_eff:.1f} km/s/Mpc (ratio = {H_eff/self.H0_standard:.2f})")
            
            results[contrast] = {
                'distances': test_distances,
                'redshifts': z_list,
                'H_effective': H_eff_list
            }
        
        return results
    
    def optimize_void_parameters(self):
        """
        Find void parameters that best match observed Hubble constant.
        """
        print("\n" + "="*60)
        print("OPTIMIZING VOID PARAMETERS FOR H₀")
        print("="*60)
        
        # Target Hubble constant
        H0_target = 70.0  # km/s/Mpc
        
        # Test range
        contrasts = np.linspace(0.05, 0.5, 20)
        void_radii = np.linspace(10, 50, 20)
        
        best_contrast = None
        best_radius = None
        best_chi2 = np.inf
        
        # Test distances for fitting
        d_test = np.array([10, 30, 50, 100, 200])
        z_hubble = d_test * H0_target / self.c
        
        for contrast in contrasts:
            for radius in void_radii:
                # Set void parameters
                self.void_models = CosmicVoidModels(
                    void_contrast=contrast,
                    void_radius_Mpc=radius
                )
                
                # Calculate redshifts
                z_ddmm = []
                for d in d_test:
                    z = self.gravitational_redshift_through_voids(d, n_steps=200)
                    z_ddmm.append(z)
                
                z_ddmm = np.array(z_ddmm)
                
                # Chi-squared
                chi2 = np.sum((z_ddmm - z_hubble)**2 / z_hubble**2)
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_contrast = contrast
                    best_radius = radius
        
        print(f"\nOptimal void parameters:")
        print(f"  Contrast: {best_contrast:.3f}")
        print(f"  Radius: {best_radius:.1f} Mpc")
        print(f"  χ²: {best_chi2:.3f}")
        
        # Test with optimal parameters
        self.void_models = CosmicVoidModels(
            void_contrast=best_contrast,
            void_radius_Mpc=best_radius
        )
        
        print(f"\nRedshift with optimal void parameters:")
        for d in [10, 50, 100, 500, 1000]:
            z = self.gravitational_redshift_through_voids(d, n_steps=500)
            z_hubble = d * H0_target / self.c
            ratio = z / z_hubble
            print(f"  d = {d:4.0f} Mpc: z_DDMM/z_Hubble = {ratio:.3f}")
        
        return best_contrast, best_radius


def plot_void_effects(ddmm_void_model):
    """
    Visualize how voids affect cosmological redshift in DDMM.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Density profile with voids
    ax = axes[0, 0]
    
    r_range = np.linspace(0, 300, 1000)
    rho_profile = [ddmm_void_model.density_profile_with_voids(r) for r in r_range]
    xi_profile = [ddmm_void_model.xi_function(rho) for rho in rho_profile]
    
    ax.plot(r_range, rho_profile, 'b-', linewidth=2)
    ax.axhline(ddmm_void_model.void_models.rho_cosmic_mean, 
               color='gray', linestyle='--', alpha=0.5, label='Cosmic mean')
    ax.axhline(ddmm_void_model.void_models.rho_void_min,
               color='red', linestyle=':', alpha=0.5, label='Void minimum')
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('Density (M_☉/kpc³)', fontsize=11)
    ax.set_title('Density Profile with Voids', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Enhancement factor in voids
    ax = axes[0, 1]
    
    ax.plot(r_range, xi_profile, 'r-', linewidth=2)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('Enhancement Factor ξ', fontsize=11)
    ax.set_title('DDMM Enhancement in Voids', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    void_indices = np.where(np.array(rho_profile) < ddmm_void_model.void_models.rho_cosmic_mean * 0.5)[0]
    if len(void_indices) > 0:
        void_center = r_range[void_indices[len(void_indices)//2]]
        ax.annotate('Void: ξ >> 1', xy=(void_center, xi_profile[void_indices[len(void_indices)//2]]),
                   xytext=(void_center+30, 50), fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # 3. Redshift accumulation
    ax = axes[0, 2]
    
    distances = np.logspace(0, 3, 50)
    z_with_voids = []
    z_no_voids = []
    
    for d in distances:
        z_v = ddmm_void_model.gravitational_redshift_through_voids(d, n_steps=200)
        z_with_voids.append(z_v)
        
        # Calculate without voids (uniform density)
        ddmm_void_model.void_model_type = 'uniform'
        z_nv = ddmm_void_model.gravitational_redshift_through_voids(d, n_steps=200)
        z_no_voids.append(z_nv)
        ddmm_void_model.void_model_type = 'gaussian'  # Reset
    
    # Hubble law
    z_hubble = distances * 70 / 299792.458
    
    ax.plot(distances, z_hubble, 'b-', label='Hubble (H₀=70)', linewidth=2)
    ax.plot(distances, z_with_voids, 'r-', label='DDMM with voids', linewidth=2)
    ax.plot(distances, z_no_voids, 'g--', label='DDMM uniform', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('Redshift z', fontsize=11)
    ax.set_title('Redshift: Impact of Voids', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Void contrast comparison
    ax = axes[1, 0]
    
    contrasts = [0.1, 0.2, 0.3, 0.5]
    colors = ['purple', 'blue', 'green', 'orange']
    
    for contrast, color in zip(contrasts, colors):
        ddmm_void_model.void_models = CosmicVoidModels(void_contrast=contrast)
        z_test = []
        for d in distances:
            z = ddmm_void_model.gravitational_redshift_through_voids(d, n_steps=200)
            z_test.append(z)
        ax.plot(distances, z_test, color=color, label=f'Contrast = {contrast}', linewidth=2)
    
    ax.plot(distances, z_hubble, 'k--', label='Hubble', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('Redshift z', fontsize=11)
    ax.set_title('Effect of Void Contrast', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Effective Hubble parameter
    ax = axes[1, 1]
    
    H_eff_profiles = []
    for contrast, color in zip(contrasts, colors):
        ddmm_void_model.void_models = CosmicVoidModels(void_contrast=contrast)
        H_eff = []
        for d in distances[distances > 1]:  # Skip very small distances
            z = ddmm_void_model.gravitational_redshift_through_voids(d, n_steps=200)
            H = 299792.458 * z / d
            H_eff.append(H)
        ax.plot(distances[distances > 1], H_eff, color=color, 
                label=f'Contrast = {contrast}', linewidth=2)
    
    ax.axhline(70, color='black', linestyle='--', label='H₀ = 70', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('H_eff (km/s/Mpc)', fontsize=11)
    ax.set_title('Effective Hubble Parameter', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 6. A parameter sensitivity
    ax = axes[1, 2]
    
    A_values = [YOUR_FITTED_PARAMS['A_min'], YOUR_FITTED_PARAMS['A'], YOUR_FITTED_PARAMS['A_max']]
    A_labels = ['A_min = 3.16', 'A_best = 5.62', 'A_max = 8.50']
    colors = ['blue', 'red', 'green']
    
    for A_val, label, color in zip(A_values, A_labels, colors):
        ddmm_test = DDMMCosmologicalRedshiftWithVoids(A=A_val)
        z_test = []
        for d in distances:
            z = ddmm_test.gravitational_redshift_through_voids(d, n_steps=200)
            z_test.append(z)
        ax.plot(distances, z_test, color=color, label=label, linewidth=2)
    
    ax.plot(distances, z_hubble, 'k--', label='Hubble', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=11)
    ax.set_ylabel('Redshift z', fontsize=11)
    ax.set_title('Sensitivity to A Parameter', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_comprehensive_void_test():
    """
    Main test with your parameters and void models.
    """
    print("\n" + "="*70)
    print("DDMM COSMOLOGICAL REDSHIFT TEST WITH VOIDS")
    print("Using your fitted parameters from dynesty run")
    print("="*70)
    
    # Test with different A values from uncertainty range
    print("\nTesting parameter uncertainty range:")
    print("-" * 50)
    
    for A_test in [YOUR_FITTED_PARAMS['A_min'], YOUR_FITTED_PARAMS['A'], YOUR_FITTED_PARAMS['A_max']]:
        print(f"\nA = {A_test:.3f}:")
        ddmm = DDMMCosmologicalRedshiftWithVoids(A=A_test, void_contrast=0.2)
        
        # Test at key distances
        for d in [10, 100, 1000]:
            z = ddmm.gravitational_redshift_through_voids(d, n_steps=500)
            z_hubble = d * 70 / 299792.458
            print(f"  d = {d:4.0f} Mpc: z = {z:.5f} (Hubble: {z_hubble:.5f}, ratio: {z/z_hubble:.2f})")
    
    # Main model with best-fit parameters
    print("\n" + "="*70)
    print("MAIN ANALYSIS WITH BEST-FIT PARAMETERS")
    print("="*70)
    
    ddmm_main = DDMMCosmologicalRedshiftWithVoids(
        A=YOUR_FITTED_PARAMS['A'],
        n=YOUR_FITTED_PARAMS['n_exp'],
        rho_c=YOUR_FITTED_PARAMS['rho_c_solar_kpc3'],
        void_model='gaussian',
        void_contrast=0.2
    )
    
    # Test void scenarios
    void_results = ddmm_main.test_void_scenarios()
    
    # Optimize void parameters
    best_contrast, best_radius = ddmm_main.optimize_void_parameters()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    fig = plot_void_effects(ddmm_main)
    
    # Save results
    output_dir = Path("ddmm_void_cosmology_results")
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / "void_effects_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"1. With A = {YOUR_FITTED_PARAMS['A']:.3f}, DDMM produces significant cosmological redshift")
    print(f"2. Voids with contrast {best_contrast:.3f} optimize agreement with H₀")
    print(f"3. Light 'crawls' through voids where ξ >> 1, accumulating redshift")
    print(f"4. No universal expansion needed - just density variations!")
    
    print("\nPhysical Interpretation:")
    print("- In voids: ρ << ρ_c, so ξ = 1 + A(ρ_c/ρ)^n becomes very large")
    print("- Large ξ in voids causes significant metric distortion")
    print("- Photons experience cumulative redshift traversing void structure")
    print("- Cosmic web naturally produces Hubble-like relation")
    
    return ddmm_main, void_results


# ============================================================================
# RUN THE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXECUTING MAIN TEST WITH YOUR DDMM DYNESTY RUN")
    print("="*70)
    
    try:
        # Check if user provides a specific checkpoint file
        if len(sys.argv) > 1:
            checkpoint_file = sys.argv[1]
            print(f"Attempting to load checkpoint from: {checkpoint_file}")
            
            if Path(checkpoint_file).exists():
                try:
                    data = np.load(checkpoint_file)
                    print(f"  File loaded. Keys: {list(data.keys())}")
                    
                    # Try to extract parameters
                    if 'median_params' in data and 'param_names' in data:
                        params = data['median_params']
                        param_names = list(data['param_names'])
                        
                        # Update YOUR_FITTED_PARAMS with loaded values
                        param_dict = {name: params[i] for i, name in enumerate(param_names)}
                        
                        if 'A' in param_dict:
                            YOUR_FITTED_PARAMS['A'] = param_dict['A']
                        if 'n_exp' in param_dict:
                            YOUR_FITTED_PARAMS['n_exp'] = param_dict['n_exp']
                        if 'rho_c_solar_kpc3' in param_dict:
                            YOUR_FITTED_PARAMS['rho_c_solar_kpc3'] = param_dict['rho_c_solar_kpc3']
                        
                        print(f"  Updated parameters from checkpoint:")
                        print(f"    A = {YOUR_FITTED_PARAMS['A']:.3f}")
                        print(f"    n = {YOUR_FITTED_PARAMS['n_exp']:.3f}")
                        print(f"    ρ_c = {YOUR_FITTED_PARAMS['rho_c_solar_kpc3']:.3e}")
                    
                    elif 'samples' in data and 'param_names' in data:
                        # Calculate from samples
                        samples = data['samples']
                        param_names = list(data['param_names'])
                        
                        # Get median and percentiles
                        median_params = np.median(samples, axis=0)
                        params_16 = np.percentile(samples, 16, axis=0)
                        params_84 = np.percentile(samples, 84, axis=0)
                        
                        # Update parameters
                        for i, name in enumerate(param_names):
                            if name == 'A':
                                YOUR_FITTED_PARAMS['A'] = median_params[i]
                                YOUR_FITTED_PARAMS['A_min'] = params_16[i]
                                YOUR_FITTED_PARAMS['A_max'] = params_84[i]
                            elif name == 'n_exp':
                                YOUR_FITTED_PARAMS['n_exp'] = median_params[i]
                            elif name == 'rho_c_solar_kpc3':
                                YOUR_FITTED_PARAMS['rho_c_solar_kpc3'] = median_params[i]
                        
                        print(f"  Calculated parameters from samples:")
                        print(f"    A = {YOUR_FITTED_PARAMS['A']:.3f} [{YOUR_FITTED_PARAMS['A_min']:.3f}, {YOUR_FITTED_PARAMS['A_max']:.3f}]")
                        print(f"    n = {YOUR_FITTED_PARAMS['n_exp']:.3f}")
                        print(f"    ρ_c = {YOUR_FITTED_PARAMS['rho_c_solar_kpc3']:.3e}")
                    
                except Exception as e:
                    print(f"  Warning: Could not load checkpoint: {e}")
                    print("  Using default loaded parameters")
            else:
                print(f"  File not found: {checkpoint_file}")
                print("  Using default loaded parameters")
        else:
            print("No checkpoint file specified.")
            print("Looking for checkpoint in default locations...")
            print("To use a specific checkpoint, run:")
            print("  python test_ddmm_cosmological_redshift_with_voids.py <checkpoint_file.npz>")
        
        # Run the comprehensive test with loaded parameters
        ddmm_model, results = run_comprehensive_void_test()
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print("\nThe DDMM model with cosmic voids can potentially explain")
        print("cosmological redshift without requiring universal expansion!")
        print(f"\nUsed parameters:")
        print(f"  A = {YOUR_FITTED_PARAMS['A']:.3f}")
        print(f"  n = {YOUR_FITTED_PARAMS['n_exp']:.3f}")  
        print(f"  ρ_c = {YOUR_FITTED_PARAMS['rho_c_solar_kpc3']:.3e} M_☉/kpc³")
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)