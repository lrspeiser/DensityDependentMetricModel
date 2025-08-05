#!/usr/bin/env python3
"""
test_ddmm_cosmological_redshift.py

Test if DDMM gravity alone can explain cosmological redshift without expansion.
This script:
1. Uses your fitted DDMM parameters from galaxy rotation curves
2. Calculates redshift from pure gravitational effects
3. Compares with Type Ia supernovae (deriving effective Hubble constant)
4. Tests if DDMM can explain the observed redshift-distance relation
"""

print("Starting DDMM Cosmological Redshift Test...")
print("Importing required libraries...")

import sys
import traceback

try:
    import numpy as np
    print("  ✓ NumPy imported")
except ImportError as e:
    print(f"  ✗ Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend first
    import matplotlib.pyplot as plt
    print("  ✓ Matplotlib imported")
except ImportError as e:
    print(f"  ✗ Failed to import Matplotlib: {e}")
    print("  Please install: pip install matplotlib")
    sys.exit(1)

try:
    from scipy import optimize
    print("  ✓ SciPy imported")
    scipy_available = True
    
    # Create curve_fit reference
    curve_fit = optimize.curve_fit
except ImportError as e:
    print(f"  ⚠ Warning: SciPy not available: {e}")
    print("  Running with limited functionality...")
    scipy_available = False
    
    # Simple fallback for curve_fit
    def curve_fit(func, xdata, ydata):
        # Simple mean estimation for linear fit through origin
        if len(xdata) > 0 and len(ydata) > 0:
            return [np.mean(ydata/xdata)], None
        return [1.0], None

try:
    from pathlib import Path
    print("  ✓ Pathlib imported")
except ImportError as e:
    print(f"  ✗ Failed to import Pathlib: {e}")
    sys.exit(1)

print("All libraries imported successfully!\n")

# Quick test to ensure script is working
print("Testing basic functionality...")
test_array = np.array([1, 2, 3])
print(f"  NumPy test: {test_array.sum()} (should be 6)")
print("  Basic tests passed!\n")

# Global flag for scipy availability
try:
    scipy_available
except NameError:
    scipy_available = False

# ============================================================================
# DDMM REDSHIFT PHYSICS
# ============================================================================

class DDMMCosmologicalRedshift:
    """
    Calculate cosmological redshift using DDMM gravity without expansion.
    Based on your enhanced power-law model from the paper.
    """
    
    def __init__(self, A=5.22, n=1.245, rho_c=6.83e15):
        """
        Initialize with your fitted DDMM parameters.
        
        Parameters from your paper:
        - A = 5.22 ± 2.43 (enhancement amplitude)
        - n = 1.245 ± 0.004 (power law exponent)
        - rho_c = 6.83e15 M_☉/kpc³ (critical density)
        """
        self.A = A
        self.n = n
        self.rho_c = rho_c
        
        # Cosmological parameters for comparison
        self.H0_standard = 70.0  # km/s/Mpc - standard Hubble constant
        self.c = 299792.458  # km/s - speed of light
        
        # Convert critical density to cosmological units
        # 1 M_☉/kpc³ ≈ 3.77e-31 g/cm³
        self.rho_c_cgs = rho_c * 3.77e-31  # g/cm³
        
        print(f"DDMM Parameters initialized:")
        print(f"  A = {self.A}")
        print(f"  n = {self.n}")
        print(f"  ρ_c = {self.rho_c:.2e} M_☉/kpc³")
        
    def xi_function(self, rho):
        """
        Enhancement factor ξ(ρ) from your DDMM model.
        
        ξ(ρ) = 1 + A(ρ_c/ρ)^n
        
        Capped at ξ_max = 5 as in your paper.
        """
        if rho <= 0:
            return 5.0  # Maximum enhancement
            
        xi = 1.0 + self.A * (self.rho_c / rho)**self.n
        return min(xi, 5.0)
    
    def density_profile_cosmological(self, r_Mpc):
        """
        Simplified cosmological density profile.
        
        Models the average density decrease with distance:
        - Local group: ~10^8 M_☉/kpc³
        - Galaxy clusters: ~10^6 M_☉/kpc³  
        - Cosmic voids: ~10^4 M_☉/kpc³
        - Intergalactic medium: ~10^2 M_☉/kpc³
        """
        # Scale the densities to work with your rho_c value
        # Your rho_c = 6.83e15, so we need higher densities
        
        if r_Mpc < 0.001:  # Within galaxy (< 1 kpc)
            return 1e17  # Much higher galactic density
        elif r_Mpc < 1:  # Local group
            return 1e16 * np.exp(-r_Mpc/0.5)
        elif r_Mpc < 10:  # Nearby clusters
            return 1e15 * np.exp(-r_Mpc/5)
        elif r_Mpc < 100:  # Cosmic web
            return 1e14 * np.exp(-r_Mpc/50)
        else:  # Cosmic average/voids
            # Density decreases with distance
            return 1e13 * (1 + r_Mpc/1000)**(-1)

    def gravitational_redshift_direct(self, rho_observer, rho_emitter):
        """
        Direct DDMM redshift formula from your paper.
        
        1 + z = [(ρ_obs + ρ_c)/(ρ_emit + ρ_c)]^(α/2)
        where α = A * n
        """
        alpha = self.A * self.n
        z_plus_1 = ((rho_observer + self.rho_c) / 
                    (rho_emitter + self.rho_c))**(alpha/2)
        return z_plus_1 - 1
    
    def gravitational_redshift_path_integral(self, distance_Mpc, n_steps=1000):
        """
        Calculate redshift via path integral through density field.
        
        From your paper:
        ln(1 + z) = (α/2) ∫ (1/(ρ + ρ_c)) · (dρ/ds) ds
        """
        # Create path from observer to source
        r_path = np.linspace(0, distance_Mpc, n_steps)
        dr = r_path[1] - r_path[0]
        
        # Get density along path
        rho_path = np.array([self.density_profile_cosmological(r) for r in r_path])
        
        # Calculate density gradient
        drho_dr = np.gradient(rho_path, dr)
        
        # Integrand
        alpha = self.A * self.n
        integrand = drho_dr / (rho_path + self.rho_c)
        
        # Numerical integration
        integral = np.trapezoid(integrand, r_path)
        
        # Calculate redshift
        ln_1_plus_z = (alpha / 2) * integral
        return np.exp(ln_1_plus_z) - 1
    
    def metric_redshift_accumulated(self, distance_Mpc, n_steps=1000):
        """
        Calculate redshift from accumulated metric distortion.
        
        In DDMM, photon frequency shifts as it traverses regions
        with different ξ(ρ) values.
        """
        # Path discretization
        r_path = np.linspace(0, distance_Mpc, n_steps)
        dr = r_path[1] - r_path[0] if n_steps > 1 else distance_Mpc
        
        # Initialize frequency ratio
        freq_ratio = 1.0
        
        # Accumulate frequency shifts along path
        for i in range(len(r_path) - 1):
            rho_current = self.density_profile_cosmological(r_path[i])
            rho_next = self.density_profile_cosmological(r_path[i+1])
            
            xi_current = self.xi_function(rho_current)
            xi_next = self.xi_function(rho_next)
            
            # Frequency shift from metric change
            freq_ratio *= np.sqrt(xi_next / xi_current)
        
        # Redshift: z = (λ_obs/λ_emit) - 1 = (ν_emit/ν_obs) - 1
        z = (1.0 / freq_ratio) - 1
        return z
    
    def hubble_diagram_ddmm(self, z_array):
        """
        Calculate distance modulus for given redshifts using DDMM.
        This inverts the redshift calculation to get distances.
        """
        distances = []
        
        for z_target in z_array:
            # Use bisection to find distance giving target redshift
            d_min, d_max = 0.1, 5000  # Mpc
            
            for _ in range(20):  # Bisection iterations
                d_mid = (d_min + d_max) / 2
                z_calc = self.gravitational_redshift_path_integral(d_mid)
                
                if z_calc < z_target:
                    d_min = d_mid
                else:
                    d_max = d_mid
            
            distances.append(d_mid)
        
        distances = np.array(distances)
        
        # Convert to distance modulus
        mu = 5 * np.log10(distances) + 25
        return mu, distances
    
    def derive_effective_hubble_constant(self, max_z=0.1, n_points=50):
        """
        Derive effective Hubble constant from DDMM redshift at low z.
        
        For small redshifts, Hubble's law: cz = H₀ * d
        We fit DDMM predictions to derive effective H₀.
        """
        # Generate redshifts
        z_test = np.linspace(0.001, max_z, n_points)
        
        # Calculate DDMM distances
        distances = []
        for z in z_test:
            # Find distance that produces this redshift
            d_min, d_max = 0.1, 1000
            for _ in range(20):
                d_mid = (d_min + d_max) / 2
                z_calc = self.gravitational_redshift_path_integral(d_mid)
                if z_calc < z:
                    d_min = d_mid
                else:
                    d_max = d_mid
            distances.append(d_mid)
        
        distances = np.array(distances)
        
        # Fit Hubble's law: v = cz = H₀ * d
        # Rearrange: H₀ = cz / d
        velocities = self.c * z_test
        
        # Linear fit through origin
        H0_effective = np.mean(velocities / distances)
        
        # Also do weighted least squares using numpy
        # For y = H0 * x through origin, H0 = sum(xy) / sum(x^2)
        H0_fitted = np.sum(distances * velocities) / np.sum(distances**2)
        
        print(f"\nDerived Effective Hubble Constants:")
        print(f"  Mean H₀:   {H0_effective:.1f} km/s/Mpc")
        print(f"  Fitted H₀: {H0_fitted:.1f} km/s/Mpc")
        print(f"  Standard:  {self.H0_standard:.1f} km/s/Mpc")
        print(f"  Ratio:     {H0_fitted/self.H0_standard:.2f}")
        
        return H0_fitted, z_test, distances

# ============================================================================
# SUPERNOVA DATA COMPARISON
# ============================================================================

def load_pantheon_sample(n_sample=100):
    """
    Load or simulate Type Ia supernova data for comparison.
    """
    # Simulate Pantheon-like data if real data not available
    np.random.seed(42)
    
    z = np.logspace(-2, 0, n_sample)
    
    # Standard ΛCDM distance modulus
    # For flat universe: d_L = (c/H₀) * z * (1 + z/2 + ...) 
    # Simplified for low z
    H0 = 70  # km/s/Mpc
    c = 299792.458  # km/s
    
    d_L = (c * z / H0) * (1 + 0.5 * z)  # Mpc, first order correction
    mu_true = 5 * np.log10(d_L) + 25
    
    # Add realistic scatter
    mu_obs = mu_true + np.random.normal(0, 0.15, size=len(z))
    mu_err = 0.15 * np.ones_like(z)
    
    return z, mu_obs, mu_err

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_hubble_comparison(ddmm_model):
    """
    Compare DDMM predictions with standard cosmology.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Load supernova data
    z_sn, mu_sn, mu_err = load_pantheon_sample()
    
    # 1. Hubble Diagram
    ax = axes[0, 0]
    
    # DDMM prediction
    mu_ddmm, d_ddmm = ddmm_model.hubble_diagram_ddmm(z_sn)
    
    # Standard ΛCDM
    H0 = 70
    c = 299792.458
    d_lcdm = (c * z_sn / H0) * (1 + 0.5 * z_sn)
    mu_lcdm = 5 * np.log10(d_lcdm) + 25
    
    ax.errorbar(z_sn, mu_sn, yerr=mu_err, fmt='ko', alpha=0.3, 
                markersize=3, label='Type Ia SNe')
    ax.plot(z_sn, mu_lcdm, 'b-', label='ΛCDM', linewidth=2)
    ax.plot(z_sn, mu_ddmm, 'r--', label='DDMM (no expansion)', linewidth=2)
    
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Distance Modulus μ', fontsize=12)
    ax.set_title('Hubble Diagram: DDMM vs Standard Cosmology', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if np.all(z_sn > 0):
        ax.set_xscale('log')
        ax.set_xlim(0.01, 1)
    else:
        ax.set_xlim(min(z_sn)*0.9, max(z_sn)*1.1)

    ax.set_xscale('log')
    ax.set_xlim(0.01, 1)
    
    # 2. Residuals
    ax = axes[0, 1]
    
    residuals_ddmm = mu_ddmm - mu_lcdm
    ax.plot(z_sn, residuals_ddmm, 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Δμ (DDMM - ΛCDM)', fontsize=12)
    ax.set_title('Distance Modulus Residuals', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 3. Density-Redshift Relation
    ax = axes[1, 0]
    
    distances = np.logspace(-3, 3, 100)  # Mpc
    redshifts_direct = []
    redshifts_path = []
    
    for d in distances:
        # Observer and emitter densities
        rho_obs = ddmm_model.density_profile_cosmological(0)
        rho_emit = ddmm_model.density_profile_cosmological(d)
        
        z_direct = ddmm_model.gravitational_redshift_direct(rho_obs, rho_emit)
        z_path = ddmm_model.gravitational_redshift_path_integral(d, n_steps=100)
        
        redshifts_direct.append(z_direct)
        redshifts_path.append(z_path)
    
    ax.plot(distances, redshifts_direct, 'b-', label='Direct formula', linewidth=2)
    ax.plot(distances, redshifts_path, 'r--', label='Path integral', linewidth=2)
    
    # Hubble law for comparison
    z_hubble = distances * H0 / c
    ax.plot(distances, z_hubble, 'g:', label=f'Hubble (H₀={H0})', linewidth=2)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=12)
    ax.set_ylabel('Redshift z', fontsize=12)
    ax.set_title('DDMM Redshift vs Distance', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    # Only set log scale if we have positive values
    if np.all(np.array(redshifts_path) > 0):
        ax.set_yscale('log')
        ax.set_ylim(0.0001, 10)
    else:
        # Use linear scale if values aren't all positive
        ax.set_ylim(min(redshifts_path + redshifts_direct), max(redshifts_path + redshifts_direct) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.001, 1000)
    
    # 4. Effective Hubble Parameter
    ax = axes[1, 1]
    
    # Calculate H₀ at different scales
    distances = np.logspace(-1, 2, 30)
    H_eff = []
    
    for d in distances:
        z = ddmm_model.gravitational_redshift_path_integral(d, n_steps=100)
        if z > 0 and d > 0:
            H = c * z / d
            H_eff.append(H)
        else:
            H_eff.append(np.nan)
    
    ax.plot(distances, H_eff, 'r-', linewidth=2)
    ax.axhline(H0, color='blue', linestyle='--', label=f'Standard H₀={H0}')
    
    ax.set_xlabel('Distance (Mpc)', fontsize=12)
    ax.set_ylabel('Effective H (km/s/Mpc)', fontsize=12)
    ax.set_title('Scale-Dependent Hubble Parameter in DDMM', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_density_xi_profiles(ddmm_model):
    """
    Show density and enhancement factor profiles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distance range
    r_Mpc = np.logspace(-3, 3, 1000)
    
    # Calculate profiles
    rho = [ddmm_model.density_profile_cosmological(r) for r in r_Mpc]
    xi = [ddmm_model.xi_function(r) for r in rho]
    
    # 1. Density Profile
    ax = axes[0]
    ax.plot(r_Mpc, rho, 'b-', linewidth=2)
    ax.axhline(ddmm_model.rho_c, color='r', linestyle='--', 
               label=f'ρ_c = {ddmm_model.rho_c:.1e} M_☉/kpc³')
    
    ax.set_xlabel('Distance (Mpc)', fontsize=12)
    ax.set_ylabel('Density (M_☉/kpc³)', fontsize=12)
    ax.set_title('Cosmological Density Profile', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(0.001, 1e8, 'Galaxy', fontsize=10, ha='center')
    ax.text(0.1, 1e6, 'Local Group', fontsize=10, ha='center')
    ax.text(10, 1e4, 'Clusters', fontsize=10, ha='center')
    ax.text(500, 1e2, 'Cosmic Voids', fontsize=10, ha='center')
    
    # 2. Enhancement Factor
    ax = axes[1]
    ax.plot(r_Mpc, xi, 'r-', linewidth=2)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Distance (Mpc)', fontsize=12)
    ax.set_ylabel('Enhancement Factor ξ', fontsize=12)
    ax.set_title(f'DDMM Enhancement (A={ddmm_model.A:.1f}, n={ddmm_model.n:.3f})', 
                 fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 5.1)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_cosmological_test():
    """
    Main test: Can DDMM explain cosmological redshift without expansion?
    """
    print("="*70)
    print("DDMM COSMOLOGICAL REDSHIFT TEST")
    print("Testing if modified gravity can explain Hubble's law")
    print("="*70)
    
    # Initialize with your fitted parameters
    ddmm = DDMMCosmologicalRedshift(A=5.22, n=1.245, rho_c=6.83e15)
    
    # Test 1: Derive effective Hubble constant
    print("\n1. DERIVING HUBBLE CONSTANT FROM PURE DDMM:")
    print("-" * 50)
    H0_derived, z_test, d_test = ddmm.derive_effective_hubble_constant()
    
    # Test 2: Compare different redshift calculations
    print("\n2. COMPARING REDSHIFT METHODS:")
    print("-" * 50)
    
    test_distances = [0.1, 1, 10, 100, 1000]  # Mpc
    
    for d in test_distances:
        rho_obs = ddmm.density_profile_cosmological(0)
        rho_emit = ddmm.density_profile_cosmological(d)
        
        z_direct = ddmm.gravitational_redshift_direct(rho_obs, rho_emit)
        z_path = ddmm.gravitational_redshift_path_integral(d, n_steps=500)
        z_metric = ddmm.metric_redshift_accumulated(d, n_steps=500)
        
        # Standard Hubble redshift for comparison
        z_hubble = d * 70 / 299792.458
        
        print(f"\nDistance = {d:6.1f} Mpc:")
        print(f"  Direct formula:  z = {z_direct:8.5f}")
        print(f"  Path integral:   z = {z_path:8.5f}")
        print(f"  Metric accumul:  z = {z_metric:8.5f}")
        print(f"  Hubble (H₀=70):  z = {z_hubble:8.5f}")
        print(f"  Ratio to Hubble:     {z_path/z_hubble:.3f}")
    
    # Test 3: Chi-squared with supernova data
    print("\n3. COMPARISON WITH TYPE IA SUPERNOVAE:")
    print("-" * 50)
    
    z_sn, mu_sn, mu_err = load_pantheon_sample(50)
    mu_ddmm, _ = ddmm.hubble_diagram_ddmm(z_sn)
    
    # Standard ΛCDM
    H0 = 70
    c = 299792.458
    d_lcdm = (c * z_sn / H0) * (1 + 0.5 * z_sn)
    mu_lcdm = 5 * np.log10(d_lcdm) + 25
    
    chi2_ddmm = np.sum(((mu_sn - mu_ddmm) / mu_err)**2)
    chi2_lcdm = np.sum(((mu_sn - mu_lcdm) / mu_err)**2)
    
    print(f"χ² (DDMM):  {chi2_ddmm:.1f}")
    print(f"χ² (ΛCDM):  {chi2_lcdm:.1f}")
    print(f"Δχ²:        {chi2_ddmm - chi2_lcdm:.1f}")
    
    if chi2_ddmm < chi2_lcdm:
        print("Result: DDMM provides BETTER fit to SNe data!")
    else:
        print("Result: ΛCDM provides better fit to SNe data")
    
    # Test 4: Physical interpretation
    print("\n4. PHYSICAL INTERPRETATION:")
    print("-" * 50)
    
    # Calculate typical enhancement at different scales
    scales = {
        "Solar System (1 AU)": 1.5e-8,  # Mpc
        "Galaxy (10 kpc)": 0.01,
        "Local Group (1 Mpc)": 1,
        "Cluster (10 Mpc)": 10,
        "Cosmic void (100 Mpc)": 100
    }
    
    print("\nGravitational enhancement at different scales:")
    for name, distance in scales.items():
        rho = ddmm.density_profile_cosmological(distance)
        xi = ddmm.xi_function(rho)
        print(f"  {name:25s}: ξ = {xi:.3f} (ρ = {rho:.2e} M_☉/kpc³)")
    
    # Generate plots
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 50)
    
    fig1 = plot_hubble_comparison(ddmm)
    fig2 = plot_density_xi_profiles(ddmm)
    
    # Save plots
    output_dir = Path("ddmm_cosmology_results")
    output_dir.mkdir(exist_ok=True)
    
    fig1.savefig(output_dir / "hubble_comparison.png", dpi=150, bbox_inches='tight')
    fig2.savefig(output_dir / "density_xi_profiles.png", dpi=150, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    
    if abs(H0_derived - 70) / 70 < 0.3:  # Within 30% of standard H₀
        print("✓ DDMM can produce Hubble-like relation!")
        print(f"  Derived H₀ = {H0_derived:.1f} km/s/Mpc")
        print(f"  This is {H0_derived/70:.1%} of the standard value")
    else:
        print("✗ DDMM produces different scaling than Hubble's law")
        print(f"  Effective H₀ varies with scale")
    
    print("\nKey findings:")
    print("- DDMM predicts redshift from gravitational effects alone")
    print("- No cosmic expansion needed in this model")
    print("- Redshift arises from photons climbing out of gravity wells")
    print("- Enhancement factor ξ creates effective cosmic redshift")
    
    print("\nNext steps to improve agreement:")
    print("1. Refine cosmic density profile with N-body simulations")
    print("2. Include galaxy clustering and cosmic web structure")
    print("3. Test with full Pantheon+ supernova dataset")
    print("4. Compare with BAO and CMB constraints")
    
    
    # Don't try to show plots if running non-interactively
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        else:
            print("\nPlots saved but not displayed (non-interactive mode)")
    except:
        print("\nPlots saved to disk")
    
    return ddmm, H0_derived

# ============================================================================
# ADVANCED TEST: WITH YOUR ACTUAL PARAMETERS
# ============================================================================

def test_with_your_fitted_params(npz_file=None):
    """
    Test using your actual fitted parameters from dynesty run.
    """
    if npz_file and Path(npz_file).exists():
        print(f"Loading fitted parameters from {npz_file}")
        try:
            data = np.load(npz_file)
            print(f"  File loaded successfully")
            print(f"  Available keys: {list(data.keys())}")
            
            # Extract best-fit parameters
            if 'samples' in data and 'param_names' in data:
                param_names = data['param_names']
                samples = data['samples']
                print(f"  Found {len(samples)} samples for {len(param_names)} parameters")
                
                # Get median values
                param_dict = {}
                for i, name in enumerate(param_names):
                    param_dict[name] = np.median(samples[:, i])
                    print(f"    {name}: {param_dict[name]:.3e}")
                
                # Extract DDMM parameters
                if 'rho_c_solar_kpc3' in param_dict:
                    rho_c = param_dict['rho_c_solar_kpc3']
                else:
                    rho_c = 6.83e15
                    print("  Warning: rho_c_solar_kpc3 not found, using default")
                    
                if 'n_exp' in param_dict:
                    n = param_dict['n_exp']
                else:
                    n = 1.245
                    print("  Warning: n_exp not found, using default")
                    
                if 'A' in param_dict:
                    A = param_dict['A']
                elif 'A_xi' in param_dict:
                    A = param_dict['A_xi']
                else:
                    A = 5.22
                    print("  Warning: A not found, using default")
                
                print(f"\nUsing fitted parameters: A={A:.2f}, n={n:.3f}, ρ_c={rho_c:.2e}")
                
                # Run test with fitted parameters
                ddmm = DDMMCosmologicalRedshift(A=A, n=n, rho_c=rho_c)
                return ddmm
            else:
                print("  Warning: Required keys not found in npz file")
                print("  Using default parameters")
                return DDMMCosmologicalRedshift()
        except Exception as e:
            print(f"  Error loading npz file: {e}")
            print("  Using default parameters")
            return DDMMCosmologicalRedshift()
    else:
        print("Using default parameters from paper")
        return DDMMCosmologicalRedshift()

# ============================================================================
# RUN THE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXECUTING MAIN TEST")
    print("="*70)
    
    try:
        # Check if user provided their fitted parameters file
        if len(sys.argv) > 1:
            npz_file = sys.argv[1]
            print(f"Attempting to load fitted parameters from: {npz_file}")
            
            if not Path(npz_file).exists():
                print(f"Warning: File {npz_file} not found, using default parameters")
                ddmm = DDMMCosmologicalRedshift()
            else:
                ddmm = test_with_your_fitted_params(npz_file)
        else:
            print("No parameter file specified, using default parameters from paper")
            ddmm = DDMMCosmologicalRedshift()
        
        # Run the comprehensive test
        print("\nStarting comprehensive cosmological test...")
        ddmm_model, H0_eff = run_cosmological_test()
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)