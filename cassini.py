#!/usr/bin/env python3
"""
cassini_formula_tester.py - Test various DDMM ξ formulas against Cassini constraint

Theory: Below a critical baryonic mass, gravity is "elastic" (ξ > 1)
        Above that mass, gravity is normal (ξ ≈ 1)
        
The Sun has M☉ = 2×10^30 kg, well above any reasonable threshold,
so ξ should be very close to 1 for the Cassini test.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import json

# Physical constants
C_KMS = 299792.458  # km/s
M_SUN = 1.989e30  # kg
G_SI = 6.67430e-11  # m^3/kg/s^2
AU_M = 1.496e11  # meters
PC_M = 3.086e16  # meters
KPC_M = 3.086e19  # meters

@dataclass
class CassiniTest:
    """Container for Cassini test results"""
    formula_name: str
    gamma_parameter: float  # The (1 + γ)/2 parameter that Cassini measured
    max_xi_deviation: float
    passes_cassini: bool
    xi_at_sun: float
    xi_at_earth: float
    xi_path: np.ndarray
    r_path: np.ndarray

class DDMMFormulaTester:
    """Test various DDMM formulas against Cassini constraint"""
    
    def __init__(self):
        # Cassini constraint: γ = 1 + (2.1 ± 2.3) × 10^-5
        self.cassini_gamma_limit = 2.3e-5
        
        # Solar System model
        self.setup_solar_system()
        
        # Results storage
        self.results = []
        
    def setup_solar_system(self):
        """Setup Solar System density and mass distribution"""
        # Distances along Earth-Saturn line when close to Sun
        self.r_min = 0.01  # AU (close to Sun)
        self.r_max = 10.0  # AU (Saturn distance)
        self.r_path_au = np.linspace(self.r_min, self.r_max, 1000)
        self.r_path_m = self.r_path_au * AU_M
        self.r_path_kpc = self.r_path_m / KPC_M
        
        # Mass distribution (simplified)
        # Sun dominates within ~1 AU
        self.M_enclosed = np.zeros_like(self.r_path_m)
        for i, r in enumerate(self.r_path_m):
            if r < AU_M:
                # Within Sun's orbit - essentially all Sun's mass
                self.M_enclosed[i] = M_SUN
            else:
                # Add planets (simplified)
                self.M_enclosed[i] = M_SUN + 1e-3 * M_SUN  # Sun + planets
        
        # Density at each point (rough approximation)
        # ρ(r) ≈ M(<r) / (4πr³/3) for average density within radius r
        self.rho_kg_m3 = 3 * self.M_enclosed / (4 * np.pi * self.r_path_m**3)
        
        # Convert to M☉/kpc³
        self.rho_msun_kpc3 = self.rho_kg_m3 * (KPC_M**3) / M_SUN
        
        print(f"Solar System density range: {self.rho_msun_kpc3.min():.2e} to {self.rho_msun_kpc3.max():.2e} M☉/kpc³")
        print(f"Mass enclosed range: {self.M_enclosed.min()/M_SUN:.2e} to {self.M_enclosed.max()/M_SUN:.2e} M☉")
    
    def cassini_light_deflection_test(self, xi_func: Callable, formula_name: str, 
                                     params: Dict, plot: bool = True) -> CassiniTest:
        """
        Test a ξ formula against Cassini constraint.
        
        In GR, the effective metric parameter γ appears in the Shapiro delay:
        Δt = (1 + γ)/2 * (2GM/c³) * ln(path geometry)
        
        In DDMM, this gets modified by ξ along the path.
        """
        # Calculate ξ along the path
        xi_path = xi_func(self.r_path_kpc, self.rho_msun_kpc3, self.M_enclosed/M_SUN, params)
        
        # Key values
        xi_at_sun = xi_path[0]  # Near Sun
        xi_at_earth = xi_path[np.argmin(np.abs(self.r_path_au - 1.0))]  # At Earth
        
        # In DDMM, the effective γ parameter is modified by ξ
        # For a simple estimate: γ_eff ≈ ξ * γ_GR
        # More precisely, we need to integrate along the light path
        
        # The Shapiro delay depends on the integral of the metric perturbation
        # In DDMM: γ_eff = path-averaged ξ value (weighted by 1/r)
        # Weight by 1/r because Shapiro delay ∝ ∫(1/r)dr along path
        
        weights = 1.0 / self.r_path_m
        gamma_eff = np.average(xi_path, weights=weights) - 1.0  # γ = ξ - 1
        
        # Maximum deviation from ξ = 1
        max_xi_deviation = np.max(np.abs(xi_path - 1.0))
        
        # Check if passes Cassini constraint
        passes = abs(gamma_eff) < self.cassini_gamma_limit
        
        result = CassiniTest(
            formula_name=formula_name,
            gamma_parameter=gamma_eff,
            max_xi_deviation=max_xi_deviation,
            passes_cassini=passes,
            xi_at_sun=xi_at_sun,
            xi_at_earth=xi_at_earth,
            xi_path=xi_path,
            r_path=self.r_path_au
        )
        
        if plot:
            self.plot_xi_profile(result)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Formula: {formula_name}")
        print(f"Parameters: {params}")
        print(f"γ_eff = {gamma_eff:.2e} (limit: ±{self.cassini_gamma_limit:.2e})")
        print(f"Max |ξ-1| = {max_xi_deviation:.2e}")
        print(f"ξ at Sun center: {xi_at_sun:.8f}")
        print(f"ξ at Earth: {xi_at_earth:.8f}")
        print(f"Status: {'✅ PASS' if passes else '❌ FAIL'}")
        
        self.results.append(result)
        return result
    
    def plot_xi_profile(self, result: CassiniTest):
        """Plot ξ profile along Cassini radio path"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top: ξ value
        ax1.semilogx(result.r_path, result.xi_path, 'b-', lw=2)
        ax1.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax1.fill_between(result.r_path, 
                         1 - self.cassini_gamma_limit, 
                         1 + self.cassini_gamma_limit,
                         alpha=0.2, color='green', label='Cassini allowed')
        ax1.set_ylabel('ξ(r)', fontsize=12)
        ax1.set_title(f'{result.formula_name}: {"PASS" if result.passes_cassini else "FAIL"}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom: Mass enclosed
        ax2.semilogx(self.r_path_au, self.M_enclosed/M_SUN, 'r-', lw=2)
        ax2.set_xlabel('Distance from Sun (AU)', fontsize=12)
        ax2.set_ylabel('Enclosed Mass (M☉)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Mark key locations
        for ax in [ax1, ax2]:
            ax.axvline(1.0, color='gray', ls=':', alpha=0.5, label='Earth')
            ax.axvline(9.5, color='gray', ls=':', alpha=0.5, label='Saturn')
        
        plt.tight_layout()
        plt.savefig(f'cassini_test_{result.formula_name.replace(" ", "_")}.png', dpi=150)
        plt.close()
    
    def test_galaxy_compatibility(self, xi_func: Callable, params: Dict) -> Dict:
        """Quick test of galaxy rotation curve compatibility"""
        # Typical galaxy parameters
        r_gal = np.logspace(0, 1.5, 50)  # 1-30 kpc
        M_gal = 1e11  # M☉ total
        
        # Simple exponential disk model
        R_d = 3.0  # kpc
        M_enclosed_gal = M_gal * (1 - (1 + r_gal/R_d) * np.exp(-r_gal/R_d))
        
        # Average density
        rho_gal = 3 * M_enclosed_gal / (4 * np.pi * r_gal**3)
        rho_gal[0] = rho_gal[1]  # Fix r=0 singularity
        
        # Calculate ξ
        xi_gal = xi_func(r_gal, rho_gal, M_enclosed_gal/M_SUN, params)
        
        # Check if provides enough boost
        xi_avg = np.mean(xi_gal[5:15])  # Average in 5-15 kpc range
        
        return {
            'xi_avg_galaxy': xi_avg,
            'xi_min': np.min(xi_gal),
            'xi_max': np.max(xi_gal),
            'sufficient_boost': xi_avg > 2.0
        }
    
    # ========================================================================
    # FORMULA DEFINITIONS
    # ========================================================================
    
    @staticmethod
    def xi_mass_threshold(r_kpc, rho, M_enclosed_msun, params):
        """
        Mass threshold model: ξ = 1 for M > M_crit, enhanced below
        
        This directly implements your theory: above a critical mass,
        gravity is normal, below it gravity is "elastic"
        """
        M_crit = params['M_crit_msun']  # Critical mass in M☉
        xi_boost = params['xi_boost']   # Boost factor for low mass
        width = params.get('width', 0.1)  # Transition width
        
        # Smooth transition using tanh
        xi = 1 + (xi_boost - 1) * 0.5 * (1 - np.tanh((M_enclosed_msun - M_crit) / (width * M_crit)))
        
        return xi
    
    @staticmethod
    def xi_mass_power_law(r_kpc, rho, M_enclosed_msun, params):
        """
        Power law in mass: ξ = 1 + A * (M_crit/M)^n for M > M_min
        """
        M_crit = params['M_crit_msun']
        n_exp = params['n_exp']
        A = params['A']
        M_min = params.get('M_min_msun', 1e-10)  # Minimum mass to avoid singularity
        
        M_safe = np.maximum(M_enclosed_msun, M_min)
        xi = 1 + A * (M_crit / M_safe) ** n_exp
        
        # Cap at reasonable values
        return np.minimum(xi, 5.0)
    
    @staticmethod
    def xi_density_screened(r_kpc, rho, M_enclosed_msun, params):
        """
        Density-dependent but screened in high-mass environments
        """
        rho_c = params['rho_c_msun_kpc3']
        n_exp = params['n_exp']
        A = params['A']
        M_screen = params['M_screen_msun']  # Screening mass
        
        # Basic density dependence
        xi_base = 1 + A * (rho_c / rho) ** n_exp
        
        # Screening factor: reduces to 1 for large masses
        screen_factor = np.exp(-M_enclosed_msun / M_screen)
        
        xi = 1 + (xi_base - 1) * screen_factor
        return xi
    
    @staticmethod
    def xi_mixed_model(r_kpc, rho, M_enclosed_msun, params):
        """
        Mixed density and mass dependence
        ξ = 1 + A * (ρ_c/ρ)^n * exp(-M/M_screen)
        """
        rho_c = params['rho_c_msun_kpc3']
        n_exp = params['n_exp']
        A = params['A']
        M_screen = params['M_screen_msun']
        
        density_factor = (rho_c / rho) ** n_exp
        mass_screen = np.exp(-M_enclosed_msun / M_screen)
        
        xi = 1 + A * density_factor * mass_screen
        return np.minimum(xi, 5.0)
    
    @staticmethod
    def xi_yukawa_like(r_kpc, rho, M_enclosed_msun, params):
        """
        Yukawa-like screening with mass dependence
        ξ = 1 + (ξ_0 - 1) * exp(-M/M_screen) * exp(-r/λ)
        """
        xi_0 = params['xi_0']  # Unscreened value
        M_screen = params['M_screen_msun']
        lambda_kpc = params['lambda_kpc']  # Range in kpc
        
        mass_screen = np.exp(-M_enclosed_msun / M_screen)
        range_screen = np.exp(-r_kpc / lambda_kpc)
        
        xi = 1 + (xi_0 - 1) * mass_screen * range_screen
        return xi
    
    @staticmethod
    def xi_critical_density(r_kpc, rho, M_enclosed_msun, params):
        """
        Critical density model: ξ changes at specific density threshold
        But only for low-mass systems
        """
        rho_crit = params['rho_crit_msun_kpc3']
        xi_low = params['xi_low']  # ξ for ρ < ρ_crit
        xi_high = params.get('xi_high', 1.0)  # ξ for ρ > ρ_crit
        M_apply = params['M_apply_msun']  # Only apply for M < M_apply
        
        # Base ξ from density
        xi_base = np.where(rho < rho_crit, xi_low, xi_high)
        
        # Mass-dependent application
        apply_factor = np.exp(-M_enclosed_msun / M_apply)
        
        xi = 1 + (xi_base - 1) * apply_factor
        return xi
    
    def run_all_tests(self):
        """Test all formulas against Cassini constraint"""
        
        # Define parameter sets to test for each formula
        test_cases = [
            # Mass threshold model - your primary theory
            {
                'formula': self.xi_mass_threshold,
                'name': 'Mass Threshold',
                'param_sets': [
                    {'M_crit_msun': 0.001, 'xi_boost': 3.0, 'width': 0.1},
                    {'M_crit_msun': 0.01, 'xi_boost': 3.0, 'width': 0.1},
                    {'M_crit_msun': 0.1, 'xi_boost': 2.5, 'width': 0.2},
                    {'M_crit_msun': 1.0, 'xi_boost': 2.5, 'width': 0.5},
                ]
            },
            
            # Mass power law
            {
                'formula': self.xi_mass_power_law,
                'name': 'Mass Power Law',
                'param_sets': [
                    {'M_crit_msun': 1e-3, 'n_exp': 1.0, 'A': 2.0},
                    {'M_crit_msun': 1e-2, 'n_exp': 0.5, 'A': 3.0},
                    {'M_crit_msun': 1e-4, 'n_exp': 1.5, 'A': 1.5},
                ]
            },
            
            # Density with mass screening
            {
                'formula': self.xi_density_screened,
                'name': 'Density + Mass Screen',
                'param_sets': [
                    {'rho_c_msun_kpc3': 1e20, 'n_exp': 1.0, 'A': 2.0, 'M_screen_msun': 0.1},
                    {'rho_c_msun_kpc3': 1e18, 'n_exp': 1.5, 'A': 1.5, 'M_screen_msun': 0.01},
                    {'rho_c_msun_kpc3': 1e22, 'n_exp': 0.8, 'A': 3.0, 'M_screen_msun': 1.0},
                ]
            },
            
            # Mixed model
            {
                'formula': self.xi_mixed_model,
                'name': 'Mixed Density-Mass',
                'param_sets': [
                    {'rho_c_msun_kpc3': 1e16, 'n_exp': 1.0, 'A': 2.0, 'M_screen_msun': 0.1},
                    {'rho_c_msun_kpc3': 1e18, 'n_exp': 0.8, 'A': 2.5, 'M_screen_msun': 0.5},
                ]
            },
            
            # Yukawa-like
            {
                'formula': self.xi_yukawa_like,
                'name': 'Yukawa Screening',
                'param_sets': [
                    {'xi_0': 3.0, 'M_screen_msun': 0.1, 'lambda_kpc': 10.0},
                    {'xi_0': 2.5, 'M_screen_msun': 0.01, 'lambda_kpc': 20.0},
                ]
            },
            
            # Critical density
            {
                'formula': self.xi_critical_density,
                'name': 'Critical Density',
                'param_sets': [
                    {'rho_crit_msun_kpc3': 1e12, 'xi_low': 3.0, 'xi_high': 1.0, 'M_apply_msun': 0.1},
                    {'rho_crit_msun_kpc3': 1e10, 'xi_low': 2.5, 'xi_high': 1.0, 'M_apply_msun': 0.01},
                ]
            },
        ]
        
        # Test each formula with each parameter set
        passing_formulas = []
        
        for test in test_cases:
            formula = test['formula']
            formula_name = test['name']
            
            print(f"\n{'='*70}")
            print(f"Testing {formula_name} Formula")
            print(f"{'='*70}")
            
            for i, params in enumerate(test['param_sets']):
                name = f"{formula_name}_{i+1}"
                result = self.cassini_light_deflection_test(
                    formula, name, params, plot=True
                )
                
                if result.passes_cassini:
                    # Also test galaxy compatibility
                    galaxy_test = self.test_galaxy_compatibility(formula, params)
                    
                    print(f"\nGalaxy compatibility:")
                    print(f"  Average ξ (5-15 kpc): {galaxy_test['xi_avg_galaxy']:.2f}")
                    print(f"  Sufficient boost: {'✅' if galaxy_test['sufficient_boost'] else '❌'}")
                    
                    if galaxy_test['sufficient_boost']:
                        passing_formulas.append({
                            'formula': formula,
                            'name': name,
                            'params': params,
                            'cassini_result': result,
                            'galaxy_test': galaxy_test
                        })
        
        return passing_formulas
    
    def export_best_formulas(self, passing_formulas: List[Dict], 
                            filename: str = 'cassini_passing_formulas.json'):
        """Export formulas that pass both tests"""
        
        print(f"\n{'='*70}")
        print(f"FORMULAS PASSING BOTH CASSINI AND GALAXY TESTS")
        print(f"{'='*70}")
        
        export_data = []
        
        for pf in passing_formulas:
            print(f"\n{pf['name']}:")
            print(f"  Parameters: {pf['params']}")
            print(f"  Cassini γ_eff: {pf['cassini_result'].gamma_parameter:.2e}")
            print(f"  Galaxy <ξ>: {pf['galaxy_test']['xi_avg_galaxy']:.2f}")
            
            export_data.append({
                'name': pf['name'],
                'formula_type': pf['name'].split('_')[0],
                'params': pf['params'],
                'cassini_gamma': float(pf['cassini_result'].gamma_parameter),
                'galaxy_xi_avg': float(pf['galaxy_test']['xi_avg_galaxy']),
                'xi_at_sun': float(pf['cassini_result'].xi_at_sun),
                'xi_at_earth': float(pf['cassini_result'].xi_at_earth)
            })
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nExported {len(export_data)} passing formulas to {filename}")
        
        return export_data

def main():
    """Run comprehensive Cassini tests"""
    tester = DDMMFormulaTester()
    
    # Run all tests
    passing_formulas = tester.run_all_tests()
    
    # Export results
    if passing_formulas:
        tester.export_best_formulas(passing_formulas)
        print(f"\n✅ Found {len(passing_formulas)} formulas that pass both tests!")
    else:
        print(f"\n❌ No formulas passed both Cassini and galaxy tests.")
        print("Consider adjusting parameter ranges or trying new functional forms.")
    
    # Create summary plot
    if tester.results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cassini performance
        names = [r.formula_name for r in tester.results]
        gammas = [abs(r.gamma_parameter) for r in tester.results]
        colors = ['green' if r.passes_cassini else 'red' for r in tester.results]
        
        ax1.bar(range(len(names)), gammas, color=colors)
        ax1.axhline(tester.cassini_gamma_limit, color='k', ls='--', label='Cassini limit')
        ax1.set_yscale('log')
        ax1.set_ylabel('|γ_eff - 1|')
        ax1.set_title('Cassini Constraint Test')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        ax1.legend()
        
        # Xi at different locations
        xi_sun = [r.xi_at_sun for r in tester.results]
        xi_earth = [r.xi_at_earth for r in tester.results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax2.bar(x - width/2, xi_sun, width, label='ξ at Sun')
        ax2.bar(x + width/2, xi_earth, width, label='ξ at Earth')
        ax2.axhline(1.0, color='k', ls='--')
        ax2.set_ylabel('ξ value')
        ax2.set_title('ξ Values in Solar System')
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('cassini_test_summary.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main()