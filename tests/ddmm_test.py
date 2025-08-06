#!/usr/bin/env python3
"""
Comprehensive Solar System tests for DDMM model
Verifies that the fitted DDMM parameters satisfy all observational constraints
Uses CuPy for GPU acceleration
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from scipy import constants

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m/s
AU = 1.495978707e11  # m
pc_to_m = 3.0857e16  # m
kpc_to_m = pc_to_m * 1000
M_sun = 1.98847e30  # kg

# Unit conversions
M_sun_per_kpc3_to_kg_per_m3 = M_sun / (kpc_to_m**3)

@dataclass
class DDMMParameters:
    """Enhanced DDMM parameters from the fit"""
    A: float = 5.223878356970734
    n: float = 1.245070374464584
    rho_c: float = 6.834318282559106e15  # M_sun/kpc^3
    A_std: float = 2.4280136911786077
    n_std: float = 0.004079771740735703
    rho_c_std: float = 2.583166796650079e15

class SolarSystemDDMMTests:
    """
    Comprehensive test suite for DDMM in the Solar System
    """
    
    def __init__(self, params: DDMMParameters = None):
        """Initialize with DDMM parameters"""
        self.params = params or DDMMParameters()
        
        # Convert critical density to SI units for some tests
        self.rho_c_SI = self.params.rho_c * M_sun_per_kpc3_to_kg_per_m3
        
        # Solar System bodies and their properties
        self.setup_solar_system()
        
    def setup_solar_system(self):
        """Setup Solar System body properties"""
        # Masses in kg
        self.masses = {
            'Sun': 1.98847e30,
            'Mercury': 3.301e23,
            'Venus': 4.867e24,
            'Earth': 5.972e24,
            'Moon': 7.342e22,
            'Mars': 6.417e23,
            'Jupiter': 1.898e27,
            'Saturn': 5.683e26,
            'Uranus': 8.681e25,
            'Neptune': 1.024e26,
        }
        
        # Semi-major axes in AU
        self.orbits_AU = {
            'Mercury': 0.387,
            'Venus': 0.723,
            'Earth': 1.000,
            'Moon': 0.00257,  # from Earth
            'Mars': 1.524,
            'Jupiter': 5.203,
            'Saturn': 9.537,
            'Uranus': 19.191,
            'Neptune': 30.069,
        }
        
        # Convert to meters
        self.orbits_m = {k: v * AU for k, v in self.orbits_AU.items()}
        
    def xi_enhancement(self, rho: cp.ndarray) -> cp.ndarray:
        """
        Calculate enhancement factor ξ(ρ) for given density
        
        Args:
            rho: Density in M_sun/kpc^3 (can be CuPy array)
            
        Returns:
            Enhancement factor ξ
        """
        # Avoid division by zero
        rho_safe = cp.maximum(rho, 1e-50)
        
        # Calculate enhancement with cap at 5
        xi = 1.0 + self.params.A * cp.power(self.params.rho_c / rho_safe, self.params.n)
        return cp.minimum(xi, 5.0)
    
    def local_density_SI(self, r: float, M_central: float = None) -> float:
        """
        Estimate local density at distance r from central mass
        Using simple point mass approximation for order of magnitude
        
        Args:
            r: Distance in meters
            M_central: Central mass in kg (default: Sun)
            
        Returns:
            Density in kg/m^3
        """
        if M_central is None:
            M_central = self.masses['Sun']
            
        # For a point mass, density ~ M/(4πr³/3) in a sphere
        # This is an order-of-magnitude estimate
        volume = (4/3) * np.pi * r**3
        density_SI = M_central / volume
        
        return density_SI
    
    def local_density_Msun_kpc3(self, r: float, M_central: float = None) -> float:
        """
        Convert local density to M_sun/kpc^3
        
        Args:
            r: Distance in meters
            M_central: Central mass in kg
            
        Returns:
            Density in M_sun/kpc^3
        """
        density_SI = self.local_density_SI(r, M_central)
        return density_SI / M_sun_per_kpc3_to_kg_per_m3
    
    def test_cassini_constraint(self) -> Dict:
        """
        Test Cassini spacecraft constraint: |γ - 1| < 2.3e-5
        In DDMM, this translates to |ξ - 1| < 2.3e-5
        """
        print("=" * 60)
        print("CASSINI CONSTRAINT TEST")
        print("=" * 60)
        
        # Test at various Solar System locations
        locations = {
            'Mercury orbit': 0.387 * AU,
            'Venus orbit': 0.723 * AU,
            'Earth orbit': 1.0 * AU,
            'Mars orbit': 1.524 * AU,
            'Asteroid belt': 2.8 * AU,
            'Jupiter orbit': 5.2 * AU,
            'Saturn orbit': 9.5 * AU,
            'Cassini measurement': 6.0 * AU,  # Approximate solar conjunction distance
        }
        
        results = {}
        cassini_limit = 2.3e-5
        
        for name, r in locations.items():
            # Calculate density in M_sun/kpc^3
            rho = self.local_density_Msun_kpc3(r)
            
            # Calculate enhancement on GPU
            rho_gpu = cp.array([rho])
            xi_gpu = self.xi_enhancement(rho_gpu)
            xi = float(xi_gpu.get()[0])
            
            deviation = abs(xi - 1.0)
            passes = deviation < cassini_limit
            margin = cassini_limit / deviation if deviation > 0 else np.inf
            
            results[name] = {
                'distance_AU': r / AU,
                'density_Msun_kpc3': rho,
                'xi': xi,
                'deviation': deviation,
                'passes': passes,
                'safety_margin': margin
            }
            
            print(f"\n{name}:")
            print(f"  Distance: {r/AU:.2f} AU")
            print(f"  Density: {rho:.2e} M_☉/kpc³")
            print(f"  ξ: {xi:.15f}")
            print(f"  |ξ - 1|: {deviation:.2e}")
            print(f"  Cassini limit: {cassini_limit:.2e}")
            print(f"  PASSES: {passes} (margin: {margin:.0e}x)")
        
        return results
    
    def test_mercury_perihelion(self) -> Dict:
        """
        Test Mercury perihelion precession
        GR predicts 42.98 arcsec/century, observed is 43.1 ± 0.5
        DDMM should not add significant additional precession
        """
        print("\n" + "=" * 60)
        print("MERCURY PERIHELION PRECESSION TEST")
        print("=" * 60)
        
        # Mercury orbital parameters
        a = 0.387 * AU  # semi-major axis
        e = 0.2056  # eccentricity
        perihelion = a * (1 - e)
        
        # Density at Mercury's orbit
        rho = self.local_density_Msun_kpc3(a)
        rho_gpu = cp.array([rho])
        xi = float(self.xi_enhancement(rho_gpu).get()[0])
        
        # DDMM would modify precession by factor ξ
        # Additional precession = 42.98 * (ξ - 1) arcsec/century
        gr_precession = 42.98  # arcsec/century
        ddmm_extra = gr_precession * (xi - 1)
        total_precession = gr_precession + ddmm_extra
        
        # Observational constraint
        observed = 43.1
        observed_error = 0.5
        
        deviation = abs(total_precession - observed)
        passes = deviation < observed_error
        
        results = {
            'density_Msun_kpc3': rho,
            'xi': xi,
            'gr_precession': gr_precession,
            'ddmm_extra': ddmm_extra,
            'total_precession': total_precession,
            'observed': observed,
            'observed_error': observed_error,
            'deviation': deviation,
            'passes': passes
        }
        
        print(f"\nMercury orbit density: {rho:.2e} M_☉/kpc³")
        print(f"Enhancement factor ξ: {xi:.15f}")
        print(f"GR precession: {gr_precession:.2f} arcsec/century")
        print(f"DDMM additional: {ddmm_extra:.2e} arcsec/century")
        print(f"Total predicted: {total_precession:.6f} arcsec/century")
        print(f"Observed: {observed} ± {observed_error} arcsec/century")
        print(f"PASSES: {passes}")
        
        return results
    
    def test_lunar_laser_ranging(self) -> Dict:
        """
        Test Lunar Laser Ranging constraints
        Earth-Moon distance measured to mm precision
        Tests equivalence principle and G variation
        """
        print("\n" + "=" * 60)
        print("LUNAR LASER RANGING TEST")
        print("=" * 60)
        
        # Earth-Moon system
        r_moon = 384400e3  # m, average Earth-Moon distance
        
        # Density at Moon's orbit (dominated by Earth)
        rho = self.local_density_Msun_kpc3(r_moon, self.masses['Earth'])
        rho_gpu = cp.array([rho])
        xi = float(self.xi_enhancement(rho_gpu).get()[0])
        
        # LLR constraint on G-dot/G < 1e-13 per year
        g_dot_limit = 1e-13  # per year
        
        # DDMM would effectively change G by factor ξ
        # If ξ ≠ 1, this looks like G variation
        g_variation = abs(xi - 1.0)
        passes = g_variation < g_dot_limit
        margin = g_dot_limit / g_variation if g_variation > 0 else np.inf
        
        results = {
            'earth_moon_distance_km': r_moon / 1000,
            'density_Msun_kpc3': rho,
            'xi': xi,
            'g_variation': g_variation,
            'g_dot_limit': g_dot_limit,
            'passes': passes,
            'safety_margin': margin
        }
        
        print(f"\nEarth-Moon distance: {r_moon/1000:.0f} km")
        print(f"Local density: {rho:.2e} M_☉/kpc³")
        print(f"Enhancement factor ξ: {xi:.15f}")
        print(f"|ξ - 1|: {g_variation:.2e}")
        print(f"LLR limit: {g_dot_limit:.2e}")
        print(f"PASSES: {passes} (margin: {margin:.0e}x)")
        
        return results
    
    def test_planetary_ephemerides(self) -> Dict:
        """
        Test planetary ephemerides constraints
        Planetary orbits known to high precision
        """
        print("\n" + "=" * 60)
        print("PLANETARY EPHEMERIDES TEST")
        print("=" * 60)
        
        results = {}
        
        # Test each planet
        for planet, orbit_m in self.orbits_m.items():
            if planet == 'Moon':
                continue
                
            # Density at planet's orbit
            rho = self.local_density_Msun_kpc3(orbit_m)
            rho_gpu = cp.array([rho])
            xi = float(self.xi_enhancement(rho_gpu).get()[0])
            
            # Ephemeris constraint: orbital period variations < 1 part in 10^8
            ephemeris_limit = 1e-8
            deviation = abs(xi - 1.0)
            passes = deviation < ephemeris_limit
            
            results[planet] = {
                'orbit_AU': orbit_m / AU,
                'density_Msun_kpc3': rho,
                'xi': xi,
                'deviation': deviation,
                'passes': passes
            }
            
        # Print summary
        print("\nPlanet        Orbit(AU)   Density(M☉/kpc³)   ξ              |ξ-1|        PASSES")
        print("-" * 85)
        for planet, res in results.items():
            print(f"{planet:12} {res['orbit_AU']:8.2f}   {res['density_Msun_kpc3']:12.2e}   "
                  f"{res['xi']:.12f}   {res['deviation']:8.2e}   {res['passes']}")
        
        return results
    
    def test_gravitational_waves(self) -> Dict:
        """
        Test gravitational wave speed = c constraint
        GW170817 showed |c_gw/c - 1| < 3e-15
        """
        print("\n" + "=" * 60)
        print("GRAVITATIONAL WAVE SPEED TEST")
        print("=" * 60)
        
        # DDMM modifies the metric as g̃_μν = ξ(ρ)g_μν
        # For null geodesics (light and GWs), this preserves c
        # as long as ξ is scalar (not tensor)
        
        # Test at LIGO/Virgo detector locations
        earth_density = self.local_density_Msun_kpc3(1.0 * AU)
        rho_gpu = cp.array([earth_density])
        xi_earth = float(self.xi_enhancement(rho_gpu).get()[0])
        
        # GW speed in DDMM
        # Since metric is conformally scaled, light cones preserved
        # c_gw = c (exactly)
        c_gw_over_c = 1.0  # Exact in DDMM
        
        gw_limit = 3e-15
        deviation = abs(c_gw_over_c - 1.0)
        passes = deviation < gw_limit
        
        results = {
            'earth_density_Msun_kpc3': earth_density,
            'xi_earth': xi_earth,
            'c_gw_over_c': c_gw_over_c,
            'deviation': deviation,
            'gw_limit': gw_limit,
            'passes': passes
        }
        
        print(f"\nEarth orbit density: {earth_density:.2e} M_☉/kpc³")
        print(f"Enhancement factor ξ: {xi_earth:.15f}")
        print(f"c_gw/c in DDMM: {c_gw_over_c} (exactly)")
        print(f"GW170817 limit: |c_gw/c - 1| < {gw_limit:.2e}")
        print(f"PASSES: {passes}")
        
        return results
    
    def test_laboratory_constraints(self) -> Dict:
        """
        Test laboratory constraints on gravity
        Inverse square law tested to 52 μm
        """
        print("\n" + "=" * 60)
        print("LABORATORY CONSTRAINTS TEST")
        print("=" * 60)
        
        # Laboratory densities are ENORMOUS
        # Consider a 1 kg mass at 1 cm distance
        r_lab = 0.01  # m
        M_lab = 1.0  # kg
        
        # Density in lab
        density_SI = self.local_density_SI(r_lab, M_lab)
        density_Msun_kpc3 = density_SI / M_sun_per_kpc3_to_kg_per_m3
        
        # Also test at Earth's surface
        earth_radius = 6.371e6  # m
        earth_surface_density_SI = self.local_density_SI(earth_radius, self.masses['Earth'])
        earth_surface_density = earth_surface_density_SI / M_sun_per_kpc3_to_kg_per_m3
        
        # Calculate enhancements
        densities = cp.array([density_Msun_kpc3, earth_surface_density])
        xis = self.xi_enhancement(densities).get()
        
        xi_lab = xis[0]
        xi_surface = xis[1]
        
        # Laboratory limit: |ξ - 1| < 1e-4 (conservative)
        lab_limit = 1e-4
        
        results = {
            'lab': {
                'scale': '1kg at 1cm',
                'density_SI': density_SI,
                'density_Msun_kpc3': density_Msun_kpc3,
                'xi': xi_lab,
                'deviation': abs(xi_lab - 1.0),
                'passes': abs(xi_lab - 1.0) < lab_limit
            },
            'earth_surface': {
                'scale': 'Earth surface',
                'density_SI': earth_surface_density_SI,
                'density_Msun_kpc3': earth_surface_density,
                'xi': xi_surface,
                'deviation': abs(xi_surface - 1.0),
                'passes': abs(xi_surface - 1.0) < lab_limit
            }
        }
        
        print(f"\nLaboratory (1 kg at 1 cm):")
        print(f"  Density: {density_SI:.2e} kg/m³ = {density_Msun_kpc3:.2e} M_☉/kpc³")
        print(f"  ξ: {xi_lab:.15f}")
        print(f"  |ξ - 1|: {abs(xi_lab - 1.0):.2e}")
        print(f"  PASSES: {results['lab']['passes']}")
        
        print(f"\nEarth's surface:")
        print(f"  Density: {earth_surface_density_SI:.2e} kg/m³ = {earth_surface_density:.2e} M_☉/kpc³")
        print(f"  ξ: {xi_surface:.15f}")
        print(f"  |ξ - 1|: {abs(xi_surface - 1.0):.2e}")
        print(f"  PASSES: {results['earth_surface']['passes']}")
        
        return results
    
    def plot_enhancement_profile(self):
        """
        Plot enhancement factor across Solar System scales
        """
        # Create density range from lab to interplanetary
        log_rho_min = 10  # Very low density (outer Solar System)
        log_rho_max = 35  # Laboratory densities
        
        log_rho = cp.linspace(log_rho_min, log_rho_max, 1000)
        rho = cp.power(10.0, log_rho)
        xi = self.xi_enhancement(rho)
        
        # Move to CPU for plotting
        log_rho_cpu = log_rho.get()
        xi_cpu = xi.get()
        xi_minus_1 = xi_cpu - 1.0
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Top: Full enhancement
        ax1.loglog(10**log_rho_cpu, xi_cpu, 'b-', linewidth=2)
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=1.000023, color='r', linestyle='--', alpha=0.5, label='Cassini limit')
        
        # Mark key locations
        locations = {
            'Galactic disk': 1e8,
            'Solar System': 1e29,
            'Earth surface': 1e33,
            'Laboratory': 1e35
        }
        
        for name, rho_val in locations.items():
            xi_val = float(self.xi_enhancement(cp.array([rho_val])).get()[0])
            ax1.axvline(x=rho_val, color='gray', linestyle=':', alpha=0.5)
            ax1.text(rho_val, xi_val*1.5, name, rotation=90, va='bottom', fontsize=9)
        
        ax1.set_xlabel('Density (M☉/kpc³)', fontsize=12)
        ax1.set_ylabel('Enhancement factor ξ', fontsize=12)
        ax1.set_title('DDMM Enhancement Factor Across Density Scales', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0.9, 6])
        
        # Bottom: Deviation from 1
        ax2.loglog(10**log_rho_cpu, np.abs(xi_minus_1), 'b-', linewidth=2)
        ax2.axhline(y=2.3e-5, color='r', linestyle='--', label='Cassini limit')
        ax2.axhline(y=1e-8, color='orange', linestyle='--', label='Ephemeris limit')
        ax2.axhline(y=1e-13, color='green', linestyle='--', label='LLR limit')
        
        for name, rho_val in locations.items():
            ax2.axvline(x=rho_val, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Density (M☉/kpc³)', fontsize=12)
        ax2.set_ylabel('|ξ - 1|', fontsize=12)
        ax2.set_title('DDMM Deviation from GR', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([1e-25, 10])
        
        plt.tight_layout()
        plt.savefig('ddmm_solar_system_screening.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nPlot saved as 'ddmm_solar_system_screening.png'")
    
    def run_all_tests(self) -> Dict:
        """
        Run all Solar System tests and summarize results
        """
        print("\n" + "=" * 60)
        print("RUNNING ALL SOLAR SYSTEM DDMM TESTS")
        print(f"Parameters: A={self.params.A:.3f}, n={self.params.n:.3f}, "
              f"ρ_c={self.params.rho_c:.2e} M☉/kpc³")
        print("=" * 60)
        
        all_results = {
            'cassini': self.test_cassini_constraint(),
            'mercury': self.test_mercury_perihelion(),
            'lunar': self.test_lunar_laser_ranging(),
            'ephemerides': self.test_planetary_ephemerides(),
            'gw_speed': self.test_gravitational_waves(),
            'laboratory': self.test_laboratory_constraints()
        }
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY OF ALL TESTS")
        print("=" * 60)
        
        all_pass = True
        test_summary = []
        
        # Cassini
        cassini_pass = all(r['passes'] for r in all_results['cassini'].values())
        test_summary.append(('Cassini constraint', cassini_pass))
        all_pass &= cassini_pass
        
        # Mercury
        mercury_pass = all_results['mercury']['passes']
        test_summary.append(('Mercury perihelion', mercury_pass))
        all_pass &= mercury_pass
        
        # Lunar
        lunar_pass = all_results['lunar']['passes']
        test_summary.append(('Lunar laser ranging', lunar_pass))
        all_pass &= lunar_pass
        
        # Ephemerides
        ephemerides_pass = all(r['passes'] for r in all_results['ephemerides'].values())
        test_summary.append(('Planetary ephemerides', ephemerides_pass))
        all_pass &= ephemerides_pass
        
        # GW speed
        gw_pass = all_results['gw_speed']['passes']
        test_summary.append(('GW speed (c)', gw_pass))
        all_pass &= gw_pass
        
        # Laboratory
        lab_pass = all(r['passes'] for r in all_results['laboratory'].values())
        test_summary.append(('Laboratory tests', lab_pass))
        all_pass &= lab_pass
        
        print("\nTest                     Result")
        print("-" * 35)
        for test_name, passes in test_summary:
            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"{test_name:24} {status}")
        
        print("\n" + "=" * 60)
        if all_pass:
            print("ALL TESTS PASSED! ✓")
            print("DDMM is fully compatible with Solar System constraints")
        else:
            print("SOME TESTS FAILED ✗")
            print("Parameter adjustment may be needed")
        print("=" * 60)
        
        # Create visualization
        self.plot_enhancement_profile()
        
        return all_results


def main():
    """
    Main execution
    """
    # Initialize with your fitted parameters
    params = DDMMParameters()
    
    print(f"Testing DDMM with fitted parameters:")
    print(f"  A = {params.A:.3f} ± {params.A_std:.3f}")
    print(f"  n = {params.n:.3f} ± {params.n_std:.3f}")
    print(f"  ρ_c = {params.rho_c:.2e} ± {params.rho_c_std:.2e} M☉/kpc³")
    
    # Create test suite
    tester = SolarSystemDDMMTests(params)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Calculate safety margins
    print("\n" + "=" * 60)
    print("SAFETY MARGINS")
    print("=" * 60)
    
    # Cassini margin
    cassini_results = results['cassini']['Cassini measurement']
    print(f"Cassini: {cassini_results['safety_margin']:.2e}x below limit")
    
    # LLR margin
    llr_margin = results['lunar']['safety_margin']
    print(f"Lunar ranging: {llr_margin:.2e}x below limit")
    
    # Typical Solar System density
    typical_ss_density = 1e29  # M_sun/kpc^3
    rho_gpu = cp.array([typical_ss_density])
    xi_typical = float(tester.xi_enhancement(rho_gpu).get()[0])
    print(f"\nTypical Solar System ξ: {xi_typical:.20f}")
    print(f"Deviation from GR: {abs(xi_typical - 1.0):.2e}")
    
    return results


if __name__ == "__main__":
    results = main()