#!/usr/bin/env python3
"""
validate_ddmm.py - Comprehensive validation suite for the Density-Dependent Metric Model (DDMM)

This script implements rigorous tests across multiple scales to validate DDMM against
observational data and theoretical constraints. It tests:
1. Solar System precision tests
2. Galaxy rotation curves (SPARC dataset)
3. Gravitational lensing
4. CMB predictions
5. Large-scale structure
6. Type Ia supernovae
7. Laboratory constraints
8. Model self-consistency

Author: DDMM Validation Team
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import h5py

# Import your existing modules
try:
    from density_metric2 import (
        v_baryon_total_newtonian_kms, 
        rho_baryon_total_midplane_solar_kpc3,
        XI_FUNCTION_MAP,
        G_ASTRO_UNITS,
        R_SUN_KPC
    )
    from data_io import load_gaia
    import emcee
    from dynesty import utils as dyfunc
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all DDMM modules are in the Python path")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Physical constants
C_KMS = 299792.458  # Speed of light in km/s
H0 = 70.0  # Hubble constant in km/s/Mpc
OMEGA_M = 0.3  # Matter density parameter
OMEGA_LAMBDA = 0.7  # Dark energy density parameter
A0_MOND = 1.2e-10  # MOND acceleration scale in m/s^2
A0_MOND_KPC_S2 = A0_MOND * 1e-3 * (3.156e7)**2 / 3.086e19  # Convert to kpc/s^2

# Test result structure
@dataclass
class TestResult:
    """Structured test result with metadata"""
    test_name: str
    passed: bool
    score: float  # 0-1, where 1 is perfect
    details: Dict[str, Any]
    recommendations: List[str]

class DDMMValidator:
    """Main validation class for DDMM tests"""
    
    def __init__(self, model_params: Dict[str, float], output_dir: str = "validation_results"):
        """
        Initialize validator with DDMM model parameters.
        
        Parameters
        ----------
        model_params : dict
            Must contain at minimum:
            - rho_c_solar_kpc3: Critical density
            - n_exp: Power law exponent
            - Component masses and scale parameters
        output_dir : str
            Directory for saving results
        """
        self.model_params = model_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Set xi function type
        self.xi_func = XI_FUNCTION_MAP.get(
            model_params.get('xi_type', 'power'), 
            XI_FUNCTION_MAP['power']
        )
        
        logger.info(f"Initialized DDMM Validator with parameters:")
        logger.info(f"  ρ_c = {model_params.get('rho_c_solar_kpc3', 0):.2e} M☉/kpc³")
        logger.info(f"  n = {model_params.get('n_exp', 0):.2f}")
        
    # ========================================================================
    # TEST 1: Solar System Consistency
    # ========================================================================
    
    def test_solar_system(self) -> TestResult:
        """Test DDMM predictions in Solar System regime"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: SOLAR SYSTEM CONSISTENCY")
        logger.info("="*60)
        
        test_details = {}
        recommendations = []
        all_passed = True
        
        # 1A. Mercury perihelion precession
        logger.info("\n1A. Mercury Perihelion Precession Test")
        mercury_result = self._test_mercury_precession()
        test_details['mercury_precession'] = mercury_result
        if not mercury_result['passed']:
            all_passed = False
            recommendations.append("Adjust ρ_c to ensure ξ ≈ 1 at Solar System densities")
        
        # 1B. Cassini time delay
        logger.info("\n1B. Cassini Time Delay Test")
        cassini_result = self._test_cassini_delay()
        test_details['cassini_delay'] = cassini_result
        if not cassini_result['passed']:
            all_passed = False
            recommendations.append("Check light deflection calculations with modified metric")
        
        # 1C. Lunar laser ranging
        logger.info("\n1C. Lunar Laser Ranging Test")
        llr_result = self._test_lunar_ranging()
        test_details['lunar_ranging'] = llr_result
        if not llr_result['passed']:
            all_passed = False
            recommendations.append("Verify Earth-Moon dynamics preservation")
        
        # Calculate overall score
        score = sum([
            mercury_result['score'] * 0.4,
            cassini_result['score'] * 0.3,
            llr_result['score'] * 0.3
        ])
        
        result = TestResult(
            test_name="Solar System Consistency",
            passed=all_passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_mercury_precession(self) -> Dict[str, Any]:
        """Test Mercury perihelion advance"""
        # Solar System typical density ~ 10^-7 M☉/pc³ = 10^2 M☉/kpc³
        rho_solar_system = 1e2  # M☉/kpc³
        
        xi_ss = self.xi_func(
            rho_solar_system, 
            self.model_params['rho_c_solar_kpc3'],
            self.model_params['n_exp']
        )[0]
        
        # GR predicts 43 arcsec/century excess
        gr_prediction = 43.0  # arcsec/century
        ddmm_factor = xi_ss  # Simplified: actual calculation would be more complex
        ddmm_prediction = gr_prediction * ddmm_factor
        
        deviation = abs(ddmm_prediction - gr_prediction) / gr_prediction
        passed = deviation < 1e-6  # Must match to 1 ppm
        
        logger.info(f"  Solar System density: {rho_solar_system:.2e} M☉/kpc³")
        logger.info(f"  ξ(ρ_SS) = {xi_ss:.8f}")
        logger.info(f"  GR prediction: {gr_prediction:.2f} arcsec/century")
        logger.info(f"  DDMM prediction: {ddmm_prediction:.2f} arcsec/century")
        logger.info(f"  Deviation: {deviation:.2e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 1e6, 1.0),  # Score based on ppm deviation
            'xi_solar_system': xi_ss,
            'deviation_ppm': deviation * 1e6
        }
    
    def _test_cassini_delay(self) -> Dict[str, Any]:
        """Test Shapiro delay consistency"""
        # Simplified test: check that ξ ≈ 1 along Sun-Earth-Saturn path
        distances = np.linspace(0.1, 10, 100)  # AU, converted to kpc
        au_to_kpc = 4.84814e-9
        r_kpc = distances * au_to_kpc
        
        # Very rough density model for Solar System
        rho_path = 1e3 * np.exp(-r_kpc / (1 * au_to_kpc))  # M☉/kpc³
        
        xi_path = self.xi_func(
            rho_path,
            self.model_params['rho_c_solar_kpc3'],
            self.model_params['n_exp']
        )
        
        xi_deviation = np.max(np.abs(xi_path - 1.0))
        passed = xi_deviation < 2e-7  # Cassini constraint
        
        logger.info(f"  Max ξ deviation along path: {xi_deviation:.2e}")
        logger.info(f"  Cassini constraint: < 2×10⁻⁷")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(xi_deviation / 2e-7, 1.0),
            'max_xi_deviation': xi_deviation
        }
    
    def _test_lunar_ranging(self) -> Dict[str, Any]:
        """Test Earth-Moon system dynamics"""
        # Earth-Moon densities
        rho_earth_orbit = 5e2  # M☉/kpc³ (rough estimate)
        
        xi_em = self.xi_func(
            rho_earth_orbit,
            self.model_params['rho_c_solar_kpc3'],
            self.model_params['n_exp']
        )[0]
        
        # Nordtvedt effect constraint
        eta_limit = 1e-4
        eta_ddmm = abs(1 - xi_em)  # Simplified
        passed = eta_ddmm < eta_limit
        
        logger.info(f"  Earth-Moon ξ = {xi_em:.8f}")
        logger.info(f"  Nordtvedt parameter η = {eta_ddmm:.2e}")
        logger.info(f"  Limit: < {eta_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(eta_ddmm / eta_limit, 1.0),
            'nordtvedt_eta': eta_ddmm
        }
    
    # ========================================================================
    # TEST 2: Universal Galaxy Rotation Curves (SPARC)
    # ========================================================================
    
    def test_sparc_galaxies(self, sparc_data_path: str, n_galaxies: int = 20) -> TestResult:
        """Test on real SPARC galaxy dataset"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: SPARC GALAXY ROTATION CURVES")
        logger.info("="*60)
        
        # Load SPARC data
        loader = SPARCDataLoader(sparc_data_path)
        loader.load_all_galaxies()
        
        # Select diverse sample
        galaxies = loader.get_galaxy_sample(n_galaxies)
        logger.info(f"Testing on {len(galaxies)} SPARC galaxies")
        
        test_details = {
            'n_galaxies': len(galaxies),
            'individual_results': {},
            'statistics': {}
        }
        
        rms_values = []
        chi2_values = []
        successful_fits = 0
        
        for galaxy in galaxies:
            logger.info(f"\nTesting galaxy: {galaxy['name']}")
            
            try:
                # Fit with DDMM
                fit_result = self._fit_sparc_galaxy(galaxy)
                test_details['individual_results'][galaxy['name']] = fit_result
                
                if fit_result['converged']:
                    rms_values.append(fit_result['rms'])
                    chi2_values.append(fit_result['chi2_reduced'])
                    successful_fits += 1
                    
            except Exception as e:
                logger.error(f"Failed to fit {galaxy['name']}: {e}")
                test_details['individual_results'][galaxy['name']] = {
                    'converged': False,
                    'error': str(e)
                }
        
        # Statistics
        if successful_fits > 0:
            mean_rms = np.mean(rms_values)
            median_rms = np.median(rms_values)
            fraction_good = sum(1 for r in rms_values if r < 10) / successful_fits
        else:
            mean_rms = median_rms = fraction_good = 0
        
        test_details['statistics'] = {
            'successful_fits': successful_fits,
            'mean_rms': mean_rms,
            'median_rms': median_rms,
            'fraction_rms_below_10': fraction_good
        }
        
        # Pass criteria: median RMS < 12 km/s (competitive with MOND)
        passed = median_rms < 12 and successful_fits > n_galaxies * 0.8
        score = np.exp(-median_rms / 10) * (successful_fits / n_galaxies)
        
        return TestResult(
            test_name="SPARC Galaxy Fitting",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=self._get_sparc_recommendations(test_details)
        )
    
    def _fit_sparc_galaxy(self, galaxy: Dict) -> Dict:
        """Fit single SPARC galaxy with DDMM"""
        from scipy.optimize import minimize
        
        r = galaxy['r_kpc']
        v_obs = galaxy['v_obs']
        v_err = galaxy['v_err']
        v_baryon = galaxy['v_baryon']
        
        # For DDMM, we need to estimate the density profile
        # Use surface brightness to estimate stellar mass-to-light ratio
        
        def objective(params):
            ml_star = params[0]  # Stellar M/L ratio
            
            # Scale baryonic components
            v_star_sq = (galaxy['v_disk']**2 + galaxy['v_bulge']**2) * ml_star
            v_gas_sq = galaxy['v_gas']**2
            v_newton_sq = v_star_sq + v_gas_sq
            v_newton = np.sqrt(np.maximum(v_newton_sq, 0))
            
            # Estimate density (simplified - would need proper integration)
            # Using exponential disk approximation
            sigma_star = galaxy['sb_disk'] * ml_star  # Surface density
            h_z = 0.3  # Assumed scale height in kpc
            rho = sigma_star / (2 * h_z) * np.exp(-r / 3.0)  # Rough approximation
            
            # Apply DDMM
            xi = self.xi_func(rho, self.model_params['rho_c_solar_kpc3'],
                            self.model_params['n_exp'])
            v_model = v_newton * np.sqrt(xi)
            
            # Chi-square
            chi2 = np.sum(((v_obs - v_model) / v_err)**2)
            return chi2
        
        # Fit M/L ratio
        result = minimize(objective, x0=[0.5], bounds=[(0.1, 5.0)])
        
        # Calculate final fit
        ml_star_best = result.x[0]
        v_model_best = self._calculate_ddmm_curve(galaxy, ml_star_best)
        
        residuals = v_obs - v_model_best
        rms = np.sqrt(np.mean(residuals**2))
        chi2_reduced = result.fun / (len(r) - 1)
        
        return {
            'converged': result.success,
            'ml_star': ml_star_best,
            'rms': rms,
            'chi2_reduced': chi2_reduced,
            'v_model': v_model_best
        }
    
    def _generate_mock_sparc_data(self, n_galaxies: int) -> List[Dict]:
        """Generate mock galaxy data for testing"""
        np.random.seed(42)
        galaxies = []
        
        for i in range(n_galaxies):
            # Generate realistic galaxy parameters
            M_star = 10**(np.random.uniform(9, 11))  # M☉
            M_gas = 10**(np.random.uniform(8.5, 10.5))  # M☉
            R_d = np.random.uniform(1, 5)  # kpc
            
            # Generate rotation curve
            r = np.logspace(-0.5, 1.5, 30)  # kpc
            
            # True rotation curve (with some "dark matter")
            v_star = np.sqrt(G_ASTRO_UNITS * M_star * r / (r + R_d)**2)
            v_gas = np.sqrt(G_ASTRO_UNITS * M_gas * r / (r + 2*R_d)**2)
            v_dm = 150 * np.sqrt(r / (r + 10))  # Mock dark matter
            v_true = np.sqrt(v_star**2 + v_gas**2 + v_dm**2)
            
            # Add noise
            v_obs = v_true + np.random.normal(0, 5, len(r))
            v_err = np.full_like(v_obs, 5.0)
            
            galaxies.append({
                'name': f'MockGalaxy_{i+1}',
                'r_kpc': r,
                'v_obs': v_obs,
                'v_err': v_err,
                'M_star': M_star,
                'M_gas': M_gas,
                'R_d': R_d
            })
        
        return galaxies
    
    def _fit_galaxy_rotation_curve(self, galaxy: Dict) -> Dict[str, Any]:
        """Fit single galaxy with DDMM"""
        r = galaxy['r_kpc']
        v_obs = galaxy['v_obs']
        v_err = galaxy['v_err']
        
        # Simple model: single disk approximation
        M_baryon = galaxy['M_star'] + galaxy['M_gas']
        R_d = galaxy['R_d']
        
        # Calculate Newtonian prediction
        params = {
            'M_disk_thin_solar': M_baryon,
            'R_d_thin_kpc': R_d,
            'h_z_thin_kpc': 0.3,
            'include_disk_thin': True,
            'include_bulge': False,
            'include_disk_thick': False,
            'include_gas': False
        }
        
        v_newton = v_baryon_total_newtonian_kms(r, params)
        
        # Calculate density
        rho = rho_baryon_total_midplane_solar_kpc3(r, params)
        
        # Apply DDMM modification
        xi = self.xi_func(rho, self.model_params['rho_c_solar_kpc3'], 
                         self.model_params['n_exp'])
        v_model = v_newton * np.sqrt(xi)
        
        # Calculate fit quality
        residuals = v_obs - v_model
        rms = np.sqrt(np.mean(residuals**2))
        chi2 = np.sum((residuals / v_err)**2)
        chi2_reduced = chi2 / (len(r) - 2)  # 2 free parameters
        
        return {
            'rms': rms,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'v_model': v_model,
            'v_newton': v_newton,
            'xi': xi
        }
    
    # ========================================================================
    # TEST 3: Gravitational Lensing (Bullet Cluster)
    # ========================================================================
    
    def test_gravitational_lensing(self) -> TestResult:
        """Test lensing predictions"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: GRAVITATIONAL LENSING")
        logger.info("="*60)
        
        test_details = {}
        
        # Test simplified Bullet Cluster scenario
        logger.info("\n3A. Bullet Cluster Mass Reconstruction")
        bullet_result = self._test_bullet_cluster_lensing()
        test_details['bullet_cluster'] = bullet_result
        
        # Test galaxy-galaxy lensing
        logger.info("\n3B. Galaxy-Galaxy Lensing")
        ggl_result = self._test_galaxy_galaxy_lensing()
        test_details['galaxy_galaxy'] = ggl_result
        
        passed = bullet_result['passed'] and ggl_result['passed']
        score = (bullet_result['score'] + ggl_result['score']) / 2
        
        recommendations = []
        if not bullet_result['passed']:
            recommendations.append("DDMM may need cluster-scale modifications")
        if not ggl_result['passed']:
            recommendations.append("Check lensing efficiency parameter")
        
        result = TestResult(
            test_name="Gravitational Lensing",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_bullet_cluster_lensing(self) -> Dict[str, Any]:
        """Test Bullet Cluster lensing map reconstruction"""
        # Simplified 1D model along collision axis
        x = np.linspace(-1000, 1000, 200)  # kpc from center
        
        # Gas distribution (X-ray observed)
        M_gas1 = 1e14  # M☉
        M_gas2 = 0.5e14
        sigma_gas = 200  # kpc
        rho_gas = (M_gas1 * np.exp(-(x + 300)**2 / (2*sigma_gas**2)) + 
                  M_gas2 * np.exp(-(x - 400)**2 / (2*sigma_gas**2))) / (np.sqrt(2*np.pi) * sigma_gas)
        
        # Galaxies (offset from gas)
        M_gal1 = 2e14
        M_gal2 = 1e14
        sigma_gal = 150
        rho_gal = (M_gal1 * np.exp(-(x + 200)**2 / (2*sigma_gal**2)) + 
                  M_gal2 * np.exp(-(x - 300)**2 / (2*sigma_gal**2))) / (np.sqrt(2*np.pi) * sigma_gal)
        
        rho_total = rho_gas + rho_gal
        
        # Calculate lensing convergence with DDMM
        xi = self.xi_func(rho_total, self.model_params['rho_c_solar_kpc3'],
                         self.model_params['n_exp'])
        
        # Effective lensing mass
        M_lens_eff = np.trapz(rho_total * xi, x)
        M_true = np.trapz(rho_total, x)
        
        # Compare to observed offset
        mass_peak_obs = x[np.argmax(rho_gal)]
        mass_peak_lens = x[np.argmax(rho_total * xi)]
        offset = abs(mass_peak_lens - mass_peak_obs)
        
        # Criteria: lensing peak should follow galaxies, not gas
        passed = offset < 50  # kpc
        score = np.exp(-offset / 100)
        
        logger.info(f"  True mass: {M_true:.2e} M☉")
        logger.info(f"  Effective lensing mass: {M_lens_eff:.2e} M☉")
        logger.info(f"  Peak offset: {offset:.1f} kpc")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        # Generate plot
        self._plot_bullet_cluster(x, rho_gas, rho_gal, xi)
        
        return {
            'passed': passed,
            'score': score,
            'mass_ratio': M_lens_eff / M_true,
            'peak_offset_kpc': offset
        }
    
    def _test_galaxy_galaxy_lensing(self) -> Dict[str, Any]:
        """Test galaxy-galaxy lensing signal"""
        # Simplified: check if DDMM produces correct Einstein radius
        M_lens = 1e12  # M☉
        z_lens = 0.3
        z_source = 1.0
        
        # NFW-like density profile
        r = np.logspace(-1, 3, 100)  # kpc
        r_s = 20  # kpc
        rho_s = 1e7  # M☉/kpc³
        rho_nfw = rho_s / ((r/r_s) * (1 + r/r_s)**2)
        
        # Apply DDMM
        xi = self.xi_func(rho_nfw, self.model_params['rho_c_solar_kpc3'],
                         self.model_params['n_exp'])
        
        # Calculate projected mass within Einstein radius (simplified)
        R_ein_obs = 1.5  # arcsec at z_lens
        kpc_per_arcsec = 4.5  # at z_lens
        R_ein_kpc = R_ein_obs * kpc_per_arcsec
        
        # Integrate mass within R_ein
        mask = r < R_ein_kpc
        M_ein_ddmm = 4 * np.pi * np.trapz(rho_nfw[mask] * xi[mask] * r[mask]**2, r[mask])
        M_ein_true = 4 * np.pi * np.trapz(rho_nfw[mask] * r[mask]**2, r[mask])
        
        # Expected from lensing
        M_ein_expected = 1e11  # M☉ (typical)
        
        deviation = abs(M_ein_ddmm - M_ein_expected) / M_ein_expected
        passed = deviation < 0.3  # 30% tolerance
        
        logger.info(f"  Einstein radius: {R_ein_obs:.1f} arcsec ({R_ein_kpc:.1f} kpc)")
        logger.info(f"  M(<R_Ein) expected: {M_ein_expected:.2e} M☉")
        logger.info(f"  M(<R_Ein) DDMM: {M_ein_ddmm:.2e} M☉")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation, 1.0),
            'mass_deviation': deviation
        }
    
    # ========================================================================
    # TEST 4: CMB Power Spectrum
    # ========================================================================
    
    def test_cmb_predictions(self) -> TestResult:
        """Test CMB acoustic peak predictions"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: CMB POWER SPECTRUM")
        logger.info("="*60)
        
        # Simplified: check key physics at recombination
        test_details = {}
        
        # Sound horizon
        logger.info("\n4A. Sound Horizon at Recombination")
        sound_horizon_result = self._test_sound_horizon()
        test_details['sound_horizon'] = sound_horizon_result
        
        # Peak ratios
        logger.info("\n4B. Acoustic Peak Ratios")
        peak_ratio_result = self._test_peak_ratios()
        test_details['peak_ratios'] = peak_ratio_result
        
        passed = sound_horizon_result['passed'] and peak_ratio_result['passed']
        score = (sound_horizon_result['score'] + peak_ratio_result['score']) / 2
        
        recommendations = []
        if not passed:
            recommendations.append("DDMM needs careful early universe implementation")
            recommendations.append("Consider running full Boltzmann code (CLASS/CAMB)")
        
        result = TestResult(
            test_name="CMB Predictions",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_sound_horizon(self) -> Dict[str, Any]:
        """Test sound horizon scale"""
        # At recombination: z ~ 1100
        z_rec = 1100
        
        # Baryon density at recombination
        rho_b_rec = 3e4  # M☉/kpc³ (rough estimate)
        
        xi_rec = self.xi_func(rho_b_rec, self.model_params['rho_c_solar_kpc3'],
                             self.model_params['n_exp'])[0]
        
        # Sound horizon scaling
        r_s_standard = 147  # Mpc (standard cosmology)
        r_s_ddmm = r_s_standard * np.sqrt(xi_rec)  # Simplified
        
        deviation = abs(r_s_ddmm - r_s_standard) / r_s_standard
        passed = deviation < 0.01  # 1% tolerance
        
        logger.info(f"  ρ_b at recombination: {rho_b_rec:.2e} M☉/kpc³")
        logger.info(f"  ξ(z=1100) = {xi_rec:.6f}")
        logger.info(f"  Standard r_s: {r_s_standard:.1f} Mpc")
        logger.info(f"  DDMM r_s: {r_s_ddmm:.1f} Mpc")
        logger.info(f"  Deviation: {deviation*100:.2f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 100, 1.0),
            'xi_recombination': xi_rec,
            'sound_horizon_deviation': deviation
        }
    
    def _test_peak_ratios(self) -> Dict[str, Any]:
        """Test acoustic peak height ratios"""
        # Simplified: check if baryon-photon coupling is preserved
        
        # First/second peak ratio depends on baryon density
        ratio_observed = 0.75  # Approximate
        
        # With DDMM, effective baryon density changes
        rho_cmb = 5e4  # M☉/kpc³
        xi_cmb = self.xi_func(rho_cmb, self.model_params['rho_c_solar_kpc3'],
                             self.model_params['n_exp'])[0]
        
        # Rough scaling
        ratio_ddmm = ratio_observed * (1 + 0.1 * (1 - xi_cmb))
        
        deviation = abs(ratio_ddmm - ratio_observed) / ratio_observed
        passed = deviation < 0.05
        
        logger.info(f"  Observed peak ratio: {ratio_observed:.3f}")
        logger.info(f"  DDMM peak ratio: {ratio_ddmm:.3f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 20, 1.0),
            'peak_ratio_deviation': deviation
        }
    
    # ========================================================================
    # TEST 5: Large Scale Structure
    # ========================================================================
    
    def test_large_scale_structure(self) -> TestResult:
        """Test structure formation and growth"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: LARGE SCALE STRUCTURE")
        logger.info("="*60)
        
        test_details = {}
        
        # Growth rate
        logger.info("\n5A. Growth Rate f(z)")
        growth_result = self._test_growth_rate()
        test_details['growth_rate'] = growth_result
        
        # BAO scale
        logger.info("\n5B. BAO Scale")
        bao_result = self._test_bao_scale()
        test_details['bao'] = bao_result
        
        passed = growth_result['passed'] and bao_result['passed']
        score = (growth_result['score'] + bao_result['score']) / 2
        
        recommendations = []
        if not growth_result['passed']:
            recommendations.append("Structure growth may be suppressed/enhanced")
        if not bao_result['passed']:
            recommendations.append("BAO scale shift indicates early universe issues")
        
        result = TestResult(
            test_name="Large Scale Structure",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_growth_rate(self) -> Dict[str, Any]:
        """Test linear growth rate"""
        # Solve growth equation with DDMM
        z_array = np.array([0, 0.5, 1.0, 2.0])
        
        # Standard f(z) ≈ Ω_m(z)^0.55
        f_standard = OMEGA_M * (1 + z_array)**3 / (OMEGA_M * (1 + z_array)**3 + OMEGA_LAMBDA)**0.55
        
        # DDMM modification (simplified)
        f_ddmm = []
        for z in z_array:
            # Average density at redshift z
            rho_z = 1e6 * (1 + z)**3  # M☉/kpc³
            xi_z = self.xi_func(rho_z, self.model_params['rho_c_solar_kpc3'],
                               self.model_params['n_exp'])[0]
            f_ddmm.append(f_standard[list(z_array).index(z)] * np.sqrt(xi_z))
        
        f_ddmm = np.array(f_ddmm)
        
        # RSD constraint at z=0
        f_obs_z0 = 0.45  # Observed
        deviation = abs(f_ddmm[0] - f_obs_z0) / f_obs_z0
        passed = deviation < 0.1
        
        logger.info("  Redshift | f_standard | f_DDMM | f_observed")
        logger.info("  ---------|------------|--------|------------")
        for i, z in enumerate(z_array):
            f_obs = f_obs_z0 if z == 0 else np.nan
            logger.info(f"  z={z:<7.1f} | {f_standard[i]:<10.3f} | {f_ddmm[i]:<6.3f} | "
                       f"{f_obs if not np.isnan(f_obs) else 'N/A'}")
        
        logger.info(f"\n  Deviation at z=0: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 10, 1.0),
            'growth_rate_z0': f_ddmm[0],
            'deviation': deviation
        }
    
    def _test_bao_scale(self) -> Dict[str, Any]:
        """Test BAO scale preservation"""
        # BAO scale should match sound horizon
        r_bao_observed = 147.5  # Mpc
        
        # Average density at z~0.5 (typical BAO measurement)
        rho_bao = 2e5  # M☉/kpc³
        xi_bao = self.xi_func(rho_bao, self.model_params['rho_c_solar_kpc3'],
                             self.model_params['n_exp'])[0]
        
        # Scaling
        r_bao_ddmm = r_bao_observed * np.sqrt(xi_bao)
        
        deviation = abs(r_bao_ddmm - r_bao_observed) / r_bao_observed
        passed = deviation < 0.02
        
        logger.info(f"  Observed BAO scale: {r_bao_observed:.1f} Mpc")
        logger.info(f"  DDMM BAO scale: {r_bao_ddmm:.1f} Mpc")
        logger.info(f"  ξ at BAO epoch: {xi_bao:.4f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 50, 1.0),
            'bao_scale_ddmm': r_bao_ddmm,
            'deviation': deviation
        }
        
    def test_large_scale_structure_with_sdss_bao(self, bao_dir: str = 'bao') -> TestResult:
        """Test structure formation using real SDSS BAO data"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: LARGE SCALE STRUCTURE (SDSS BAO)")
        logger.info("="*60)
        
        # Load BAO measurements
        from bao_data_loader import BAODataLoader
        loader = BAODataLoader(bao_dir)
        measurements = loader.load_all_measurements()
        distances = loader.get_distance_measurements()
        
        test_details = {
            'n_measurements': len(distances),
            'chi2_total': 0,
            'individual_tests': {}
        }
        
        # Sound horizon at drag epoch (rd)
        # In ΛCDM: rd ≈ 147.5 Mpc
        rd_fiducial = 147.5  # Mpc
        
        # For DDMM, rd might be modified
        # Estimate average ξ at drag epoch (z ~ 1060)
        z_drag = 1060
        rho_drag = 1e8 * (1 + z_drag)**3  # Rough estimate
        xi_drag = self.xi_func(rho_drag, self.model_params['rho_c_solar_kpc3'],
                            self.model_params['n_exp'])[0]
        
        rd_ddmm = rd_fiducial * np.sqrt(xi_drag)  # Simplified scaling
        
        logger.info(f"\nSound horizon analysis:")
        logger.info(f"  Fiducial rd = {rd_fiducial:.1f} Mpc")
        logger.info(f"  ξ at drag = {xi_drag:.4f}")
        logger.info(f"  DDMM rd = {rd_ddmm:.1f} Mpc")
        
        # Test each measurement
        chi2_total = 0
        n_tests = 0
        
        for meas in distances:
            logger.info(f"\nTesting {meas['name']} at z={meas['z']:.2f}")
            
            # Calculate theoretical prediction with DDMM
            z = meas['z']
            
            # Average density at this redshift
            rho_z = 1e6 * (1 + z)**3  # M☉/kpc³
            xi_z = self.xi_func(rho_z, self.model_params['rho_c_solar_kpc3'],
                            self.model_params['n_exp'])[0]
            
            # Distance predictions (simplified - should integrate)
            if 'DV_over_rd' in meas:
                # DV is volume-averaged distance
                # In ΛCDM: DV(z) = [cz(1+z)²DA²(z)/H(z)]^(1/3)
                DV_over_rd_lcdm = meas['DV_over_rd']  # Use observed as baseline
                DV_over_rd_ddmm = DV_over_rd_lcdm * (rd_fiducial / rd_ddmm) * np.sqrt(xi_z)
                
                chi2 = ((meas['DV_over_rd'] - DV_over_rd_ddmm) / meas['DV_over_rd_err'])**2
                chi2_total += chi2
                n_tests += 1
                
                logger.info(f"  DV/rd observed: {meas['DV_over_rd']:.2f} ± {meas['DV_over_rd_err']:.2f}")
                logger.info(f"  DV/rd DDMM: {DV_over_rd_ddmm:.2f}")
                logger.info(f"  χ² contribution: {chi2:.2f}")
            
            # Growth rate test
            if 'fs8' in meas:
                # f(z) = d ln D / d ln a, where D is growth factor
                # In DDMM, growth is modified by ξ
                fs8_obs = meas['fs8']
                fs8_ddmm = fs8_obs * np.sqrt(xi_z)  # Simplified
                
                chi2_growth = ((fs8_obs - fs8_ddmm) / meas['fs8_err'])**2
                chi2_total += chi2_growth
                n_tests += 1
                
                logger.info(f"  fs8 observed: {fs8_obs:.3f} ± {meas['fs8_err']:.3f}")
                logger.info(f"  fs8 DDMM: {fs8_ddmm:.3f}")
                logger.info(f"  χ² contribution: {chi2_growth:.2f}")
            
            test_details['individual_tests'][meas['name']] = {
                'z': z,
                'xi': xi_z,
                'chi2': chi2 if 'DV_over_rd' in meas else chi2_growth
            }
        
        # Overall assessment
        chi2_reduced = chi2_total / max(1, n_tests)
        passed = chi2_reduced < 3.0  # Reasonable fit
        score = np.exp(-chi2_reduced / 2)
        
        test_details['chi2_total'] = chi2_total
        test_details['chi2_reduced'] = chi2_reduced
        test_details['n_tests'] = n_tests
        
        logger.info(f"\nOverall BAO test results:")
        logger.info(f"  Total χ² = {chi2_total:.2f}")
        logger.info(f"  Reduced χ² = {chi2_reduced:.2f}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        recommendations = []
        if not passed:
            recommendations.append("DDMM modification of distances needs refinement")
            recommendations.append("Consider full numerical integration of modified Friedmann equations")
        
        return TestResult(
            test_name="Large Scale Structure (SDSS BAO)",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )        
    
    # ========================================================================
    # TEST 6: Type Ia Supernovae
    # ========================================================================
    
    def test_supernovae(self) -> TestResult:
        """Test distance-redshift relation"""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: TYPE IA SUPERNOVAE")
        logger.info("="*60)
        
        # Generate mock SN Ia data
        z_sn = np.logspace(-2, 0.3, 50)
        
        # Standard luminosity distance
        d_L_standard = self._luminosity_distance_lcdm(z_sn)
        
        # DDMM modification (simplified - needs full integration)
        d_L_ddmm = []
        for z in z_sn:
            # Average ξ along light path
            z_path = np.linspace(0, z, 100)
            rho_path = 1e6 * (1 + z_path)**3
            xi_path = self.xi_func(rho_path, self.model_params['rho_c_solar_kpc3'],
                                  self.model_params['n_exp'])
            xi_avg = np.mean(xi_path)
            
            # Modified distance (very simplified)
            d_L_ddmm.append(d_L_standard[list(z_sn).index(z)] / np.sqrt(xi_avg))
        
        d_L_ddmm = np.array(d_L_ddmm)
        
        # Distance modulus
        mu_standard = 5 * np.log10(d_L_standard) + 25
        mu_ddmm = 5 * np.log10(d_L_ddmm) + 25
        
        # Mock observational errors
        mu_err = 0.15 * np.ones_like(mu_standard)
        
        # Chi-square test
        chi2 = np.sum(((mu_ddmm - mu_standard) / mu_err)**2)
        chi2_reduced = chi2 / len(z_sn)
        
        passed = chi2_reduced < 2.0
        score = np.exp(-chi2_reduced)
        
        logger.info(f"  Number of SNe: {len(z_sn)}")
        logger.info(f"  χ²/dof = {chi2_reduced:.2f}")
        logger.info(f"  Max Δμ = {np.max(np.abs(mu_ddmm - mu_standard)):.2f} mag")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        # Plot Hubble diagram
        self._plot_hubble_diagram(z_sn, mu_standard, mu_ddmm, mu_err)
        
        test_details = {
            'chi2_reduced': chi2_reduced,
            'max_distance_modulus_diff': np.max(np.abs(mu_ddmm - mu_standard)),
            'n_supernovae': len(z_sn)
        }
        
        recommendations = []
        if not passed:
            recommendations.append("DDMM cosmological distance measure needs refinement")
        
        result = TestResult(
            test_name="Type Ia Supernovae",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _luminosity_distance_lcdm(self, z: np.ndarray) -> np.ndarray:
        """Calculate luminosity distance in ΛCDM"""
        c_over_H0 = C_KMS / H0 * 1e-3  # Mpc
        
        def integrand(zp):
            return 1 / np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_LAMBDA)
        
        d_L = []
        for zi in z:
            integral, _ = quad(integrand, 0, zi)
            d_L.append(c_over_H0 * (1 + zi) * integral)
        
        return np.array(d_L)
    
    # ========================================================================
    # TEST 7: Laboratory Tests
    # ========================================================================
    
    def test_laboratory_constraints(self) -> TestResult:
        """Test laboratory and space-based constraints"""
        logger.info("\n" + "="*60)
        logger.info("TEST 7: LABORATORY CONSTRAINTS")
        logger.info("="*60)
        
        test_details = {}
        
        # Eöt-Wash torsion balance
        logger.info("\n7A. Eöt-Wash Torsion Balance")
        eotwash_result = self._test_eotwash()
        test_details['eotwash'] = eotwash_result
        
        # MICROSCOPE
        logger.info("\n7B. MICROSCOPE Equivalence Principle")
        microscope_result = self._test_microscope()
        test_details['microscope'] = microscope_result
        
        passed = eotwash_result['passed'] and microscope_result['passed']
        score = (eotwash_result['score'] + microscope_result['score']) / 2
        
        recommendations = []
        if not passed:
            recommendations.append("Consider screening mechanisms for high-density environments")
        
        result = TestResult(
            test_name="Laboratory Constraints",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_eotwash(self) -> Dict[str, Any]:
        """Test Eöt-Wash constraints"""
        # Laboratory density ~ 10^3 g/cm³ ~ 10^24 M☉/kpc³
        rho_lab = 1e24  # M☉/kpc³
        
        xi_lab = self.xi_func(rho_lab, self.model_params['rho_c_solar_kpc3'],
                             self.model_params['n_exp'])[0]
        
        # Fifth force constraint
        alpha = abs(1 - xi_lab)
        alpha_limit = 1e-6
        
        passed = alpha < alpha_limit
        
        logger.info(f"  Lab density: {rho_lab:.2e} M☉/kpc³")
        logger.info(f"  ξ(ρ_lab) = {xi_lab:.12f}")
        logger.info(f"  Fifth force α = {alpha:.2e}")
        logger.info(f"  Limit: < {alpha_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(alpha / alpha_limit, 1.0),
            'fifth_force_alpha': alpha
        }
    
    def _test_microscope(self) -> Dict[str, Any]:
        """Test MICROSCOPE constraints"""
        # Satellite orbit density
        rho_orbit = 1e-10  # M☉/kpc³ (very low)
        
        xi_orbit = self.xi_func(rho_orbit, self.model_params['rho_c_solar_kpc3'],
                               self.model_params['n_exp'])[0]
        
        # Eötvös parameter
        eta = abs(1 - xi_orbit) * 0.01  # Material dependence factor
        eta_limit = 1e-15
        
        passed = eta < eta_limit
        
        logger.info(f"  Orbit density: {rho_orbit:.2e} M☉/kpc³")
        logger.info(f"  ξ(ρ_orbit) = {xi_orbit:.12f}")
        logger.info(f"  Eötvös η = {eta:.2e}")
        logger.info(f"  MICROSCOPE limit: < {eta_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(eta / eta_limit, 1.0),
            'eotvos_eta': eta
        }
    
    # ========================================================================
    # TEST 8: Self-Consistency
    # ========================================================================
    
    def test_self_consistency(self) -> TestResult:
        """Test internal consistency and invariance"""
        logger.info("\n" + "="*60)
        logger.info("TEST 8: SELF-CONSISTENCY & INVARIANCE")
        logger.info("="*60)
        
        test_details = {}
        
        # Conservation of xi integral
        logger.info("\n8A. Xi Conservation Test")
        conservation_result = self._test_xi_conservation()
        test_details['xi_conservation'] = conservation_result
        
        # Parameter invariance
        logger.info("\n8B. Effective Mass Invariance")
        invariance_result = self._test_mass_invariance()
        test_details['mass_invariance'] = invariance_result
        
        passed = conservation_result['passed'] and invariance_result['passed']
        score = (conservation_result['score'] + invariance_result['score']) / 2
        
        recommendations = []
        if not conservation_result['passed']:
            recommendations.append("Check xi normalization across models")
        if not invariance_result['passed']:
            recommendations.append("Effective mass principle may be violated")
        
        result = TestResult(
            test_name="Self-Consistency",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        self._log_result(result)
        return result
    
    def _test_xi_conservation(self) -> Dict[str, Any]:
        """Test xi integral conservation"""
        # Test that ∫ξ(ρ)ρ dV is conserved across different decompositions
        
        # Model 1: Single thick disk
        params1 = {
            'M_disk_thin_solar': 1.5e11,
            'R_d_thin_kpc': 4.0,
            'h_z_thin_kpc': 0.6,
            'include_disk_thin': True,
            'include_bulge': False,
            'include_disk_thick': False,
            'include_gas': False
        }
        
        # Model 2: Thin + thick disk (same total mass)
        params2 = {
            'M_disk_thin_solar': 1.0e11,
            'R_d_thin_kpc': 3.5,
            'h_z_thin_kpc': 0.3,
            'M_disk_thick_solar': 0.5e11,
            'R_d_thick_kpc': 5.0,
            'h_z_thick_kpc': 0.9,
            'include_disk_thin': True,
            'include_bulge': False,
            'include_disk_thick': True,
            'include_gas': False
        }
        
        # Calculate integral over relevant region
        r = np.linspace(0.1, 30, 100)
        
        rho1 = rho_baryon_total_midplane_solar_kpc3(r, params1)
        xi1 = self.xi_func(rho1, self.model_params['rho_c_solar_kpc3'],
                          self.model_params['n_exp'])
        integral1 = np.trapz(xi1 * rho1 * r, r)  # Cylindrical integral
        
        rho2 = rho_baryon_total_midplane_solar_kpc3(r, params2)
        xi2 = self.xi_func(rho2, self.model_params['rho_c_solar_kpc3'],
                          self.model_params['n_exp'])
        integral2 = np.trapz(xi2 * rho2 * r, r)
        
        ratio = integral2 / integral1
        deviation = abs(ratio - 1.0)
        passed = deviation < 0.1
        
        logger.info(f"  Model 1 integral: {integral1:.2e}")
        logger.info(f"  Model 2 integral: {integral2:.2e}")
        logger.info(f"  Ratio: {ratio:.3f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 10, 1.0),
            'integral_ratio': ratio
        }
    
    def _test_mass_invariance(self) -> Dict[str, Any]:
        """Test effective mass invariance principle"""
        # Test M_eff = M_baryon × <ξ> invariance
        
        r_test = np.linspace(5, 15, 50)  # Test range
        
        # Configuration 1
        M1 = 1.27e11
        params1 = {
            'M_disk_thin_solar': M1,
            'R_d_thin_kpc': 4.0,
            'h_z_thin_kpc': 0.6,
            'include_disk_thin': True,
            'include_bulge': False,
            'include_disk_thick': False,
            'include_gas': False
        }
        
        rho1 = rho_baryon_total_midplane_solar_kpc3(r_test, params1)
        xi1 = self.xi_func(rho1, self.model_params['rho_c_solar_kpc3'],
                          self.model_params['n_exp'])
        xi1_avg = np.mean(xi1)
        M_eff1 = M1 * xi1_avg
        
        # Configuration 2
        M2 = 1.67e11
        params2 = {
            'M_disk_thin_solar': 1.1e11,
            'R_d_thin_kpc': 3.5,
            'h_z_thin_kpc': 0.3,
            'M_disk_thick_solar': 0.57e11,
            'R_d_thick_kpc': 5.0,
            'h_z_thick_kpc': 0.9,
            'include_disk_thin': True,
            'include_bulge': False,
            'include_disk_thick': True,
            'include_gas': False
        }
        
        rho2 = rho_baryon_total_midplane_solar_kpc3(r_test, params2)
        xi2 = self.xi_func(rho2, self.model_params['rho_c_solar_kpc3'],
                          self.model_params['n_exp'])
        xi2_avg = np.mean(xi2)
        M_eff2 = M2 * xi2_avg
        
        ratio = M_eff2 / M_eff1
        deviation = abs(ratio - 1.0)
        passed = deviation < 0.05
        
        logger.info(f"  Config 1: M={M1:.2e}, <ξ>={xi1_avg:.3f}, M_eff={M_eff1:.2e}")
        logger.info(f"  Config 2: M={M2:.2e}, <ξ>={xi2_avg:.3f}, M_eff={M_eff2:.2e}")
        logger.info(f"  M_eff ratio: {ratio:.3f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': 1.0 - min(deviation * 20, 1.0),
            'M_eff_ratio': ratio
        }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _log_result(self, result: TestResult):
        """Log test result summary"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"\n{status} | {result.test_name} | Score: {result.score:.2f}")
        
        if result.recommendations:
            logger.info("Recommendations:")
            for rec in result.recommendations:
                logger.info(f"  - {rec}")
    
    def _plot_sparc_comparison(self, galaxies: List[Dict], test_details: Dict):
        """Plot SPARC galaxy fits"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(min(4, len(galaxies))):
            ax = axes[i]
            galaxy = galaxies[i]
            fit = test_details['individual_results'][galaxy['name']]
            
            ax.errorbar(galaxy['r_kpc'], galaxy['v_obs'], yerr=galaxy['v_err'],
                       fmt='o', alpha=0.5, label='Observed')
            ax.plot(galaxy['r_kpc'], fit['v_model'], 'r-', lw=2, label='DDMM')
            ax.plot(galaxy['r_kpc'], fit['v_newton'], 'g--', label='Newton')
            
            ax.set_xlabel('R (kpc)')
            ax.set_ylabel('V (km/s)')
            ax.set_title(f"{galaxy['name']} (RMS={fit['rms']:.1f} km/s)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sparc_galaxy_fits.png', dpi=150)
        plt.close()
    
    def _plot_bullet_cluster(self, x: np.ndarray, rho_gas: np.ndarray, 
                            rho_gal: np.ndarray, xi: np.ndarray):
        """Plot Bullet Cluster density and lensing"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Density distributions
        ax1.plot(x, rho_gas / 1e12, 'b-', lw=2, label='Gas (X-ray)')
        ax1.plot(x, rho_gal / 1e12, 'r-', lw=2, label='Galaxies')
        ax1.plot(x, (rho_gas + rho_gal) / 1e12, 'k--', label='Total')
        ax1.set_ylabel('ρ (10¹² M☉/kpc³)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lensing signal
        lensing_signal = (rho_gas + rho_gal) * xi
        ax2.plot(x, lensing_signal / 1e12, 'purple', lw=2, label='Lensing κ ∝ ρ×ξ')
        ax2.axvline(x[np.argmax(rho_gal)], color='r', ls=':', label='Galaxy peak')
        ax2.axvline(x[np.argmax(lensing_signal)], color='purple', ls=':', 
                   label='Lensing peak')
        ax2.set_xlabel('Distance along collision axis (kpc)')
        ax2.set_ylabel('Lensing signal (arb.)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Bullet Cluster: DDMM Lensing Prediction')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bullet_cluster_lensing.png', dpi=150)
        plt.close()
    
    def _plot_hubble_diagram(self, z: np.ndarray, mu_standard: np.ndarray,
                            mu_ddmm: np.ndarray, mu_err: np.ndarray):
        """Plot Hubble diagram comparison"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Hubble diagram
        ax1.errorbar(z, mu_standard, yerr=mu_err, fmt='o', alpha=0.5, 
                    label='ΛCDM expectation')
        ax1.plot(z, mu_ddmm, 'r-', lw=2, label='DDMM prediction')
        ax1.set_ylabel('Distance Modulus μ')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2.errorbar(z, mu_ddmm - mu_standard, yerr=mu_err, fmt='o', alpha=0.5)
        ax2.axhline(0, color='k', ls='--')
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Δμ (mag)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Type Ia Supernovae: DDMM vs ΛCDM')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hubble_diagram.png', dpi=150)
        plt.close()
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive test report"""
        if output_file is None:
            output_file = self.output_dir / 'ddmm_validation_report.json'
        
        report = {
            'model_parameters': self.model_params,
            'test_results': [asdict(r) for r in self.results],
            'overall_score': np.mean([r.score for r in self.results]),
            'tests_passed': sum(1 for r in self.results if r.passed),
            'tests_failed': sum(1 for r in self.results if not r.passed),
            'critical_issues': [],
            'recommendations': []
        }
        
        # Identify critical issues
        for result in self.results:
            if not result.passed and result.score < 0.5:
                report['critical_issues'].append({
                    'test': result.test_name,
                    'score': result.score,
                    'details': result.details
                })
        
        # Aggregate recommendations
        all_recs = []
        for result in self.results:
            all_recs.extend(result.recommendations)
        report['recommendations'] = list(set(all_recs))
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nValidation report saved to: {output_file}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print test summary to console"""
        print("\n" + "="*70)
        print("DDMM VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\nModel Parameters:")
        print(f"  ρ_c = {self.model_params.get('rho_c_solar_kpc3', 0):.2e} M☉/kpc³")
        print(f"  n = {self.model_params.get('n_exp', 0):.2f}")
        
        print(f"\nTest Results:")
        print(f"  {'Test Name':<30} {'Status':<10} {'Score':<10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.test_name:<30} {status:<10} {result.score:>6.2f}")
        
        overall_score = np.mean([r.score for r in self.results])
        tests_passed = sum(1 for r in self.results if r.passed)
        
        print(f"\nOverall Score: {overall_score:.2f}/1.00")
        print(f"Tests Passed: {tests_passed}/{len(self.results)}")
        
        if overall_score > 0.8:
            print("\n✅ DDMM shows excellent agreement with observations!")
        elif overall_score > 0.6:
            print("\n⚠️  DDMM shows reasonable agreement but needs refinement.")
        else:
            print("\n❌ DDMM has significant issues that need addressing.")
        
        print("="*70)


def run_full_validation(model_params_file: str):
    """Run complete validation suite"""
    logger.info("Starting DDMM validation suite...")
    
    # Load model parameters
    if model_params_file.endswith('.npz'):
        # Load from dynesty results
        data = np.load(model_params_file)
        samples = data['samples']
        weights = data['weights']
        
        # Get median parameters
        param_names = [
            'rho_c_solar_kpc3', 'n_exp',
            'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
            'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
            'M_bulge_solar', 'a_bulge_kpc',
            'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
        ]
        
        median_params = np.average(samples, weights=weights, axis=0)
        model_params = dict(zip(param_names[:len(median_params)], median_params))
        
        # Add component flags
        model_params['include_disk_thin'] = True
        model_params['include_disk_thick'] = 'M_disk_thick_solar' in param_names
        model_params['include_bulge'] = 'M_bulge_solar' in param_names
        model_params['include_gas'] = 'M_gas_solar' in param_names
        model_params['xi_type'] = 'power'
        
    else:
        # Load from JSON
        with open(model_params_file, 'r') as f:
            model_params = json.load(f)
    
    # Initialize validator
    validator = DDMMValidator(model_params)
    
    # Run all tests
    logger.info("\nRunning comprehensive DDMM validation...")
    
    # 1. Solar System
    validator.test_solar_system()
    
    # 2. Galaxy rotation curves (mock data)
    validator.test_sparc_galaxies(n_galaxies=10)
    
    # 3. Gravitational lensing
    validator.test_gravitational_lensing()
    
    # 4. CMB
    validator.test_cmb_predictions()
    
    # 5. Large scale structure
    validator.test_large_scale_structure()
    
    # 6. Supernovae
    validator.test_supernovae()
    
    # 7. Laboratory
    validator.test_laboratory_constraints()
    
    # 8. Self-consistency
    validator.test_self_consistency()
    
    # Generate report
    validator.generate_report()
    
    logger.info("\nValidation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate DDMM against observational tests")
    parser.add_argument('params_file', help="Model parameters file (.npz from dynesty or .json)")
    parser.add_argument('--output_dir', default='validation_results', 
                       help="Output directory for results")
    parser.add_argument('--sparc_data', default=None,
                       help="Path to SPARC data (optional)")
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run validation
    run_full_validation(args.params_file)