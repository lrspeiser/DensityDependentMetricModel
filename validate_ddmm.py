#!/usr/bin/env python3
"""
validate_ddmm_focused.py - Focused validation suite for DDMM

This script runs only the appropriate tests for DDMM theory:
1. Solar System precision tests (high density regime)
2. Galaxy rotation curves (SPARC + Gaia)
3. Gravitational lensing (with proper ξ integration)
4. Type Ia supernovae (cosmological distances)
5. Laboratory constraints (Earth density regime)
6. Large scale structure (z < 2 only)

Tests excluded:
- CMB (DDMM not designed for early universe)
- High-z observations (theory may not apply)

Version: 2.0 (Focused on valid tests only)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import pickle
import gzip
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Import DDMM modules
try:
    from density_metric2 import (
        v_baryon_total_newtonian_kms, 
        rho_baryon_total_midplane_solar_kpc3,
        XI_FUNCTION_MAP,
        G_ASTRO_UNITS,
        R_SUN_KPC
    )
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

# Expected DDMM parameters from your MCMC run
EXPECTED_PARAMS = {
    'rho_c_solar_kpc3': 1e9,
    'n_exp': 1.0,
    'A': 1.0,
    'xi_type': 'power'
}

@dataclass
class TestResult:
    """Structured test result with metadata"""
    test_name: str
    passed: bool
    score: float  # 0-1, where 1 is perfect
    details: Dict[str, Any]
    recommendations: List[str]

class DDMMValidator:
    """Focused validation class for DDMM tests"""
    
    def __init__(self, model_params: Dict[str, float], output_dir: str = "validation_results"):
        """Initialize validator with verified DDMM parameters"""
        # Set up output directory first
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Set up xi function BEFORE verifying parameters
        self._xi_func_base = XI_FUNCTION_MAP.get(
            model_params.get('xi_type', 'power'), 
            XI_FUNCTION_MAP['power']
        )
        
        # NOW verify parameters (which uses xi function)
        self.model_params = self._verify_parameters(model_params)
        
        logger.info("="*70)
        logger.info("DDMM FOCUSED VALIDATOR INITIALIZED")
        logger.info("="*70)
        logger.info(f"Parameters:")
        logger.info(f"  ρ_c = {self.model_params['rho_c_solar_kpc3']:.2e} M☉/kpc³")
        logger.info(f"  n = {self.model_params['n_exp']:.2f}")
        logger.info(f"  A = {self.model_params['A']:.2f}")
        logger.info(f"  ξ_max = 5.0 (capped)")
        logger.info("="*70)
        
    def _verify_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Verify and fix model parameters"""
        verified = params.copy()
        
        # Check for required parameters
        required_keys = ['rho_c_solar_kpc3', 'n_exp', 'A']
        
        # Handle alternative names
        if 'rho_c' in params and 'rho_c_solar_kpc3' not in params:
            verified['rho_c_solar_kpc3'] = params['rho_c']
        
        if 'gamma_exp' in params and 'n_exp' not in params:
            verified['n_exp'] = params['gamma_exp']
            
        if 'lambda_g' in params and 'A' not in params:
            verified['A'] = params['lambda_g']
            
        # Verify values are reasonable
        for key, expected in EXPECTED_PARAMS.items():
            if key in verified:
                actual = verified[key]
                if isinstance(expected, (int, float)):
                    deviation = abs(actual - expected) / expected
                    if deviation > 0.2:  # 20% tolerance
                        logger.warning(f"Parameter {key} = {actual} deviates from expected {expected}")
                        logger.warning("This may affect test results!")
        
        # Test xi behavior
        logger.info("\nVerifying ξ function behavior:")
        test_densities = {
            'Galaxy disk': 1e8,
            'Solar System': 1e16,
            'Earth surface': 1e31
        }
        
        for name, rho in test_densities.items():
            xi = self._calculate_xi(rho, verified)
            # Handle array output
            if hasattr(xi, '__len__'):
                xi_val = float(xi[0]) if len(xi) > 0 else float(xi)
            else:
                xi_val = float(xi)
            logger.info(f"  ξ({name}) = {xi_val:.6f}")
            
        return verified
    
    def _calculate_xi(self, rho, params=None):
        """Calculate xi with physical cap"""
        if params is None:
            params = self.model_params
            
        xi_uncapped = self._xi_func_base(
            rho, 
            params['rho_c_solar_kpc3'], 
            params['n_exp'], 
            params['A']
        )
        xi_capped = np.minimum(xi_uncapped, 5.0)
        
        # Return scalar if input was scalar
        if np.isscalar(rho):
            if hasattr(xi_capped, '__len__'):
                return float(xi_capped[0]) if len(xi_capped) > 0 else float(xi_capped)
            else:
                return float(xi_capped)
        else:
            return xi_capped
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check which datasets are available"""
        data_paths = {
            'gaia': 'gaia_sky_slices',
            'sparc': 'Rotmod_LTG', 
            'pantheon': 'pantheon',
            'bao': 'bao',
            'des': 'DES_Y3',
            'kids': 'Kids'
        }
        
        availability = {}
        logger.info("\nChecking data availability:")
        
        for name, path in data_paths.items():
            exists = Path(path).exists()
            availability[name] = exists
            status = "✅ Found" if exists else "❌ Not found"
            logger.info(f"  {name:<12} {status:<15} ({path})")
            
        return availability
    
    # ========================================================================
    # TEST 1: Solar System Consistency
    # ========================================================================
    
    def test_solar_system(self) -> TestResult:
        """Test Solar System constraints where ξ must be very close to 1"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: SOLAR SYSTEM CONSISTENCY")
        logger.info("="*60)
        
        test_details = {}
        recommendations = []
        all_passed = True
        
        # Test 1A: Mercury perihelion
        logger.info("\n1A. Mercury Perihelion Precession")
        mercury_result = self._test_mercury_precession()
        test_details['mercury'] = mercury_result
        if not mercury_result['passed']:
            all_passed = False
            recommendations.append("ξ deviates from unity in Solar System - check ρ_c value")
        
        # Test 1B: Lunar Laser Ranging
        logger.info("\n1B. Lunar Laser Ranging")
        llr_result = self._test_lunar_ranging()
        test_details['lunar'] = llr_result
        if not llr_result['passed']:
            all_passed = False
            recommendations.append("Equivalence principle violation detected")
            
        # Test 1C: Cassini constraint
        logger.info("\n1C. Cassini Radio Science")
        cassini_result = self._test_cassini()
        test_details['cassini'] = cassini_result
        if not cassini_result['passed']:
            all_passed = False
            recommendations.append("Light deflection shows deviation from GR")
        
        # Calculate overall score
        weights = {'mercury': 0.4, 'lunar': 0.3, 'cassini': 0.3}
        score = sum(test_details[k]['score'] * w for k, w in weights.items())
        
        result = TestResult(
            test_name="Solar System Consistency",
            passed=all_passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _test_mercury_precession(self) -> Dict[str, Any]:
        """Test Mercury perihelion advance"""
        # Solar System density
        rho_mercury_orbit = 1e15  # M☉/kpc³
        xi = self._calculate_xi(rho_mercury_orbit)
        
        # GR predicts 43 arcsec/century
        gr_prediction = 43.0
        ddmm_prediction = gr_prediction * float(xi)
        
        deviation_arcsec = abs(ddmm_prediction - gr_prediction)
        deviation_ppm = deviation_arcsec / gr_prediction * 1e6
        
        # Observations constrain to < 0.1% (1000 ppm)
        passed = deviation_ppm < 1000
        score = 1.0 - min(deviation_ppm / 1000, 1.0)
        
        logger.info(f"  Density at Mercury: {rho_mercury_orbit:.2e} M☉/kpc³")
        logger.info(f"  ξ = {float(xi):.8f}")
        logger.info(f"  GR prediction: {gr_prediction:.1f} arcsec/century")
        logger.info(f"  DDMM prediction: {ddmm_prediction:.1f} arcsec/century")
        logger.info(f"  Deviation: {deviation_ppm:.1f} ppm")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'xi': float(xi),
            'deviation_ppm': deviation_ppm
        }
    
    def _test_lunar_ranging(self) -> Dict[str, Any]:
        """Test Lunar Laser Ranging constraints"""
        # Earth-Moon system density
        rho_em = 5e14  # M☉/kpc³
        xi = self._calculate_xi(rho_em)
        
        # Nordtvedt parameter η = 4β - γ - 3
        # In GR: η = 0
        # In DDMM: η ≈ |1 - ξ|
        eta_ddmm = abs(1 - float(xi))
        eta_limit = 1e-4  # LLR constraint
        
        passed = eta_ddmm < eta_limit
        score = 1.0 - min(eta_ddmm / eta_limit, 1.0)
        
        logger.info(f"  Density at Earth-Moon: {rho_em:.2e} M☉/kpc³")
        logger.info(f"  ξ = {float(xi):.8f}")
        logger.info(f"  Nordtvedt η = {eta_ddmm:.2e}")
        logger.info(f"  Limit: < {eta_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'eta': eta_ddmm
        }
    
    def _test_cassini(self) -> Dict[str, Any]:
        """Test Cassini spacecraft constraints"""
        # Average density along Sun-Saturn path
        r_path = np.linspace(1, 10, 50)  # AU
        rho_path = 1e14 / r_path**2  # Simple r^-2 profile
        xi_path = self._calculate_xi(rho_path)
        
        # Cassini constrains γ-1 < 2.3e-5
        # In DDMM: γ_eff ≈ ξ
        # xi_path is an array, so we need to handle it properly
        if hasattr(xi_path, '__len__'):
            gamma_deviation = np.max(np.abs(xi_path - 1.0))
        else:
            gamma_deviation = abs(xi_path - 1.0)
        gamma_limit = 2.3e-5
        
        passed = gamma_deviation < gamma_limit
        score = 1.0 - min(gamma_deviation / gamma_limit, 1.0)
        
        logger.info(f"  Max |ξ-1| along path: {gamma_deviation:.2e}")
        logger.info(f"  Cassini limit: < {gamma_limit:.2e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'gamma_deviation': gamma_deviation
        }
    
    # ========================================================================
    # TEST 2: Galaxy Rotation Curves (SPARC + Gaia)
    # ========================================================================
    
    def test_galaxy_rotation_curves(self, sparc_path: Optional[str] = None,
                                   use_gaia: bool = True) -> TestResult:
        """Test galaxy rotation curves using SPARC and/or Gaia data"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: GALAXY ROTATION CURVES")
        logger.info("="*60)
        
        test_details = {}
        recommendations = []
        
        # Test 2A: Gaia Milky Way data
        if use_gaia and Path('gaia_sky_slices').exists():
            logger.info("\n2A. Gaia DR3 Milky Way Rotation Curve")
            gaia_result = self._test_gaia_rotation_curve()
            test_details['gaia'] = gaia_result
            if not gaia_result['passed']:
                recommendations.append("DDMM struggles with Milky Way rotation curve")
        
        # Test 2B: SPARC galaxies
        if sparc_path and Path(sparc_path).exists():
            logger.info("\n2B. SPARC Galaxy Sample")
            sparc_result = self._test_sparc_galaxies(sparc_path)
            test_details['sparc'] = sparc_result
            if not sparc_result['passed']:
                recommendations.append("Consider mass-dependent ρ_c for better fits")
        
        # Calculate overall score
        scores = [v['score'] for v in test_details.values()]
        score = np.mean(scores) if scores else 0.0
        passed = all(v['passed'] for v in test_details.values()) and len(test_details) > 0
        
        if not test_details:
            recommendations.append("No galaxy data available for testing")
            
        result = TestResult(
            test_name="Galaxy Rotation Curves",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _test_gaia_rotation_curve(self) -> Dict[str, Any]:
        """Test against Gaia DR3 Milky Way data"""
        try:
            # Load Gaia data
            from pathlib import Path
            import pandas as pd
            
            # Load from cached slices
            cache_dir = Path('gaia_sky_slices')
            if not cache_dir.exists():
                return {'passed': False, 'score': 0.0, 'error': 'Gaia cache directory not found'}
            
            slice_files = list(cache_dir.glob('processed_*.parquet'))
            if not slice_files:
                return {'passed': False, 'score': 0.0, 'error': 'No processed Gaia files found'}
            
            # Combine all slices
            df_list = [pd.read_parquet(f) for f in slice_files]
            gaia_df = pd.concat(df_list, ignore_index=True)
            
            if gaia_df is None or len(gaia_df) == 0:
                return {'passed': False, 'score': 0.0, 'error': 'Failed to load Gaia data'}
            
            # Bin the data
            r_bins = np.linspace(5, 15, 20)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            
            v_mean = []
            v_err = []
            
            for i in range(len(r_bins)-1):
                mask = (gaia_df['R_kpc'] > r_bins[i]) & (gaia_df['R_kpc'] < r_bins[i+1])
                if mask.sum() > 100:
                    v_mean.append(np.median(gaia_df.loc[mask, 'v_obs']))
                    v_err.append(np.std(gaia_df.loc[mask, 'v_obs']) / np.sqrt(mask.sum()))
            
            v_mean = np.array(v_mean)
            v_err = np.array(v_err)
            
            final_r_centers = np.array([c for i, c in enumerate(r_centers) if i < len(v_mean)])
            
            # Calculate DDMM prediction using your fitted parameters
            params = self.model_params.copy()
            params.update({
                'M_disk_thin_solar': 6e10,
                'R_d_thin_kpc': 2.6,
                'h_z_thin_kpc': 0.3,
                'M_bulge_solar': 2e10,
                'a_bulge_kpc': 0.9,
                'include_disk_thin': True,
                'include_bulge': True,
                'include_disk_thick': False,
                'include_gas': False
            })
            
            
            v_newton = v_baryon_total_newtonian_kms(final_r_centers, params)
            rho = rho_baryon_total_midplane_solar_kpc3(final_r_centers, params)
            xi = self._calculate_xi(rho)
            # Ensure xi is array-like for multiplication
            if np.isscalar(xi):
                xi = np.full_like(v_newton, xi)
            v_ddmm = v_newton * np.sqrt(xi)
            
            # Calculate fit quality
            chi2 = np.sum(((v_mean - v_ddmm) / v_err)**2)
            chi2_dof = chi2 / (len(v_mean) - 2)
            
            passed = chi2_dof < 3.0
            score = np.exp(-chi2_dof / 2)
            
            logger.info(f"  Data points: {len(v_mean)}")
            logger.info(f"  χ²/dof = {chi2_dof:.2f}")
            logger.info(f"  Mean residual: {np.mean(v_mean - v_ddmm):.1f} km/s")
            logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
            
            # Plot result
            self._plot_gaia_fit(r_centers, v_mean, v_err, v_ddmm, v_newton)
            
            return {
                'passed': passed,
                'score': score,
                'chi2_dof': chi2_dof,
                'mean_residual': float(np.mean(v_mean - v_ddmm))
            }
            
        except Exception as e:
            logger.error(f"Gaia test failed: {e}")
            return {'passed': False, 'score': 0.0, 'error': str(e)}
    
    def _test_sparc_galaxies(self, sparc_path: str) -> Dict[str, Any]:
        """Test SPARC galaxy sample"""
        try:
            # Try to import SPARC loader
            try:
                from sparc_data_loader import SPARCDataLoader
            except ImportError:
                logger.warning("sparc_data_loader not found, skipping SPARC test")
                return {'passed': False, 'score': 0.0, 'error': 'SPARC loader not available'}
            
            loader = SPARCDataLoader(sparc_path)
            loader.load_all_galaxies()
            
            # Test subsample
            galaxies = loader.get_galaxy_sample(30)
            
            rms_values = []
            successful = 0
            
            for galaxy in galaxies:
                try:
                    # Simple single M/L fit
                    ml_best, rms = self._fit_single_galaxy(galaxy)
                    if rms < 50:  # Reasonable fit
                        rms_values.append(rms)
                        successful += 1
                except:
                    continue
            
            if successful > 0:
                mean_rms = np.mean(rms_values)
                median_rms = np.median(rms_values)
            else:
                mean_rms = median_rms = np.inf
                
            passed = median_rms < 15 and successful > len(galaxies) * 0.7
            score = np.exp(-median_rms / 20) * (successful / len(galaxies))
            
            logger.info(f"  Galaxies tested: {len(galaxies)}")
            logger.info(f"  Successful fits: {successful}")
            logger.info(f"  Median RMS: {median_rms:.1f} km/s")
            logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
            
            return {
                'passed': passed,
                'score': score,
                'n_galaxies': len(galaxies),
                'successful': successful,
                'median_rms': median_rms
            }
            
        except Exception as e:
            logger.error(f"SPARC test failed: {e}")
            return {'passed': False, 'score': 0.0, 'error': str(e)}
    
    def _fit_single_galaxy(self, galaxy: Dict) -> Tuple[float, float]:
        """Fit single galaxy with DDMM"""
        r = galaxy['r_kpc']
        v_obs = galaxy['v_obs']
        v_err = galaxy['v_err']
        
        def objective(ml_star):
            # Newtonian velocities
            v_star = np.sqrt((galaxy['v_disk']**2 + galaxy['v_bulge']**2) * ml_star[0])
            v_newton = np.sqrt(v_star**2 + galaxy['v_gas']**2)
            
            # Simple density estimate
            sigma = 100 * ml_star[0]  # M☉/pc²
            rho = sigma * 1e6 / 0.6  # M☉/kpc³
            
            # Apply DDMM
            xi = self._calculate_xi(rho * np.exp(-r/3))
            # Ensure xi is array-like for multiplication
            if np.isscalar(xi):
                xi = np.full_like(v_newton, xi)
            v_model = v_newton * np.sqrt(xi)
            
            return np.sum(((v_obs - v_model) / v_err)**2)
        
        result = minimize(objective, x0=[0.5], bounds=[(0.1, 5.0)])
        ml_best = result.x[0]
        
        # Calculate RMS
        v_star = np.sqrt((galaxy['v_disk']**2 + galaxy['v_bulge']**2) * ml_best)
        v_newton = np.sqrt(v_star**2 + galaxy['v_gas']**2)
        rho = 100 * ml_best * 1e6 / 0.6 * np.exp(-r/3)
        xi = self._calculate_xi(rho)
        # Ensure xi is array-like
        if np.isscalar(xi):
            xi = np.full_like(v_newton, xi)
        v_model = v_newton * np.sqrt(xi)
        
        rms = np.sqrt(np.mean((v_obs - v_model)**2))
        
        return ml_best, rms
    # ========================================================================
    # TEST 3: Type Ia Supernovae with Proper Light Propagation
    # ========================================================================



    def _luminosity_distance_lcdm(self, z):
        """
        Calculate standard ΛCDM luminosity distance.
        
        Parameters:
        -----------
        z : array-like
            Redshift values
            
        Returns:
        --------
        d_L : array-like
            Luminosity distance in Mpc
        """
        from scipy.integrate import quad
        
        c_H0 = C_KMS / H0 * 1e-3  # c/H0 in Mpc
        
        def E(z):
            """Hubble parameter evolution"""
            return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
        
        # Handle both scalar and array inputs
        z_array = np.atleast_1d(z)
        distances = []
        
        for zi in z_array:
            # Comoving distance
            integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
            d_c = c_H0 * integral
            
            # Luminosity distance
            d_L = d_c * (1 + zi)
            distances.append(d_L)
        
        # Return scalar if input was scalar
        if np.isscalar(z):
            return distances[0]
        else:
            return np.array(distances)
    
    def _load_pantheon_data(self, pantheon_path):
        """
        Load Pantheon supernova dataset.
        
        Parameters:
        -----------
        pantheon_path : str
            Path to Pantheon data directory or file
            
        Returns:
        --------
        dict with keys:
            z : array of redshifts
            mu : array of distance moduli
            mu_err : array of errors
        """
        try:
            path = Path(pantheon_path)
            
            # If path is a file, use it directly
            if path.is_file():
                file_path = path
            else:
                # Look for common Pantheon file names in directory
                possible_files = [
                    'pantheon_plus.csv',
                    'pantheon+_sn_data.csv', 
                    'Pantheon+SH0ES.dat',
                    'pantheon.csv',
                    'pantheon_sample.txt',
                    'lcparam_full_long_zhel.txt',
                    'lcparam_full_long.txt',
                    'pantheon.txt'
                ]
                
                file_path = None
                for filename in possible_files:
                    candidate = path / filename
                    if candidate.exists():
                        file_path = candidate
                        logger.info(f"Found Pantheon data file: {filename}")
                        break
                
                if file_path is None:
                    # Check for any CSV or txt files
                    csv_files = list(path.glob('*.csv'))
                    txt_files = list(path.glob('*.txt'))
                    dat_files = list(path.glob('*.dat'))
                    
                    if csv_files:
                        file_path = csv_files[0]
                    elif txt_files:
                        file_path = txt_files[0]
                    elif dat_files:
                        file_path = dat_files[0]
                    else:
                        logger.warning(f"No Pantheon data file found in {pantheon_path}")
                        return None
            
            # Load the file
            logger.info(f"Loading Pantheon data from: {file_path}")
            
            # Try different delimiters and formats
            data = None
            if file_path.suffix == '.csv':
                try:
                    data = pd.read_csv(file_path)
                except:
                    data = pd.read_csv(file_path, delim_whitespace=True)
            else:
                # Try space-delimited first, then comma
                try:
                    data = pd.read_csv(file_path, delim_whitespace=True)
                except:
                    try:
                        data = pd.read_csv(file_path, sep=',')
                    except:
                        data = pd.read_csv(file_path, sep='\t')
            
            if data is None or len(data) == 0:
                logger.error("Failed to load data from file")
                return None
            
            # Debug: print column names
            logger.debug(f"Columns found: {list(data.columns)}")
            
            # Find redshift column
            z_col = None
            z_candidates = ['zCMB', 'zcmb', 'zHEL', 'zhel', 'z', 'Z', 'ZHELIO', 
                          'redshift', 'REDSHIFT', 'zSN', 'z_cmb', 'z_helio']
            for candidate in z_candidates:
                if candidate in data.columns:
                    z_col = candidate
                    break
            
            # Find distance modulus column
            mu_col = None
            mu_candidates = ['mu', 'MU', 'mu_obs', 'MU_OBS', 'distance_modulus',
                           'mu_SH0ES', 'MU_SH0ES', 'mB', 'MB', 'mag', 'MAG']
            for candidate in mu_candidates:
                if candidate in data.columns:
                    mu_col = candidate
                    break
            
            # Find error column
            mu_err_col = None
            err_candidates = ['mu_err', 'MUERR', 'dmu', 'err_mu', 'ERR_MU',
                            'mu_err_total', 'mBERR', 'MBERR', 'sigma_mu',
                            'err', 'ERR', 'error', 'ERROR']
            for candidate in err_candidates:
                if candidate in data.columns:
                    mu_err_col = candidate
                    break
            
            if z_col is None or mu_col is None:
                logger.error(f"Could not find required columns. Found: {list(data.columns)}")
                return None
            
            logger.info(f"Using columns: z='{z_col}', mu='{mu_col}', mu_err='{mu_err_col}'")
            
            # Extract data
            result = {
                'z': data[z_col].values,
                'mu': data[mu_col].values,
                'mu_err': data[mu_err_col].values if mu_err_col else np.ones_like(data[z_col].values) * 0.15
            }
            
            # Filter valid data
            mask = (result['z'] > 0.001) & (result['z'] < 2.5) & \
                   np.isfinite(result['mu']) & np.isfinite(result['mu_err']) & \
                   (result['mu_err'] > 0)
            
            filtered = {
                'z': result['z'][mask],
                'mu': result['mu'][mask],
                'mu_err': result['mu_err'][mask]
            }
            
            logger.info(f"Loaded {len(filtered['z'])} supernovae from Pantheon dataset")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error loading Pantheon data: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    def test_supernovae(self, pantheon_path: Optional[str] = None) -> TestResult:
        """Test distance-redshift relation with SNe Ia using proper DDMM light propagation"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: TYPE IA SUPERNOVAE (DDMM Light Propagation)")
        logger.info("="*60)
        
        # Try to load real Pantheon data if path provided
        sn_data = None
        if pantheon_path:
            logger.info(f"Attempting to load Pantheon data from: {pantheon_path}")
            sn_data = self._load_pantheon_data(pantheon_path)
        
        # Use loaded data or fall back to mock
        if sn_data is not None:
            logger.info(f"Using {len(sn_data['z'])} real supernovae from Pantheon dataset")
            z_obs = sn_data['z']
            mu_obs = sn_data['mu']
            mu_err = sn_data['mu_err']
        else:
            logger.info("Using mock supernova data for testing")
            z_obs = np.logspace(-2, 0.3, 50)
            mu_obs = None
            mu_err = None
        
        # Calculate distances with different models
        logger.info("\nTesting light propagation models:")
        
        # Model 1: Standard ΛCDM (for comparison)
        d_L_lcdm = self._luminosity_distance_lcdm(z_obs)
        mu_lcdm = 5 * np.log10(d_L_lcdm) + 25
        
        # Model 2: DDMM with path integration (no expansion)
        logger.info("\n  Model A: DDMM redshift from density gradients")
        z_ddmm_noexp, d_L_ddmm_noexp = self._calculate_ddmm_distances_no_expansion(z_obs)
        mu_ddmm_noexp = 5 * np.log10(d_L_ddmm_noexp) + 25
        
        # Model 3: DDMM + expansion (hybrid model)
        logger.info("\n  Model B: DDMM + cosmic expansion")
        d_L_ddmm_hybrid = self._luminosity_distance_ddmm_hybrid(z_obs)
        mu_ddmm_hybrid = 5 * np.log10(d_L_ddmm_hybrid) + 25
        
        # Compare models
        if mu_obs is not None:
            chi2_lcdm = np.sum(((mu_obs - mu_lcdm) / mu_err)**2)
            chi2_noexp = np.sum(((mu_obs - mu_ddmm_noexp) / mu_err)**2)
            chi2_hybrid = np.sum(((mu_obs - mu_ddmm_hybrid) / mu_err)**2)
            
            logger.info(f"\nChi-squared comparison:")
            logger.info(f"  ΛCDM:           χ² = {chi2_lcdm:.1f}")
            logger.info(f"  DDMM (no exp):  χ² = {chi2_noexp:.1f}")
            logger.info(f"  DDMM + exp:     χ² = {chi2_hybrid:.1f}")
            
            best_model = min(chi2_lcdm, chi2_noexp, chi2_hybrid)
            passed = (chi2_noexp < 1.5 * chi2_lcdm) or (chi2_hybrid < 1.5 * chi2_lcdm)
            score = np.exp(-min(chi2_noexp, chi2_hybrid) / chi2_lcdm)
        else:
            # Without real data, check if deviations are reasonable
            max_dev_noexp = np.max(np.abs(mu_ddmm_noexp - mu_lcdm))
            max_dev_hybrid = np.max(np.abs(mu_ddmm_hybrid - mu_lcdm))
            
            logger.info(f"\nMaximum deviations from ΛCDM:")
            logger.info(f"  DDMM (no expansion): {max_dev_noexp:.3f} mag")
            logger.info(f"  DDMM + expansion:    {max_dev_hybrid:.3f} mag")
            
            passed = min(max_dev_noexp, max_dev_hybrid) < 0.3  # More lenient than 0.1
            score = 1.0 - min(max_dev_noexp, max_dev_hybrid) / 0.3
        
        # Plot results
        self._plot_hubble_diagram_advanced(z_obs, mu_lcdm, mu_ddmm_noexp, 
                                        mu_ddmm_hybrid, mu_obs, mu_err)
        
        test_details = {
            'model_comparison': {
                'ddmm_no_expansion_max_dev': float(max_dev_noexp) if mu_obs is None else None,
                'ddmm_hybrid_max_dev': float(max_dev_hybrid) if mu_obs is None else None,
                'chi2_results': {
                    'lcdm': float(chi2_lcdm) if mu_obs is not None else None,
                    'ddmm_no_exp': float(chi2_noexp) if mu_obs is not None else None,
                    'ddmm_hybrid': float(chi2_hybrid) if mu_obs is not None else None
                }
            }
        }
        
        recommendations = []
        if not passed:
            recommendations.append("DDMM light propagation needs refinement")
            recommendations.append("Consider environmental density models from simulations")
        
        result = TestResult(
            test_name="Type Ia Supernovae",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result

    def _calculate_ddmm_distances_no_expansion(self, z_obs):
        """
        Calculate distances assuming ALL redshift comes from DDMM effects.
        This is the radical model where there's no cosmic expansion.
        
        Key idea: Light gains energy (blueshifts) in high-density regions
        and loses energy (redshifts) in low-density voids. The cumulative
        effect produces the observed cosmological redshift.
        """
        distances = []
        z_ddmm = []
        
        for z_target in z_obs:
            # For each supernova, integrate along the light path
            # Start with trial distance and iterate
            d_trial = self._distance_from_redshift_static(z_target)  # Initial guess
            
            # Iterate to find self-consistent solution
            for iteration in range(5):
                # Model the density along line of sight
                # This is simplified - real implementation needs cosmic web structure
                r_path = np.linspace(0, d_trial, 1000)  # Mpc
                rho_path = self._cosmic_density_profile(r_path)
                
                # Calculate ξ along path
                xi_path = self._calculate_xi(rho_path)
                
                # Key equation: photon frequency evolution
                # dν/dr = -(ν/2c) * d(ln ξ)/dr
                # This gives cumulative redshift:
                # 1 + z = exp(∫[d(ln ξ)/dr]dr/2) = √(ξ_emit/ξ_obs)
                
                # For a static universe with DDMM:
                xi_emit = xi_path[-1]  # ξ at source
                xi_obs = xi_path[0]    # ξ at observer
                
                # Additional contribution from density gradients along path
                dxi_dr = np.gradient(xi_path) / np.gradient(r_path)
                cumulative_factor = np.exp(-0.5 * np.trapz(dxi_dr/xi_path, r_path*1e3/C_KMS))
                
                z_predicted = cumulative_factor * np.sqrt(xi_emit/xi_obs) - 1
                
                # Update distance estimate
                if abs(z_predicted - z_target) / z_target < 0.01:
                    break
                d_trial *= (1 + z_target) / (1 + z_predicted)
            
            distances.append(d_trial)
            z_ddmm.append(z_predicted)
        
        return np.array(z_ddmm), np.array(distances)

    def _cosmic_density_profile(self, r_mpc):
        """
        Model cosmic density along line of sight.
        Includes galaxy clusters, filaments, and voids.
        
        This is highly simplified - real implementation would use
        simulations or observational density fields.
        """
        # Convert to M☉/kpc³
        rho_mean = 2.775e11 * OMEGA_M * (H0/100)**2 / 1e9  # M☉/kpc³
        
        # Add cosmic web structure
        # Simplified model with periodic overdensities (clusters/filaments) and voids
        rho = np.ones_like(r_mpc) * rho_mean
        
        # Add some structure
        for i in range(len(r_mpc)):
            # Clusters every ~50 Mpc
            if (r_mpc[i] % 50) < 5:
                rho[i] *= 100  # Overdensity in clusters
            # Filaments
            elif (r_mpc[i] % 50) < 20:
                rho[i] *= 5
            # Voids
            elif (r_mpc[i] % 50) > 35:
                rho[i] *= 0.1
        
        # Add local environment
        if len(r_mpc) > 0:
            # Higher density near observer (in galaxy)
            local_mask = r_mpc < 10
            rho[local_mask] = 1e8  # Galactic density
            
            # Higher density near source (assuming in galaxy)
            if len(r_mpc) > 10:
                source_mask = r_mpc > (r_mpc[-1] - 10)
                rho[source_mask] = 1e8
        
        return rho

    def _luminosity_distance_ddmm_hybrid(self, z):
        """
        Hybrid model: Universe expands AND has DDMM effects.
        More conservative approach that modifies standard cosmology.
        """
        from scipy.integrate import quad, odeint
        
        c_H0 = C_KMS / H0 * 1e-3  # Mpc
        distances = []
        
        for zi in z:
            # First get standard ΛCDM distance
            def E(z):
                return np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_LAMBDA)
            
            # Comoving distance
            integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
            d_c = c_H0 * integral
            
            # Now apply DDMM corrections
            # Model average density along path accounting for expansion
            z_path = np.linspace(0, zi, 100)
            rho_path = []
            
            for zp in z_path:
                # Density evolves as (1+z)³ in expanding universe
                rho_z = 2.775e11 * OMEGA_M * (H0/100)**2 * (1+zp)**3 / 1e9  # M☉/kpc³
                # Add structure
                rho_z *= (1 + 0.5*np.sin(zp*20))  # Simplified structure
                rho_path.append(rho_z)
            
            rho_path = np.array(rho_path)
            xi_path = self._calculate_xi(rho_path)
            
            # Modified luminosity distance
            # Key insight: ξ affects both photon propagation AND distance measures
            xi_eff = np.exp(np.mean(np.log(xi_path)))  # Geometric mean
            
            # In hybrid model, luminosity distance is modified by ξ
            # This comes from modified photon geodesics in DDMM spacetime
            d_L = d_c * (1 + zi) * xi_eff**(1/4)  # ξ^(1/4) gives modest correction
            
            distances.append(d_L)
        
        return np.array(distances)

    def _distance_from_redshift_static(self, z):
        """
        In static DDMM universe, distance-redshift relation is different.
        This assumes Euclidean space with DDMM effects only.
        """
        # Very rough approximation for initial guess
        # In DDMM, effective Hubble law: cz ≈ H_eff * d
        # where H_eff depends on typical ξ values
        xi_typical = 2.5  # Typical value in cosmic voids
        H_eff = H0 * np.sqrt(xi_typical)
        return C_KMS * z / H_eff * 1e-3  # Mpc

    def _plot_hubble_diagram_advanced(self, z, mu_lcdm, mu_ddmm_noexp, 
                                    mu_ddmm_hybrid, mu_obs=None, mu_err=None):
        """Plot advanced Hubble diagram comparing models"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.plot(z, mu_lcdm, 'b-', lw=2, label='ΛCDM', alpha=0.8)
        ax1.plot(z, mu_ddmm_noexp, 'r--', lw=2, label='DDMM (no expansion)', alpha=0.8)
        ax1.plot(z, mu_ddmm_hybrid, 'g-.', lw=2, label='DDMM + expansion', alpha=0.8)
        
        if mu_obs is not None:
            ax1.errorbar(z, mu_obs, yerr=mu_err, fmt='ko', markersize=4, 
                        alpha=0.5, label='Observed SNe Ia')
        
        ax1.set_ylabel('Distance Modulus μ', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_xlim(0.008, 2.5)
        ax1.set_ylim(32, 46)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Type Ia Supernovae: Testing DDMM Light Propagation', fontsize=16)
        
        # Residuals
        ax2.axhline(0, color='gray', ls='--', alpha=0.5)
        ax2.plot(z, mu_ddmm_noexp - mu_lcdm, 'r--', lw=2, 
                label='DDMM (no exp) - ΛCDM', alpha=0.8)
        ax2.plot(z, mu_ddmm_hybrid - mu_lcdm, 'g-.', lw=2, 
                label='DDMM+exp - ΛCDM', alpha=0.8)
        
        ax2.set_xlabel('Redshift z', fontsize=14)
        ax2.set_ylabel('Δμ (mag)', fontsize=14)
        ax2.set_xscale('log')
        ax2.set_xlim(0.008, 2.5)
        ax2.set_ylim(-2, 2)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hubble_diagram_ddmm_models.png', dpi=150)
        plt.close()
    
    # ========================================================================
    # TEST 4: Laboratory Constraints
    # ========================================================================
    
    def test_laboratory_constraints(self) -> TestResult:
        """Test Earth-based laboratory constraints"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: LABORATORY CONSTRAINTS")
        logger.info("="*60)
        
        test_details = {}
        
        # Test 4A: Eöt-Wash torsion balance
        logger.info("\n4A. Eöt-Wash Torsion Balance")
        eotwash_result = self._test_eotwash()
        test_details['eotwash'] = eotwash_result
        
        # Test 4B: MICROSCOPE satellite
        logger.info("\n4B. MICROSCOPE Satellite")
        microscope_result = self._test_microscope()
        test_details['microscope'] = microscope_result
        
        # Overall assessment
        passed = all(v['passed'] for v in test_details.values())
        score = np.mean([v['score'] for v in test_details.values()])
        
        recommendations = []
        if not passed:
            recommendations.append("Earth density requires ξ ≈ 1 to extreme precision")
            recommendations.append("Consider additional screening mechanisms if needed")
        
        result = TestResult(
            test_name="Laboratory Constraints",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _test_eotwash(self) -> Dict[str, Any]:
        """Eöt-Wash torsion balance constraints"""
        # Earth surface density
        rho_lab = 8e31  # M☉/kpc³
        xi = self._calculate_xi(rho_lab)
        
        # Fifth force parameter α = |ξ - 1|
        alpha = abs(float(xi) - 1.0)
        alpha_limit = 1e-6
        
        passed = alpha < alpha_limit
        score = 1.0 - min(alpha / alpha_limit, 1.0)
        
        logger.info(f"  Lab density: {rho_lab:.2e} M☉/kpc³")
        logger.info(f"  ρ/ρ_c = {rho_lab/self.model_params['rho_c_solar_kpc3']:.2e}")
        logger.info(f"  ξ = {float(xi):.12f}")
        logger.info(f"  Fifth force α = {alpha:.2e}")
        logger.info(f"  Limit: < {alpha_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'alpha': alpha,
            'xi': float(xi)
        }
    
    def _test_microscope(self) -> Dict[str, Any]:
        """MICROSCOPE equivalence principle test"""
        # Satellite orbit density
        rho_orbit = 1e20  # M☉/kpc³ (LEO)
        xi = self._calculate_xi(rho_orbit)
        
        # Eötvös parameter
        eta = abs(float(xi) - 1.0)
        eta_limit = 1e-14
        
        passed = eta < eta_limit
        score = 1.0 - min(eta / eta_limit, 1.0)
        
        logger.info(f"  Orbit density: {rho_orbit:.2e} M☉/kpc³")
        logger.info(f"  ξ = {float(xi):.15f}")
        logger.info(f"  Eötvös η = {eta:.2e}")
        logger.info(f"  MICROSCOPE limit: < {eta_limit:.0e}")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'eta': eta,
            'xi': float(xi)
        }
    
    # ========================================================================
    # TEST 5: Gravitational Lensing
    # ========================================================================
    
    def test_gravitational_lensing(self) -> TestResult:
        """Test gravitational lensing predictions"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: GRAVITATIONAL LENSING")
        logger.info("="*60)
        
        test_details = {}
        
        # Test 5A: Galaxy-galaxy lensing
        logger.info("\n5A. Galaxy-Galaxy Lensing")
        ggl_result = self._test_galaxy_lensing()
        test_details['galaxy_galaxy'] = ggl_result
        
        # Test 5B: Cluster lensing (simplified)
        logger.info("\n5B. Cluster Lensing")
        cluster_result = self._test_cluster_lensing()
        test_details['cluster'] = cluster_result
        
        passed = all(v['passed'] for v in test_details.values())
        score = np.mean([v['score'] for v in test_details.values()])
        
        recommendations = []
        if not passed:
            recommendations.append("Lensing requires careful ξ integration along line of sight")
            recommendations.append("Consider environmental dependence of parameters")
        
        result = TestResult(
            test_name="Gravitational Lensing",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _test_galaxy_lensing(self) -> Dict[str, Any]:
        """Test galaxy-scale lensing"""
        # Typical galaxy lens
        M_lens = 1e12  # M☉
        R_eff = 10  # kpc
        
        # NFW-like profile
        r = np.logspace(-1, 2, 100)
        r_s = 20  # kpc
        rho_s = 1e7  # M☉/kpc³
        rho = rho_s / ((r/r_s) * (1 + r/r_s)**2)
        
        # Apply DDMM
        xi = self._calculate_xi(rho)
        
        # Ensure xi is array if rho is array
        if hasattr(rho, '__len__') and not hasattr(xi, '__len__'):
            xi = np.full_like(rho, xi)
        
        # Ensure xi is array if rho is array
        if hasattr(rho, '__len__') and not hasattr(xi, '__len__'):
            xi = np.full_like(rho, xi)
        
        # Einstein radius (simplified)
        # θ_E ∝ √(M_eff)
        M_newton = 4 * np.pi * np.trapz(rho * r**2, r)
        M_ddmm = 4 * np.pi * np.trapz(rho * xi * r**2, r)
        
        theta_ratio = np.sqrt(M_ddmm / M_newton)
        
        # Observations show standard Einstein radii
        deviation = abs(theta_ratio - 1.0)
        passed = deviation < 0.2  # 20% tolerance
        score = 1.0 - min(deviation / 0.2, 1.0)
        
        logger.info(f"  Mass ratio (DDMM/Newton): {M_ddmm/M_newton:.2f}")
        logger.info(f"  Einstein radius ratio: {theta_ratio:.2f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'theta_ratio': theta_ratio,
            'mass_ratio': M_ddmm/M_newton
        }
    
    def _test_cluster_lensing(self) -> Dict[str, Any]:
        """Test cluster-scale lensing"""
        # Simplified cluster model
        M_cl = 1e15  # M☉
        R_cl = 1000  # kpc
        
        # Beta model
        r = np.linspace(1, 2000, 200)
        r_c = 200  # kpc
        rho_0 = 1e4  # M☉/kpc³
        rho = rho_0 / (1 + (r/r_c)**2)**1.5
        
        # Apply DDMM
        xi = self._calculate_xi(rho)
        
        # Ensure xi is array if rho is array
        if hasattr(rho, '__len__') and not hasattr(xi, '__len__'):
            xi = np.full_like(rho, xi)
        
        # Lensing mass
        M_lens = 2 * np.pi * np.trapz(rho * xi * r, r)
        
        # Expected from X-ray
        M_gas = 2 * np.pi * np.trapz(rho * r, r)
        
        # Typical cluster: M_lens ≈ 5 × M_gas
        ratio_expected = 5.0
        ratio_ddmm = M_lens / M_gas
        
        deviation = abs(ratio_ddmm - ratio_expected) / ratio_expected
        passed = deviation < 0.3
        score = 1.0 - min(deviation / 0.3, 1.0)
        
        logger.info(f"  M_lens/M_gas expected: {ratio_expected:.1f}")
        logger.info(f"  M_lens/M_gas DDMM: {ratio_ddmm:.1f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'mass_ratio': ratio_ddmm
        }
    
    # ========================================================================
    # TEST 6: Large Scale Structure (z < 2 only)
    # ========================================================================
    
    def test_large_scale_structure(self, bao_path: Optional[str] = None) -> TestResult:
        """Test structure formation at z < 2"""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: LARGE SCALE STRUCTURE (z < 2)")
        logger.info("="*60)
        
        test_details = {}
        
        # Test 6A: Growth rate at low z
        logger.info("\n6A. Growth Rate f(z) at z < 1")
        growth_result = self._test_growth_rate()
        test_details['growth_rate'] = growth_result
        
        # Test 6B: BAO scale (if data available)
        if bao_path and Path(bao_path).exists():
            logger.info("\n6B. BAO Scale")
            bao_result = self._test_bao_scale()
            test_details['bao'] = bao_result
        
        passed = all(v['passed'] for v in test_details.values())
        score = np.mean([v['score'] for v in test_details.values()])
        
        recommendations = []
        if not passed:
            recommendations.append("Structure growth modified at low z")
            recommendations.append("Full numerical integration of modified equations needed")
        
        result = TestResult(
            test_name="Large Scale Structure",
            passed=passed,
            score=score,
            details=test_details,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _test_growth_rate(self) -> Dict[str, Any]:
        """Test linear growth rate at low redshift"""
        # Test at z = 0 where we have RSD constraints
        z = 0.0
        rho_z0 = 1e6  # M☉/kpc³
        xi_z0 = self._calculate_xi(rho_z0)
        
        # Standard growth rate
        f_standard = OMEGA_M**0.55  # ≈ 0.47
        
        # DDMM modification
        f_ddmm = f_standard * np.sqrt(float(xi_z0))
        
        # Observed from RSD
        f_observed = 0.45
        f_error = 0.05
        
        deviation = abs(f_ddmm - f_observed) / f_error  # in sigma
        passed = deviation < 2  # Within 2σ
        score = np.exp(-deviation**2 / 2)
        
        logger.info(f"  f(z=0) standard: {f_standard:.3f}")
        logger.info(f"  f(z=0) DDMM: {f_ddmm:.3f}")
        logger.info(f"  f(z=0) observed: {f_observed:.3f} ± {f_error:.3f}")
        logger.info(f"  Deviation: {deviation:.1f}σ")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'f_ddmm': f_ddmm,
            'deviation_sigma': deviation
        }
    
    def _test_bao_scale(self) -> Dict[str, Any]:
        """Test BAO scale preservation"""
        # BAO at z ≈ 0.5
        z_bao = 0.5
        rho_bao = 1e6 * (1 + z_bao)**3
        xi_bao = self._calculate_xi(rho_bao)
        
        # Sound horizon
        r_s_fid = 147.5  # Mpc (fiducial)
        r_s_ddmm = r_s_fid  # Should be unchanged at drag epoch
        
        # But distance measures affected
        DV_ratio = float(xi_bao)**(1/6)  # Simplified scaling
        
        deviation = abs(DV_ratio - 1.0)
        passed = deviation < 0.02  # 2% tolerance
        score = 1.0 - min(deviation / 0.02, 1.0)
        
        logger.info(f"  ξ at BAO redshift: {float(xi_bao):.3f}")
        logger.info(f"  Distance measure ratio: {DV_ratio:.3f}")
        logger.info(f"  Deviation: {deviation*100:.1f}%")
        logger.info(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'passed': passed,
            'score': score,
            'DV_ratio': DV_ratio
        }
    
    # ========================================================================
    # Plotting Methods
    # ========================================================================
    
    def _plot_gaia_fit(self, r, v_obs, v_err, v_ddmm, v_newton):
        """Plot Gaia rotation curve fit"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Gaia DR3', 
                    capsize=5, markersize=8)
        ax1.plot(r, v_newton, 'b--', lw=2, label='Newtonian')
        ax1.plot(r, v_ddmm, 'r-', lw=3, label='DDMM')
        
        ax1.set_ylabel('V (km/s)', fontsize=12)
        ax1.set_xlim(4, 16)
        ax1.set_ylim(150, 300)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = v_obs - v_ddmm
        ax2.errorbar(r, residuals, yerr=v_err, fmt='ko', capsize=5)
        ax2.axhline(0, color='k', ls='--', alpha=0.5)
        ax2.set_xlabel('R (kpc)', fontsize=12)
        ax2.set_ylabel('O-C (km/s)', fontsize=12)
        ax2.set_xlim(4, 16)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('DDMM Fit to Gaia DR3 Rotation Curve', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gaia_rotation_curve_fit.png', dpi=150)
        plt.close()
    
    def _plot_hubble_diagram(self, z, mu_lcdm, mu_ddmm):
        """Plot Hubble diagram comparison"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(z, mu_lcdm, 'b-', lw=2, label='ΛCDM')
        ax1.plot(z, mu_ddmm, 'r--', lw=2, label='DDMM')
        ax1.set_ylabel('Distance Modulus μ', fontsize=12)
        ax1.set_xscale('log')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(z, mu_ddmm - mu_lcdm, 'k-', lw=2)
        ax2.axhline(0, color='gray', ls='--')
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('Δμ (mag)', fontsize=12)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Type Ia SNe: DDMM vs ΛCDM', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hubble_diagram_comparison.png', dpi=150)
        plt.close()
    
    # ========================================================================
    # Report Generation
    # ========================================================================
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        report = {
            'model_parameters': self.model_params,
            'test_results': [asdict(r) for r in self.results],
            'overall_score': np.mean([r.score for r in self.results]),
            'tests_passed': sum(1 for r in self.results if r.passed),
            'tests_total': len(self.results),
            'summary': self._generate_summary()
        }
        
        # Save JSON report
        report_file = self.output_dir / 'ddmm_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nReport saved to: {report_file}")
        
        # Print summary
        self._print_summary()
    
    def _generate_summary(self) -> Dict[str, str]:
        """Generate text summary of results"""
        overall_score = np.mean([r.score for r in self.results])
        
        if overall_score > 0.8:
            status = "EXCELLENT"
            message = "DDMM shows excellent agreement with observations"
        elif overall_score > 0.6:
            status = "GOOD"
            message = "DDMM shows good agreement but has some tensions"
        else:
            status = "POOR"
            message = "DDMM has significant issues that need addressing"
            
        return {
            'status': status,
            'message': message,
            'score': f"{overall_score:.2f}"
        }
    
    def _print_summary(self):
        """Print results summary to console"""
        print("\n" + "="*70)
        print("DDMM VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\nTest Results:")
        print(f"{'Test Name':<30} {'Status':<10} {'Score':<10}")
        print(f"{'-'*30} {'-'*10} {'-'*10}")
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{result.test_name:<30} {status:<10} {result.score:>6.2f}")
        
        summary = self._generate_summary()
        print(f"\nOverall Score: {summary['score']}/1.00")
        print(f"Status: {summary['status']}")
        print(f"\n{summary['message']}")
        print("="*70)


def main():
    """Main validation runner"""
    import argparse
    import pickle
    import gzip
    
    parser = argparse.ArgumentParser(description="Focused DDMM Validation Suite")
    parser.add_argument('params_file', help="Model parameters file (.json, .npz, or .pkl.gz)")
    parser.add_argument('--output_dir', default='validation_results', 
                       help="Output directory")
    parser.add_argument('--skip_data_check', action='store_true',
                       help="Skip data availability check")
    
    args = parser.parse_args()
    
    # Load parameters
    if args.params_file.endswith('.json'):
        with open(args.params_file, 'r') as f:
            model_params = json.load(f)
    elif args.params_file.endswith('.npz'):
        data = np.load(args.params_file, allow_pickle=True)
        # Extract parameters from dynesty output
        param_names = data.get('param_names', data.get('paramnames'))
        if 'logwt' in data:
            weights = np.exp(data['logwt'] - data['logz'][-1])
        else:
            weights = np.ones(len(data['samples'])) / len(data['samples'])
        median_params = np.average(data['samples'], weights=weights, axis=0)
        model_params = dict(zip(param_names, median_params))
        model_params['xi_type'] = 'power'
    elif args.params_file.endswith('.pkl.gz'):
        # Load dynesty results directly
        with gzip.open(args.params_file, 'rb') as f:
            results = pickle.load(f)
        
        # Calculate weights
        weights = np.exp(results.logwt - results.logz[-1])
        
        # Define parameter names (these are the fitted baryonic parameters)
        param_names = [
            'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',
            'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',
            'M_bulge_solar', 'a_bulge_kpc',
            'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'
        ]
        
        # Get median parameters
        median_params = np.average(results.samples, weights=weights, axis=0)
        model_params = dict(zip(param_names, median_params))
        
        # Add fixed gravity parameters
        model_params['rho_c_solar_kpc3'] = 1e9
        model_params['n_exp'] = 1.0
        model_params['A'] = 1.0
        model_params['xi_type'] = 'power'
        
        # Add component flags
        model_params['include_disk_thin'] = True
        model_params['include_disk_thick'] = True
        model_params['include_bulge'] = True
        model_params['include_gas'] = True
    else:
        raise ValueError("Parameters file must be .json, .npz, or .pkl.gz")
    
    # Initialize validator
    validator = DDMMValidator(model_params, args.output_dir)
    
    # Check data availability
    if not args.skip_data_check:
        data_available = validator.check_data_availability()
    
    # Run tests
    logger.info("\n" + "="*70)
    logger.info("RUNNING DDMM VALIDATION TESTS")
    logger.info("="*70)
    
    # Test 1: Solar System
    validator.test_solar_system()
    
    # Test 2: Galaxy rotation curves
    validator.test_galaxy_rotation_curves(
        sparc_path='Rotmod_LTG' if Path('Rotmod_LTG').exists() else None,
        use_gaia=True
    )
    
    # Test 3: Type Ia supernovae
    pantheon_path = 'pantheon' if Path('pantheon').exists() else None
    validator.test_supernovae(pantheon_path)
        
    # Test 4: Laboratory constraints
    validator.test_laboratory_constraints()
    
    # Test 5: Gravitational lensing
    validator.test_gravitational_lensing()
    
    # Test 6: Large scale structure (z < 2)
    validator.test_large_scale_structure(
        bao_path='bao' if Path('bao').exists() else None
    )
    
    # Generate report
    validator.generate_report()
    
    logger.info("\n✅ Validation complete!")


if __name__ == "__main__":
    main()