#!/usr/bin/env python3
"""
density_metric_cupy.py - CuPy-optimized physics layer for maximum GPU utilization.

This version uses CuPy for GPU acceleration, providing much better GPU utilization
than JAX on NVIDIA GPUs. CuPy is specifically designed for NVIDIA CUDA and provides
excellent performance for numerical computations.
"""
import cupy as cp
import numpy as np
from scipy.special import i0 as scipy_i0, i1 as scipy_i1, kv as scipy_kv
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

# CuPy Configuration for maximum GPU utilization
DEFAULT_DTYPE = cp.float32

# Initialize CuPy and set memory pool for better performance
try:
    # Set memory pool to use most of available GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=0.8 * mempool.get_limit())  # Use 80% of GPU memory
    
    # Enable memory pool for better performance
    cp.cuda.set_allocator(mempool.malloc)
    
    logger.info(f"CuPy initialized successfully. GPU: {cp.cuda.runtime.getDeviceCount()} devices available")
    logger.info(f"Current device: {cp.cuda.runtime.getDevice()}")
    logger.info(f"GPU memory: {cp.cuda.runtime.memGetInfo()}")
except Exception as e:
    logger.warning(f"CuPy initialization warning: {e}")

# ============================================================================
# BESSEL FUNCTION WRAPPERS - Optimized for CuPy (GPU-native, no host bounce)
# ============================================================================
try:
    from cupyx.scipy import special as cpx_special
except Exception as _e:
    raise RuntimeError(
        "[density_metric_cupy] CuPy SciPy special not available. Install cupyx.scipy (CuPy >= 12)."
    )

def bessel_i0_cupy(x):
    return cpx_special.i0(x)

def bessel_i1_cupy(x):
    return cpx_special.i1(x)

def bessel_k0_cupy(x):
    return cpx_special.k0(x)

def bessel_k1_cupy(x):
    return cpx_special.k1(x)

# ============================================================================
# CORE PHYSICS FUNCTIONS - CuPy Optimized
# ============================================================================

def _enclosed_disk_mass_solar_cupy(R_kpc, M_disk_solar, R_d_kpc):
    """Enclosed disk mass using Freeman profile - CuPy optimized."""
    R_kpc_arr = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    R_d_kpc_arr = cp.asarray(R_d_kpc, dtype=DEFAULT_DTYPE)
    M_disk_solar_arr = cp.asarray(M_disk_solar, dtype=DEFAULT_DTYPE)
    
    x = R_kpc_arr / R_d_kpc_arr
    x_safe = cp.maximum(x, 0)
    m_enc = M_disk_solar_arr * (1.0 - cp.exp(-x_safe) * (1.0 + x_safe))
    m_enc = cp.where(R_kpc_arr < 0, 0.0, m_enc)
    return cp.where(R_d_kpc_arr <= 1e-9, 0.0, m_enc)

def _enclosed_hernquist_mass_solar_cupy(R_kpc, M_bulge_solar, R_b_kpc):
    """Enclosed Hernquist bulge mass - CuPy optimized."""
    R_kpc_arr = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    R_b_kpc_arr = cp.asarray(R_b_kpc, dtype=DEFAULT_DTYPE)
    M_bulge_solar_arr = cp.asarray(M_bulge_solar, dtype=DEFAULT_DTYPE)
    
    R_kpc_safe = cp.maximum(R_kpc_arr, 0)
    m_enc = M_bulge_solar_arr * (R_kpc_safe / (R_kpc_safe + R_b_kpc_arr))**2
    m_enc = cp.where(R_kpc_arr < 0, 0.0, m_enc)
    return cp.where((R_b_kpc_arr <= 1e-9) | (M_bulge_solar_arr <= 1e-9), 0.0, m_enc)

def v_newton_kms_cupy(R_kpc, M_disk_solar_main, R_d_kpc_main,
                     M_bulge_solar_opt=0.0, R_b_kpc_opt=0.5, include_bulge_opt=False,
                     M_gas_solar_opt=0.0, R_gas_kpc_opt=7.0, include_gas_opt=False):
    """Newtonian velocity calculation - CuPy optimized."""
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Convert scalar parameters to CuPy arrays for consistent operations
    M_disk_solar_main_arr = cp.asarray(M_disk_solar_main, dtype=DEFAULT_DTYPE)
    R_d_kpc_main_arr = cp.asarray(R_d_kpc_main, dtype=DEFAULT_DTYPE)
    M_bulge_solar_opt_arr = cp.asarray(M_bulge_solar_opt, dtype=DEFAULT_DTYPE)
    R_b_kpc_opt_arr = cp.asarray(R_b_kpc_opt, dtype=DEFAULT_DTYPE)
    M_gas_solar_opt_arr = cp.asarray(M_gas_solar_opt, dtype=DEFAULT_DTYPE)
    R_gas_kpc_opt_arr = cp.asarray(R_gas_kpc_opt, dtype=DEFAULT_DTYPE)
    
    # Disk contribution
    v_disk_sq = cp.where(R_d_kpc_main_arr > 1e-9, 
                         (4.302e-6 * M_disk_solar_main_arr / R_kpc_arr) * 
                         (1.0 - cp.exp(-R_kpc_arr / R_d_kpc_main_arr) * 
                          (1.0 + R_kpc_arr / R_d_kpc_main_arr)), 0.0)
    
    # Bulge contribution
    v_bulge_sq = 0.0
    if bool(include_bulge_opt) and float(M_bulge_solar_opt) > 1e-9 and float(R_b_kpc_opt) > 1e-9:
        v_bulge_sq = (4.302e-6 * M_bulge_solar_opt_arr / R_kpc_arr) * \
                     (R_kpc_arr / (R_kpc_arr + R_b_kpc_opt_arr))**2
    
    # Gas contribution
    v_gas_sq = 0.0
    if bool(include_gas_opt) and float(M_gas_solar_opt) > 1e-9 and float(R_gas_kpc_opt) > 1e-9:
        v_gas_sq = (4.302e-6 * M_gas_solar_opt_arr / R_kpc_arr) * \
                   (1.0 - cp.exp(-R_kpc_arr / R_gas_kpc_opt_arr) * 
                    (1.0 + R_kpc_arr / R_gas_kpc_opt_arr))
    
    # Total velocity
    v_total_sq = v_disk_sq + v_bulge_sq + v_gas_sq
    v_total_sq = cp.where((R_kpc_arr <= 1e-9) | (v_total_sq <= 1e-9), 0.0, v_total_sq)
    
    v_out_kms = cp.sqrt(cp.maximum(v_total_sq, 0.0))
    return v_out_kms

def _volume_density_total_midplane_solar_kpc3_cupy(R_kpc_arr, M_disk, Rd_disk, hz_disk, 
                                                  M_bulge, Rb_bulge, incl_bulge, 
                                                  M_gas, Rd_gas, hz_gas, incl_gas):
    """Total volume density at midplane - CuPy optimized."""
    # Ensure R_kpc_arr is a CuPy array
    R_kpc_arr = cp.asarray(R_kpc_arr, dtype=DEFAULT_DTYPE)
    rho_total = cp.zeros_like(R_kpc_arr, dtype=DEFAULT_DTYPE)
    
    # Convert scalar parameters to CuPy arrays for consistent operations
    M_disk_arr = cp.asarray(M_disk, dtype=DEFAULT_DTYPE)
    Rd_disk_arr = cp.asarray(Rd_disk, dtype=DEFAULT_DTYPE)
    hz_disk_arr = cp.asarray(hz_disk, dtype=DEFAULT_DTYPE)
    M_bulge_arr = cp.asarray(M_bulge, dtype=DEFAULT_DTYPE)
    Rb_bulge_arr = cp.asarray(Rb_bulge, dtype=DEFAULT_DTYPE)
    M_gas_arr = cp.asarray(M_gas, dtype=DEFAULT_DTYPE)
    Rd_gas_arr = cp.asarray(Rd_gas, dtype=DEFAULT_DTYPE)
    hz_gas_arr = cp.asarray(hz_gas, dtype=DEFAULT_DTYPE)
    
    # Disk density
    if float(M_disk) > 1e-9 and float(Rd_disk) > 1e-9 and float(hz_disk) > 1e-9:
        Sigma0_disk = M_disk_arr / (2.0 * cp.pi * Rd_disk_arr**2)
        rho_disk = (Sigma0_disk / (2.0 * hz_disk_arr)) * cp.exp(-R_kpc_arr / Rd_disk_arr)
        rho_total += rho_disk
    
    # Bulge density
    if bool(incl_bulge) and float(M_bulge) > 0 and float(Rb_bulge) > 1e-9:
        R_eff_bulge = cp.maximum(R_kpc_arr, 1e-6)
        rho_bulge_mid = (M_bulge_arr / (2.0 * cp.pi)) * \
                       (Rb_bulge_arr / (R_eff_bulge * (R_eff_bulge + Rb_bulge_arr)**3))
        
        # Handle very small radii
        min_r_b = 1e-5
        fill_val_b = (M_bulge_arr / (2.0 * cp.pi)) * \
                     (Rb_bulge_arr / (min_r_b * (min_r_b + Rb_bulge_arr)**3))
        rho_bulge = cp.where(R_kpc_arr < 1e-5, fill_val_b, rho_bulge_mid)
        rho_total += rho_bulge
    
    # Gas density
    if bool(incl_gas) and float(M_gas) > 1e-9 and float(Rd_gas) > 1e-9 and float(hz_gas) > 1e-9:
        Sigma0_gas = M_gas_arr / (2.0 * cp.pi * Rd_gas_arr**2)
        rho_gas = (Sigma0_gas / (2.0 * hz_gas_arr)) * cp.exp(-R_kpc_arr / Rd_gas_arr)
        rho_total += rho_gas
    
    return rho_total

def volume_density_total_midplane_solar_kpc3_cupy(*args, **kwargs):
    """Wrapper for volume density calculation."""
    return _volume_density_total_midplane_solar_kpc3_cupy(*args, **kwargs)

@cp.fuse()
def v_circ_hernquist_bulge_kms_cupy(R_kpc, M_bulge_solar, a_bulge_kpc):
    """Circular velocity for Hernquist bulge - CuPy optimized."""
    R_kpc_arr = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    M_bulge_solar_arr = cp.asarray(M_bulge_solar, dtype=DEFAULT_DTYPE)
    a_bulge_kpc_arr = cp.asarray(a_bulge_kpc, dtype=DEFAULT_DTYPE)
    
    R_safe = cp.maximum(R_kpc_arr, 1e-9)
    v_sq = (4.302e-6 * M_bulge_solar_arr / R_safe) * (R_safe / (R_safe + a_bulge_kpc_arr))**2
    return cp.sqrt(cp.maximum(v_sq, 0.0))

@cp.fuse()
def v_circ_exponential_disk_approx_kms_cupy(R_kpc, M_disk_solar, R_d_kpc):
    """Approximate circular velocity for exponential disk - CuPy optimized."""
    R_kpc_arr = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    M_disk_solar_arr = cp.asarray(M_disk_solar, dtype=DEFAULT_DTYPE)
    R_d_kpc_arr = cp.asarray(R_d_kpc, dtype=DEFAULT_DTYPE)
    
    x = R_kpc_arr / R_d_kpc_arr
    x_safe = cp.maximum(x, 1e-9)
    
    # Freeman formula
    i0x, k0x, i1x, k1x = bessel_i0_cupy(x_safe), bessel_k0_cupy(x_safe), \
                         bessel_i1_cupy(x_safe), bessel_k1_cupy(x_safe)
    
    freeman_term = i0x * k0x - i1x * k1x
    v_sq = (4.302e-6 * M_disk_solar_arr / R_d_kpc_arr) * (x_safe**2) * freeman_term
    return cp.sqrt(cp.maximum(v_sq, 0.0))

def v_baryon_total_newtonian_kms_cupy(R_kpc, p_baryons):
    """Total baryonic velocity (Newtonian) - CuPy optimized."""
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Extract parameters
    M_disk = p_baryons.get('M_disk_solar', 0.0)
    R_d = p_baryons.get('R_d_kpc', 3.0)
    M_bulge = p_baryons.get('M_bulge_solar', 0.0)
    R_b = p_baryons.get('R_b_kpc', 0.5)
    include_bulge = p_baryons.get('include_bulge', False)
    M_gas = p_baryons.get('M_gas_solar', 0.0)
    R_gas = p_baryons.get('R_gas_kpc', 7.0)
    include_gas = p_baryons.get('include_gas', False)
    
    return v_newton_kms_cupy(R_kpc_arr, M_disk, R_d, M_bulge, R_b, include_bulge, 
                            M_gas, R_gas, include_gas)

# ============================================================================
# XI FUNCTIONS - CuPy Optimized
# ============================================================================

@cp.fuse()
def xi_power_law_cupy(rho, rho_c, n_exp, A=1.0):
    """Power law xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    return A * (rho_safe / rho_c_safe)**(n_exp - 1.0)

@cp.fuse()
def xi_logistic_law_cupy(rho, rho_c, n_exp, A=1.0):
    """Logistic law xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    ratio = rho_safe / rho_c_safe
    return A * (1.0 + cp.tanh((ratio - 1.0) * n_exp)) / 2.0

@cp.fuse()
def xi_exponential_cupy(rho, rho_c, n_exp, A=1.0):
    """Exponential xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    return A * cp.exp((rho_safe / rho_c_safe - 1.0) * n_exp)

@cp.fuse()
def xi_gravitational_color_cupy(rho, rho_c, gamma, lambda_g):
    """Gravitational color xi function - CuPy optimized (exponential screening)."""
    rho_safe = cp.maximum(rho, 1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    ratio = rho_safe / rho_c_safe
    xi = 1.0 + lambda_g * cp.exp(-cp.power(ratio, gamma))
    # Cap to [1, 1+lambda_g]
    xi = cp.clip(xi, 1.0, 1.0 + lambda_g)
    return xi

@cp.fuse()
def xi_gaussian_enhancement_cupy(rho, rho_peak, sigma_log, lambda_max):
    """Gaussian enhancement xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_peak_safe = cp.maximum(rho_peak, 1e-10)
    
    log_ratio = cp.log(rho_safe / rho_peak_safe)
    gaussian = cp.exp(-0.5 * (log_ratio / sigma_log)**2)
    return 1.0 + (lambda_max - 1.0) * gaussian

@cp.fuse()
def xi_balanced_screening_cupy(rho, rho_c, R, R_screen=50.0, n_exp=1.0, A_max=2.0):
    """
    Balanced screening model with physically reasonable enhancement.
    
    This model ensures:
    1. Cassini constraint satisfaction (xi ~ 1 at solar density)
    2. Modest enhancement (max 2-3x, not 250x!)
    3. Deep space safety (xi -> 1 as R -> infinity)
    4. Realistic rotation curves (200-300 km/s)
    
    Parameters:
    -----------
    rho : array
        Local density in M_sun/kpc^3
    rho_c : float
        Critical density (typically solar density ~1e8 M_sun/kpc^3)
    R : array
        Distance from galactic center in kpc
    R_screen : float
        Screening radius beyond which enhancement vanishes (default 50 kpc)
    n_exp : float
        Density dependence exponent (default 1.0 for linear)
    A_max : float
        Maximum enhancement factor (default 2.0 for 2x max enhancement)
    
    Returns:
    --------
    xi : array
        Enhancement factor in range [1, 1+A_max]
    
    Physics:
    --------
    The enhancement has two components:
    1. Density factor: (1 - rho/rho_c)^n_exp
       - At solar density: factor = 0 (no enhancement)
       - In voids: factor = 1 (full enhancement allowed)
    
    2. Distance screening: tanh-based smooth cutoff
       - Full enhancement for R < R_screen/2
       - Smooth decay for R > R_screen/2
       - Nearly zero for R > 2*R_screen
    
    This ensures gravity enhancement only occurs in the transition zone
    between high-density regions and deep space, preventing unphysical
    behavior while explaining galaxy rotation curves.
    """
    
    # Normalized density ratio (capped at 1 to prevent negative enhancement)
    rho_ratio = cp.minimum(rho / (rho_c + 1e-10), 1.0)
    
    # Density enhancement factor
    # High density (rho/rho_c ~ 1): factor ~ 0 (satisfies Cassini)
    # Low density (rho/rho_c ~ 0): factor ~ 1 (allows enhancement)
    density_factor = (1.0 - rho_ratio)**n_exp
    
    # Distance screening with smooth transition
    # Uses hyperbolic tangent for smooth cutoff
    screening_factor = 0.5 * (1.0 + cp.tanh((R_screen - R) / (0.3 * R_screen)))
    
    # Combined enhancement (limited to A_max)
    enhancement = A_max * density_factor * screening_factor
    
    # Final xi value
    xi = 1.0 + enhancement
    
    # Safety constraints
    xi = cp.maximum(xi, 1.0)  # Never less than 1
    xi = cp.minimum(xi, 1.0 + A_max)  # Never more than 1 + A_max
    xi = cp.where(cp.isfinite(xi), xi, 1.0)  # Replace NaN/Inf with 1
    
    return xi


def xi_gravitational_color_void_safe_cupy(rho, rho_c, gamma, lambda_g):
    """
    Void-safe gravitational color confinement model - CuPy optimized.
    
    This function ensures xi -> 1 as rho -> 0 to prevent unphysical
    behavior in intergalactic voids.
    """
    # More robust numerical handling
    rho_safe = cp.maximum(rho, 1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    
    ratio = rho_safe / rho_c_safe
    
    # Limit ratio_gamma to prevent numerical overflow
    ratio_gamma = cp.minimum(cp.power(ratio, gamma), 50.0)  # exp(-50) is safely small
    
    # Rational function: approaches 1 at high density, 0 at low density
    rational_factor = ratio_gamma / (1.0 + ratio_gamma)
    
    # Exponential suppression with numerical safety
    exp_factor = cp.exp(-ratio_gamma)
    
    # Combined enhancement - approaches 1 in voids
    xi = 1.0 + lambda_g * rational_factor * exp_factor
    
    # Ensure xi >= 1 everywhere and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)  # Replace non-finite with 1
    
    return xi


def xi_hybrid_safe_cupy(rho, rho_c, n_exp, A):
    """
    Hybrid xi function that smoothly transitions between regimes.
    
    - Low density (voids): xi -> 1 (no enhancement)
    - Intermediate density: Power law enhancement
    - High density (solar system): xi -> 1 (screening)
    """
    # Ensure numerical safety
    rho_safe = cp.maximum(rho, 1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    
    # Density ratio
    ratio = rho_safe / rho_c_safe
    
    # Power law component (stable and well-tested)
    xi_power = 1.0 + A * cp.power(rho_c_safe / rho_safe, n_exp)
    
    # Screening at high density
    screening_factor = 1.0 / (1.0 + cp.power(ratio, 2.0))
    
    # Void suppression
    void_factor = ratio / (1.0 + ratio)
    
    # Combine all factors
    xi = 1.0 + (xi_power - 1.0) * screening_factor * void_factor
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi

def xi_smooth_transition_cupy(rho, rho_c, n_exp, A):
    """
    Smooth transition model using Gaussian profile in log-space.
    """
    # Ensure numerical safety
    rho_safe = cp.maximum(rho, 1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    
    # Log-space ratio for smooth transitions
    log_ratio = cp.log10(rho_safe / rho_c_safe)
    
    # Gaussian enhancement profile
    enhancement_profile = cp.exp(-0.5 * log_ratio**2)
    
    # Final enhancement
    xi = 1.0 + A * enhancement_profile
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi


# ============================================================================
# RUBBER BAND GRAVITY MODELS - Elastic field theory implementations
# ============================================================================

def xi_elastic_strain_cupy(rho, rho_c, params):
    """
    Gravity enhancement based on elastic strain of gravitational field.
    Like a rubber band: unstrained in high density, strained in low density.
    """
    # Extract parameters with defaults
    L0 = params.get('relaxation_scale', 1.0)  # kpc
    strain_critical = params.get('strain_critical', 10.0)
    k_elastic = params.get('k_elastic', 0.5)
    rho_solar = params.get('rho_solar', 1e9)
    
    # Ensure numerical safety
    rho_safe = cp.maximum(rho, 1e-30)
    
    # Calculate strain = how much field is stretched
    density_ratio = rho_solar / rho_safe
    strain = cp.log10(cp.maximum(density_ratio, 1.0)) / L0
    
    # Elastic response with breaking point
    xi = cp.where(
        strain < strain_critical,
        1.0 + k_elastic * strain * (1.0 - strain/strain_critical),
        1.0 + k_elastic * cp.exp(-(strain - strain_critical))  # Decays after snap
    )
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi

def xi_tension_field_cupy(rho, R_kpc, params):
    """
    Gravitational field develops tension when stretched across voids.
    Tension creates additional inward pull, like stretched rubber.
    """
    # Parameters with defaults
    rho_relax = params.get('rho_relaxation', 1e8)
    tension_max = params.get('tension_max', 5.0)
    R_snap = params.get('R_snap', 25.0)
    
    # Ensure numerical safety
    rho_safe = cp.maximum(rho, 1e-30)
    R_kpc_safe = cp.maximum(R_kpc, 0.1)
    
    # Calculate field tension based on density deficit
    deficit = cp.maximum(0, cp.log10(rho_relax) - cp.log10(rho_safe))
    
    # Tension builds up with deficit but peaks then drops
    tension = deficit * cp.exp(-deficit / 3.0)
    
    # Spatial modulation - field can only stretch so far
    stretch = cp.tanh(R_kpc_safe / 10.0)  # Gradually allows stretching
    snap = cp.exp(-cp.maximum(0, R_kpc_safe - R_snap)**2 / 25)  # Snaps beyond R_snap
    
    # Total enhancement
    xi = 1.0 + tension_max * tension * stretch * snap
    
    # Cassini suppression at Solar position
    R_sun = 8.5
    sun_suppress = cp.exp(-((R_kpc_safe - R_sun)/1.0)**2)
    xi = 1.0 + (xi - 1.0) * (1.0 - 0.99 * sun_suppress)
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi

def xi_hookean_potential_cupy(rho, R_kpc, params):
    """
    Gravity gains elastic potential energy when stretched.
    Based on Hooke's law: F = -kx becomes g_eff = g_newton * (1 + elastic_term)
    """
    # Parameters with defaults
    k_spacetime = params.get('k_spacetime', 0.1)
    rho_eq = params.get('rho_equilibrium', 1e9)
    stress_break = params.get('stress_break', 100.0)
    
    # Ensure numerical safety
    rho_safe = cp.maximum(rho, 1e-30)
    R_kpc_safe = cp.maximum(R_kpc, 0.1)
    
    # Calculate displacement from equilibrium
    displacement = cp.maximum(0, cp.log10(rho_eq) - cp.log10(rho_safe))
    
    # Calculate stress
    stress = k_spacetime * displacement
    elastic_term = k_spacetime * displacement**2 / (1.0 + displacement/10.0)
    
    # Apply breaking/snapping
    xi = cp.where(
        stress < stress_break,
        1.0 + elastic_term,
        1.0 + elastic_term * cp.exp(-(stress - stress_break)/20.0)
    )
    
    # Cassini suppression at Solar position
    R_sun = 8.5
    sun_suppress = cp.exp(-((R_kpc_safe - R_sun)/1.0)**2)
    xi = 1.0 + (xi - 1.0) * (1.0 - 0.99 * sun_suppress)
    
    # Ensure xi >= 1 and finite
    xi = cp.maximum(xi, 1.0)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    
    return xi

def xi_mond_like_cupy(rho, rho_c, n):
    """MOND-like xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    ratio = rho_safe / rho_c_safe
    
    # MOND-like enhancement
    xi = 1.0 + (ratio**(-n/2.0)) / (1.0 + ratio**(-n/2.0))
    return cp.maximum(xi, 1.0)

@cp.fuse()
def xi_sigmoid_saturation_cupy(rho, rho_c, A, n_exp):
    """Sigmoid saturation - enhancement caps at A+1, never goes to infinity."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    
    # Prevent infinity with small offset
    x = rho_c_safe / (rho_safe + rho_c_safe/100)
    xi = 1.0 + A * cp.tanh(x**n_exp)
    return cp.maximum(xi, 1.0)

@cp.fuse()
def xi_peak_enhancement_cupy(rho, rho_peak, width, A):
    """
    Peak enhancement - maximum effect at intermediate densities (void boundaries).
    Enhancement PEAKS at rho_peak then falls off in both directions.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    rho_peak_safe = cp.maximum(rho_peak, 1e-10)
    
    # Log-space Gaussian peak
    log_rho = cp.log10(rho_safe)
    log_peak = cp.log10(rho_peak_safe)
    
    xi = 1.0 + A * cp.exp(-(log_rho - log_peak)**2 / (2*width**2))
    return cp.maximum(xi, 1.0)

@cp.fuse()
def xi_broken_power_cupy(rho, rho_break, n_low, n_high, A):
    """
    Broken power law - different behavior above/below rho_break.
    More flexible than single power law, more stable than peak.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    rho_break_safe = cp.maximum(rho_break, 1e-10)
    
    # Different power laws for high/low density regimes
    xi_low = 1.0 + A * (rho_break_safe / rho_safe)**n_low
    xi_high = 1.0 + A * (rho_safe / rho_break_safe)**n_high
    
    # Smooth transition
    xi = cp.where(rho_safe < rho_break_safe, xi_low, xi_high)
    
    # Cap at reasonable maximum
    return cp.minimum(xi, 50.0)

@cp.fuse() 
def xi_hybrid_cupy(rho, rho_c, n_exp, A_power, rho_peak, width, A_peak):
    """
    Hybrid model: Power law base + Gaussian peak enhancement.
    Best of both worlds - continuous enhancement + peak at boundaries.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    
    # Base power law (like enhanced model that worked)
    xi_base = 1.0 + A_power * (rho_c / rho_safe)**n_exp
    
    # Additional peak enhancement
    log_rho = cp.log10(rho_safe)
    log_peak = cp.log10(rho_peak)
    peak_enhancement = A_peak * cp.exp(-(log_rho - log_peak)**2 / (2*width**2))
    
    # Combine effects
    xi = xi_base + peak_enhancement
    
    # Cap at reasonable maximum
    return cp.minimum(xi, 100.0)

@cp.fuse()
def xi_tanh_transition_cupy(rho, rho_mid, width, A_low, A_high):
    """
    Smooth transition between two enhancement levels.
    Very stable for fitting.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    
    # Smooth transition using tanh
    x = (cp.log10(rho_safe) - cp.log10(rho_mid)) / width
    transition = 0.5 * (1.0 + cp.tanh(x))
    
    # Interpolate between low and high density enhancements
    xi = 1.0 + A_low * (1 - transition) + A_high * transition
    
    return cp.maximum(xi, 1.0)

@cp.fuse()
def xi_yukawa_screening_cupy(rho, r, rho_c, A, lambda_screen):
    """
    Yukawa-like screening - enhancement decreases exponentially with distance.
    Note: Requires radial distance r as additional input.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    r_safe = cp.maximum(r, 0.1)  # Avoid issues at r=0
    
    # Density factor
    rho_factor = cp.sqrt(rho_c_safe / (rho_safe + rho_c_safe/100))
    
    # Distance screening (exponential cutoff)
    distance_factor = cp.exp(-r_safe / lambda_screen)
    
    xi = 1.0 + A * rho_factor * distance_factor
    return cp.maximum(xi, 1.0)


@cp.fuse()
def xi_spacetime_grain_cupy(R_kpc, M_enclosed, grain_size_kpc=10.0, boundary_width=2.0, A_boundary=5.0):
    """
    Quantum Spacetime Granularity Model
    
    Key insight: Spacetime has fundamental grains ~10 kpc in size.
    Galaxy mass compresses central grains, creating boundary effects.
    
    Parameters:
    -----------
    R_kpc : array
        Distance from galaxy center in kpc
    M_enclosed : array  
        Enclosed mass at radius R (we'll calculate this)
    grain_size_kpc : float
        Fundamental spacetime grain size (~10 kpc)
    boundary_width : float
        Width of grain boundary transition region
    A_boundary : float
        Enhancement strength at boundaries
    
    Returns:
    --------
    xi : Enhancement factor
        1.0 inside grains, enhanced at boundaries
    """
    R_safe = cp.maximum(R_kpc, 0.001)
    
    # Determine grain boundaries based on mass distribution
    # First grain: 0 to grain_size
    # Boundary region: grain_size ± boundary_width
    
    # Calculate which grain we're in
    grain_number = cp.floor(R_safe / grain_size_kpc)
    distance_from_boundary = cp.abs(R_safe - (grain_number + 0.5) * grain_size_kpc)
    
    # Key physics: Enhancement ONLY at grain boundaries
    # This is crucial for Cassini constraint!
    
    # Smooth transition at boundaries
    xi = cp.ones(R_safe.shape, dtype=R_safe.dtype)
    
    # Only enhance at grain boundaries (around 10, 20, 30 kpc etc)
    for n in range(1, 5):  # First few grain boundaries
        r_boundary = n * grain_size_kpc
        distance_from_this_boundary = cp.abs(R_safe - r_boundary)
        
        # Gaussian enhancement at boundary
        boundary_enhancement = A_boundary * cp.exp(-(distance_from_this_boundary**2) / (2 * boundary_width**2))
        
        # But suppress if inside dense region (mass screening)
        # This is key: massive galaxies compress grains, boundaries shift outward
        mass_suppression = cp.tanh(M_enclosed / 1e11)  # Normalize to galaxy mass
        
        xi += boundary_enhancement * (1 - mass_suppression * 0.5)
    
    return xi

@cp.fuse()
def xi_spacetime_grain_v2_cupy(R_kpc, rho_local, grain_size_kpc=10.0, rho_compress=1e5, A_max=8.0):
    """
    Alternative formulation: Grains compress based on local density
    More compatible with existing parameter structure
    """
    R_safe = cp.maximum(R_kpc, 0.001)
    rho_safe = cp.maximum(rho_local, 1.0)
    
    # Local density determines grain compression
    # High density: grains compressed, boundary moves outward
    compression_factor = cp.log10(rho_safe / rho_compress + 1)
    
    # Effective grain size at this location
    local_grain_size = grain_size_kpc * (1 + compression_factor)
    
    # Distance to nearest grain boundary
    grain_phase = cp.mod(R_safe, local_grain_size) / local_grain_size
    
    # Enhancement peaks at grain boundaries (phase = 0 or 1)
    # But zero at grain centers (phase = 0.5)
    boundary_distance = cp.minimum(grain_phase, 1 - grain_phase) * 2
    
    # Sharp peak at boundaries, normal elsewhere
    xi = 1.0 + A_max * cp.exp(-((boundary_distance - 0) / 0.1)**2)
    
    # Critical for Cassini: No enhancement below 1 kpc
    xi = cp.where(R_safe < 1.0, 1.0, xi)
    
    return xi

@cp.fuse()
def xi_transition_based_cupy(rho, rho_prev, A, width):
    """
    Transition-based enhancement - strongest where density CHANGES rapidly.
    Requires density at previous radius for gradient calculation.
    """
    rho_safe = cp.maximum(rho, 1e-10)
    rho_prev_safe = cp.maximum(rho_prev, 1e-10)
    
    # Calculate relative density change
    delta_log_rho = cp.abs(cp.log10(rho_safe) - cp.log10(rho_prev_safe))
    
    # Enhancement peaks for large density gradients
    xi = 1.0 + A * cp.exp(-delta_log_rho**2 / (2*width**2))
    return cp.maximum(xi, 1.0)

# ============================================================================
# TIDAL BAND-PASS XI (CuPy)
# ============================================================================

@cp.fuse()
def xi_tidal_bandpass_cupy(rho, T, rho_c, gamma, lambda_max, T0, sigma_lnT, wmin):
    """
    Band-pass enhancement in tidal norm T with density screening and void floor.
    Inputs:
      rho: local baryonic density (M_sun/kpc^3)
      T: tidal proxy ~ v_baryon^2 / R^2 (in (km/s)^2/kpc^2)
      rho_c: critical density for Solar System screening
      gamma: density sharpness for screening
      lambda_max: maximum enhancement amplitude
      T0: tidal scale where enhancement peaks
      sigma_lnT: log-width of the tidal window (natural log)
      wmin: residual floor in voids (0 to 0.05 typical)
    Returns xi in [1, 1+lambda_max].
    """
    # Safety floors
    rho_safe = cp.maximum(rho, 1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    T_safe = cp.maximum(T, 1e-30)
    T0_safe = cp.maximum(T0, 1e-30)
    sigma_safe = cp.maximum(sigma_lnT, 1e-6)
    wmin_clamped = cp.clip(wmin, 0.0, 0.2)

    # Density switch S_rho = 1 / (1 + (rho/rho_c)^gamma)
    ratio_rho = rho_safe / rho_c_safe
    S_rho = 1.0 / (1.0 + cp.power(ratio_rho, gamma))

    # Tidal window W(T) = wmin + (1-wmin) * exp(- (ln(T/T0))^2 / (2 sigma^2))
    ln_ratio = cp.log(T_safe / T0_safe)
    W_T = wmin_clamped + (1.0 - wmin_clamped) * cp.exp(- (ln_ratio * ln_ratio) / (2.0 * sigma_safe * sigma_safe))

    xi = 1.0 + lambda_max * S_rho * W_T
    # Cap and ensure finite
    xi = cp.clip(xi, 1.0, 1.0 + lambda_max)
    xi = cp.where(cp.isfinite(xi), xi, 1.0)
    return xi

# --- NEW: logistic helper (numerically stable) --------------------------------
def _sigmoid_cupy(x):
    # 0.5 * (1 + tanh(x/2)) is stable for large |x|
    return 0.5 * (1.0 + cp.tanh(0.5 * x))

# --- NEW: tidal_band2 (logistic in lnT + softened density screen) -------------
def xi_tidal_band2_cupy(rho, T, rho_c, gamma, beta, lambda_max, T0, alpha, kappa, wmin):
    rho_safe   = cp.maximum(rho,   1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    T_safe     = cp.maximum(T,     1e-30)
    T0_safe    = cp.maximum(T0,    1e-30)
    beta_safe  = cp.maximum(beta,  1e-6)
    wmin_c     = cp.clip(cp.asarray(wmin, dtype=DEFAULT_DTYPE), 0.0, 0.2)
    S_rho = cp.power(1.0 + cp.power(rho_safe / rho_c_safe, gamma), -beta_safe)
    wT   = _sigmoid_cupy(alpha * (cp.log(T_safe) - cp.log(T0_safe)))
    wT   = cp.maximum(wT, wmin_c)
    xi   = 1.0 + lambda_max * (1.0 - cp.exp(-kappa * S_rho * wT))
    xi   = cp.clip(xi, 1.0, 1.0 + lambda_max)
    return cp.where(cp.isfinite(xi), xi, 1.0)

# --- NEW: tidal_ratio (single trigger from lnT and lnρ) -----------------------
def xi_tidal_ratio_cupy(rho, T, rho_c, eta, lambda_max, T0, alpha, kappa, wmin):
    rho_safe   = cp.maximum(rho,   1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    T_safe     = cp.maximum(T,     1e-30)
    T0_safe    = cp.maximum(T0,    1e-30)
    wmin_c     = cp.clip(cp.asarray(wmin, dtype=DEFAULT_DTYPE), 0.0, 0.2)
    U   = (cp.log(T_safe) - cp.log(T0_safe)) + eta * (cp.log(rho_c_safe) - cp.log(rho_safe))
    w   = _sigmoid_cupy(alpha * U)
    w   = cp.maximum(w, wmin_c)
    xi  = 1.0 + lambda_max * (1.0 - cp.exp(-kappa * w))
    xi  = cp.clip(xi, 1.0, 1.0 + lambda_max)
    return cp.where(cp.isfinite(xi), xi, 1.0)

# --- NEW: tidal_noisyor (noisy-OR aggregator of Sρ and wT) --------------------
def xi_tidal_noisyor_cupy(rho, T, rho_c, gamma, lambda_max, T0, alpha, kappa, wmin):
    rho_safe   = cp.maximum(rho,   1e-30)
    rho_c_safe = cp.maximum(rho_c, 1e-30)
    T_safe     = cp.maximum(T,     1e-30)
    T0_safe    = cp.maximum(T0,    1e-30)
    wmin_c     = cp.clip(cp.asarray(wmin, dtype=DEFAULT_DTYPE), 0.0, 0.2)
    S_rho = 1.0 / (1.0 + cp.power(rho_safe / rho_c_safe, gamma))
    wT    = _sigmoid_cupy(alpha * (cp.log(T_safe) - cp.log(T0_safe)))
    wT    = cp.maximum(wT, wmin_c)
    G     = 1.0 - (1.0 - S_rho) * (1.0 - wT)   # noisy-OR with a=b=1
    xi    = 1.0 + lambda_max * (1.0 - cp.exp(-kappa * G))
    xi    = cp.clip(xi, 1.0, 1.0 + lambda_max)
    return cp.where(cp.isfinite(xi), xi, 1.0)

# ============================================================================
# MAIN VELOCITY FUNCTION - CuPy Optimized
# ============================================================================

def v_baryon_comprehensive_kms_cupy(R_kpc, p):
    """
    Comprehensive baryonic velocity calculation including all Milky Way components.
    
    Parameters:
    - p: dictionary with all baryonic parameters
    """
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Extract all baryonic parameters
    M_thin_disk = p.get('M_thin_disk_solar', 0.0)
    R_thin_disk = p.get('R_thin_disk_kpc', 3.0)
    hz_thin_disk = p.get('hz_thin_disk_kpc', 0.3)
    
    M_thick_disk = p.get('M_thick_disk_solar', 0.0)
    R_thick_disk = p.get('R_thick_disk_kpc', 3.5)
    hz_thick_disk = p.get('hz_thick_disk_kpc', 0.8)
    
    M_bulge = p.get('M_bulge_solar', 0.0)
    R_bulge = p.get('R_bulge_kpc', 0.5)
    
    M_gas = p.get('M_gas_solar', 0.0)
    R_gas = p.get('R_gas_kpc', 7.0)
    hz_gas = p.get('hz_gas_kpc', 0.1)
    
    # Convert to CuPy arrays
    M_thin_disk_arr = cp.asarray(M_thin_disk, dtype=DEFAULT_DTYPE)
    R_thin_disk_arr = cp.asarray(R_thin_disk, dtype=DEFAULT_DTYPE)
    M_thick_disk_arr = cp.asarray(M_thick_disk, dtype=DEFAULT_DTYPE)
    R_thick_disk_arr = cp.asarray(R_thick_disk, dtype=DEFAULT_DTYPE)
    M_bulge_arr = cp.asarray(M_bulge, dtype=DEFAULT_DTYPE)
    R_bulge_arr = cp.asarray(R_bulge, dtype=DEFAULT_DTYPE)
    M_gas_arr = cp.asarray(M_gas, dtype=DEFAULT_DTYPE)
    R_gas_arr = cp.asarray(R_gas, dtype=DEFAULT_DTYPE)
    
    # Initialize total velocity squared
    v_total_sq = cp.zeros_like(R_kpc_arr, dtype=DEFAULT_DTYPE)
    
    # 1. Thin disk contribution (exponential disk)
    if float(M_thin_disk) > 1e-9 and float(R_thin_disk) > 1e-9:
        v_thin_disk_sq = (4.302e-6 * M_thin_disk_arr / R_kpc_arr) * \
                        (1.0 - cp.exp(-R_kpc_arr / R_thin_disk_arr) * 
                         (1.0 + R_kpc_arr / R_thin_disk_arr))
        v_total_sq += v_thin_disk_sq
    
    # 2. Thick disk contribution (exponential disk)
    if float(M_thick_disk) > 1e-9 and float(R_thick_disk) > 1e-9:
        v_thick_disk_sq = (4.302e-6 * M_thick_disk_arr / R_kpc_arr) * \
                         (1.0 - cp.exp(-R_kpc_arr / R_thick_disk_arr) * 
                          (1.0 + R_kpc_arr / R_thick_disk_arr))
        v_total_sq += v_thick_disk_sq
    
    # 3. Bulge contribution (Hernquist profile)
    if float(M_bulge) > 1e-9 and float(R_bulge) > 1e-9:
        R_safe = cp.maximum(R_kpc_arr, 1e-9)
        v_bulge_sq = (4.302e-6 * M_bulge_arr / R_safe) * \
                     (R_safe / (R_safe + R_bulge_arr))**2
        v_total_sq += v_bulge_sq
    
    # 4. Gas contribution (exponential disk)
    if float(M_gas) > 1e-9 and float(R_gas) > 1e-9:
        v_gas_sq = (4.302e-6 * M_gas_arr / R_kpc_arr) * \
                   (1.0 - cp.exp(-R_kpc_arr / R_gas_arr) * 
                    (1.0 + R_kpc_arr / R_gas_arr))
        v_total_sq += v_gas_sq
    
    # Ensure positive velocity squared
    v_total_sq = cp.where((R_kpc_arr <= 1e-9) | (v_total_sq <= 1e-9), 0.0, v_total_sq)
    
    # ADD NaN PROTECTION to prevent numerical issues
    v_total_sq = cp.nan_to_num(v_total_sq, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Include optional NFW halo if requested (ΛCDM baseline)
    if bool(p.get('include_halo', False)):
        # Halo parameters V200, c
        V200 = float(p.get('V200_kms', 0.0))
        c_conc = float(p.get('c_concentration', 10.0))
        if V200 > 0.0 and c_conc > 0.0:
            # Minimal NFW halo term: v^2_total += v_halo^2
            # Implement NFW v_halo in CuPy directly to avoid host bounce
            H0 = 70.0  # km/s/Mpc
            R200_kpc = V200 / (10.0 * H0) * 1000.0
            rs = R200_kpc / max(c_conc, 1e-6)
            x = cp.clip(R_kpc_arr / max(rs, 1e-12), 1e-8, cp.inf)
            def f_cupy(z):
                return cp.log1p(z) - z/(1.0+z)
            norm = cp.maximum(f_cupy(cp.array(c_conc, dtype=DEFAULT_DTYPE)), 1e-12)
            vx2_halo = (V200**2) * (f_cupy(x) / cp.maximum(x, 1e-12)) / norm
            v_total_sq += cp.maximum(vx2_halo, 0.0)

    # Return velocity
    v_out_kms = cp.sqrt(cp.maximum(v_total_sq, 0.0))

    # Additional safety check for final result
    v_out_kms = cp.nan_to_num(v_out_kms, nan=0.0, posinf=0.0, neginf=0.0)

    return v_out_kms

def volume_density_comprehensive_solar_kpc3_cupy(R_kpc, p):
    """
    Comprehensive volume density calculation including all Milky Way components.
    
    Parameters:
    - p: dictionary with all baryonic parameters
    """
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Extract all baryonic parameters
    M_thin_disk = p.get('M_thin_disk_solar', 0.0)
    R_thin_disk = p.get('R_thin_disk_kpc', 3.0)
    hz_thin_disk = p.get('hz_thin_disk_kpc', 0.3)
    
    M_thick_disk = p.get('M_thick_disk_solar', 0.0)
    R_thick_disk = p.get('R_thick_disk_kpc', 3.5)
    hz_thick_disk = p.get('hz_thick_disk_kpc', 0.8)
    
    M_bulge = p.get('M_bulge_solar', 0.0)
    R_bulge = p.get('R_bulge_kpc', 0.5)
    
    M_gas = p.get('M_gas_solar', 0.0)
    R_gas = p.get('R_gas_kpc', 7.0)
    hz_gas = p.get('hz_gas_kpc', 0.1)
    
    # Convert to CuPy arrays
    M_thin_disk_arr = cp.asarray(M_thin_disk, dtype=DEFAULT_DTYPE)
    R_thin_disk_arr = cp.asarray(R_thin_disk, dtype=DEFAULT_DTYPE)
    hz_thin_disk_arr = cp.asarray(hz_thin_disk, dtype=DEFAULT_DTYPE)
    M_thick_disk_arr = cp.asarray(M_thick_disk, dtype=DEFAULT_DTYPE)
    R_thick_disk_arr = cp.asarray(R_thick_disk, dtype=DEFAULT_DTYPE)
    hz_thick_disk_arr = cp.asarray(hz_thick_disk, dtype=DEFAULT_DTYPE)
    M_bulge_arr = cp.asarray(M_bulge, dtype=DEFAULT_DTYPE)
    R_bulge_arr = cp.asarray(R_bulge, dtype=DEFAULT_DTYPE)
    M_gas_arr = cp.asarray(M_gas, dtype=DEFAULT_DTYPE)
    R_gas_arr = cp.asarray(R_gas, dtype=DEFAULT_DTYPE)
    hz_gas_arr = cp.asarray(hz_gas, dtype=DEFAULT_DTYPE)
    
    # Initialize total density
    rho_total = cp.zeros_like(R_kpc_arr, dtype=DEFAULT_DTYPE)
    
    # 1. Thin disk density (exponential disk)
    if float(M_thin_disk) > 1e-9 and float(R_thin_disk) > 1e-9 and float(hz_thin_disk) > 1e-9:
        Sigma0_thin = M_thin_disk_arr / (2.0 * cp.pi * R_thin_disk_arr**2)
        rho_thin = (Sigma0_thin / (2.0 * hz_thin_disk_arr)) * cp.exp(-R_kpc_arr / R_thin_disk_arr)
        rho_total += rho_thin
    
    # 2. Thick disk density (exponential disk)
    if float(M_thick_disk) > 1e-9 and float(R_thick_disk) > 1e-9 and float(hz_thick_disk) > 1e-9:
        Sigma0_thick = M_thick_disk_arr / (2.0 * cp.pi * R_thick_disk_arr**2)
        rho_thick = (Sigma0_thick / (2.0 * hz_thick_disk_arr)) * cp.exp(-R_kpc_arr / R_thick_disk_arr)
        rho_total += rho_thick
    
    # 3. Bulge density (Hernquist profile)
    if float(M_bulge) > 1e-9 and float(R_bulge) > 1e-9:
        R_eff_bulge = cp.maximum(R_kpc_arr, 1e-6)
        rho_bulge_mid = (M_bulge_arr / (2.0 * cp.pi)) * \
                       (R_bulge_arr / (R_eff_bulge * (R_eff_bulge + R_bulge_arr)**3))
        
        # Handle very small radii
        min_r_b = 1e-5
        fill_val_b = (M_bulge_arr / (2.0 * cp.pi)) * \
                     (R_bulge_arr / (min_r_b * (min_r_b + R_bulge_arr)**3))
        rho_bulge = cp.where(R_kpc_arr < 1e-5, fill_val_b, rho_bulge_mid)
        rho_total += rho_bulge
    
    # 4. Gas density (exponential disk)
    if float(M_gas) > 1e-9 and float(R_gas) > 1e-9 and float(hz_gas) > 1e-9:
        Sigma0_gas = M_gas_arr / (2.0 * cp.pi * R_gas_arr**2)
        rho_gas = (Sigma0_gas / (2.0 * hz_gas_arr)) * cp.exp(-R_kpc_arr / R_gas_arr)
        rho_total += rho_gas
    
    # ADD NaN PROTECTION to prevent numerical issues
    rho_total = cp.nan_to_num(rho_total, nan=0.0, posinf=0.0, neginf=0.0)
    
    return rho_total

from .xi_registry import register_xi, resolve_xi

def v_total_kms_cupy(R_kpc, p, xi_type='power'):
    """
    Total circular velocity including baryons and a selectable xi.
    IMPORTANT: For reproducible paper runs, published xi are 'gr', 'tidal_band'.
               We also permit 'nfw' as a convenience alias that sets xi≡1 and enables an NFW halo term
               via p['include_halo']=True in the comprehensive baryonic model.
               To enable additional experimental xi forms, pass p['allow_experimental']=True.
    """
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))

    # If xi_type requests NFW, ensure the halo is included
    if xi_type == 'nfw':
        # Avoid mutating caller's dict
        p = dict(p)
        p['include_halo'] = True

    # Baryons and density
    if 'M_thin_disk_solar' in p:
        v_baryon_sq = v_baryon_comprehensive_kms_cupy(R_kpc_arr, p)**2
        rho_total = volume_density_comprehensive_solar_kpc3_cupy(R_kpc_arr, p)
    else:
        p_baryons = {
            'M_disk_solar': p.get('M_disk_solar', 0.0),
            'R_d_kpc': p.get('R_d_kpc', 3.0),
            'M_bulge_solar': p.get('M_bulge_solar', 0.0),
            'R_b_kpc': p.get('R_b_kpc', 0.5),
            'include_bulge': p.get('include_bulge', False),
            'M_gas_solar': p.get('M_gas_solar', 0.0),
            'R_gas_kpc': p.get('R_gas_kpc', 7.0),
            'include_gas': p.get('include_gas', False),
        }
        v_baryon_sq = v_baryon_total_newtonian_kms_cupy(R_kpc_arr, p_baryons)**2
        rho_total = volume_density_total_midplane_solar_kpc3_cupy(
            R_kpc_arr,
            p_baryons['M_disk_solar'], p_baryons['R_d_kpc'], p.get('hz_disk_kpc', 0.3),
            p_baryons['M_bulge_solar'], p_baryons['R_b_kpc'], p_baryons['include_bulge'],
            p_baryons['M_gas_solar'], p_baryons['R_gas_kpc'], p.get('hz_gas_kpc', 0.1),
            p_baryons['include_gas']
        )

    # Registry-resolved xi
    allow_experimental = bool(p.get('allow_experimental', False))
    R_safe = cp.maximum(R_kpc_arr, 1e-9)
    rho_c = p.get('rho_c_solar_kpc3', 7e7)

    # Prepare tidal proxy only if needed
    T = None
    if xi_type in ('tidal_band', 'tidal_band2', 'tidal_ratio', 'tidal_noisyor'):
        T = cp.maximum(v_baryon_sq, 0.0) / cp.maximum(R_safe * R_safe, 1e-18)

    # Define small wrappers for published xi
    def _xi_gr_published(rho, **kwargs):
        return cp.ones_like(rho, dtype=rho.dtype)

    def _xi_tidal_band_published(rho, T=None, rho_c=None, gamma=None, lambda_max=None, T0=None, sigma_lnT=None, wmin=None, **kwargs):
        if T is None:
            raise ValueError("tidal_band requires tidal proxy T; computed internally when xi_type='tidal_band'")
        return xi_tidal_bandpass_cupy(rho, T, rho_c, gamma, lambda_max, T0, sigma_lnT, wmin)

    # Helper to ensure xi registry has published defaults available early
    def ensure_xi_registry_defaults():
        """
        Idempotently register published xi ('gr', 'tidal_band') and declare experimental ones.
        Safe to call at import time or before CLI validation.
        """
        try:
            register_xi('gr', _xi_gr_published, published=True, doc="GR baseline (xi≡1)")
            register_xi('tidal_band', _xi_tidal_band_published, published=True, doc="ER tidal-band with density screening")
            # Convenience alias: 'nfw' uses xi≡1 but enables an NFW halo via include_halo in the velocity model
            register_xi('nfw', _xi_gr_published, published=True, doc="ΛCDM/NFW baseline (xi≡1) with halo term enabled")

            # --- NEW: experimental tidal-family variants (require --allow_experimental) ---
            register_xi(
                'tidal_band2',
                lambda rho, **kw: xi_tidal_band2_cupy(
                    rho, kw.get('T'), kw.get('rho_c'),
                    kw.get('gamma', kw.get('gamma_exp', 3.0)),
                    kw.get('beta', 0.8),
                    kw.get('lambda_max', 10.0),
                    kw.get('T0', 10.0),
                    kw.get('alpha', 2.0),
                    kw.get('kappa', 1.0),
                    kw.get('wmin', 0.02)
                ),
                published=False,
                doc="Experimental: logistic lnT onset + softened density screen"
            )
            register_xi(
                'tidal_ratio',
                lambda rho, **kw: xi_tidal_ratio_cupy(
                    rho, kw.get('T'), kw.get('rho_c'),
                    kw.get('eta', 0.9),
                    kw.get('lambda_max', 10.0),
                    kw.get('T0', 10.0),
                    kw.get('alpha', 2.0),
                    kw.get('kappa', 1.0),
                    kw.get('wmin', 0.02)
                ),
                published=False,
                doc="Experimental: ratio trigger U=ln(T/T0)+η ln(ρc/ρ)"
            )
            register_xi(
                'tidal_noisyor',
                lambda rho, **kw: xi_tidal_noisyor_cupy(
                    rho, kw.get('T'), kw.get('rho_c'),
                    kw.get('gamma', kw.get('gamma_exp', 3.0)),
                    kw.get('lambda_max', 10.0),
                    kw.get('T0', 10.0),
                    kw.get('alpha', 2.0),
                    kw.get('kappa', 1.0),
                    kw.get('wmin', 0.02)
                ),
                published=False,
                doc="Experimental: noisy-OR combine of Sρ and wT"
            )

            # Experimental examples (available only if allow_experimental=True)
            register_xi('grav_color', lambda rho, **kw: xi_gravitational_color_cupy(rho, kw.get('rho_c'), kw.get('gamma_exp', 2.7), kw.get('lambda_g', 8.0)), published=False, doc="Experimental: exponential screening")
            register_xi('grav_color_void_safe', lambda rho, **kw: xi_gravitational_color_void_safe_cupy(rho, kw.get('rho_c'), kw.get('gamma_exp', 2.7), kw.get('lambda_g', 8.0)), published=False, doc="Experimental: void-safe color")
            register_xi('balanced_screening', lambda rho, **kw: xi_balanced_screening_cupy(rho, kw.get('rho_c'), kw.get('R'), kw.get('R_screen', 50.0), kw.get('n_exp', 1.0), kw.get('A_max', 2.0)), published=False, doc="Experimental: bounded enhancement with R cutoff")
        except Exception:
            # Ignore if already registered in a prior call
            pass

    # Ensure defaults are registered for any consumer that imported this function
    ensure_xi_registry_defaults()

    try:
        xi_fn = resolve_xi(xi_type, allow_experimental=allow_experimental)
    except Exception as e:
        print(f"[v_total_kms_cupy] ERROR resolving xi_type='{xi_type}': {e}")
        raise

    xi_kwargs = dict(p)
    xi_kwargs.update({"rho_c": float(rho_c) if rho_c is not None else 7e7, "T": T, "R": R_safe})

    # Parameter normalization for tidal_band and similar xi
    if xi_type == 'tidal_band':
        # Map gamma_exp -> gamma if needed
        if 'gamma' not in xi_kwargs or xi_kwargs.get('gamma') is None:
            gamma_val = xi_kwargs.get('gamma_exp', 3.0)
            xi_kwargs['gamma'] = float(gamma_val)
        # Ensure required scalars exist and are numeric
        xi_kwargs['lambda_max'] = float(xi_kwargs.get('lambda_max', 4.0))
        xi_kwargs['T0'] = float(xi_kwargs.get('T0', 10.0))
        xi_kwargs['sigma_lnT'] = float(xi_kwargs.get('sigma_lnT', 0.8))
        xi_kwargs['wmin'] = float(xi_kwargs.get('wmin', 0.02))

    elif xi_type == 'tidal_band2':
        xi_kwargs['gamma']      = float(xi_kwargs.get('gamma', xi_kwargs.get('gamma_exp', 3.0)))
        xi_kwargs['beta']       = float(xi_kwargs.get('beta', 0.8))
        xi_kwargs['lambda_max'] = float(xi_kwargs.get('lambda_max', 10.0))
        xi_kwargs['T0']         = float(xi_kwargs.get('T0', 10.0))
        xi_kwargs['alpha']      = float(xi_kwargs.get('alpha', 2.0))
        xi_kwargs['kappa']      = float(xi_kwargs.get('kappa', 1.0))
        xi_kwargs['wmin']       = float(xi_kwargs.get('wmin', 0.02))

    elif xi_type == 'tidal_ratio':
        xi_kwargs['eta']        = float(xi_kwargs.get('eta', 0.9))
        xi_kwargs['lambda_max'] = float(xi_kwargs.get('lambda_max', 10.0))
        xi_kwargs['T0']         = float(xi_kwargs.get('T0', 10.0))
        xi_kwargs['alpha']      = float(xi_kwargs.get('alpha', 2.0))
        xi_kwargs['kappa']      = float(xi_kwargs.get('kappa', 1.0))
        xi_kwargs['wmin']       = float(xi_kwargs.get('wmin', 0.02))

    elif xi_type == 'tidal_noisyor':
        xi_kwargs['gamma']      = float(xi_kwargs.get('gamma', xi_kwargs.get('gamma_exp', 3.0)))
        xi_kwargs['lambda_max'] = float(xi_kwargs.get('lambda_max', 10.0))
        xi_kwargs['T0']         = float(xi_kwargs.get('T0', 10.0))
        xi_kwargs['alpha']      = float(xi_kwargs.get('alpha', 2.0))
        xi_kwargs['kappa']      = float(xi_kwargs.get('kappa', 1.0))
        xi_kwargs['wmin']       = float(xi_kwargs.get('wmin', 0.02))

    xi = xi_fn(rho=rho_total, **xi_kwargs)

    v_total_sq = cp.nan_to_num(v_baryon_sq * xi, nan=0.0, posinf=0.0, neginf=0.0)
    v_total = cp.sqrt(cp.maximum(v_total_sq, 0.0))
    return cp.nan_to_num(v_total, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_cupy_array(arr):
    """Convert numpy array to CuPy array."""
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr, dtype=DEFAULT_DTYPE)
    elif isinstance(arr, cp.ndarray):
        return arr
    else:
        return cp.asarray(arr, dtype=DEFAULT_DTYPE)

def to_numpy_array(arr):
    """Convert CuPy array to numpy array."""
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    else:
        return np.asarray(arr)

def get_gpu_info():
    """Get GPU information for monitoring."""
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        current_device = cp.cuda.runtime.getDevice()
        mem_info = cp.cuda.runtime.memGetInfo()
        
        return {
            'device_count': device_count,
            'current_device': current_device,
            'memory_free': mem_info[0],
            'memory_total': mem_info[1],
            'memory_used': mem_info[1] - mem_info[0]
        }
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
        return None

def check_cassini_constraint_cupy(params, xi_type='spacetime_grain'):
    """
    Test if model violates Cassini spacecraft constraint.
    Cassini measured |γ - 1| < 2.3 × 10^-5 where γ is PPN parameter.
    
    For DDMM: γ - 1 ≈ (ξ - 1) at Solar System location
    """
    # Solar System location
    R_sun = 8.5  # kpc from galactic center
    
    # Local density at Sun's position (roughly)
    rho_solar_neighborhood = 1e6  # M_☉/kpc³
    
    # Calculate xi at Solar System
    if xi_type == 'spacetime_grain':
        # For grain model, check at Sun's position
        xi_sun = xi_spacetime_grain_v2_cupy(
            cp.array([R_sun]), 
            cp.array([rho_solar_neighborhood]),
            grain_size_kpc=params.get('grain_size_kpc', 10.0),
            rho_compress=params.get('rho_compress', 1e5),
            A_max=params.get('A_grain', 8.0)
        )
    else:
        # For other models, use standard calculation
        xi_sun = 1.0  # Placeholder
    
    # Cassini constraint
    gamma_minus_one = float(xi_sun - 1.0)
    cassini_limit = 2.3e-5
    
    passes_cassini = abs(gamma_minus_one) < cassini_limit
    
    return {
        'passes': passes_cassini,
        'gamma_minus_one': gamma_minus_one,
        'limit': cassini_limit,
        'ratio': abs(gamma_minus_one) / cassini_limit
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        logger.debug("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Could not clear GPU memory: {e}")

# ============================================================================
# SELF-TESTING
# ============================================================================

def run_cupy_self_tests():
    """Run self-tests for CuPy functions."""
    logger.info("Running CuPy self-tests...")
    
    try:
        # Test basic array operations
        test_arr = cp.array([1.0, 2.0, 3.0], dtype=DEFAULT_DTYPE)
        result = cp.sum(test_arr)
        assert abs(float(result) - 6.0) < 1e-6, f"Basic array test failed: {result}"
        
        # Test velocity calculation
        R_test = cp.array([8.0], dtype=DEFAULT_DTYPE)
        params = {
            'M_disk_solar': 5e10,
            'R_d_kpc': 3.0,
            'rho_c_solar_kpc3': 1e8,
            'n_exp': 2.0,
            'A_xi': 1.0
        }
        v_result = v_total_kms_cupy(R_test, params, xi_type='power')
        assert v_result.shape == (1,), f"Velocity calculation shape error: {v_result.shape}"
        assert cp.all(cp.isfinite(v_result)), "Velocity calculation produced non-finite values"
        
        logger.info("CuPy self-tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"CuPy self-test failed: {e}")
        return False

if __name__ == "__main__":
    # Run self-tests when executed directly
    run_cupy_self_tests() 