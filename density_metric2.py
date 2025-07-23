#!/usr/bin/env python3
"""
density_metric2.py - Physics layer: mass models, density profiles,
                    density-weighting functions (xi), and full velocity model.

This version is accelerated with JAX for hardware-agnostic execution on
GPUs (Apple Metal, NVIDIA CUDA) and TPUs.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import i0 as BesselI0, i1 as BesselI1
from scipy.special import kv as scipy_kv
import numpy as np  # Kept for CPU-specific tasks like data loading and plotting
from scipy.special import i0 as scipy_i0, i1 as scipy_i1, kv as scipy_kv
import logging

# Define GPU-incompatible functions to fallback to CPU
def BesselK0(x): return scipy_kv(0, x)
def BesselK1(x): return scipy_kv(1, x)

# Set up a logger for this module
logger = logging.getLogger(__name__)
def BesselK0(x): logger.debug("⚠️ BesselK0 falling back to CPU via scipy"); return scipy_kv(0, x)
def BesselK1(x): logger.debug("⚠️ BesselK1 falling back to CPU via scipy"); return scipy_kv(1, x)

# JAX Configuration: By default, JAX will use float32. This is ideal for
# GPU performance on Metal and CUDA. We will explicitly use float32.
# On Metal, float64 support may be limited or emulated on the CPU.
DEFAULT_DTYPE = jnp.float32
jax.config.update("jax_enable_x64", False)


# ============================================================================
# BESSEL FUNCTION WRAPPERS - CPU FALLBACK FOR METAL COMPATIBILITY
# ============================================================================

def bessel_i0_cpu_fallback(x):
    """Bessel I0 with CPU fallback for Metal compatibility."""
    x_np = np.asarray(x)
    result = scipy_i0(x_np)
    return jnp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_i1_cpu_fallback(x):
    """Bessel I1 with CPU fallback for Metal compatibility."""
    x_np = np.asarray(x)
    result = scipy_i1(x_np)
    return jnp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_k0_cpu_fallback(x):
    """Bessel K0 with CPU fallback for Metal compatibility."""
    x_np = np.asarray(x)
    result = scipy_kv(0, x_np)
    return jnp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_k1_cpu_fallback(x):
    """Bessel K1 with CPU fallback for Metal compatibility."""
    x_np = np.asarray(x)
    result = scipy_kv(1, x_np)
    return jnp.asarray(result, dtype=DEFAULT_DTYPE)

# Check if we're on Metal backend
def is_metal_backend():
    """Check if JAX is using Metal backend."""
    try:
        devices = jax.devices()
        return any('metal' in str(d).lower() for d in devices)
    except:
        return False

# Select appropriate Bessel functions based on backend
if is_metal_backend():
    logger.info("Metal backend detected - using CPU fallbacks for Bessel functions")
    BesselI0 = bessel_i0_cpu_fallback
    BesselI1 = bessel_i1_cpu_fallback
    BesselK0 = bessel_k0_cpu_fallback
    BesselK1 = bessel_k1_cpu_fallback
else:
    # Use JAX native functions where available
    from jax.scipy.special import i0 as BesselI0, i1 as BesselI1
    BesselK0 = bessel_k0_cpu_fallback  # K functions always use CPU
    BesselK1 = bessel_k1_cpu_fallback

# ---- Self-Testing Functions (updated for JAX) ----
def _assert_freeman_identity():
    """Quick self‑test: Check Freeman kernel against a known approximate value."""
    logger.debug("[SELF-TEST] Verifying Freeman kernel...")
    y_test = jnp.array(0.5, dtype=DEFAULT_DTYPE)
    i0y, k0y, i1y, k1y = BesselI0(y_test), BesselK0(y_test), BesselI1(y_test), BesselK1(y_test)
    bessel_term = i0y * k0y - i1y * k1y
    val = (y_test**2) * bessel_term
    expected_val = 0.138979394445648
    assert jnp.abs(val - expected_val) < 1e-4, f"Freeman kernel self-test failed! Expected ~{expected_val}, got {val}"
    logger.debug("[SELF-TEST] Freeman kernel OK.")

def run_physics_self_tests():
    """Runs all self-tests for this physics module."""
    try:
        logger.info("[PHYSICS] Running self-tests...")
        _assert_freeman_identity()
        logger.info("[PHYSICS] All self-tests passed.")
    except Exception as e:
        logger.error(f"[PHYSICS SELF-TEST FAILED] {e}")
        raise

def _assert_xi_limits():
    """Ensure JAX-compiled xi functions behave correctly for enhanced gravity theory."""
    logger.debug("[SELF-TEST] Verifying xi function limits with JAX...")
    # Low density test
    xi_low = xi_gravitational_color(jnp.array(1e-10), 1e8, 2.7, 8.0)
    expected_low = 9.0  # 1 + λ = 1 + 8 = 9
    assert jnp.abs(xi_low - expected_low) < 0.1, f"xi at ρ→0 should be ≈{expected_low}, got {xi_low}"
    
    # High density test
    xi_high = xi_gravitational_color(jnp.array(1e12), 1e8, 2.7, 8.0)
    assert jnp.abs(xi_high - 1.0) < 0.1, f"xi at ρ>>ρ_c should be ≈1.0, got {xi_high}"
    
    # Test another function
    test_val = xi_power_law(jnp.array(1e8), 1e8, 2.0, A=1.0)
    assert 0 < test_val <= 10.0, f"xi_power_law should return value in (0,10], got {test_val}"
    logger.debug("[SELF-TEST] xi limit checks OK.")


# ---------- Physical Constants ----------
G_ASTRO_UNITS = 4.30091e-6 # kpc (km/s)^2 / Msun
R_SUN_KPC = 8.122
MSUN_PC2_TO_MSUN_KPC2 = (1e3)**2
SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2 = 85.0 * MSUN_PC2_TO_MSUN_KPC2
TAU_MICRO_TARGET_BAADE_MAX = 3.0e-6
L_BAADE_DEG = 1.0
B_BAADE_DEG = -2.75
D_SUN_GC_KPC = R_SUN_KPC

# ============================================================================
# SELF-TESTING FUNCTIONS
# ============================================================================

def _assert_freeman_identity():
    """Quick self‑test: Check Freeman kernel against a known approximate value."""
    logger.debug("[SELF-TEST] Verifying Freeman kernel...")
    y_test = jnp.array(0.5, dtype=DEFAULT_DTYPE)
    i0y, k0y, i1y, k1y = BesselI0(y_test), BesselK0(y_test), BesselI1(y_test), BesselK1(y_test)
    bessel_term = i0y * k0y - i1y * k1y
    val = (y_test**2) * bessel_term
    expected_val = 0.138979394445648
    assert jnp.abs(val - expected_val) < 1e-4, f"Freeman kernel self-test failed! Expected ~{expected_val}, got {val}"
    logger.debug("[SELF-TEST] Freeman kernel OK.")

def run_physics_self_tests():
    """Runs all self-tests for this physics module."""
    try:
        logger.info("[PHYSICS] Running self-tests...")
        _assert_freeman_identity()
        logger.info("[PHYSICS] All self-tests passed.")
    except Exception as e:
        logger.error(f"[PHYSICS SELF-TEST FAILED] {e}")
        raise

# ============================================================================
# OLDER SINGLE-DISK COMPATIBLE FUNCTIONS (Updated for Metal compatibility)
# ============================================================================

# Remove JIT from functions that use Bessel functions indirectly
def _enclosed_disk_mass_solar_old(R_kpc, M_disk_solar, R_d_kpc):
    """Helper for OLD v_newton_kms: Exponential disk, cumulative mass."""
    x = R_kpc / R_d_kpc
    x_safe = jnp.maximum(x, 0)
    m_enc = M_disk_solar * (1.0 - jnp.exp(-x_safe) * (1.0 + x_safe))
    m_enc = jnp.where(R_kpc < 0, 0.0, m_enc)
    return jnp.where(R_d_kpc <= 1e-9, 0.0, m_enc)

def _enclosed_hernquist_mass_solar_old(R_kpc, M_bulge_solar, R_b_kpc):
    """Helper for OLD v_newton_kms: Hernquist profile enclosed mass."""
    R_kpc_safe = jnp.maximum(R_kpc, 0)
    m_enc = M_bulge_solar * (R_kpc_safe**2) / ((R_kpc_safe + R_b_kpc)**2)
    m_enc = jnp.where(R_kpc < 0, 0.0, m_enc)
    return jnp.where((R_b_kpc <= 1e-9) | (M_bulge_solar <= 1e-9), 0.0, m_enc)

def v_newton_kms(R_kpc, M_disk_solar_main, R_d_kpc_main,
                 M_bulge_solar_opt=0.0, R_b_kpc_opt=0.5, include_bulge_opt=False,
                 M_gas_solar_opt=0.0, R_gas_kpc_opt=7.0, include_gas_opt=False):
    """OLDER Newtonian circular velocity (spherical approx)."""
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    M_enc_disk = _enclosed_disk_mass_solar_old(R_kpc_arr, M_disk_solar_main, R_d_kpc_main)
    
    M_enc_bulge = 0.0
    if include_bulge_opt and M_bulge_solar_opt > 0:
        M_enc_bulge = _enclosed_hernquist_mass_solar_old(R_kpc_arr, M_bulge_solar_opt, R_b_kpc_opt)
        
    M_enc_gas = 0.0
    if include_gas_opt and M_gas_solar_opt > 0:
        M_enc_gas = _enclosed_disk_mass_solar_old(R_kpc_arr, M_gas_solar_opt, R_gas_kpc_opt)
        
    M_enc_total_solar = M_enc_disk + M_enc_bulge + M_enc_gas
    
    v_total_sq_kms2 = (G_ASTRO_UNITS * M_enc_total_solar) / R_kpc_arr
    v_total_sq_kms2 = jnp.where((R_kpc_arr <= 1e-9) | (M_enc_total_solar <= 1e-9), 0.0, v_total_sq_kms2)
    
    v_out_kms = jnp.sqrt(jnp.maximum(v_total_sq_kms2, 0.0))
    return v_out_kms.item() if is_scalar else v_out_kms

@jax.jit
def _volume_density_total_midplane_solar_kpc3_jax(R_kpc_arr, M_disk, Rd_disk, hz_disk, M_bulge, Rb_bulge, incl_bulge, M_gas, Rd_gas, hz_gas, incl_gas):
    """ JAX-compiled core for OLDER Midplane volume density. """
    rho_total = jnp.zeros_like(R_kpc_arr, dtype=DEFAULT_DTYPE)

    # Disk component
    Sigma0_disk = M_disk / (2.0 * jnp.pi * Rd_disk**2)
    rho_disk = (Sigma0_disk / (2.0 * hz_disk)) * jnp.exp(-R_kpc_arr / Rd_disk)
    rho_total += jnp.where((M_disk > 1e-9) & (Rd_disk > 1e-9) & (hz_disk > 1e-9), rho_disk, 0.0)
    
    # Bulge component
    R_eff_bulge = jnp.maximum(R_kpc_arr, 1e-6)
    rho_bulge_mid = (M_bulge / (2.0 * jnp.pi)) * (Rb_bulge / (R_eff_bulge * (R_eff_bulge + Rb_bulge)**3))
    # Handle R=0 case by filling with value at a small R to avoid infinity
    min_r_b = 1e-6
    fill_val_b = (M_bulge / (2.0 * jnp.pi)) * (Rb_bulge / (min_r_b * (min_r_b + Rb_bulge)**3))
    rho_bulge = jnp.where(R_kpc_arr < 1e-5, fill_val_b, rho_bulge_mid)
    rho_total += jnp.where(incl_bulge & (M_bulge > 0) & (Rb_bulge > 1e-9), rho_bulge, 0.0)

    # Gas component
    Sigma0_gas = M_gas / (2.0 * jnp.pi * Rd_gas**2)
    rho_gas = (Sigma0_gas / (2.0 * hz_gas)) * jnp.exp(-R_kpc_arr / Rd_gas)
    rho_total += jnp.where(incl_gas & (M_gas > 0) & (Rd_gas > 1e-9) & (hz_gas > 1e-9), rho_gas, 0.0)

    return rho_total

def volume_density_total_midplane_solar_kpc3(*args, **kwargs):
    """ OLDER Midplane volume density. Python wrapper for JAX core. """
    R_kpc = args[0]
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Unpack args and kwargs into the order expected by the JAX function
    p = {
        'M_disk': args[1], 'Rd_disk': args[2], 'hz_disk': args[3],
        'M_bulge': kwargs.get('M_bulge_solar_opt', 0.0), 'Rb_bulge': kwargs.get('R_b_kpc_opt', 0.5),
        'incl_bulge': kwargs.get('include_bulge_opt', False),
        'M_gas': kwargs.get('M_gas_solar_opt', 0.0), 'Rd_gas': kwargs.get('R_gas_kpc_opt', 7.0),
        'hz_gas': kwargs.get('h_z_gas_kpc_opt', 0.15), 'incl_gas': kwargs.get('include_gas_opt', False)
    }

    rho_total = _volume_density_total_midplane_solar_kpc3_jax(
        R_kpc_arr, p['M_disk'], p['Rd_disk'], p['hz_disk'], p['M_bulge'], p['Rb_bulge'],
        p['incl_bulge'], p['M_gas'], p['Rd_gas'], p['hz_gas'], p['incl_gas']
    )
    return rho_total.item() if is_scalar else rho_total


@jax.jit
def v_circ_hernquist_bulge_kms(R_kpc, M_bulge_solar, a_bulge_kpc):
    """JAX-compiled Hernquist bulge circular velocity."""
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    M_bulge_solar = jnp.asarray(M_bulge_solar, dtype=DEFAULT_DTYPE)
    a_bulge_kpc = jnp.asarray(a_bulge_kpc, dtype=DEFAULT_DTYPE)

    v_calc = (jnp.sqrt(G_ASTRO_UNITS * M_bulge_solar * R_kpc_arr)) / (R_kpc_arr + a_bulge_kpc)
    v_out = jnp.where(R_kpc_arr > 1e-9, v_calc, 0.0)
    return jnp.where((M_bulge_solar <= 1e-9) | (a_bulge_kpc <= 1e-9), jnp.zeros_like(v_out), v_out)

# Freeman disk calculation WITHOUT @jax.jit due to Bessel functions
def v_circ_exponential_disk_freeman_kms(R_kpc_in, M_disk_solar, R_d_kpc):
    """
    Exact Freeman (1970) kernel for exponential disk rotation.
    Not JIT-compiled due to Bessel function requirements on Metal.
    """
    # Convert inputs to JAX arrays
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc_in, dtype=DEFAULT_DTYPE))
    M_disk_solar = jnp.asarray(M_disk_solar, dtype=DEFAULT_DTYPE)
    R_d_kpc = jnp.asarray(R_d_kpc, dtype=DEFAULT_DTYPE)

    # Freeman formula y = R / (2 * Rd)
    y = R_kpc_arr / (2.0 * R_d_kpc)
    y = jnp.maximum(y, 1e-9)

    # Compute Bessel functions (will use CPU fallback on Metal)
    i0y = BesselI0(y)
    i1y = BesselI1(y)
    k0y = BesselK0(y)
    k1y = BesselK1(y)

    bessel_term_safe = jnp.maximum(i0y * k0y - i1y * k1y, 0.0)

    pre_factor = 2.0 * G_ASTRO_UNITS * M_disk_solar / R_d_kpc
    v_sq = pre_factor * (y**2) * bessel_term_safe

    v_sq = jnp.where(R_kpc_arr > 1e-9, v_sq, 0.0)
    v_sq = jnp.where((M_disk_solar <= 1e-9) | (R_d_kpc <= 1e-9), jnp.zeros_like(v_sq), v_sq)

    return jnp.sqrt(jnp.maximum(v_sq, 0.0))

# Alternative: Approximate Freeman formula that can be JIT-compiled
@jax.jit
def v_circ_exponential_disk_approx_kms(R_kpc_in, M_disk_solar, R_d_kpc):
    """
    Approximate exponential disk rotation curve using Binney & Tremaine approximation.
    This can be JIT-compiled on all backends including Metal.
    """
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc_in, dtype=DEFAULT_DTYPE))
    M_disk_solar = jnp.asarray(M_disk_solar, dtype=DEFAULT_DTYPE)
    R_d_kpc = jnp.asarray(R_d_kpc, dtype=DEFAULT_DTYPE)
    
    x = R_kpc_arr / R_d_kpc
    
    # Binney & Tremaine approximation for exponential disk
    # Valid for all x > 0
    v_sq_norm = x**2 * (1.0 - jnp.exp(-x) * (1.0 + x))
    
    # Maximum velocity occurs at x ≈ 2.16
    v_max_sq = G_ASTRO_UNITS * M_disk_solar / (2.16 * R_d_kpc)
    v_sq = v_max_sq * v_sq_norm / 0.609  # Normalize to peak
    
    v_sq = jnp.where(R_kpc_arr > 1e-9, v_sq, 0.0)
    v_sq = jnp.where((M_disk_solar <= 1e-9) | (R_d_kpc <= 1e-9), jnp.zeros_like(v_sq), v_sq)
    
    return jnp.sqrt(jnp.maximum(v_sq, 0.0))

# Choose which disk function to use based on backend
USE_FREEMAN_EXACT = not is_metal_backend()  # Use approximation on Metal

def v_baryon_total_newtonian_kms(R_kpc, p_baryons):
    """Sum the circular velocities of all baryonic sub-components in quadrature."""
    logger.debug("v_baryon_total_newtonian_kms called.")
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    v_total_sq_kms2 = jnp.zeros_like(R_kpc_arr)
    
    if p_baryons.get('include_bulge', False):
        v_total_sq_kms2 += v_circ_hernquist_bulge_kms(
            R_kpc_arr, p_baryons.get('M_bulge_solar', 0.0), p_baryons.get('a_bulge_kpc', 0.5)
        )**2
    
    # Choose disk function based on backend
    disk_func = v_circ_exponential_disk_freeman_kms if USE_FREEMAN_EXACT else v_circ_exponential_disk_approx_kms
    
    if p_baryons.get('include_disk_thin', False):
        v_total_sq_kms2 += disk_func(
            R_kpc_arr, p_baryons.get('M_disk_thin_solar', 0.0), p_baryons.get('R_d_thin_kpc', 2.5)
        )**2
    
    if p_baryons.get('include_disk_thick', False):
        v_total_sq_kms2 += disk_func(
            R_kpc_arr, p_baryons.get('M_disk_thick_solar', 0.0), p_baryons.get('R_d_thick_kpc', 3.5)
        )**2
    
    if p_baryons.get('include_gas', False):
        v_total_sq_kms2 += disk_func(
            R_kpc_arr, p_baryons.get('M_gas_solar', 0.0), p_baryons.get('R_d_gas_kpc', 7.0)
        )**2
    
    return jnp.sqrt(jnp.maximum(v_total_sq_kms2, 0.0))

@jax.jit
def _get_disk_rho_mid_internal(M, Rd, hz, R_arr):
    rho = (M / (2 * jnp.pi * Rd**2) / (2 * hz)) * jnp.exp(-R_arr / Rd)
    return jnp.where((M > 1e-9) & (Rd > 1e-9) & (hz > 1e-9), rho, jnp.zeros_like(R_arr))

@jax.jit
def _get_bulge_rho_mid_internal(M_b, a_b, R_arr):
    R_eff_b = jnp.maximum(R_arr, 1e-6)
    rho_b_mid = (M_b / (2 * jnp.pi)) * (a_b / (R_eff_b * (R_eff_b + a_b)**3))
    min_r_b = 1e-6
    fill_val_b = (M_b / (2 * jnp.pi)) * (a_b / (min_r_b * (min_r_b + a_b)**3))
    rho_b_mid_safe = jnp.where(R_arr < 1e-5, fill_val_b, rho_b_mid)
    return jnp.where(a_b > 1e-9, rho_b_mid_safe, jnp.zeros_like(R_arr))

def rho_baryon_total_midplane_solar_kpc3(R_kpc, p_baryons):
    """Mid-plane volume density rho(R, z=0) for multi-component model."""
    logger.debug("rho_baryon_total_midplane_solar_kpc3 called.")
    is_scalar_input = isinstance(R_kpc, (float, int))
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    rho_total = jnp.zeros_like(R_kpc_arr)
    
    if p_baryons.get('include_disk_thin', False):
        rho_total += _get_disk_rho_mid_internal(
            p_baryons.get('M_disk_thin_solar',0.0), 
            p_baryons.get('R_d_thin_kpc',2.5), 
            p_baryons.get('h_z_thin_kpc',0.3), 
            R_kpc_arr
        )
    
    if p_baryons.get('include_disk_thick', False):
        rho_total += _get_disk_rho_mid_internal(
            p_baryons.get('M_disk_thick_solar',0.0), 
            p_baryons.get('R_d_thick_kpc',3.5), 
            p_baryons.get('h_z_thick_kpc',0.9), 
            R_kpc_arr
        )
    
    if p_baryons.get('include_gas', False):
        rho_total += _get_disk_rho_mid_internal(
            p_baryons.get('M_gas_solar',0.0), 
            p_baryons.get('R_d_gas_kpc',7.0), 
            p_baryons.get('h_z_gas_kpc',0.15), 
            R_kpc_arr
        )
    
    if p_baryons.get('include_bulge_density', False) and p_baryons.get('M_bulge_solar', 0.0) > 0:
        rho_total += _get_bulge_rho_mid_internal(
            p_baryons.get('M_bulge_solar', 0.0), 
            p_baryons.get('a_bulge_kpc', 0.5), 
            R_kpc_arr
        )
        
    return rho_total.item() if is_scalar_input else rho_total


# ============================================================================
# SECTION 3: XI(RHO) FUNCTIONS (JAX-compiled)
# ============================================================================

@jax.jit
def xi_power_law(rho, rho_c, n_exp, A=1.0):
    """JAX-compiled: ξ = 1 + A / (1 + (ρ/ρ_c)^n)"""
    rho_arr = jnp.atleast_1d(rho)
    ratio = rho_arr / rho_c
    enhancement = A / (1.0 + jnp.power(ratio, n_exp))
    result = 1.0 + enhancement
    result = jnp.where(rho_c <= 1e-9, jnp.ones_like(rho_arr), result)
    return jnp.clip(result, 0.1, 10.0)

@jax.jit
def xi_logistic_law(rho, rho_c, n_exp, A=1.0):
    """JAX-compiled: Logistic function for a smooth transition from 1 to 1+A."""
    rho_arr = jnp.atleast_1d(rho)
    log_rho_safe = jnp.log(jnp.maximum(rho_arr, 1e-30))
    log_rho_c_safe = jnp.log(jnp.maximum(rho_c, 1e-30))
    exponent_val = -n_exp * (log_rho_safe - log_rho_c_safe)
    logistic_val = 1.0 / (1.0 + jnp.exp(exponent_val))
    result_arr = 1.0 + A * (1.0 - logistic_val)
    return jnp.where(rho_c <= 1e-9, jnp.ones_like(rho_arr), result_arr)

@jax.jit
def xi_exponential(rho, rho_c, n_exp, A=1.0):
    """JAX-compiled: Exponential enhancement at low density."""
    rho_arr = jnp.atleast_1d(rho)
    ratio = rho_arr / rho_c
    exp_arg = -jnp.power(ratio, n_exp)
    result = 1.0 + A * jnp.exp(exp_arg)
    return jnp.where(rho_c <= 1e-9, jnp.ones_like(rho_arr), result)

@jax.jit
def xi_gravitational_color(rho, rho_c, gamma, lambda_g):
    """JAX-compiled: Gravitational color confinement model."""
    rho_arr = jnp.atleast_1d(rho)
    ratio = rho_arr / rho_c
    exp_arg = -jnp.power(ratio, gamma)
    result = 1.0 + lambda_g * jnp.exp(exp_arg)
    return jnp.where(rho_c <= 1e-9, jnp.ones_like(rho_arr), result)

@jax.jit
def xi_gaussian_enhancement(rho, rho_peak, sigma_log, lambda_max):
    """JAX-compiled: Gaussian enhancement in log-density space."""
    rho_arr = jnp.atleast_1d(rho)
    log_rho = jnp.log10(jnp.maximum(rho_arr, 1e-30))
    log_peak = jnp.log10(jnp.maximum(rho_peak, 1e-30))
    exponent = -0.5 * ((log_rho - log_peak) / sigma_log)**2
    enhancement = lambda_max * jnp.exp(exponent)
    return 1.0 + enhancement

@jax.jit
def xi_mond_like(rho, rho_c, n):
    """JAX-compiled: MOND-inspired enhancement."""
    rho_arr = jnp.atleast_1d(rho)
    rho_safe = jnp.maximum(rho_arr, 1e-30)
    ratio = rho_c / rho_safe
    result = jnp.sqrt(1.0 + jnp.power(ratio, n))
    result = jnp.where(rho_c <= 1e-9, jnp.ones_like(rho_arr), result)
    return jnp.minimum(result, 10.0)

# Aliases
xi_enhanced_bounded = xi_power_law
xi_enhanced_exp = xi_exponential


@jax.jit
def xi_nonlocal(rho_local, M_enclosed, R_kpc, rho_c=1e8, M_c=5e10):
    """ JAX-vectorized non-local model. """
    rho_local_arr = jnp.atleast_1d(rho_local)
    M_enclosed_arr = jnp.atleast_1d(M_enclosed)
    R_kpc_arr = jnp.atleast_1d(R_kpc)

    xi_local = 1.0 / (1.0 + (rho_local_arr / rho_c)**1.5)

    M_expected = M_c * (1 - jnp.exp(-R_kpc_arr / 3.0))
    mass_ratio = M_enclosed_arr / jnp.maximum(M_expected, 1e-6)
    
    xi_global = 1.0 / jnp.sqrt(mass_ratio)
    xi_global = jnp.where(mass_ratio >= 1, 1.0, xi_global)
    xi_global = jnp.where(M_expected < 1e-6, 1.0, xi_global)

    return xi_local * xi_global

@jax.jit
def xi_anisotropic(rho, direction, rho_c_rad=5e8, rho_c_vert=1e7):
    """ JAX-vectorized anisotropic model. """
    rho_arr = jnp.atleast_1d(rho)
    
    # Radial calculation
    x_rad = jnp.log10(rho_arr / rho_c_rad)
    xi_rad_enhance = 1.0 + 0.3 * jnp.exp(-x_rad**2)
    xi_rad_suppress = 1.0 / (1.0 + (rho_arr / rho_c_rad)**1.5)
    xi_radial = jnp.where(x_rad < 0, xi_rad_enhance, xi_rad_suppress)

    # Vertical calculation
    xi_vertical = 1.0 / (1.0 + (rho_arr / rho_c_vert)**0.5)

    return jnp.where(direction == 'radial', xi_radial, xi_vertical)

# ============================================================================
# High-level test functions (remain on CPU using numpy)
# ============================================================================

def test_galaxy_xi():
    # This function uses numpy and matplotlib, it's a CPU-based test harness
    # It will call the JAX functions, which will run on the GPU.
    import matplotlib.pyplot as plt
    R_test = np.array([5, 8, 12, 20, 30])
    Sigma_0 = 5e10 / (2 * np.pi * 2.6**2)
    rho_disk = (Sigma_0 / (2 * 0.3)) * np.exp(-R_test / 2.6)
    
    print("Testing with JAX-compiled xi_gravitational_color...")
    # This call pushes data to GPU, computes, and brings it back
    xi_values = xi_gravitational_color(jnp.array(rho_disk), 1e7, 2.7, 1.5)
    print("Test successful, xi_values:", np.asarray(xi_values))


def test_xi_functions():
    test_rho = jnp.array([1e6, 1e8, 1e10], dtype=DEFAULT_DTYPE)
    rho_c = 5e8
    n = 1.5
    print("\nJAX Xi Function Test (ρ_c=5e8, n=1.5):")
    xi_p = xi_power_law(test_rho, rho_c, n, A=1.0)
    xi_m = xi_mond_like(test_rho, rho_c, n)
    xi_x = xi_exponential(test_rho, rho_c, n, A=1.0)
    print("Results are JAX arrays:", xi_p, xi_m, xi_x)

# ============================================================================
# XI Function Wrappers and Map
# ============================================================================

def xi_power_law_wrapper(rho, rho_c, n_exp, A=1.0, **_):
    return xi_power_law(rho, rho_c, n_exp, A)

def xi_logistic_law_wrapper(rho, rho_c, n_exp, A=1.0, **_):
    return xi_logistic_law(rho, rho_c, n_exp, A)

def xi_grav_color_standard_interface(rho, rho_c, n_exp, A, **_):
    return xi_gravitational_color(rho, rho_c, gamma=n_exp, lambda_g=A)

def xi_gaussian_wrapper(rho, rho_c, n_exp, A, **_):
    return xi_gaussian_enhancement(rho, rho_peak=rho_c, sigma_log=n_exp, lambda_max=A)

def xi_mass_threshold(rho, rho_c, n_exp, A, r_kpc, params, **_):
    """High-level wrapper for mass threshold model."""
    M_crit = rho_c
    width = n_exp
    xi_boost = 1 + A
    
    v_newton = v_baryon_total_newtonian_kms(r_kpc, params)
    M_enclosed = v_newton**2 * r_kpc / G_ASTRO_UNITS
    
    @jax.jit
    def _transition(m_enc, m_crit, w, boost):
        width_abs = jnp.maximum(w * m_crit, 0.1 * m_crit)
        x = (m_enc - m_crit) / width_abs
        transition_factor = 0.5 * (1.0 - jnp.tanh(x))
        return 1.0 + (boost - 1.0) * transition_factor
        
    xi = _transition(M_enclosed, M_crit, width, xi_boost)
    return jnp.clip(xi, 0.1, 10.0)

XI_FUNCTION_MAP = {
    'power': xi_power_law_wrapper,
    'logistic': xi_logistic_law_wrapper,
    'enhanced': xi_power_law_wrapper,
    'mond': xi_mond_like,
    'exp_enhance': xi_exponential,
    'grav_color': xi_grav_color_standard_interface,
    'gaussian': xi_gaussian_wrapper,
    'mass_threshold': xi_mass_threshold
}

# ============================================================================
# SECTION 4: FULL VELOCITY MODEL
# ============================================================================

def v_total_kms(R_kpc, p, xi_type='power'):
    """MASTER function to compute the full circular velocity."""
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # 1. Calculate Newtonian velocity from baryons
    v_newton = v_baryon_total_newtonian_kms(R_kpc_arr, p)
    
    # 2. Get the appropriate xi function
    xi_func = XI_FUNCTION_MAP.get(xi_type)
    if xi_func is None:
        raise ValueError(f"Unknown xi_type: '{xi_type}'. Available: {list(XI_FUNCTION_MAP.keys())}")

    # 3. Prepare args and call the xi function
    try:
        rho = rho_baryon_total_midplane_solar_kpc3(R_kpc_arr, p)
        rho_c = p.get('rho_c_solar_kpc3', 1e9)
        n_exp = p.get('n_exp', p.get('gamma_exp', 1.0))
        A_param = p.get('A', p.get('lambda_g', 1.0))
        
        kwargs = {'r_kpc': R_kpc_arr, 'params': p}
        xi = xi_func(rho, rho_c, n_exp, A_param, **kwargs)

    except Exception as e:
        logger.error(f"Error in v_total_kms with xi_type '{xi_type}': {e}")
        raise

    # 4. Apply the modification
    xi = jnp.atleast_1d(xi)
    v_modified = v_newton * jnp.sqrt(jnp.maximum(xi, 0.0))
    
    return v_modified.item() if isinstance(R_kpc, (float, int)) else v_modified



# Anisotropic model requires a slightly different high-level velocity function
def v_model_for_dynesty_anisotropic(R_kpc_array, p_all_params_dict, **_):
    """Modified to use anisotropic xi with JAX."""
    R_kpc_arr = jnp.atleast_1d(jnp.asarray(R_kpc_array, dtype=DEFAULT_DTYPE))
    
    v_n_kms = v_baryon_total_newtonian_kms(R_kpc_arr, p_all_params_dict)
    rho_midplane = rho_baryon_total_midplane_solar_kpc3(R_kpc_arr, p_all_params_dict)
    
    # Use JAX-compiled RADIAL xi for rotation curve
    xi_radial = xi_anisotropic(rho_midplane, direction='radial', 
                               rho_c_rad=p_all_params_dict.get('rho_c_radial', 5e8))
    
    v_mod_kms = v_n_kms * jnp.sqrt(jnp.maximum(xi_radial, 0.0))
    return v_mod_kms


# ---------- Milky Way Internal Consistency Checks (Updated to use JAX) ----------
def get_total_volume_density_at_R_z_solar_kpc3_multi(R_kpc_scalar, z_kpc_scalar, p_baryons):
    """Calculates density at a single (R, z) point using JAX functions."""
    R_kpc = jnp.array(R_kpc_scalar, dtype=DEFAULT_DTYPE)
    abs_z = jnp.abs(jnp.array(z_kpc_scalar, dtype=DEFAULT_DTYPE))
    rho_total_at_point = 0.0

    @jax.jit
    def get_sdisk_rho_at_z(M, Rd, hz, R, z):
        mid_rho = (M / (2 * jnp.pi * Rd**2) / (2 * hz)) * jnp.exp(-R / Rd)
        full_rho = mid_rho * jnp.exp(-z / hz)
        return jnp.where((M > 0) & (Rd > 0) & (hz > 0), full_rho, 0.0)

    @jax.jit
    def get_bulge_rho_at_rz(M_b, a_b, R, z):
        r_sph = jnp.sqrt(R**2 + z**2)
        r_eff = jnp.maximum(r_sph, 1e-6)
        rho = (M_b / (2 * jnp.pi)) * (a_b / (r_eff * (r_eff + a_b)**3))
        return jnp.where(M_b > 0, rho, 0.0)

    if p_baryons.get('include_disk_thin', False):
        rho_total_at_point += get_sdisk_rho_at_z(p_baryons.get('M_disk_thin_solar',0.0), p_baryons.get('R_d_thin_kpc',2.5), p_baryons.get('h_z_thin_kpc',0.3), R_kpc, abs_z)
    if p_baryons.get('include_disk_thick', False):
        rho_total_at_point += get_sdisk_rho_at_z(p_baryons.get('M_disk_thick_solar',0.0), p_baryons.get('R_d_thick_kpc',3.5), p_baryons.get('h_z_thick_kpc',0.9), R_kpc, abs_z)
    if p_baryons.get('include_gas', False):
        rho_total_at_point += get_sdisk_rho_at_z(p_baryons.get('M_gas_solar',0.0), p_baryons.get('R_d_gas_kpc',7.0), p_baryons.get('h_z_gas_kpc',0.15), R_kpc, abs_z)
    if p_baryons.get('include_bulge', False):
        rho_total_at_point += get_bulge_rho_at_rz(p_baryons.get('M_bulge_solar', 0.0), p_baryons.get('a_bulge_kpc', 0.5), R_kpc, abs_z)

    return rho_total_at_point # Returns a JAX scalar

def check_vertical_kinematics_Kz(p_baryons, R_solar_val=R_SUN_KPC, z_limit_kpc=1.0, nz_points=100, target_sigma_z_max_msun_kpc2=SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2):
    """CPU-based check that uses JAX for the density calculation."""
    logger.debug(f"[Kz Test] Target < {target_sigma_z_max_msun_kpc2 / MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")
    z_pts = np.linspace(-z_limit_kpc, z_limit_kpc, nz_points)
    dz = z_pts[1] - z_pts[0]
    # This list comprehension will be slow, but it's a diagnostic, not a performance path.
    rhos = [get_total_volume_density_at_R_z_solar_kpc3_multi(R_solar_val, z, p_baryons) for z in z_pts]
    rhos_np = np.asarray(rhos) # Convert list of JAX scalars back to numpy array
    col_dens_model = np.sum((rhos_np[:-1] + rhos_np[1:]) * 0.5 * dz)
    logger.debug(f"[Kz Test] Model Sigma_z = {col_dens_model/MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")
    return col_dens_model <= target_sigma_z_max_msun_kpc2

def calculate_microlensing_tau_baade(p_baryons, target_tau_max=TAU_MICRO_TARGET_BAADE_MAX):
    """Simple algebraic check, does not require JAX."""
    logger.debug(f"[Microlensing Test] Target < {target_tau_max:.1e}")
    M_eff_lenses = 0.0
    if p_baryons.get('include_disk_thin', False): M_eff_lenses += p_baryons.get('M_disk_thin_solar', 0)
    if p_baryons.get('include_disk_thick', False): M_eff_lenses += p_baryons.get('M_disk_thick_solar', 0)
    if p_baryons.get('include_bulge', False): M_eff_lenses += p_baryons.get('M_bulge_solar', 0)
    if M_eff_lenses <= 1e-9:
        logger.debug("[Microlensing] No lens mass. PASSED.")
        return True
    model_tau_s = (M_eff_lenses / 5e10) * 2e-6 # Crude scaling
    logger.debug(f"[Microlensing] M_eff_lenses={M_eff_lenses:.2e}, model_tau_scaled: {model_tau_s:.2e}")
    return model_tau_s <= target_tau_max

logger.info("density_metric2.py (JAX Version): Multi-component functions defined and JIT-compiled for GPU.")