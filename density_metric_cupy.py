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
# BESSEL FUNCTION WRAPPERS - Optimized for CuPy
# ============================================================================

def bessel_i0_cupy(x):
    """Bessel I0 optimized for CuPy arrays."""
    if isinstance(x, cp.ndarray):
        # For CuPy arrays, we need to use scipy on CPU then transfer back
        x_cpu = cp.asnumpy(x)
        result_cpu = scipy_i0(x_cpu)
        return cp.asarray(result_cpu, dtype=DEFAULT_DTYPE)
    else:
        # For scalar or numpy arrays
        result = scipy_i0(x)
        return cp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_i1_cupy(x):
    """Bessel I1 optimized for CuPy arrays."""
    if isinstance(x, cp.ndarray):
        x_cpu = cp.asnumpy(x)
        result_cpu = scipy_i1(x_cpu)
        return cp.asarray(result_cpu, dtype=DEFAULT_DTYPE)
    else:
        result = scipy_i1(x)
        return cp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_k0_cupy(x):
    """Bessel K0 optimized for CuPy arrays."""
    if isinstance(x, cp.ndarray):
        x_cpu = cp.asnumpy(x)
        result_cpu = scipy_kv(0, x_cpu)
        return cp.asarray(result_cpu, dtype=DEFAULT_DTYPE)
    else:
        result = scipy_kv(0, x)
        return cp.asarray(result, dtype=DEFAULT_DTYPE)

def bessel_k1_cupy(x):
    """Bessel K1 optimized for CuPy arrays."""
    if isinstance(x, cp.ndarray):
        x_cpu = cp.asnumpy(x)
        result_cpu = scipy_kv(1, x_cpu)
        return cp.asarray(result_cpu, dtype=DEFAULT_DTYPE)
    else:
        result = scipy_kv(1, x)
        return cp.asarray(result, dtype=DEFAULT_DTYPE)

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
                         (4.302e-3 * M_disk_solar_main_arr / R_kpc_arr) * 
                         (1.0 - cp.exp(-R_kpc_arr / R_d_kpc_main_arr) * 
                          (1.0 + R_kpc_arr / R_d_kpc_main_arr)), 0.0)
    
    # Bulge contribution
    v_bulge_sq = 0.0
    if bool(include_bulge_opt) and float(M_bulge_solar_opt) > 1e-9 and float(R_b_kpc_opt) > 1e-9:
        v_bulge_sq = (4.302e-3 * M_bulge_solar_opt_arr / R_kpc_arr) * \
                     (R_kpc_arr / (R_kpc_arr + R_b_kpc_opt_arr))**2
    
    # Gas contribution
    v_gas_sq = 0.0
    if bool(include_gas_opt) and float(M_gas_solar_opt) > 1e-9 and float(R_gas_kpc_opt) > 1e-9:
        v_gas_sq = (4.302e-3 * M_gas_solar_opt_arr / R_kpc_arr) * \
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
    v_sq = (4.302e-3 * M_bulge_solar_arr / R_safe) * (R_safe / (R_safe + a_bulge_kpc_arr))**2
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
    v_sq = (4.302e-3 * M_disk_solar_arr / R_d_kpc_arr) * (x_safe**2) * freeman_term
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
    """Gravitational color xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    ratio = rho_safe / rho_c_safe
    
    # Gravitational color formula
    xi = 1.0 + lambda_g * (ratio**gamma) / (1.0 + ratio**gamma)
    return cp.maximum(xi, 1.0)  # Ensure xi >= 1

@cp.fuse()
def xi_gaussian_enhancement_cupy(rho, rho_peak, sigma_log, lambda_max):
    """Gaussian enhancement xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_peak_safe = cp.maximum(rho_peak, 1e-10)
    
    log_ratio = cp.log(rho_safe / rho_peak_safe)
    gaussian = cp.exp(-0.5 * (log_ratio / sigma_log)**2)
    return 1.0 + (lambda_max - 1.0) * gaussian

@cp.fuse()
def xi_mond_like_cupy(rho, rho_c, n):
    """MOND-like xi function - CuPy optimized."""
    rho_safe = cp.maximum(rho, 1e-10)
    rho_c_safe = cp.maximum(rho_c, 1e-10)
    ratio = rho_safe / rho_c_safe
    
    # MOND-like enhancement
    xi = 1.0 + (ratio**(-n/2.0)) / (1.0 + ratio**(-n/2.0))
    return cp.maximum(xi, 1.0)

# ============================================================================
# MAIN VELOCITY FUNCTION - CuPy Optimized
# ============================================================================

def v_total_kms_cupy(R_kpc, p, xi_type='power'):
    """
    Total velocity including modified gravity effects - CuPy optimized.
    
    This is the main function that calculates the total circular velocity
    including baryonic components and modified gravity effects.
    """
    R_kpc_arr = cp.atleast_1d(cp.asarray(R_kpc, dtype=DEFAULT_DTYPE))
    
    # Extract baryonic parameters
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
    
    # Calculate baryonic velocity (Newtonian)
    v_baryon_sq = v_baryon_total_newtonian_kms_cupy(R_kpc_arr, p_baryons)**2
    
    # Calculate total density for xi function
    rho_total = volume_density_total_midplane_solar_kpc3_cupy(
        R_kpc_arr,
        p_baryons['M_disk_solar'], p_baryons['R_d_kpc'], p.get('hz_disk_kpc', 0.3),
        p_baryons['M_bulge_solar'], p_baryons['R_b_kpc'], p_baryons['include_bulge'],
        p_baryons['M_gas_solar'], p_baryons['R_gas_kpc'], p.get('hz_gas_kpc', 0.1),
        p_baryons['include_gas']
    )
    
    # Calculate xi enhancement factor
    rho_c = p.get('rho_c_solar_kpc3', 1e8)
    
    if xi_type == 'power':
        n_exp = p.get('n_exp', 2.0)
        A = p.get('A_xi', 1.0)
        xi = xi_power_law_cupy(rho_total, rho_c, n_exp, A)
    elif xi_type == 'logistic':
        n_exp = p.get('n_exp', 2.0)
        A = p.get('A_xi', 1.0)
        xi = xi_logistic_law_cupy(rho_total, rho_c, n_exp, A)
    elif xi_type == 'exponential':
        n_exp = p.get('n_exp', 2.0)
        A = p.get('A_xi', 1.0)
        xi = xi_exponential_cupy(rho_total, rho_c, n_exp, A)
    elif xi_type == 'grav_color':
        gamma = p.get('gamma_exp', 2.7)
        lambda_g = p.get('lambda_g', 8.0)
        xi = xi_gravitational_color_cupy(rho_total, rho_c, gamma, lambda_g)
    elif xi_type == 'gaussian':
        rho_peak = p.get('rho_peak_solar_kpc3', 1e8)
        sigma_log = p.get('sigma_log', 1.0)
        lambda_max = p.get('lambda_max', 2.0)
        xi = xi_gaussian_enhancement_cupy(rho_total, rho_peak, sigma_log, lambda_max)
    elif xi_type == 'mond':
        n = p.get('n_mond', 1.0)
        xi = xi_mond_like_cupy(rho_total, rho_c, n)
    elif xi_type == 'gr':
        # Standard GR - no enhancement
        xi = cp.ones_like(rho_total)
    else:
        # Default to power law
        n_exp = p.get('n_exp', 2.0)
        A = p.get('A_xi', 1.0)
        xi = xi_power_law_cupy(rho_total, rho_c, n_exp, A)
    
    # Apply xi enhancement to velocity
    v_total_sq = v_baryon_sq * xi
    v_total = cp.sqrt(cp.maximum(v_total_sq, 0.0))
    
    return v_total

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