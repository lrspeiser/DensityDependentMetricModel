#!/usr/bin/env python3
"""
density_metric2.py - Physics layer: mass models, density profiles,
                    density-weighting functions (xi), and full velocity model.
                    Contains BOTH old single-disk functions (for backward compatibility
                    in main2.py) AND new multi-component baryonic model functions.
"""
import numpy as np
import numba as nb
from numba import vectorize, float64
from numba import njit
from scipy.special import iv as BesselI, kv as BesselK 
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

# ---- Self-Testing Functions (as recommended by audit) ----
def _assert_freeman_identity():
    """Quick self‑test: Check Freeman kernel against a known approximate value."""
    logger.debug("[SELF-TEST] Verifying Freeman kernel...")
    # At R=R_d, y=0.5. The expression y^2 * (I0K0 - I1K1) should be ~0.1386.
    # The full v^2 = (2*G*M/R_d) * (y^2*(...))
    y_test = 0.5
    i0y, k0y, i1y, k1y = BesselI(0, y_test), BesselK(0, y_test), BesselI(1, y_test), BesselK(1, y_test)
    bessel_term = i0y * k0y - i1y * k1y
    val = (y_test**2) * bessel_term
    expected_val = 0.138979394445648
    assert np.isclose(val, expected_val, rtol=1e-6), f"Freeman kernel self-test failed! Expected ~{expected_val}, got {val}"
    logger.debug("[SELF-TEST] Freeman kernel OK.")

def _assert_xi_limits():
    """Ensure xi_power_law hits expected limits."""
    logger.debug("[SELF-TEST] Verifying xi_power_law limits...")
    # The function now returns an array, so we access the first element.
    test_val = xi_power_law(1e40, 1.0, 2.0)[0]
    print(f"DEBUG: xi(1e40) = {test_val}, type = {type(test_val)}")
    assert np.isclose(xi_power_law(0.0, 1.0, 2.0)[0], 1.0), "xi at ρ→0 should be 1.0"
    assert xi_power_law(1e40, 1.0, 2.0)[0] < 1e-5, "xi at ρ≫ρ_c should be ~0.0"
    logger.debug("[SELF-TEST] xi limit checks OK.")


def run_physics_self_tests():
    """
    Runs all self-tests for this physics module.
    Call this from the main script after logging is configured.
    """
    try:
        logger.info("[PHYSICS] Running self-tests...")
        _assert_freeman_identity()
        _assert_xi_limits() # This will call the Numba-compiled function
        logger.info("[PHYSICS] All self-tests passed.")
    except Exception as e:
        logger.error(f"[PHYSICS SELF-TEST FAILED] {e}")
        # Re-raise the exception to halt execution if a core formula is broken
        raise

# ---------- Physical Constants ----------
G_ASTRO_UNITS = 4.30091e-6 # kpc (km/s)^2 / Msun
R_SUN_KPC = 8.122
MSUN_PC2_TO_MSUN_KPC2 = (1e3)**2
SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2 = 85.0 * MSUN_PC2_TO_MSUN_KPC2
TAU_MICRO_TARGET_BAADE_MAX = 3.0e-6
L_BAADE_DEG = 1.0
B_BAADE_DEG = -2.75
D_SUN_GC_KPC = R_SUN_KPC

# --- START: OLDER SINGLE-DISK COMPATIBLE FUNCTIONS (Unchanged) ---
@nb.njit
def _enclosed_disk_mass_solar_old(R_kpc, M_disk_solar, R_d_kpc):
    """ Helper for OLD v_newton_kms: Exponential disk, cumulative mass (spherical approx). """
    if R_d_kpc <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    x = R_kpc_arr / R_d_kpc
    x_safe = np.maximum(x, 0)
    m_enc_arr = M_disk_solar * (1.0 - np.exp(-x_safe) * (1.0 + x_safe))
    m_enc_arr[R_kpc_arr < 0] = 0.0
    return m_enc_arr[0] if is_scalar else m_enc_arr

@nb.njit
def _enclosed_hernquist_mass_solar_old(R_kpc, M_bulge_solar, R_b_kpc):
    """ Helper for OLD v_newton_kms: Hernquist profile enclosed mass. """
    if R_b_kpc <= 1e-9 or M_bulge_solar <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    R_kpc_safe = np.maximum(R_kpc_arr, 0)
    m_enc_arr = M_bulge_solar * (R_kpc_safe**2) / ((R_kpc_safe + R_b_kpc)**2)
    m_enc_arr[R_kpc_arr < 0] = 0.0
    return m_enc_arr[0] if is_scalar else m_enc_arr

def v_newton_kms(R_kpc, M_disk_solar_main, R_d_kpc_main,
                 M_bulge_solar_opt=0.0, R_b_kpc_opt=0.5, include_bulge_opt=False,
                 M_gas_solar_opt=0.0, R_gas_kpc_opt=7.0, include_gas_opt=False):
    """ OLDER Newtonian circular velocity (spherical approx). For backward compatibility. """
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    v_total_sq_kms2 = np.zeros_like(R_kpc_arr, dtype=np.float64)
    M_enc_disk = _enclosed_disk_mass_solar_old(R_kpc_arr, M_disk_solar_main, R_d_kpc_main)
    M_enc_bulge = np.zeros_like(R_kpc_arr, dtype=np.float64)
    if include_bulge_opt and M_bulge_solar_opt > 0:
        M_enc_bulge = _enclosed_hernquist_mass_solar_old(R_kpc_arr, M_bulge_solar_opt, R_b_kpc_opt)
    M_enc_gas = np.zeros_like(R_kpc_arr, dtype=np.float64)
    if include_gas_opt and M_gas_solar_opt > 0:
        M_enc_gas = _enclosed_disk_mass_solar_old(R_kpc_arr, M_gas_solar_opt, R_gas_kpc_opt)
    M_enc_total_solar = M_enc_disk + M_enc_bulge + M_enc_gas
    valid_R_mask = (R_kpc_arr > 1e-9) & (M_enc_total_solar > 1e-9)
    if np.any(valid_R_mask):
        v_total_sq_kms2[valid_R_mask] = G_ASTRO_UNITS * M_enc_total_solar[valid_R_mask] / R_kpc_arr[valid_R_mask]
    v_out_kms = np.sqrt(np.maximum(v_total_sq_kms2, 0.0))
    return v_out_kms[0] if is_scalar else v_out_kms

@nb.njit
def volume_density_total_midplane_solar_kpc3(R_kpc,
                                             M_disk_solar_main, R_d_kpc_main, h_z_disk_kpc_main,
                                             M_bulge_solar_opt=0.0, R_b_kpc_opt=0.5, h_z_bulge_eff_kpc_opt=0.3, include_bulge_opt=False,
                                             M_gas_solar_opt=0.0, R_gas_kpc_opt=7.0, h_z_gas_kpc_opt=0.15, include_gas_opt=False):
    """ OLDER Midplane volume density. For backward compatibility. """
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    rho_total = np.zeros_like(R_kpc_arr, dtype=np.float64)
    if M_disk_solar_main > 1e-9 and R_d_kpc_main > 1e-9 and h_z_disk_kpc_main > 1e-9:
        Sigma0_disk = M_disk_solar_main / (2.0 * np.pi * R_d_kpc_main**2)
        rho_total += (Sigma0_disk / (2.0 * h_z_disk_kpc_main)) * np.exp(-R_kpc_arr / R_d_kpc_main)
    if include_bulge_opt and M_bulge_solar_opt > 0 and R_b_kpc_opt > 1e-9:
        R_eff_bulge = np.maximum(R_kpc_arr, 1e-6)
        rho_bulge_mid_vals = (M_bulge_solar_opt / (2.0 * np.pi)) * (R_b_kpc_opt / (R_eff_bulge * (R_eff_bulge + R_b_kpc_opt)**3))
        mask_zero_R_bulge = R_kpc_arr < 1e-5
        if np.any(mask_zero_R_bulge):
            non_zero_R_eff_b = R_eff_bulge[R_eff_bulge > 1e-7]
            if len(non_zero_R_eff_b) > 0:
                min_r_b = np.min(non_zero_R_eff_b)
                fill_val_b = (M_bulge_solar_opt / (2.0 * np.pi)) * (R_b_kpc_opt / (min_r_b * (min_r_b + R_b_kpc_opt)**3))
                rho_bulge_mid_vals[mask_zero_R_bulge] = fill_val_b
        rho_total += rho_bulge_mid_vals
    if include_gas_opt and M_gas_solar_opt > 0 and R_gas_kpc_opt > 1e-9 and h_z_gas_kpc_opt > 1e-9:
        Sigma0_gas = M_gas_solar_opt / (2.0 * np.pi * R_gas_kpc_opt**2)
        rho_total += (Sigma0_gas / (2.0 * h_z_gas_kpc_opt)) * np.exp(-R_kpc_arr / R_gas_kpc_opt)
    return rho_total[0] if is_scalar else rho_total

# --- END: OLDER FUNCTIONS ---


# --- START: NEW MULTI-COMPONENT BARYONIC MODEL FUNCTIONS (with added logging) ---
def v_circ_hernquist_bulge_kms(R_kpc, M_bulge_solar, a_bulge_kpc):
    """Hernquist bulge circular velocity. v = sqrt(G M R) / (R+a)"""
    logger.debug(f"v_circ_hernquist called with M={M_bulge_solar:.2e}, a={a_bulge_kpc:.2f}")
    if M_bulge_solar <= 1e-9 or a_bulge_kpc <= 1e-9:
        return np.zeros_like(np.atleast_1d(R_kpc), dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    v_out = np.zeros_like(R_kpc_arr, dtype=np.float64)
    valid_mask = (R_kpc_arr > 1e-9)
    if np.any(valid_mask):
        R_valid = R_kpc_arr[valid_mask]
        v_calc = (np.sqrt(G_ASTRO_UNITS * M_bulge_solar * R_valid)) / (R_valid + a_bulge_kpc)
        v_out[valid_mask] = v_calc
    return v_out[0] if is_scalar else v_out

def v_circ_exponential_disk_freeman_kms(R_kpc_in, M_disk_solar, R_d_kpc):
    """Exact Freeman (1970) kernel."""
    logger.debug(f"v_circ_exp_disk (Freeman) called with M={M_disk_solar:.2e}, Rd={R_d_kpc:.2f}")
    if R_d_kpc <= 1e-9 or M_disk_solar <= 1e-9:
        return np.zeros_like(np.atleast_1d(R_kpc_in), dtype=np.float64)
    is_scalar_input = isinstance(R_kpc_in, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc_in).astype(np.float64)
    v_sq_out = np.zeros_like(R_kpc_arr, dtype=np.float64)
    valid_mask = (R_kpc_arr > 1e-9)
    if np.any(valid_mask):
        R_kpc_valid = R_kpc_arr[valid_mask]
        y = R_kpc_valid / (2.0 * R_d_kpc); y = np.maximum(y, 1e-9)
        i0y, k0y, i1y, k1y = BesselI(0,y), BesselK(0,y), BesselI(1,y), BesselK(1,y)
        bessel_term_safe = np.maximum(i0y * k0y - i1y * k1y, 0.0)
        v_sq_out[valid_mask] = (2.0*G_ASTRO_UNITS*M_disk_solar/R_d_kpc)*(y**2)*bessel_term_safe
    v_kms = np.sqrt(np.maximum(v_sq_out, 0.0))
    return v_kms[0] if is_scalar_input else v_kms

def v_baryon_total_newtonian_kms(R_kpc, p_baryons):
    """Sum the circular velocities of all baryonic sub-components in quadrature."""
    logger.debug("v_baryon_total_newtonian_kms called.")
    v_total_sq_kms2 = np.zeros_like(np.atleast_1d(R_kpc), dtype=np.float64)
    if p_baryons.get('include_bulge', False):
        v_total_sq_kms2 += v_circ_hernquist_bulge_kms(R_kpc, p_baryons.get('M_bulge_solar', 0), p_baryons.get('a_bulge_kpc', 0.5))**2
    if p_baryons.get('include_disk_thin', False):
        v_total_sq_kms2 += v_circ_exponential_disk_freeman_kms(R_kpc, p_baryons.get('M_disk_thin_solar', 0), p_baryons.get('R_d_thin_kpc', 2.5))**2
    if p_baryons.get('include_disk_thick', False):
        v_total_sq_kms2 += v_circ_exponential_disk_freeman_kms(R_kpc, p_baryons.get('M_disk_thick_solar', 0), p_baryons.get('R_d_thick_kpc', 3.5))**2
    if p_baryons.get('include_gas', False):
        v_total_sq_kms2 += v_circ_exponential_disk_freeman_kms(R_kpc, p_baryons.get('M_gas_solar', 0), p_baryons.get('R_d_gas_kpc', 7.0))**2
    return np.sqrt(np.maximum(v_total_sq_kms2, 0.0))

def rho_baryon_total_midplane_solar_kpc3(R_kpc, p_baryons_for_density):
    """Mid-plane volume density rho(R, z=0) for multi-component disks and optionally bulge."""
    logger.debug("rho_baryon_total_midplane_solar_kpc3 called.")
    is_scalar_input = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    rho_total = np.zeros_like(R_kpc_arr, dtype=np.float64)
    
    def get_disk_rho_mid_internal(M, Rd, hz, R_arr, name="disk"):
        logger.debug(f"get_disk_rho_mid for {name} with M={M:.2e}, Rd={Rd:.2f}, hz={hz:.2f}")
        if M <= 1e-9 or Rd <= 1e-9 or hz <= 1e-9: return np.zeros_like(R_arr)
        return (M/(2*np.pi*Rd**2)/(2*hz))*np.exp(-R_arr/Rd)

    if p_baryons_for_density.get('include_disk_thin', False):
        rho_total += get_disk_rho_mid_internal(p_baryons_for_density.get('M_disk_thin_solar',0), p_baryons_for_density.get('R_d_thin_kpc',2.5), p_baryons_for_density.get('h_z_thin_kpc',0.3), R_kpc_arr, "thin_disk")
    if p_baryons_for_density.get('include_disk_thick', False):
        rho_total += get_disk_rho_mid_internal(p_baryons_for_density.get('M_disk_thick_solar',0), p_baryons_for_density.get('R_d_thick_kpc',3.5), p_baryons_for_density.get('h_z_thick_kpc',0.9), R_kpc_arr, "thick_disk")
    if p_baryons_for_density.get('include_gas', False):
        rho_total += get_disk_rho_mid_internal(p_baryons_for_density.get('M_gas_solar',0), p_baryons_for_density.get('R_d_gas_kpc',7.0), p_baryons_for_density.get('h_z_gas_kpc',0.15), R_kpc_arr, "gas")
    
    if p_baryons_for_density.get('include_bulge_density', False) and p_baryons_for_density.get('M_bulge_solar',0)>0:
        M_b, a_b = p_baryons_for_density.get('M_bulge_solar', 0), p_baryons_for_density.get('a_bulge_kpc', 0.5)
        logger.debug(f"Adding bulge density with M={M_b:.2e}, a={a_b:.2f}")
        if a_b > 1e-9:
            R_eff_b=np.maximum(R_kpc_arr,1e-6)
            rho_b_mid=(M_b/(2*np.pi))*(a_b/(R_eff_b*(R_eff_b+a_b)**3))
            m_zero_R_b = R_kpc_arr < 1e-5
            if np.any(m_zero_R_b):
                non_zero_R_eff_b = R_eff_b[R_eff_b > 1e-7]
                if len(non_zero_R_eff_b)>0: min_r_b=np.min(non_zero_R_eff_b); fill_val_b=(M_b/(2*np.pi))*(a_b/(min_r_b*(min_r_b+a_b)**3)); rho_b_mid[m_zero_R_b]=fill_val_b
            rho_total += rho_b_mid
            
    return rho_total[0] if is_scalar_input and len(rho_total)==1 else rho_total

# --- END: NEW MULTI-COMPONENT FUNCTIONS ---


# ---------- Candidate xi(rho) Laws (Corrected for Numba) ----------
@nb.njit(cache=True)
def xi_power_law(rho, rho_c, n_exp):
    """
    Numerically stable, bounded version of ξ(ρ) = 1 / (1 + (ρ/ρ_c)^n_exp)
    
    - Returns an array, handles scalar input correctly via np.atleast_1d
    - Prevents ξ from going to exactly 0 or above 1
    - Safe for use with extremely high or low density values
    """
    # Convert to array, dtype-safe
    rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))

    # Edge case: if rho_c is too small, we default to Newtonian everywhere
    if rho_c <= 1e-9:
        return np.ones_like(rho_arr, dtype=np.float64)

    # Clip densities to a wide range to avoid overflow in (rho / rho_c)**n_exp
    rho_clipped = np.clip(rho_arr, 1e-15, 1e25)
    safe_rho_c = np.maximum(rho_c, 1e-15)
    ratio = rho_clipped / safe_rho_c

    # Compute (rho / rho_c)^n
    term_power = np.power(ratio, n_exp)

    # Standard power-law denominator
    denominator = 1.0 + term_power

    # Initialize ξ with ones
    result_arr = np.ones_like(rho_arr, dtype=np.float64)

    # Where denominator is valid, compute 1 / (1 + (rho/rho_c)^n)
    safe_mask = np.isfinite(denominator) & (denominator > 1e-100)
    result_arr[safe_mask] = 1.0 / denominator[safe_mask]

    # Clip ξ to physically meaningful range to avoid zero-gravity regions
    result_arr = np.clip(result_arr, 1e-6, 1.0)

    return result_arr

@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def xi_enhanced_bounded(rho, rho_c, n, A=1.0):
    """
    Enhanced gravity at low density, bounded maximum
    ξ → 1 as ρ → ∞ (normal gravity in dense regions)
    ξ → 1+A as ρ → 0 (enhanced gravity in voids)
    """
    val = 1.0 + A / (1.0 + (rho / rho_c)**n)
    return np.maximum(1e-3, np.minimum(val, 2.0))  # Numba-safe

def xi_enhanced(rho, rho_c, n):
    return xi_enhanced_bounded(rho, rho_c, n, 1.0)

@njit
def xi_mond_like(rho, rho_c, n):
    """
    MOND-inspired enhancement
    ξ → 1 as ρ → ∞
    ξ → sqrt(1 + (ρ_c/ρ)^n) as ρ → 0
    """
    return np.sqrt(1.0 + (rho_c/rho)**n)

@njit  
def xi_enhanced_exp(rho, rho_c, n):
    """
    Exponential enhancement at low density
    ξ = exp(-(ρ/ρ_c)^n)
    """
    return np.exp(-(rho/rho_c)**n) + 1.0

@nb.njit(cache=True)
def xi_logistic_law(rho, rho_c, n_exp):
    """
    Numba-compiled logistic law for xi.
    This version always works with arrays and returns an array.
    """
    rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))

    if rho_c <= 1e-9:
        return np.ones_like(rho_arr, dtype=np.float64)
        
    log_rho_safe = np.log(np.maximum(rho_arr, 1e-30))
    log_rho_c_safe = np.log(np.maximum(rho_c, 1e-30))
    exponent_val = n_exp * (log_rho_safe - log_rho_c_safe)
    clipped_exponent = np.clip(exponent_val, -709, 709) # Prevents np.exp overflow
    exp_term = np.exp(clipped_exponent)
    result_arr = 1.0 / (1.0 + exp_term)
    
    return result_arr

def xi_nonlocal(rho_local, M_enclosed, R_kpc, rho_c=1e8, M_c=5e10):
    """
    Non-local model: depends on both local density and mass interior to R.
    """
    rho_local = np.atleast_1d(np.asarray(rho_local, dtype=np.float64))
    M_enclosed = np.atleast_1d(np.asarray(M_enclosed, dtype=np.float64))
    R_kpc = np.atleast_1d(np.asarray(R_kpc, dtype=np.float64))

    xi_arr = np.ones_like(rho_local)

    for i in range(len(rho_local)):
        rho = rho_local[i]
        M_enc = M_enclosed[i]
        R = R_kpc[i]

        xi_local = 1.0 / (1.0 + (rho / rho_c)**1.5)

        M_expected = M_c * (1 - np.exp(-R / 3.0))
        if M_expected < 1e-6:
            xi_global = 1.0
        else:
            mass_ratio = M_enc / M_expected
            if mass_ratio < 1:
                xi_global = 1.0 / np.sqrt(mass_ratio)
            else:
                xi_global = 1.0

        xi_arr[i] = xi_local * xi_global

    return xi_arr

def xi_anisotropic(rho, direction='radial', rho_c_rad=5e8, rho_c_vert=1e7):
    """
    Different behavior depending on direction.
    'radial' can enhance; 'vertical' always suppresses.
    """
    rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    xi_arr = np.ones_like(rho_arr)

    for i in range(len(rho_arr)):
        rho_val = rho_arr[i]
        if direction == 'radial':
            x = np.log10(rho_val / rho_c_rad)
            if x < 0:
                xi_arr[i] = 1.0 + 0.3 * np.exp(-x**2)
            else:
                xi_arr[i] = 1.0 / (1.0 + (rho_val / rho_c_rad)**1.5)
        elif direction == 'vertical':
            xi_arr[i] = 1.0 / (1.0 + (rho_val / rho_c_vert)**0.5)

    return xi_arr

def v_model_for_dynesty_anisotropic(R_kpc_array, p_all_params_dict, ARGS_obj_dynesty):
    """Modified to use anisotropic xi."""
    
    # Get Newtonian velocities
    v_n_kms = v_baryon_total_newtonian_kms(R_kpc_array, p_all_params_dict)
    rho_midplane = rho_baryon_total_midplane_solar_kpc3(R_kpc_array, p_all_params_dict)
    
    # Use RADIAL xi for rotation curve
    xi_radial = xi_anisotropic(rho_midplane, direction='radial', 
                               rho_c_rad=p_all_params_dict.get('rho_c_radial', 5e8))
    
    v_mod_kms = v_n_kms * np.sqrt(np.maximum(xi_radial, 0.0))
    return v_mod_kms

# And for K_z calculation, use VERTICAL xi

XI_FUNCTION_MAP = {
    'power': xi_power_law,
    'enhanced': xi_enhanced,  # ✅ now safe: uses 3 args
    'mond': xi_mond_like,
    'exp_enhance': xi_enhanced_exp,
    'logistic': xi_logistic_law
}


# ---------- Milky Way Internal Consistency Checks (Unchanged) ----------
def get_total_volume_density_at_R_z_solar_kpc3_multi(R_kpc_scalar, z_kpc_scalar, p_baryons):
    rho_total_at_point = 0.0; abs_z = np.abs(z_kpc_scalar)
    def get_sdisk_mid_rho(M,Rd,hz,R):
        if M<=0 or Rd<=0 or hz<=0: return 0.0
        return (M/(2*np.pi*Rd**2)/(2*hz))*np.exp(-R/Rd)
    if p_baryons.get('include_disk_thin',False) and p_baryons.get('M_disk_thin_solar',0)>0: rho_total_at_point += get_sdisk_mid_rho(p_baryons['M_disk_thin_solar'],p_baryons['R_d_thin_kpc'],p_baryons['h_z_thin_kpc'],R_kpc_scalar)*np.exp(-abs_z/p_baryons['h_z_thin_kpc'])
    if p_baryons.get('include_disk_thick',False) and p_baryons.get('M_disk_thick_solar',0)>0: rho_total_at_point += get_sdisk_mid_rho(p_baryons['M_disk_thick_solar'],p_baryons['R_d_thick_kpc'],p_baryons['h_z_thick_kpc'],R_kpc_scalar)*np.exp(-abs_z/p_baryons['h_z_thick_kpc'])
    if p_baryons.get('include_gas',False) and p_baryons.get('M_gas_solar',0)>0: rho_total_at_point += get_sdisk_mid_rho(p_baryons['M_gas_solar'],p_baryons['R_d_gas_kpc'],p_baryons['h_z_gas_kpc'],R_kpc_scalar)*np.exp(-abs_z/p_baryons['h_z_gas_kpc'])
    if p_baryons.get('include_bulge',False) and p_baryons.get('M_bulge_solar',0)>0:
        r_sph=np.sqrt(R_kpc_scalar**2+z_kpc_scalar**2); r_eff=np.maximum(r_sph,1e-6); M_b,a_b=p_baryons['M_bulge_solar'],p_baryons['a_bulge_kpc']
        rho_total_at_point += (M_b/(2*np.pi))*(a_b/(r_eff*(r_eff+a_b)**3))
    return rho_total_at_point

def check_vertical_kinematics_Kz(p_baryons, R_solar_val=R_SUN_KPC, z_limit_kpc=1.0, nz_points=100, target_sigma_z_max_msun_kpc2=SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2):
    logger.debug(f"[Kz Test] Target < {target_sigma_z_max_msun_kpc2 / MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")
    z_pts=np.linspace(-z_limit_kpc,z_limit_kpc,nz_points); dz=z_pts[1]-z_pts[0]
    rhos=[get_total_volume_density_at_R_z_solar_kpc3_multi(R_solar_val,z,p_baryons) for z in z_pts]
    col_dens_model=np.sum((np.array(rhos[:-1])+np.array(rhos[1:]))*0.5*dz)
    logger.debug(f"[Kz Test] Model Sigma_z = {col_dens_model/MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")
    if col_dens_model > target_sigma_z_max_msun_kpc2: logger.debug("[Kz Test] FAILED."); return False
    logger.debug("[Kz Test] PASSED."); return True

def calculate_microlensing_tau_baade(p_baryons, l_deg=L_BAADE_DEG, b_deg=B_BAADE_DEG, ds_los_kpc=0.1, target_tau_max=TAU_MICRO_TARGET_BAADE_MAX):
    logger.debug(f"[Microlensing Test] Target < {target_tau_max:.1e}")
    M_eff_lenses=0.0
    if p_baryons.get('include_disk_thin',False): M_eff_lenses += p_baryons.get('M_disk_thin_solar',0)
    if p_baryons.get('include_disk_thick',False): M_eff_lenses += p_baryons.get('M_disk_thick_solar',0)
    if p_baryons.get('include_bulge',False): M_eff_lenses += p_baryons.get('M_bulge_solar',0)
    if M_eff_lenses <= 1e-9 : logger.debug("[Microlensing] No lens mass. PASSED."); return True
    model_tau_s = (M_eff_lenses / 5e10) * 2e-6 # Crude scaling
    logger.debug(f"[Microlensing] M_eff_lenses={M_eff_lenses:.2e}, model_tau_scaled: {model_tau_s:.2e}")
    if model_tau_s > target_tau_max: logger.debug("[Microlensing] FAILED."); return False
    logger.debug("[Microlensing] PASSED."); return True

# This final info log is fine, it just confirms the module was imported.
logger.info("density_metric2.py: Multi-component and (aliased) single-disk functions defined.")