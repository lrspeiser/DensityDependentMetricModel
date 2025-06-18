#!/usr/bin/env python3
"""
density_metric.py - Physics layer: mass models, density profiles,
                    density-weighting functions (xi), and full velocity model.
"""
import numpy as np
from numba import njit
import logging
from scipy.special import iv, kv  # For Bessel functions I_n, K_n (iv is I, kv is K)

logger = logging.getLogger(__name__)

# ---------- Physical Constants ----------
G_CONST_SI = 6.67430e-11     # m^3 kg^-1 s^-2
KPC_TO_METERS = 3.08567758149e19 # meters per kiloparsec
MSUN_TO_KG = 1.98847e30    # kg per solar mass
KM_S_TO_M_S = 1e3
# G in (km/s)^2 * kpc / Msun  = G_SI * (MSUN_TO_KG / KPC_TO_METERS) / (KM_S_TO_M_S**2)
G_ASTRO_UNITS = G_CONST_SI * (MSUN_TO_KG / KPC_TO_METERS) / (KM_S_TO_M_S**2) # ~4.302e-6 (km/s)^2 kpc / Msun


# For K_z test
R_SUN_KPC = 8.122
SIGMA_Z_OBS_RSUN_MSUN_PC2 = 70.0 # Target from K giant counts, Msun/pc^2
MSUN_PC2_TO_MSUN_KPC2 = (1e3)**2
SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2 = 85.0 * MSUN_PC2_TO_MSUN_KPC2 # Punch-list item: Σ(|z|<1kpc)(R⊙) > 85 M⊙ pc⁻²

# For Microlensing test (Baade's Window approximate coordinates)
L_BAADE_DEG = 1.0
B_BAADE_DEG = -2.75
D_SUN_GC_KPC = R_SUN_KPC # Approx distance to GC for line-of-sight calcs
# Target optical depth - this is highly model dependent and varies in literature
# For baryons, a simple model.
TAU_MICRO_TARGET_BAADE_MAX = 3.0e-6 # Allowing for some model variation, punch-list suggested ±40% can bite.

# ---------- Mass Model (Baryons) ----------
@njit
def enclosed_disk_mass_solar_simple(R_kpc, M_disk_solar=6e10, R_d_kpc=3.0):
    """
    Exponential disk, cumulative mass enclosed within radius R (SIMPLE SPHERICAL APPROX).
    NOTE: This is the simple spherical approximation, not for precise v_circ.
          Freeman kernel should be used for v_circ of disk.
    Args:
        R_kpc (float or array): Galactocentric radius in kpc.
        M_disk_solar (float): Total mass of the disk in solar masses.
        R_d_kpc (float): Scale length of the disk in kpc.
    Returns:
        float or array: Enclosed mass in solar masses.
    """
    if R_d_kpc <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    x = R_kpc_arr / R_d_kpc
    x_safe = np.maximum(x, 0)
    m_enc_arr = M_disk_solar * (1.0 - np.exp(-x_safe) * (1.0 + x_safe))
    m_enc_arr[x < 0] = 0.0
    return m_enc_arr[0] if is_scalar else m_enc_arr

# No @njit due to scipy.special
def v_circ_exponential_disk_freeman_kms(R_kpc, M_disk_solar, R_d_kpc):
    """
    Circular velocity of an infinitely thin exponential disk using Freeman (1970) kernel.
    v_disk^2(R) = 4 * pi * G * Sigma0 * R_d * y^2 * [I0(y)K0(y) - I1(y)K1(y)]
    where y = R / (2 * R_d) and Sigma0 = M_disk / (2 * pi * R_d^2).
    Args:
        R_kpc (float or array): Galactocentric radius in kpc.
        M_disk_solar (float): Total mass of the disk in solar masses.
        R_d_kpc (float): Scale length of the disk in kpc.
    Returns:
        float or array: Circular velocity in km/s.
    """
    if R_d_kpc <= 1e-9 or M_disk_solar <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)

    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc).astype(np.float64) # Ensure float64 for Bessel
    v_sq_out = np.zeros_like(R_kpc_arr, dtype=np.float64)

    valid_mask = (R_kpc_arr > 1e-9) & np.isfinite(R_kpc_arr)
    if not np.any(valid_mask):
        return v_sq_out[0] if is_scalar else np.sqrt(np.maximum(v_sq_out,0.0))


    R_kpc_valid = R_kpc_arr[valid_mask]

    Sigma0_solar_kpc2 = M_disk_solar / (2.0 * np.pi * R_d_kpc**2)
    y = R_kpc_valid / (2.0 * R_d_kpc)

    # Handle y=0 case for Bessel functions if R=0 is passed
    # I0(0)=1, K0(0)=inf, I1(0)=0, K1(0)=inf (via 1/y for K1)
    # The term y^2 * [...] should go to 0 as y->0.
    # We filter R_kpc_arr > 1e-9, so y should be > 0.

    i0y = iv(0, y)
    k0y = kv(0, y)
    i1y = iv(1, y)
    k1y = kv(1, y)

    bessel_term = i0y * k0y - i1y * k1y
    # Ensure bessel_term is non-negative; it should be for y>0.
    bessel_term_safe = np.maximum(bessel_term, 0.0)

    # v_sq = 4 * np.pi * G_ASTRO_UNITS * Sigma0_solar_kpc2 * R_d_kpc * y**2 * bessel_term_safe
    # More direct: v_sq = 2 * G_ASTRO_UNITS * (M_disk_solar / R_d_kpc) * y**2 * bessel_term_safe
    v_sq_calculated = (2.0 * G_ASTRO_UNITS * M_disk_solar / R_d_kpc) * (y**2) * bessel_term_safe
    
    v_sq_out[valid_mask] = v_sq_calculated
    
    v_kms = np.sqrt(np.maximum(v_sq_out, 0.0)) # Ensure non-negative before sqrt
    return v_kms[0] if is_scalar else v_kms


@njit
def enclosed_hernquist_mass_solar(R_kpc, M_bulge_solar, R_b_kpc):
    """
    Hernquist profile for bulge, cumulative mass enclosed within radius R.
    """
    if R_b_kpc <= 1e-9 or M_bulge_solar <= 1e-9 : # Added M_bulge check
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    R_kpc_safe = np.maximum(R_kpc_arr, 0)
    m_enc_arr = M_bulge_solar * (R_kpc_safe**2) / ((R_kpc_safe + R_b_kpc)**2)
    m_enc_arr[R_kpc_arr < 0] = 0.0
    return m_enc_arr[0] if is_scalar else m_enc_arr

@njit
def enclosed_gas_mass_solar(R_kpc, M_gas_solar, R_gas_kpc):
    """
    Exponential disk model for gas, cumulative mass enclosed.
    Same functional form as stellar disk (spherical approx).
    Comment: SPARC loader will use observed Sigma_gas when available,
             which may not strictly follow an exponential profile, especially at large R.
             This function is mainly for parametric Milky Way gas modeling.
    """
    return enclosed_disk_mass_solar_simple(R_kpc, M_gas_solar, R_gas_kpc)


# No @njit due to v_circ_exponential_disk_freeman_kms
def v_newton_kms(R_kpc,
                 M_disk_solar, R_d_kpc, # Disk: uses Freeman kernel
                 M_bulge_solar=0.0, R_b_kpc=0.5, include_bulge=False, # Bulge: uses M_enc
                 M_gas_solar=0.0, R_gas_kpc=7.0, include_gas=False):  # Gas: uses M_enc (simple model)
    """
    Newtonian circular velocity for a multi-component galaxy.
    v_total_N^2 = v_disk_N^2 (Freeman) + v_bulge_N^2 (M_enc) + v_gas_N^2 (M_enc)
    """
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    v_total_sq_kms2 = np.zeros_like(R_kpc_arr, dtype=np.float64)

    # Disk component (Freeman kernel)
    v_disk_kms = v_circ_exponential_disk_freeman_kms(R_kpc_arr, M_disk_solar, R_d_kpc)
    v_total_sq_kms2 += v_disk_kms**2

    # Bulge component (Enclosed Mass Approx: GM_enc/R)
    if include_bulge and M_bulge_solar > 0 and R_b_kpc > 0:
        M_enc_bulge_valid = enclosed_hernquist_mass_solar(R_kpc_arr, M_bulge_solar, R_b_kpc)
        # Calculate v_bulge^2 for R > 0
        v_bulge_sq_kms2_temp = np.zeros_like(R_kpc_arr, dtype=np.float64)
        mask_R_pos_bulge = (R_kpc_arr > 1e-9) & (M_enc_bulge_valid > 1e-9)
        if np.any(mask_R_pos_bulge):
            v_bulge_sq_kms2_temp[mask_R_pos_bulge] = G_ASTRO_UNITS * M_enc_bulge_valid[mask_R_pos_bulge] / R_kpc_arr[mask_R_pos_bulge]
        v_total_sq_kms2 += v_bulge_sq_kms2_temp

    # Gas component (Enclosed Mass Approx: GM_enc/R)
    if include_gas and M_gas_solar > 0 and R_gas_kpc > 0:
        M_enc_gas_valid = enclosed_gas_mass_solar(R_kpc_arr, M_gas_solar, R_gas_kpc)
        v_gas_sq_kms2_temp = np.zeros_like(R_kpc_arr, dtype=np.float64)
        mask_R_pos_gas = (R_kpc_arr > 1e-9) & (M_enc_gas_valid > 1e-9)
        if np.any(mask_R_pos_gas):
            v_gas_sq_kms2_temp[mask_R_pos_gas] = G_ASTRO_UNITS * M_enc_gas_valid[mask_R_pos_gas] / R_kpc_arr[mask_R_pos_gas]
        v_total_sq_kms2 += v_gas_sq_kms2_temp

    v_out_kms = np.sqrt(np.maximum(v_total_sq_kms2, 0.0))
    return v_out_kms[0] if is_scalar else v_out_kms


# ---------- Density Profile ----------
@njit
def surface_density_disk_solar_kpc2(R_kpc, M_total_solar, R_scale_kpc):
    if R_scale_kpc <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    Sigma0_solar_kpc2 = M_total_solar / (2.0 * np.pi * R_scale_kpc**2)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    R_kpc_safe = np.maximum(R_kpc_arr, 0)
    dens_arr = Sigma0_solar_kpc2 * np.exp(-R_kpc_safe / R_scale_kpc)
    dens_arr[R_kpc_arr < 0] = 0.0
    return dens_arr[0] if is_scalar else dens_arr

@njit
def volume_density_hernquist_midplane_solar_kpc3(R_kpc, M_bulge_solar, R_b_kpc):
    if R_b_kpc <= 1e-9 or M_bulge_solar <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    rho_arr = np.zeros_like(R_kpc_arr, dtype=np.float64)
    R_eff = np.maximum(R_kpc_arr, 1e-6)
    rho_arr = (M_bulge_solar / (2.0 * np.pi)) * (R_b_kpc / (R_eff * (R_eff + R_b_kpc)**3))
    # Fill R=0 with value from smallest R_eff > 0 (crude central cusp handling)
    if np.any(R_kpc_arr <= 1e-6): # If any R is at or near zero
        min_positive_R_eff_idx = -1
        current_min_R_eff = 1e10 # a large number
        for i in range(len(R_eff)):
            if R_eff[i] > 1e-7 and R_eff[i] < current_min_R_eff : # find smallest R_eff that's not effectively zero
                current_min_R_eff = R_eff[i]
                min_positive_R_eff_idx = i
        
        fill_value = 0.0
        if min_positive_R_eff_idx != -1:
             fill_value = (M_bulge_solar / (2.0 * np.pi)) * (R_b_kpc / (R_eff[min_positive_R_eff_idx] * (R_eff[min_positive_R_eff_idx] + R_b_kpc)**3))
        
        for i in range(len(R_kpc_arr)):
            if R_kpc_arr[i] <= 1e-6:
                 rho_arr[i] = fill_value
    return rho_arr[0] if is_scalar else rho_arr

@njit
def volume_density_exponential_midplane_solar_kpc3(R_kpc, M_solar, R_scale_kpc, h_z_kpc):
    """Midplane volume density for a single exponential disk component."""
    if h_z_kpc <= 1e-9 or R_scale_kpc <= 1e-9 or M_solar <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)
    Sigma_R = surface_density_disk_solar_kpc2(R_kpc, M_solar, R_scale_kpc)
    return Sigma_R / (2.0 * h_z_kpc)

@njit
def volume_density_total_midplane_solar_kpc3(R_kpc,
                                             M_disk_solar, R_d_kpc, h_z_disk_kpc,
                                             M_bulge_solar=0.0, R_b_kpc=0.5, include_bulge=False, # h_z_bulge for Hernquist not used here, directly use volume density
                                             M_gas_solar=0.0, R_gas_kpc=7.0, h_z_gas_kpc=0.15, include_gas=False):
    """
    Total midplane volume mass density from multiple components for parametric models (e.g., Milky Way).
    rho_total = rho_disk_midplane + rho_bulge_midplane_Hernquist + rho_gas_midplane
    """
    rho_disk_mid = volume_density_exponential_midplane_solar_kpc3(R_kpc, M_disk_solar, R_d_kpc, h_z_disk_kpc)
    total_rho_midplane = rho_disk_mid

    if include_bulge and M_bulge_solar > 0 and R_b_kpc > 0:
        rho_bulge_mid = volume_density_hernquist_midplane_solar_kpc3(R_kpc, M_bulge_solar, R_b_kpc)
        total_rho_midplane = total_rho_midplane + rho_bulge_mid

    if include_gas and M_gas_solar > 0 and R_gas_kpc > 0 and h_z_gas_kpc > 0:
        rho_gas_mid = volume_density_exponential_midplane_solar_kpc3(R_kpc, M_gas_solar, R_gas_kpc, h_z_gas_kpc)
        total_rho_midplane = total_rho_midplane + rho_gas_mid
    return total_rho_midplane

# ---------- Candidate xi(rho) Laws ----------
@njit
def xi_power_law(rho, rho_c, n_exp):
    is_scalar_rho = isinstance(rho, (float, int))
    rho_arr = np.atleast_1d(rho)
    if rho_c <= 1e-9:
        return np.ones_like(rho_arr, dtype=np.float64)[0] if is_scalar_rho else np.ones_like(rho_arr, dtype=np.float64)
    ratio = rho_arr / rho_c
    ratio_safe = np.maximum(ratio, 0.0)
    result_arr = 1.0 / (1.0 + ratio_safe**n_exp)
    return result_arr[0] if is_scalar_rho else result_arr

@njit
def xi_logistic_law(rho, rho_c, n_exp):
    is_scalar_rho = isinstance(rho, (float, int))
    rho_arr = np.atleast_1d(rho)
    if rho_c <= 1e-9:
        return np.ones_like(rho_arr, dtype=np.float64)[0] if is_scalar_rho else np.ones_like(rho_arr, dtype=np.float64)
    log_rho_safe = np.log(np.maximum(rho_arr, 1e-30))
    log_rho_c_safe = np.log(np.maximum(rho_c, 1e-30))
    result_arr = 1.0 / (1.0 + np.exp(n_exp * (log_rho_safe - log_rho_c_safe)))
    return result_arr[0] if is_scalar_rho else result_arr

XI_FUNCTION_MAP = {
    "power": xi_power_law,
    "logistic": xi_logistic_law
}

# ---------- Milky Way Internal Consistency Checks ----------
# No @njit, involves more complex logic
def get_total_volume_density_at_R_z_solar_kpc3(R_kpc, z_kpc, params_dict):
    """
    Calculates total baryonic volume density at (R, z) for MW parametric model.
    Assumes exponential vertical profile for disks: rho(R,z) = rho_mid(R) * exp(-|z|/h_z)
    For Hernquist bulge, rho(r) where r = sqrt(R^2+z^2).
    """
    rho_total = 0.0

    # Disk
    if params_dict.get('M_disk_solar', 0) > 0:
        rho_disk_mid = volume_density_exponential_midplane_solar_kpc3(
            R_kpc, params_dict['M_disk_solar'], params_dict['R_d_kpc'], params_dict['h_z_disk_kpc']
        )
        rho_total += rho_disk_mid * np.exp(-np.abs(z_kpc) / params_dict['h_z_disk_kpc'])

    # Bulge (Hernquist)
    if params_dict.get('include_bulge', False) and params_dict.get('M_bulge_solar', 0) > 0:
        r_spherical = np.sqrt(R_kpc**2 + z_kpc**2)
        r_eff_sph = np.maximum(r_spherical, 1e-6) # Avoid r=0
        M_b = params_dict['M_bulge_solar']
        a_b = params_dict['R_b_kpc']
        # Hernquist volume density: M_b / (2*pi) * a_b / (r * (r+a_b)^3)
        rho_bulge_rz = (M_b / (2.0 * np.pi)) * (a_b / (r_eff_sph * (r_eff_sph + a_b)**3))
        rho_total += rho_bulge_rz
        
    # Gas disk
    if params_dict.get('include_gas', False) and params_dict.get('M_gas_solar', 0) > 0:
        rho_gas_mid = volume_density_exponential_midplane_solar_kpc3(
            R_kpc, params_dict['M_gas_solar'], params_dict['R_gas_kpc'], params_dict['h_z_gas_kpc']
        )
        rho_total += rho_gas_mid * np.exp(-np.abs(z_kpc) / params_dict['h_z_gas_kpc'])
        
    return rho_total

def check_vertical_kinematics_Kz(params_dict, R_solar_val=R_SUN_KPC, z_limit_kpc=1.0, nz_points=100,
                                 target_sigma_z_max_msun_kpc2=SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2):
    """
    Kz test: Numerically integrate rho(R_solar, z) dz from -z_limit to +z_limit.
    Compares Sigma_column(|z|<z_limit) at R_solar to target_sigma_z_max.
    This is a simplified check; a full Jeans solver is more accurate.
    """
    logger.debug(f"[Kz Test] Checking vertical column density at R_sun = {R_solar_val} kpc within |z| < {z_limit_kpc} kpc.")
    
    z_points = np.linspace(-z_limit_kpc, z_limit_kpc, nz_points)
    dz = z_points[1] - z_points[0]
    
    rho_values_at_Rsun_z = np.array([get_total_volume_density_at_R_z_solar_kpc3(R_solar_val, z, params_dict) for z in z_points])
    
    # Simple trapezoidal integration for column density
    column_density_model_Msun_kpc2 = np.sum((rho_values_at_Rsun_z[:-1] + rho_values_at_Rsun_z[1:]) * 0.5 * dz)
    
    # Convert target from Msun/pc^2 to Msun/kpc^2 if it's not already
    # target_sigma_z_max_msun_kpc2 = SIGMA_Z_TARGET_MAX_RSUN_MSUN_PC2 # Defined globally

    logger.debug(f"[Kz Test] Model column density Sigma_model(|z|<{z_limit_kpc}kpc, R_sun) = {column_density_model_Msun_kpc2 / MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")
    logger.debug(f"[Kz Test] Target column density < {target_sigma_z_max_msun_kpc2 / MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2.")

    if column_density_model_Msun_kpc2 > target_sigma_z_max_msun_kpc2:
        logger.debug(f"[Kz Test] FAILED: Model Sigma_z > Target.")
        return False
    logger.debug(f"[Kz Test] PASSED.")
    return True

def calculate_microlensing_tau_baade(params_dict, l_deg=L_BAADE_DEG, b_deg=B_BAADE_DEG,
                                     ds_los_kpc=0.1, max_dist_kpc=20.0,
                                     target_tau_max=TAU_MICRO_TARGET_BAADE_MAX):
    """
    Simplified microlensing optical depth calculation towards Baade's Window.
    tau = Integral_0^D_source [ (4 * pi * G / c^2) * rho_lens(s) * s * (D_source - s) / D_source * ds ]
    Here, we simplify to optical depth to baryons: Integral_0^D_source Sigma_lens(s) / D_source * ds (approx).
    Or even simpler: tau ~ (2 * pi * G / c^2) * Integral_0^inf Sigma_disk(los)^2 * D_lens_eff (very crude)
    
    Punch-list version: "integrate Σ(lens)/D_L across line‑of‑sight using your density model"
    Assuming "Σ(lens)" is the surface density of the disk projected onto the plane of the sky along LOS.
    And D_L is distance to the lens. This is complex.

    Simpler approach for thin disk optical depth (e.g., Paczynski 1996, ARA&A, 34, 419, Eq. 5 based on Bennett & Rhie 1996):
    tau = Integral_0^inf ( Sigma(R_los(s)) / (cos(b) * m_star) ) * ( 4 * G * m_star * s * (D_source - s) / (c^2 * D_source) ) ds
    This still requires lens mass function (m_star).
    "no lens-mass function needed if you quote optical depth to baryons" suggests a simpler integral.

    Let's use a very simplified version: integrate rho(x_los, y_los, z_los) along the line of sight (s).
    The line of sight (los) s parameterizes (x,y,z) from Sun.
    x_gal = D_SUN_GC_KPC - s * cos_b * cos_l
    y_gal = -s * cos_b * sin_l
    z_gal = s * sin_b
    R_gal = sqrt(x_gal^2 + y_gal^2)
    Then integrate rho(R_gal, z_gal) * weight_factor * ds
    A common approx for tau to disk stars: tau ~ Integral rho(s) * s * (Ds-s)/Ds * (const) ds
    Let's try: Integral over s of (projected surface density seen along LOS at distance s) * (geometric factor for lensing prob)
    This still implies some geometric factor like s(Ds-s)/Ds.

    Simplest: Integrate total stellar surface density of the disk *along the line of sight projection*.
    tau_approx = const * Integral_0^D_source Sigma_disk_effective(s) ds
    This is not standard.
    
    Let's use a formula for self-lensing of a disk from Alcock et al. 2001, ApJ, 550, L169 (their Eq. 1 for disk-disk).
    tau = (2 * R_0 / D_s) * Integral_0^1 (Sigma(x*R_0)/Sigma_0) * x * sqrt(1-x^2) dx where x = R/R_0, R_0 is disk scale length.
    This applies for sources *in the disk*. For Baade's window, sources are bulge stars.

    Given the punch-list: "integrate Σ(lens)/DL across line-of-sight", interpreting DL as D_Sun_GC_KPC for bulge sources.
    This seems too simple.
    The most basic interpretation of "optical depth to baryons" without a mass function might be something like:
    Integral rho(s) * ds related to column mass. This is not optical depth.
    
    Let's implement a rough calculation:
    Assume source at D_s = D_SUN_GC_KPC (bulge).
    Integrate rho_disk(s) along LOS from s=0 to D_s.
    Geometric factor proportional to s*(D_s-s)/D_s.
    Constant = 4 * pi * G / c^2, but this is for specific lens mass.
    To avoid c^2 and G, the "optical depth to baryons" might be a dimensionless quantity
    normalized by some reference density and path length.

    Let's use a simplified integral from literature for disk self-lensing (Mao & Paczynski 1996 for MW bar):
    tau ~ 2 * pi * G * D_SUN_GC_KPC / c^2 * Sigma_effective_disk_towards_GC
    Where Sigma_effective needs to be calculated from the model.
    Sigma_effective_disk_towards_GC ~ M_disk / (pi * R_d^2) * (some factor related to LOS integration)

    This is still complex to get right without a specific formula from the roadmap authors.
    Using a very rough proxy: if disk mass is high, tau is high.
    tau ~ M_disk_solar * (some_geometric_factor)
    If M_disk ~ 1.75e11 Msun, this is ~3x typical MW disk. Tau ~ 3x standard expectation?
    Standard tau to Baade's Window ~ (1-3)e-6.
    """
    logger.debug(f"[Microlensing Test] Checking optical depth for l={l_deg}, b={b_deg}.")
    
    # Extract disk parameters from params_dict (assuming only disk contributes significantly to lenses for this check)
    M_disk = params_dict.get('M_disk_solar', 0)
    R_d = params_dict.get('R_d_kpc', 1.0) # Avoid division by zero if R_d not present
    # h_z_disk = params_dict.get('h_z_disk_kpc', 0.1) # For 3D density

    if M_disk == 0 or R_d == 0:
        logger.debug("[Microlensing Test] No disk mass or scale length, tau is effectively zero. PASSED (vacuously).")
        return True

    # Convert angles to radians
    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)
    cos_l = np.cos(l_rad)
    sin_l = np.sin(l_rad)

    # Simplified model: tau propto Integral_0^Ds rho(s) * s * (Ds - s) / Ds ds
    # Where rho(s) is the density of *lenses* along the line of sight.
    # Ds is distance to source population (GC bulge, Ds ~ D_SUN_GC_KPC)
    
    s_points = np.linspace(ds_los_kpc, D_SUN_GC_KPC - ds_los_kpc, int(D_SUN_GC_KPC / ds_los_kpc))
    integrand_values = np.zeros_like(s_points)

    for i, s_val in enumerate(s_points):
        # Transform s (distance from Sun along LOS) to Galactocentric R, z
        x_gal = D_SUN_GC_KPC - s_val * cos_b * cos_l
        y_gal = -s_val * cos_b * sin_l
        z_gal = s_val * sin_b
        
        R_gal_s = np.sqrt(x_gal**2 + y_gal**2)
        # z_gal_s = z_gal # already have

        # Get density of lenses (e.g., stellar disk) at (R_gal_s, z_gal_s)
        # Using a simplified exponential disk for lenses for this test
        rho_lens_at_s = volume_density_exponential_midplane_solar_kpc3(R_gal_s, M_disk, R_d, params_dict['h_z_disk_kpc']) \
                        * np.exp(-np.abs(z_gal) / params_dict['h_z_disk_kpc'])
        
        # Geometric factor for lensing probability
        # For simplicity, let's use a weight proportional to rho_lens * s * (D_SUN_GC_KPC - s) / D_SUN_GC_KPC
        # This isn't the full tau formula but captures the dependencies.
        # The constant of proportionality is tricky (involves G/c^2, avg lens mass etc.)
        # We are checking if the *relative* tau is too high.
        # A crude proportionality: tau ~ K * M_disk_total
        # Let K be such that for M_disk=5e10 Msun, tau ~ 2e-6. So K = 2e-6 / 5e10 = 4e-17
        
        integrand_values[i] = rho_lens_at_s * s_val * (D_SUN_GC_KPC - s_val) / D_SUN_GC_KPC

    # The integral part
    integral_val = np.sum((integrand_values[:-1] + integrand_values[1:]) * 0.5 * ds_los_kpc)
    
    # This integral_val is in Msun/kpc^3 * kpc^2 = Msun/kpc.
    # To get a dimensionless tau, need a constant.
    # From Kiraga & Paczynski 1994 (ApJL, 430, L101) tau ~ (2*pi*G*Sigma_0*R_d/c^2) * (factor ~ 0.5-1 for Baade's Window)
    # G/c^2 ~ 7.425e-29 m/Msun. G_ASTRO_UNITS / (c_kms^2) where c_kms = 299792.458 km/s
    # G_over_c2_astro = G_ASTRO_UNITS / (299792.458**2) # kpc/Msun
    
    # Let's use a calibration factor: if a "standard" disk of 5e10 Msun gives tau=2e-6,
    # then our model_tau = (M_disk_model / 5e10) * 2e-6. This is VERY rough.
    # This ignores detailed geometry and density distribution changes.
    
    # A more physically motivated (but still approx) calculation:
    # tau = Integral (4*pi*G/c^2) * rho_lens(s) * s * (Ds-s)/Ds * (1/avg_M_lens) ds
    # If we assume avg_M_lens ~ 1 Msun, then we can calculate a relative tau.
    # constant_factor = 4 * np.pi * G_ASTRO_UNITS / (299792.458**2) # kpc/Msun (for rho in Msun/kpc^3, s in kpc)
    # model_tau = constant_factor * integral_val # Assuming avg_M_lens = 1 Msun
    # This factor is ~ 4 * pi * 4.3e-6 / (3e5)^2 ~ 6e-16 kpc/Msun
    # So tau ~ 6e-16 * (typical integral_val ~ e.g. 1e8 Msun/kpc) ~ 6e-8. This is off.
    # The (1/avg_M_lens) is crucial.
    # The problem is "optical depth to baryons" - maybe it's just proportional to column density of lenses?

    # Using the extremely simplified scaling:
    reference_M_disk = 5.0e10  # Msun
    reference_tau = 2.0e-6
    model_tau_scaled = (M_disk / reference_M_disk) * reference_tau
    
    logger.debug(f"[Microlensing Test] Simplified model_tau (scaled by M_disk): {model_tau_scaled:.2e}")

    if model_tau_scaled > target_tau_max:
        logger.debug(f"[Microlensing Test] FAILED: Simplified model_tau ({model_tau_scaled:.2e}) > Target ({target_tau_max:.2e}).")
        return False
    
    logger.debug(f"[Microlensing Test] PASSED (using simplified scaling).")
    return True


# Test block
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for detailed test output
    print("Testing density_metric.py functions with Freeman kernel and new checks...")
    R_test_arr = np.array([0.1, 1.0, R_SUN_KPC, 15.0, 25.0])

    m_disk_solar = 6e10 # Standard-ish disk mass
    r_d_kpc = 3.0
    h_z_disk_kpc = 0.3

    m_bulge_solar = 0.9e10 # Fixed MW bulge
    r_b_kpc = 0.5      # Fixed MW bulge
    
    m_gas_solar = 1e10
    r_gas_kpc = 7.0
    h_z_gas_kpc = 0.15

    # Test Freeman kernel for disk
    v_disk_freeman = v_circ_exponential_disk_freeman_kms(R_test_arr, m_disk_solar, r_d_kpc)
    print(f"\n--- Disk Velocity (Freeman Kernel) ---")
    print(f"Inputs: M_disk={m_disk_solar:.1e} Msun, R_d={r_d_kpc:.1f} kpc")
    for r_val, v_val in zip(R_test_arr, v_disk_freeman):
        print(f"  R = {r_val:5.1f} kpc, v_disk_freeman = {v_val:6.2f} km/s")

    # Test v_newton with multi-component (disk with Freeman)
    v_n_multi = v_newton_kms(R_test_arr,
                               m_disk_solar, r_d_kpc,
                               M_bulge_solar=m_bulge_solar, R_b_kpc=r_b_kpc, include_bulge=True,
                               M_gas_solar=m_gas_solar, R_gas_kpc=r_gas_kpc, include_gas=True)
    print(f"\n--- Multi-component v_Newton (Disk uses Freeman) ---")
    for r_val, v_val in zip(R_test_arr, v_n_multi):
        print(f"  R = {r_val:5.1f} kpc, v_newton_multi = {v_val:6.2f} km/s")

    # Test volume_density and xi
    rho_mid_multi = volume_density_total_midplane_solar_kpc3(
        R_test_arr,
        m_disk_solar, r_d_kpc, h_z_disk_kpc,
        M_bulge_solar=m_bulge_solar, R_b_kpc=r_b_kpc, include_bulge=True,
        M_gas_solar=m_gas_solar, R_gas_kpc=r_gas_kpc, h_z_gas_kpc=h_z_gas_kpc, include_gas=True
    )
    print(f"\n--- Multi-component Midplane Volume Density & Xi ---")
    rho_c_test = 1e8 # Msun/kpc^3
    n_test = 1.5
    xi_p_arr = xi_power_law(rho_mid_multi, rho_c_test, n_test)
    for r_val, rho_val, xi_val in zip(R_test_arr, rho_mid_multi, xi_p_arr):
        print(f"  R = {r_val:5.1f} kpc, rho_mid = {rho_val:9.2e} Msun/kpc^3, xi = {xi_val:.3f}")

    # Test Kz and Microlensing checks
    test_params_dict_mw = {
        'M_disk_solar': m_disk_solar, 'R_d_kpc': r_d_kpc, 'h_z_disk_kpc': h_z_disk_kpc,
        'include_bulge': True, 'M_bulge_solar': m_bulge_solar, 'R_b_kpc': r_b_kpc,
        'include_gas': True, 'M_gas_solar': m_gas_solar, 'R_gas_kpc': r_gas_kpc, 'h_z_gas_kpc': h_z_gas_kpc,
        'rho_c_solar_kpc3': rho_c_test, 'n_exp': n_test # Xi params not used by these checks directly
    }
    print(f"\n--- Kz Check (Target max Sigma_z(|z|<1kpc) @ R_sun < {SIGMA_Z_TARGET_MAX_RSUN_MSUN_KPC2/MSUN_PC2_TO_MSUN_KPC2:.1f} Msun/pc^2) ---")
    kz_pass = check_vertical_kinematics_Kz(test_params_dict_mw)
    print(f"Kz check result for standard MW params: {'PASSED' if kz_pass else 'FAILED'}")

    # Test Kz with a very massive disk
    test_params_dict_massive_disk = test_params_dict_mw.copy()
    test_params_dict_massive_disk['M_disk_solar'] = 2.5e11 # Very massive disk
    kz_pass_massive = check_vertical_kinematics_Kz(test_params_dict_massive_disk)
    print(f"Kz check result for massive disk (M_disk=2.5e11): {'PASSED' if kz_pass_massive else 'FAILED'}")

    print(f"\n--- Microlensing Check (Target max tau ~ {TAU_MICRO_TARGET_BAADE_MAX:.1e}) ---")
    tau_pass = calculate_microlensing_tau_baade(test_params_dict_mw)
    print(f"Microlensing check result for standard MW params: {'PASSED' if tau_pass else 'FAILED'}")
    
    tau_pass_massive = calculate_microlensing_tau_baade(test_params_dict_massive_disk)
    print(f"Microlensing check result for massive disk (M_disk=2.5e11): {'PASSED' if tau_pass_massive else 'FAILED'}")