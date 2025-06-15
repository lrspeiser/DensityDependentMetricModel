#!/usr/bin/env python3
"""
density_metric.py - Physics layer: mass models, density profiles,
                    density-weighting functions (xi), and full velocity model.
"""
import numpy as np
from numba import njit

# ---------- Physical Constants ----------
G_CONST = 6.67430e-11     # m^3 kg^-1 s^-2
KPC_TO_METERS = 3.08567758149e19 # meters per kiloparsec
MSUN_TO_KG = 1.98847e30    # kg per solar mass
KM_S_TO_M_S = 1e3

# ---------- Mass Model (Baryons) ----------
@njit
def enclosed_disk_mass_solar(R_kpc, M_disk_solar=6e10, R_d_kpc=3.0):
    """
    Exponential disk, cumulative mass enclosed within radius R.
    Args:
        R_kpc (float or array): Galactocentric radius in kpc.
        M_disk_solar (float): Total mass of the disk in solar masses.
        R_d_kpc (float): Scale length of the disk in kpc.
    Returns:
        float or array: Enclosed mass in solar masses.
    """
    # Ensure R_d_kpc is positive
    if R_d_kpc <= 1e-9: # Use a small epsilon instead of just 0
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)

    is_scalar = isinstance(R_kpc, (float, int))
    # Work with R_kpc as an array internally for consistent operations
    R_kpc_arr = np.atleast_1d(R_kpc)

    x = R_kpc_arr / R_d_kpc
    x_safe = np.maximum(x, 0) # Ensure x is not negative for np.exp
    m_enc_arr = M_disk_solar * (1.0 - np.exp(-x_safe) * (1.0 + x_safe))
    m_enc_arr[x < 0] = 0.0 # Set mass to 0 for negative input R_kpc

    return m_enc_arr[0] if is_scalar else m_enc_arr


@njit
def v_newton_kms(R_kpc, M_disk_solar, R_d_kpc):
    """
    Newtonian circular velocity for an exponential disk.
    Args:
        R_kpc (float or array): Galactocentric radius in kpc.
        M_disk_solar (float): Total mass of the disk in solar masses.
        R_d_kpc (float): Scale length of the disk in kpc.
    Returns:
        float or array: Newtonian velocity in km/s.
    """
    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)
    v_out = np.zeros_like(R_kpc_arr, dtype=np.float64)

    valid_mask = R_kpc_arr > 1e-9 # Use a small epsilon

    if np.any(valid_mask):
        R_kpc_valid = R_kpc_arr[valid_mask]
        M_enc_solar_valid = enclosed_disk_mass_solar(R_kpc_valid, M_disk_solar, R_d_kpc)
        M_enc_kg_valid = M_enc_solar_valid * MSUN_TO_KG

        mass_positive_mask = M_enc_kg_valid > 1e-9 # Check for positive mass

        if np.any(mass_positive_mask):
            R_m_valid_mass_pos = R_kpc_valid[mass_positive_mask] * KPC_TO_METERS
            M_enc_kg_valid_mass_pos = M_enc_kg_valid[mass_positive_mask]

            # Numba requires arrays for sqrt to be of same type or castable
            v_calc = np.sqrt(G_CONST * M_enc_kg_valid_mass_pos / R_m_valid_mass_pos) / KM_S_TO_M_S

            # Place results back into v_out using careful indexing
            # Create a temporary array for results on R_kpc_valid
            temp_v_on_R_valid = np.zeros_like(R_kpc_valid, dtype=np.float64)
            temp_v_on_R_valid[mass_positive_mask] = v_calc
            v_out[valid_mask] = temp_v_on_R_valid

    return v_out[0] if is_scalar else v_out


# ---------- Density Profile ----------
@njit
def surface_density_solar_kpc2(R_kpc, Sigma0_solar_kpc2, R_d_kpc):
    """
    Exponential disk surface mass density.
    """
    if R_d_kpc <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)

    is_scalar = isinstance(R_kpc, (float, int))
    R_kpc_arr = np.atleast_1d(R_kpc)

    R_kpc_safe = np.maximum(R_kpc_arr, 0)
    dens_arr = Sigma0_solar_kpc2 * np.exp(-R_kpc_safe / R_d_kpc)
    dens_arr[R_kpc_arr < 0] = 0.0

    return dens_arr[0] if is_scalar else dens_arr


@njit
def volume_density_midplane_solar_kpc3(R_kpc, Sigma0_solar_kpc2, R_d_kpc, h_z_kpc=0.3):
    """
    Midplane volume mass density assuming an exponential disk vertically.
    """
    if h_z_kpc <= 1e-9:
        if isinstance(R_kpc, (float, int)): return 0.0
        return np.zeros_like(R_kpc, dtype=np.float64)

    Sigma_R = surface_density_solar_kpc2(R_kpc, Sigma0_solar_kpc2, R_d_kpc)
    # Sigma_R can be scalar or array, division will broadcast if needed
    return Sigma_R / (2.0 * h_z_kpc)

# ---------- Candidate xi(rho) Laws ----------
@njit
def xi_power_law(rho, rho_c, n_exp):
    """ Power law for xi: 1 / (1 + (rho/rho_c)^n) """
    is_scalar_rho = isinstance(rho, (float, int))
    rho_arr = np.atleast_1d(rho) # Ensure rho is an array for consistent operations

    if rho_c <= 1e-9: # Avoid division by zero or invalid rho_c
        # If rho_c is invalid, xi effectively has no impact (or is 1 if modification makes things weaker)
        # Returning array of 1s means no modification.
        return np.ones_like(rho_arr, dtype=np.float64)[0] if is_scalar_rho else np.ones_like(rho_arr, dtype=np.float64)

    ratio = rho_arr / rho_c
    ratio_safe = np.maximum(ratio, 0.0) # Density ratio cannot be negative

    result_arr = 1.0 / (1.0 + ratio_safe**n_exp)

    return result_arr[0] if is_scalar_rho else result_arr

@njit
def xi_logistic_law(rho, rho_c, n_exp): # n_exp here is 'k' or steepness in logistic func
    """ Logistic function for xi: 1 / (1 + exp(n*(log(rho) - log(rho_c)))) """
    is_scalar_rho = isinstance(rho, (float, int))
    rho_arr = np.atleast_1d(rho)

    if rho_c <= 1e-9:
        return np.ones_like(rho_arr, dtype=np.float64)[0] if is_scalar_rho else np.ones_like(rho_arr, dtype=np.float64)

    # Handle rho=0 or negative rho safely for log
    log_rho_safe = np.log(np.maximum(rho_arr, 1e-30)) # Use a small floor for rho before log
    log_rho_c_safe = np.log(np.maximum(rho_c, 1e-30)) # Ensure rho_c is also positive for log

    result_arr = 1.0 / (1.0 + np.exp(n_exp * (log_rho_safe - log_rho_c_safe)))

    return result_arr[0] if is_scalar_rho else result_arr

XI_FUNCTION_MAP = {
    "power": xi_power_law,
    "logistic": xi_logistic_law
}
# ... (v_circ_density_metric_kms and __main__ test block remain the same) ...
# The v_model_for_emcee in run_fit.py will call these corrected xi functions.

# Test block (for standalone testing of this file)
if __name__ == '__main__':
    print("Testing density_metric.py functions...")
    # Test v_newton
    print(f"v_newton at 8kpc (6e10 Msun, 3kpc R_d): {v_newton_kms(8.0, 6e10, 3.0):.2f} km/s")
    print(f"v_newton array: {v_newton_kms(np.array([1.0, 8.0, 15.0]), 6e10, 3.0)}")

    # Test density
    sigma0_test = 6e10 / (2 * np.pi * 3.0**2) # Msun/kpc^2
    print(f"Sigma0 for test disk: {sigma0_test:.2e} Msun/kpc^2")
    rho_test_scalar = volume_density_midplane_solar_kpc3(8.0, sigma0_test, 3.0, 0.3)
    print(f"Midplane density at 8kpc (test disk, hz=0.3): {rho_test_scalar:.2e} Msun/kpc^3")
    rho_test_array = volume_density_midplane_solar_kpc3(np.array([1.0, 8.0, 15.0]), sigma0_test, 3.0, 0.3)
    print(f"Midplane density array: {rho_test_array}")

    # Test xi
    rho_c_test = 1e7 # Msun/kpc^3
    n_test = 1.0
    xi_p_scalar = xi_power_law(rho_test_scalar, rho_c_test, n_test)
    xi_l_scalar = xi_logistic_law(rho_test_scalar, rho_c_test, n_test)
    print(f"xi_power_law scalar for rho={rho_test_scalar:.2e}, rho_c={rho_c_test:.2e}, n={n_test}: {xi_p_scalar:.3f}")
    print(f"xi_logistic_law scalar for rho={rho_test_scalar:.2e}, rho_c={rho_c_test:.2e}, n={n_test}: {xi_l_scalar:.3f}")

    xi_p_array = xi_power_law(rho_test_array, rho_c_test, n_test)
    xi_l_array = xi_logistic_law(rho_test_array, rho_c_test, n_test)
    print(f"xi_power_law array: {xi_p_array}")
    print(f"xi_logistic_law array: {xi_l_array}")

    # Test v_model_for_emcee structure (mimicking run_fit.py)
    def local_test_v_model_for_emcee(R_kpc_array, theta_params, xi_type_str="power"):
        M_disk_solar, R_d_kpc, rho_c_solar_kpc3, n_exp, h_z_kpc = theta_params
        if R_d_kpc <= 0: return np.zeros_like(R_kpc_array) if isinstance(R_kpc_array, np.ndarray) else 0.0
        Sigma0_solar_kpc2 = M_disk_solar / (2.0 * np.pi * R_d_kpc**2)
        v_n_kms = v_newton_kms(R_kpc_array, M_disk_solar, R_d_kpc)
        rho_midplane_solar_kpc3 = volume_density_midplane_solar_kpc3(
            R_kpc_array, Sigma0_solar_kpc2, R_d_kpc, h_z_kpc
        )
        xi_func = XI_FUNCTION_MAP.get(xi_type_str) # This will be a Numba dispatcher
        xi_values = xi_func(rho_midplane_solar_kpc3, rho_c_solar_kpc3, n_exp)
        xi_values_safe = np.maximum(xi_values, 0)
        xi_values_safe = np.nan_to_num(xi_values_safe, nan=0.0)
        v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
        return v_mod_kms

    test_pars = [6e10, 3.0, rho_c_test, n_test, 0.3]
    r_test_array = np.array([1.0, 8.0, 15.0])
    v_mod_test_power = local_test_v_model_for_emcee(r_test_array, test_pars, xi_type_str="power")
    print(f"v_model_for_emcee (power xi) at [1, 8, 15] kpc with test pars: {v_mod_test_power}")
    v_mod_test_logistic = local_test_v_model_for_emcee(r_test_array, test_pars, xi_type_str="logistic")
    print(f"v_model_for_emcee (logistic xi) at [1, 8, 15] kpc with test pars: {v_mod_test_logistic}")