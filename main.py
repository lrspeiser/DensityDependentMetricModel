#!/usr/bin/env python3
"""
main.py (formerly run_fit.py) - Main script to orchestrate MCMC fitting of density-dependent
             metric models to Gaia rotation curve data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import argparse
from tqdm import tqdm # For progress bar
import time # For timing
import os # For process management
import signal # For process management
import subprocess # For process management
import shlex # For safe command splitting
from multiprocessing import Pool # FOR PARALLEL MCMC
import platform # To check OS for multiprocessing

# Import from local modules
try:
    from data_io import load_gaia
    from density_metric import (
        v_newton_kms, volume_density_midplane_solar_kpc3,
        XI_FUNCTION_MAP, # Dictionary of xi functions
        G_CONST, KPC_TO_METERS, MSUN_TO_KG, KM_S_TO_M_S
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure data_io.py and density_metric.py are in the same directory or Python path.")
    sys.exit(1)
import sys

# --- Function to kill existing script instances ---
def kill_existing_instances(script_name_to_kill="main.py"):
    """
    Finds and kills other running instances of this script.
    Be careful with this function.
    """
    current_pid = os.getpid()
    print(f"🌬️  Attempting to terminate other instances of '{script_name_to_kill}' (current PID: {current_pid})...")
    try:
        cmd = f"ps aux | grep python | grep {shlex.quote(script_name_to_kill)} | grep -v grep"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        killed_count = 0
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        if pid != current_pid:
                            print(f"   Killing process {pid}: {line}")
                            os.kill(pid, signal.SIGKILL) # SIGKILL is forceful
                            killed_count += 1
                            time.sleep(0.1) 
                    except ValueError:
                        print(f"   Could not parse PID from line: {line}")
                    except ProcessLookupError:
                        print(f"   Process {pid} not found (already terminated?).")
                    except Exception as e_kill:
                        print(f"   Error killing process {pid}: {e_kill}")
        if killed_count > 0:
            print(f"   ✅ Terminated {killed_count} other instance(s).")
        else:
            print(f"   No other running instances of '{script_name_to_kill}' found to terminate.")
    except Exception as e:
        print(f"   ⚠️ Error during attempt to kill existing instances: {e}")
    print("-" * 30)


# Define v_model here, to be passed to emcee.
def v_model_for_emcee(R_kpc_array, theta_params, xi_type_str="power"):
    """
    Wrapper for the velocity model suitable for emcee, selecting xi function.
    theta_params: (M_disk_solar, R_d_kpc, rho_c_solar_kpc3, n_exp, h_z_kpc)
    """
    M_disk_solar, R_d_kpc, rho_c_solar_kpc3, n_exp, h_z_kpc = theta_params

    if R_d_kpc <= 0: return np.zeros_like(R_kpc_array) if isinstance(R_kpc_array, np.ndarray) else 0.0
    Sigma0_solar_kpc2 = M_disk_solar / (2.0 * np.pi * R_d_kpc**2)

    v_n_kms = v_newton_kms(R_kpc_array, M_disk_solar, R_d_kpc)
    rho_midplane_solar_kpc3 = volume_density_midplane_solar_kpc3(
        R_kpc_array, Sigma0_solar_kpc2, R_d_kpc, h_z_kpc
    )

    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    if xi_func is None:
        raise ValueError(f"Unknown xi_type: {xi_type_str}. Available: {list(XI_FUNCTION_MAP.keys())}")

    xi_values = xi_func(rho_midplane_solar_kpc3, rho_c_solar_kpc3, n_exp)
    xi_values_safe = np.maximum(xi_values, 0)
    xi_values_safe = np.nan_to_num(xi_values_safe, nan=0.0)

    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
    return v_mod_kms


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Fit density-dependent metric model to Gaia data.")
parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'],
                    help='Type of xi(rho) function to use (power or logistic)')
parser.add_argument('--nwalkers', type=int, default=32, help='Number of MCMC walkers')
# INCREASED DEFAULT NSTEPS, but recommend overriding for final runs
parser.add_argument('--nsteps', type=int, default=10000, help='Total MCMC steps (burn-in + sampling)')
parser.add_argument('--burnin', type=int, default=2000, help='Number of burn-in steps to discard (can be auto-adjusted)')
parser.add_argument('--thin', type=int, default=50, help='Thinning factor for chain (can be auto-adjusted)')
parser.add_argument('--max_sample_gaia', type=int, default=80000, help='Max Gaia stars to load')
parser.add_argument('--force_live_gaia', action='store_true',
                    help='Force a new LIVE Gaia ADQL query, ignoring any raw data CSV cache.')
parser.add_argument('--force_reprocess', action='store_true',
                    help='Force reprocessing of raw Gaia data, ignoring any processed Parquet cache.')
parser.add_argument('--kill_existing', action='store_true', 
                    help='Attempt to kill other running instances of this script before starting.')
parser.add_argument('--ncores', type=int, default=1, # NEW ARGUMENT FOR PARALLELISM
                    help='Number of CPU cores to use for MCMC parallelization (0 or 1 means no parallel).')


ARGS = parser.parse_args()

# --- Optionally kill existing instances ---
if ARGS.kill_existing:
    current_script_name = os.path.basename(__file__)
    kill_existing_instances(script_name_to_kill=current_script_name)
    print("Continuing with the current script run...")


# --- 1. Load Data ---
# ... (Data loading remains the same) ...
print("--- Loading Data ---")
gaia_data_dict = load_gaia(sample_max=ARGS.max_sample_gaia,
                           force_new_query_gaia=ARGS.force_live_gaia,
                           force_reprocess_raw=ARGS.force_reprocess)
if gaia_data_dict is None or gaia_data_dict.get("R_kpc") is None or len(gaia_data_dict["R_kpc"]) == 0:
    print("❌ Failed to load Gaia data or data is empty. Exiting.")
    sys.exit(1)

R_obs_kpc = gaia_data_dict["R_kpc"]
v_obs_kms = gaia_data_dict["v_obs"]
sigma_v_kms = gaia_data_dict["sigma_v"]

print(f"Loaded {len(R_obs_kpc)} stars for fitting.")
if len(R_obs_kpc) == 0:
    print("❌ No data points to fit after loading. Exiting.")
    sys.exit(1)

# --- 2. Define Log-Likelihood & Priors ---
# ... (Likelihood and Prior functions remain the same) ...
print(f"\n--- Setting up Likelihood & Priors for xi_type = '{ARGS.xi}' ---")

PARAM_LABELS = [r"$M_\mathrm{disk}$ ($M_\odot$)", r"$R_d$ (kpc)", r"$\rho_c$ ($M_\odot/kpc^3$)", r"$n$", r"$h_z$ (kpc)"]

def log_prior(theta):
    M_disk, R_d, rho_c, n, h_z = theta
    if not (1e10 < M_disk < 2e11 and
            1.5 < R_d < 5.0 and
            1e5 < rho_c < 1e9 and 
            0.1 < n < 4.0 and
            0.1 < h_z < 0.7):
        return -np.inf
    return 0.0

def log_likelihood(theta, R_data, v_data, sigma_data, xi_type_selected):
    v_predicted = v_model_for_emcee(R_data, theta, xi_type_str=xi_type_selected)
    if not np.all(np.isfinite(v_predicted)):
        return -np.inf
    sigma_data_safe = np.maximum(sigma_data, 1e-9)
    residuals_sq = ((v_data - v_predicted) / sigma_data_safe)**2
    log_L = -0.5 * np.sum(residuals_sq + np.log(2 * np.pi * sigma_data_safe**2))
    if not np.isfinite(log_L):
        return -np.inf
    return log_L

def log_posterior(theta, R_data, v_data, sigma_data, xi_type_selected):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, R_data, v_data, sigma_data, xi_type_selected)
    return lp + ll


# --- 3. Run MCMC ---
print(f"\n--- Running MCMC ({ARGS.nwalkers} walkers, {ARGS.nsteps} steps, on {ARGS.ncores if ARGS.ncores > 0 else 1} core(s)) ---")
ndim = 5

p0_guess_means = np.array([6e10, 3.0, 1e7, 1.0, 0.3])
pos0 = np.zeros((ARGS.nwalkers, ndim))
prior_bounds_low = np.array([1e10, 1.5, 1e5, 0.1, 0.1])
prior_bounds_high = np.array([2e11, 5.0, 1e9, 4.0, 0.7])

for i in range(ndim):
    pos0[:, i] = np.random.uniform(prior_bounds_low[i], prior_bounds_high[i], ARGS.nwalkers)

# Setup for parallel MCMC if ncores > 1 (and not on Windows where Pool can be tricky with some setups)
pool_obj = None
if ARGS.ncores > 1 and platform.system() != "Windows": # Pool can be problematic on Windows if not using __main__ guard
    try:
        pool_obj = Pool(processes=ARGS.ncores)
        print(f"   MCMC will run in parallel on {ARGS.ncores} cores.")
    except Exception as e_pool:
        print(f"   ⚠️ Could not initialize multiprocessing Pool for MCMC (Error: {e_pool}). Running in serial.")
        pool_obj = None
elif ARGS.ncores > 1 and platform.system() == "Windows":
    print("   Multiprocessing for MCMC on Windows requested. Ensure script is run with '__main__' guard if issues arise.")
    # On Windows, Pool might need to be in if __name__ == "__main__": block,
    # but emcee handles its own Pool context usually. Let's try.
    try:
        pool_obj = Pool(processes=ARGS.ncores) # May or may not work depending on environment
        print(f"   MCMC will attempt parallel run on {ARGS.ncores} cores (Windows).")
    except Exception as e_pool_win:
        print(f"   ⚠️ Could not initialize multiprocessing Pool for MCMC on Windows (Error: {e_pool_win}). Running in serial.")
        pool_obj = None

sampler = emcee.EnsembleSampler(ARGS.nwalkers, ndim, log_posterior,
                                args=(R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS.xi),
                                pool=pool_obj) # Pass the pool to the sampler

start_time_mcmc = time.time()
with tqdm(total=ARGS.nsteps, desc="MCMC Sampling", unit="step") as pbar:
    for _ in sampler.sample(pos0, iterations=ARGS.nsteps, progress=False):
        pbar.update(1)
end_time_mcmc = time.time()

if pool_obj: # Important to close the pool
    pool_obj.close()
    pool_obj.join()

print(f"MCMC finished in {(end_time_mcmc - start_time_mcmc)/60:.2f} minutes.")

actual_burnin = ARGS.burnin
actual_thin = ARGS.thin
try:
    autocorr_time = sampler.get_autocorr_time(tol=0) # tol=0 can be faster but less precise for final check
    print(f"Autocorrelation time estimates: {autocorr_time}")

    # Calculate recommended burn-in and thin, ensuring they are integers
    finite_autocorr = autocorr_time[np.isfinite(autocorr_time)]
    if len(finite_autocorr) > 0:
        max_autocorr = np.max(finite_autocorr)
        recommended_burnin = int(np.ceil(max_autocorr * 5)) # Use ceil to be conservative
        recommended_thin = int(np.ceil(max_autocorr / 2))
        recommended_thin = max(1, recommended_thin) # Thin must be at least 1

        print(f"Recommended burn-in based on Autocorr: ~{recommended_burnin}")
        print(f"Recommended thinning based on Autocorr: ~{recommended_thin}")

        # Override ARGS.burnin and ARGS.thin if user didn't specify them high enough
        # or if user wants to use auto-detected values (could add another flag for this)
        # For now, just warn and use user's values or auto if user's are too low.
        # Let's use the recommended values if they seem more robust, but inform user.
        if ARGS.burnin < recommended_burnin:
            print(f"⚠️ User burn-in ({ARGS.burnin}) is less than recommended ({recommended_burnin}). Using recommended.")
            actual_burnin = recommended_burnin
        if ARGS.thin < recommended_thin : # Only increase thin if user's is too small
            print(f"⚠️ User thinning ({ARGS.thin}) is less than recommended ({recommended_thin}). Using recommended.")
            actual_thin = recommended_thin
    else:
        print("⚠️ All autocorrelation times are non-finite. Using user-specified burn-in/thin.")

    actual_burnin = min(actual_burnin, ARGS.nsteps -1) # Ensure burnin is not >= nsteps
    actual_thin = max(1, actual_thin) # Ensure thin is at least 1


    print(f"Using actual burn-in: {actual_burnin}")
    print(f"Using actual thinning: {actual_thin}")


    if ARGS.nsteps < actual_burnin + actual_thin * 50 : # Need enough effective samples
        print(f"⚠️ Consider increasing total steps for more effective samples post-thinning/burn-in.")

    chain_flat = sampler.get_chain(discard=actual_burnin, thin=actual_thin, flat=True)

except emcee.autocorr.AutocorrError as e_acorr:
    print(f"⚠️ Could not estimate autocorrelation time: {e_acorr}. Using specified burn-in ({ARGS.burnin}) / thin ({ARGS.thin}).")
    chain_flat = sampler.get_chain(discard=ARGS.burnin, thin=ARGS.thin, flat=True)


chain_filename = f"chain_{ARGS.xi}.npy"
np.save(chain_filename, chain_flat)
print(f"Saved flattened chain to {chain_filename} ({len(chain_flat)} samples)")

# --- 4. Posterior Diagnostics & Rotation-Curve Envelope ---
# ... (This section remains largely the same, ensure median_params uses chain_flat) ...
print("\n--- Generating Posterior Diagnostics & Plot ---")
if len(chain_flat) < ndim * 2: 
    print(f"⚠️ Chain has very few samples ({len(chain_flat)}) after thinning. Corner plot and envelope might be unreliable or fail.")
    # Use initial guess means as fallback if chain is too small for median
    median_params = p0_guess_means if len(chain_flat) == 0 else np.median(chain_flat, axis=0)
    params_16th = median_params * 0.9 # Rough fallback
    params_84th = median_params * 1.1 # Rough fallback
    param_summary_text = "Chain too small for robust parameter estimation; using fallbacks or initial guesses.\n"
    if len(chain_flat) > 0: # If some samples exist, try to get percentiles
        try:
            params_16th, params_84th = np.percentile(chain_flat, [16, 84], axis=0)
        except IndexError: # If chain_flat is 1D or too small for percentile
             pass # Keep rough fallbacks
else:
    try:
        figure = corner.corner(chain_flat, labels=PARAM_LABELS, quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs={"fontsize": 12}, truths=None)
        corner_filename = f"corner_{ARGS.xi}.png"
        figure.savefig(corner_filename)
        print(f"Saved corner plot to {corner_filename}")
        plt.close(figure)
    except Exception as e_corner:
        print(f"⚠️ Error generating corner plot: {e_corner}")

    median_params = np.median(chain_flat, axis=0)
    params_16th, params_84th = np.percentile(chain_flat, [16, 84], axis=0)

# This block executes regardless of chain_flat size, using median_params which is now defined.
print("\nFitted Parameters (Median & 68% CI):")
param_summary_text = "" # Initialize/reset
for i, label in enumerate(PARAM_LABELS):
    summary_line = (f"  {label:<25}: {median_params[i]:.3e} "
                    f" (+{params_84th[i]-median_params[i]:.2e} / -{median_params[i]-params_16th[i]:.2e})")
    print(summary_line)
    param_summary_text += summary_line + "\n"


R_plot_curve_min = np.min(R_obs_kpc) if len(R_obs_kpc) > 0 else 0.1
R_plot_curve_max = np.max(R_obs_kpc) if len(R_obs_kpc) > 0 else 25.0
R_plot_curve_min = max(0.01, R_plot_curve_min) 
R_plot_curve = np.linspace(R_plot_curve_min, R_plot_curve_max, 300)

n_samples_for_envelope = min(1000, len(chain_flat))
if n_samples_for_envelope > 0:
    chain_subset_for_envelope = chain_flat[np.random.choice(len(chain_flat), n_samples_for_envelope, replace=False)]
    v_model_samples = np.array([v_model_for_emcee(R_plot_curve, pars, xi_type_str=ARGS.xi) for pars in chain_subset_for_envelope])
    v_median_curve = np.nanmedian(v_model_samples, axis=0)
    v_16th_curve, v_84th_curve = np.nanpercentile(v_model_samples, [16, 84], axis=0)
else: 
    v_median_curve = v_model_for_emcee(R_plot_curve, median_params, xi_type_str=ARGS.xi) # Plot with median if no samples
    v_16th_curve = v_median_curve * 0.9 # Rough envelope
    v_84th_curve = v_median_curve * 1.1


# --- 5. Plot Rotation Curve ---
# ... (Plotting section remains the same) ...
plt.figure(figsize=(10,6))
plt.errorbar(R_obs_kpc, v_obs_kms, yerr=sigma_v_kms, fmt=".k", alpha=0.02, label="Gaia DR3 Stars (Subset)", zorder=1)
plt.plot(R_plot_curve, v_median_curve, color="red", lw=2.5, label=f"Density-Metric Median ({ARGS.xi} $\\xi$)", zorder=3)
plt.fill_between(R_plot_curve, v_16th_curve, v_84th_curve, color="red", alpha=0.3, zorder=2, label="68% Credible Interval")
v_newton_plot_median_params = v_newton_kms(R_plot_curve, median_params[0], median_params[1])
plt.plot(R_plot_curve, v_newton_plot_median_params, color="green", ls="--", lw=2, label="Newtonian (Median Fitted Disk)", zorder=2.5)

plt.xlabel("Galactocentric Radius R (kpc)", fontsize=12)
plt.ylabel("Tangential Velocity v (km s$^{-1}$)", fontsize=12)
plt.title(f"Milky Way Rotation Curve Fit: Density-Dependent Metric ({ARGS.xi} $\\xi$)", fontsize=14)
plt.legend(fontsize=9)
plt.grid(True, ls=':', alpha=0.7)
valid_y_for_lim = v_84th_curve[np.isfinite(v_84th_curve)]
ylim_max = max(350, np.max(valid_y_for_lim) * 1.1 if len(valid_y_for_lim) > 0 else 350)
plt.ylim(0, ylim_max)
plt.xlim(0, np.max(R_plot_curve) if len(R_plot_curve) > 0 else 25.0)
plt.tight_layout()
plot_filename_fit = f"rotation_curve_fit_{ARGS.xi}.png"
plt.savefig(plot_filename_fit, dpi=150)
print(f"Saved rotation curve plot to {plot_filename_fit}")
plt.close()

# --- 6. Model Comparison (AIC/BIC) ---
# ... (AIC/BIC section remains the same) ...
print("\n--- Model Comparison Metrics ---")
logL_samples = sampler.get_log_prob(discard=actual_burnin, thin=actual_thin, flat=True) # Use actual_burnin/thin
if len(logL_samples) > 0 and len(chain_flat) > 0 : # Ensure chain_flat also has samples
    max_log_prob_val = np.max(logL_samples)
    max_log_prob_idx = np.argmax(logL_samples) 
    best_fit_theta_mcmc = chain_flat[max_log_prob_idx]
    max_log_likelihood_val = log_likelihood(best_fit_theta_mcmc, R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS.xi)
else:
    print("⚠️ No samples in chain_flat/logL_samples to calculate max log likelihood. Using fallback.")
    max_log_likelihood_val = -np.inf 
    best_fit_theta_mcmc = p0_guess_means 

k_params = ndim
N_data = len(R_obs_kpc)

if np.isfinite(max_log_likelihood_val):
    AIC = 2 * k_params - 2 * max_log_likelihood_val
    BIC = k_params * np.log(N_data) - 2 * max_log_likelihood_val
    print(f"Model: Density-Dependent Metric with xi_type = '{ARGS.xi}'")
    print(f"  Max Log-Likelihood (at MCMC best-fit posterior): {max_log_likelihood_val:.2f}")
    print(f"  Number of parameters (k): {k_params}")
    print(f"  Number of data points (N): {N_data}")
    print(f"  AIC: {AIC:.2f}")
    print(f"  BIC: {BIC:.2f}")
    aic_bic_text = (f"Model: {ARGS.xi}\n"
                    f"Max Log-Likelihood: {max_log_likelihood_val:.2f}\n"
                    f"k: {k_params}, N: {N_data}\n"
                    f"AIC: {AIC:.2f}\nBIC: {BIC:.2f}\n\n")
else:
    print("⚠️ Max log-likelihood is not finite. Cannot calculate AIC/BIC.")
    aic_bic_text = "AIC/BIC: Not calculable (max log-likelihood not finite).\n"

# --- Verify near-centre vs outer disc explicitly ---
# ... (RMS section remains the same) ...
print("\n--- RMS Residuals in Radial Bands ---")
mask_inner = (R_obs_kpc < 5.0)
mask_outer = (R_obs_kpc > 10.0) & (R_obs_kpc < 20.0)

def calculate_rms_residuals(data_mask, best_params_for_rms):
    if np.sum(data_mask) == 0 or len(best_params_for_rms) != ndim:
        return np.nan
    v_pred_masked = v_model_for_emcee(R_obs_kpc[data_mask], best_params_for_rms, xi_type_str=ARGS.xi)
    if np.all(np.isnan(v_pred_masked)): return np.nan
    rms_val = np.sqrt(np.nanmean((v_obs_kms[data_mask] - v_pred_masked)**2)) 
    return rms_val

rms_inner = calculate_rms_residuals(mask_inner, median_params)
rms_outer = calculate_rms_residuals(mask_outer, median_params)

print(f"RMS Δv (Median Params) for R < 5 kpc  : {rms_inner:.2f} km/s (N_stars={np.sum(mask_inner)})")
print(f"RMS Δv (Median Params) for 10 < R < 20 kpc : {rms_outer:.2f} km/s (N_stars={np.sum(mask_outer)})")

rms_text = (f"RMS inner (R < 5 kpc): {rms_inner:.2f} km/s\n"
            f"RMS outer (10 < R < 20 kpc): {rms_outer:.2f} km/s\n")

# --- Save summary info to text file ---
# ... (Info saving section remains the same) ...
info_filename = f"info_summary_{ARGS.xi}.txt"
with open(info_filename, "w") as f:
    f.write(f"--- Fit Summary for xi_type = '{ARGS.xi}' ---\n")
    f.write("Fitted Parameters (Median & 68% CI):\n")
    f.write(param_summary_text)
    f.write("\n--- Model Comparison ---\n")
    f.write(aic_bic_text)
    f.write("\n--- RMS Residuals ---\n")
    f.write(rms_text)
print(f"\nSaved summary information to {info_filename}")

print("\n--- main.py Script Finished ---")