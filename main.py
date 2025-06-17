#!/usr/bin/env python3
"""
main.py (formerly run_fit.py) - Main script to orchestrate MCMC fitting of density-dependent
             metric models to Gaia rotation curve data.
""" # <<< NO INDENTATION HERE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import argparse
from tqdm import tqdm 
import time 
import os 
import signal 
import subprocess 
import shlex 
from multiprocessing import Pool, cpu_count, freeze_support # Single import for multiprocessing
import platform 

# Import from local modules
try:
    from data_io import load_gaia
    from density_metric import (
        v_newton_kms, volume_density_midplane_solar_kpc3,
        XI_FUNCTION_MAP, 
        G_CONST, KPC_TO_METERS, MSUN_TO_KG, KM_S_TO_M_S
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure data_io.py and density_metric.py are in the same directory or Python path.")
    import sys 
    sys.exit(1)
import sys 

# --- Function to kill existing script instances ---
def kill_existing_instances(script_name_to_kill="main.py"):
    current_pid = os.getpid()
    print(f"🌬️  Attempting to terminate other instances of '{script_name_to_kill}' (current PID: {current_pid})...")
    try:
        # Constructing command carefully to avoid issues with script_name_to_kill if it has spaces (though unlikely here)
        # The grep pipeline:
        # 1. ps aux: list all processes
        # 2. grep python: filter for lines containing 'python'
        # 3. grep <script_name>: further filter for lines containing the script name
        # 4. grep -v grep: exclude the grep process itself that matches the script name
        ps_cmd = "ps aux"
        grep_python_cmd = "grep python"
        grep_script_cmd = f"grep {shlex.quote(script_name_to_kill)}" # shlex.quote for safety
        grep_exclude_cmd = "grep -v grep"
        
        # Chain the commands using pipes
        p1 = subprocess.Popen(shlex.split(ps_cmd), stdout=subprocess.PIPE)
        p2 = subprocess.Popen(shlex.split(grep_python_cmd), stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
        p3 = subprocess.Popen(shlex.split(grep_script_cmd), stdin=p2.stdout, stdout=subprocess.PIPE)
        p2.stdout.close()
        p4 = subprocess.Popen(shlex.split(grep_exclude_cmd), stdin=p3.stdout, stdout=subprocess.PIPE)
        p3.stdout.close()
        
        result_stdout, result_stderr = p4.communicate()
        
        killed_count = 0
        if result_stdout:
            lines = result_stdout.decode().strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                if len(parts) > 1: # Ensure there's a PID column
                    try:
                        pid = int(parts[1])
                        if pid != current_pid:
                            print(f"   Killing process {pid}: {' '.join(parts[10:])}") # Print command part
                            os.kill(pid, signal.SIGKILL) # SIGKILL is forceful
                            killed_count += 1
                            time.sleep(0.1) # Small pause
                    except ValueError:
                        # print(f"   Could not parse PID from line: {line}") # Can be noisy
                        pass
                    except ProcessLookupError:
                        # print(f"   Process {pid} not found (already terminated?).")
                        pass
                    except Exception as e_kill:
                        print(f"   Error killing process {pid} from line '{line}': {e_kill}")
        if killed_count > 0:
            print(f"   ✅ Terminated {killed_count} other instance(s).")
        else:
            print(f"   No other running instances of '{script_name_to_kill}' found to terminate.")
    except Exception as e: 
        print(f"   ⚠️ Error during attempt to kill existing instances: {e}")
    print("-" * 30)

# Define v_model here, to be passed to emcee.
def v_model_for_emcee(R_kpc_array, theta_params, xi_type_str="power"):
    M_disk_solar, R_d_kpc, rho_c_solar_kpc3, n_exp, h_z_kpc = theta_params
    if R_d_kpc <= 1e-9: 
        return np.zeros_like(R_kpc_array, dtype=float) if isinstance(R_kpc_array, np.ndarray) else 0.0
    
    if R_d_kpc**2 < 1e-9 : 
        Sigma0_solar_kpc2 = M_disk_solar / (2.0 * np.pi * 1e-9) 
    else:
        Sigma0_solar_kpc2 = M_disk_solar / (2.0 * np.pi * R_d_kpc**2)

    v_n_kms = v_newton_kms(R_kpc_array, M_disk_solar, R_d_kpc)
    rho_midplane_solar_kpc3 = volume_density_midplane_solar_kpc3(
        R_kpc_array, Sigma0_solar_kpc2, R_d_kpc, h_z_kpc
    )
    xi_func = XI_FUNCTION_MAP.get(xi_type_str)
    if xi_func is None: 
        raise ValueError(f"Unknown xi_type: {xi_type_str}. Available: {list(XI_FUNCTION_MAP.keys())}")
    
    xi_values = xi_func(rho_midplane_solar_kpc3, rho_c_solar_kpc3, n_exp)
    xi_values_safe = np.maximum(xi_values, 0.0) 
    xi_values_safe = np.nan_to_num(xi_values_safe, nan=0.0) 
    
    v_mod_kms = v_n_kms * np.sqrt(xi_values_safe)
    return v_mod_kms

# Define Log-Likelihood & Priors (must be picklable for multiprocessing)
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
    return log_L if np.isfinite(log_L) else -np.inf

def log_posterior(theta, R_data, v_data, sigma_data, xi_type_selected):
    lp = log_prior(theta)
    if not np.isfinite(lp): 
        return -np.inf
    ll = log_likelihood(theta, R_data, v_data, sigma_data, xi_type_selected)
    if not np.isfinite(ll): # Added this check
        return -np.inf
    return lp + ll

# === Main Execution Function ===
def run_mcmc_analysis(ARGS_in):
    print("--- Loading Data ---")
    gaia_data_dict = load_gaia(sample_max=ARGS_in.max_sample_gaia,
                               force_new_query_gaia=ARGS_in.force_live_gaia,
                               force_reprocess_raw=ARGS_in.force_reprocess)
    if gaia_data_dict is None or gaia_data_dict.get("R_kpc") is None or len(gaia_data_dict["R_kpc"]) == 0:
        print("❌ Failed to load Gaia data or data is empty. Exiting.")
        sys.exit(1)

    R_obs_kpc = gaia_data_dict["R_kpc"]
    v_obs_kms = gaia_data_dict["v_obs"]
    sigma_v_kms = gaia_data_dict["sigma_v"]
    print(f"Loaded {len(R_obs_kpc)} stars for fitting.")
    if len(R_obs_kpc) == 0: 
        sys.exit("❌ No data points to fit after loading. Exiting.")

    print(f"\n--- Setting up Likelihood & Priors for xi_type = '{ARGS_in.xi}' ---")

    n_cores_to_use = ARGS_in.ncores if ARGS_in.ncores >= 1 else 1 # Ensure at least 1
    if ARGS_in.ncores > 1: # Only check cpu_count if planning to use more than 1
        try:
            available_cpus = cpu_count()
            if n_cores_to_use > available_cpus :
                print(f"⚠️ Requested {n_cores_to_use} cores, but only {available_cpus} seem available. Using {available_cpus}.")
                n_cores_to_use = available_cpus
        except NotImplementedError:
            print("⚠️ cpu_count() not available. Defaulting to serial for MCMC if --ncores > 1 was used without a specific number.")
            if ARGS_in.ncores > 1 : n_cores_to_use = 1 # Force serial if count failed and user wanted parallel

    print(f"\n--- Running MCMC ({ARGS_in.nwalkers} walkers, {ARGS_in.nsteps} steps, on {n_cores_to_use} core(s)) ---")
    ndim = 5
    p0_guess_means = np.array([6e10, 3.0, 1e7, 1.0, 0.3])
    pos0 = np.zeros((ARGS_in.nwalkers, ndim))
    prior_bounds_low = np.array([1e10, 1.5, 1e5, 0.1, 0.1])
    prior_bounds_high = np.array([2e11, 5.0, 1e9, 4.0, 0.7])
    for i in range(ndim):
        pos0[:, i] = np.random.uniform(prior_bounds_low[i], prior_bounds_high[i], ARGS_in.nwalkers)

    sampler_pool_arg_for_emcee = None
    # The Pool context manager needs to wrap the sampler.sample call for proper cleanup
    # Emcee's internal pool management is typically for its `threads` argument (deprecated)
    # or when it takes an existing pool object.
    
    if n_cores_to_use > 1:
        print(f"   MCMC will attempt parallel run on {n_cores_to_use} cores.")
        # The Pool object must be created here to be passed to EnsembleSampler
        # It will be closed after the sampling loop
        sampler_pool_arg_for_emcee = Pool(processes=n_cores_to_use)
    else:
        print("   MCMC running in serial mode.")
        
    sampler = emcee.EnsembleSampler(ARGS_in.nwalkers, ndim, log_posterior,
                                    args=(R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi),
                                    pool=sampler_pool_arg_for_emcee) # Pass the pool object
    start_time_mcmc = time.time()
    with tqdm(total=ARGS_in.nsteps, desc="MCMC Sampling", unit="step") as pbar:
        for _ in sampler.sample(pos0, iterations=ARGS_in.nsteps, progress=False, store=True):
            pbar.update(1)
    end_time_mcmc = time.time()

    if sampler_pool_arg_for_emcee is not None: # If a pool was created, close it
        sampler_pool_arg_for_emcee.close()
        sampler_pool_arg_for_emcee.join()

    print(f"MCMC finished in {(end_time_mcmc - start_time_mcmc)/60:.2f} minutes.")

    actual_burnin = ARGS_in.burnin
    actual_thin = ARGS_in.thin
    try:
        autocorr_time = sampler.get_autocorr_time(tol=0, quiet=True) 
        print(f"Autocorrelation time estimates: {autocorr_time}")
        finite_autocorr = autocorr_time[np.isfinite(autocorr_time)]
        if len(finite_autocorr) > 0:
            max_autocorr = np.max(finite_autocorr)
            recommended_burnin = int(np.ceil(max_autocorr * 5))
            recommended_thin = int(np.ceil(max_autocorr / 2)); recommended_thin = max(1, recommended_thin)
            print(f"Recommended burn-in based on Autocorr: ~{recommended_burnin}")
            print(f"Recommended thinning based on Autocorr: ~{recommended_thin}")
            if ARGS_in.burnin < recommended_burnin:
                print(f"⚠️ User burn-in ({ARGS_in.burnin}) is less than recommended ({recommended_burnin}). Using recommended.")
                actual_burnin = recommended_burnin
            if ARGS_in.thin < recommended_thin :
                print(f"⚠️ User thinning ({ARGS_in.thin}) is less than recommended ({recommended_thin}). Using recommended.")
                actual_thin = recommended_thin
        else: print("⚠️ All autocorrelation times are non-finite. Using user-specified burn-in/thin.")
    except emcee.autocorr.AutocorrError as e_acorr:
        print(f"⚠️ Could not estimate autocorrelation time: {e_acorr}. Using user-specified values.")
    except Exception as e_autocorr_generic: # Catch other potential errors from get_autocorr_time
        print(f"⚠️ An unexpected error occurred during autocorrelation time estimation: {e_autocorr_generic}. Using user-specified values.")

    
    actual_burnin = min(actual_burnin, ARGS_in.nsteps -1); actual_burnin = max(0, actual_burnin)
    actual_thin = max(1, actual_thin)
    print(f"Using actual burn-in: {actual_burnin}")
    print(f"Using actual thinning: {actual_thin}")

    chain_flat = np.array([]) 
    if ARGS_in.nsteps > actual_burnin and (ARGS_in.nsteps - actual_burnin) >= actual_thin :
        chain_flat = sampler.get_chain(discard=actual_burnin, thin=actual_thin, flat=True)
        if ARGS_in.nsteps < actual_burnin + actual_thin * 50 and len(chain_flat) > 0 : # Check for effective samples
             print(f"⚠️ Consider increasing total steps for more effective samples. Current effective samples: {len(chain_flat)}")
    else:
        print(f"⚠️ WARNING: nsteps ({ARGS_in.nsteps}) is too small for the chosen burn-in ({actual_burnin}) and thinning ({actual_thin}). Chain will be empty or very small.")

    chain_filename = f"chain_{ARGS_in.xi}.npy"
    np.save(chain_filename, chain_flat)
    print(f"Saved flattened chain to {chain_filename} ({len(chain_flat)} samples)")
    
    # --- 4. Posterior Diagnostics & Rotation-Curve Envelope ---
    param_summary_text = "" 
    median_params = p0_guess_means 
    params_16th = prior_bounds_low
    params_84th = prior_bounds_high

    print("\n--- Generating Posterior Diagnostics & Plot ---")
    if len(chain_flat) < ndim * 2:
        print(f"⚠️ Chain has very few samples ({len(chain_flat)}) for robust analysis. Plots may be misleading or fail.")
        if len(chain_flat) > 0: median_params = np.median(chain_flat, axis=0)
        param_summary_text = "Chain too small for reliable parameter estimation; using prior ranges for errors if chain empty.\n"
        if len(chain_flat) > ndim:
            try:
                figure = corner.corner(chain_flat, labels=PARAM_LABELS, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
                figure.savefig(f"corner_{ARGS_in.xi}.png"); plt.close(figure)
                print(f"Saved corner plot to corner_{ARGS_in.xi}.png (with few samples).")
            except Exception as e_corner_few: print(f"⚠️ Error generating corner plot with few samples: {e_corner_few}")
        else: print("   Skipping corner plot due to insufficient samples.")
    else:
        figure = corner.corner(chain_flat, labels=PARAM_LABELS, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        figure.savefig(f"corner_{ARGS_in.xi}.png"); plt.close(figure)
        print(f"Saved corner plot to corner_{ARGS_in.xi}.png")
        median_params = np.median(chain_flat, axis=0)
        params_16th, params_84th = np.percentile(chain_flat, [16, 84], axis=0)

    print("\nFitted Parameters (Median & 68% CI):")
    # This logic ensures param_summary_text is built based on whether robust results were obtained
    if not param_summary_text or len(chain_flat) >= ndim *2 : 
        param_summary_text = "" 
        for i, label in enumerate(PARAM_LABELS):
            # Check if median_params and percentiles are correctly shaped
            if len(median_params) == ndim and len(params_16th) == ndim and len(params_84th) == ndim:
                summary_line = (f"  {label:<25}: {median_params[i]:.3e} "
                                f" (+{params_84th[i]-median_params[i]:.2e} / -{median_params[i]-params_16th[i]:.2e})")
            else: # Fallback if shapes are wrong (should not happen if chain is sufficient)
                summary_line = f"  {label:<25}: Error in param shapes"
            print(summary_line); param_summary_text += summary_line + "\n"
    else:
        print(param_summary_text) # Print the "Chain too small..." message


    R_plot_curve_min = np.min(R_obs_kpc) if len(R_obs_kpc) > 0 and np.any(np.isfinite(R_obs_kpc)) else 0.1
    R_plot_curve_max = np.max(R_obs_kpc) if len(R_obs_kpc) > 0 and np.any(np.isfinite(R_obs_kpc)) else 25.0
    R_plot_curve_min = max(0.01, R_plot_curve_min)
    R_plot_curve = np.linspace(R_plot_curve_min, R_plot_curve_max, 300)
    v_median_curve, v_16th_curve, v_84th_curve = [np.zeros_like(R_plot_curve)]*3

    n_samples_for_envelope = min(1000, len(chain_flat))
    if n_samples_for_envelope > 0 and len(chain_flat) > 0:
        chain_subset_for_envelope = chain_flat[np.random.choice(len(chain_flat), n_samples_for_envelope, replace=False)]
        v_model_samples = np.array([v_model_for_emcee(R_plot_curve, pars, xi_type_str=ARGS_in.xi) for pars in chain_subset_for_envelope])
        v_median_curve = np.nanmedian(v_model_samples, axis=0)
        v_16th_curve, v_84th_curve = np.nanpercentile(v_model_samples, [16, 84], axis=0)
    else:
        v_median_curve = v_model_for_emcee(R_plot_curve, median_params, xi_type_str=ARGS_in.xi)
        v_16th_curve, v_84th_curve = v_median_curve * 0.9, v_median_curve * 1.1
    
    plt.figure(figsize=(10,6))
    plt.errorbar(R_obs_kpc, v_obs_kms, yerr=sigma_v_kms, fmt=".k", alpha=0.02, label="Gaia DR3 Stars (Subset)", zorder=1)
    plt.plot(R_plot_curve, v_median_curve, color="red", lw=2.5, label=f"Density-Metric Median ({ARGS_in.xi} $\\xi$)", zorder=3)
    plt.fill_between(R_plot_curve, v_16th_curve, v_84th_curve, color="red", alpha=0.3, zorder=2, label="68% Credible Interval")
    v_newton_plot_median_params = v_newton_kms(R_plot_curve, median_params[0], median_params[1])
    plt.plot(R_plot_curve, v_newton_plot_median_params, color="green", ls="--", lw=2, label="Newtonian (Median Fitted Disk)", zorder=2.5)
    plt.xlabel("Galactocentric Radius R (kpc)", fontsize=12)
    plt.ylabel("Tangential Velocity v (km s$^{-1}$)", fontsize=12)
    plt.title(f"Milky Way Rotation Curve Fit: Density-Dependent Metric ({ARGS_in.xi} $\\xi$)", fontsize=14)
    plt.legend(fontsize=9); plt.grid(True, ls=':', alpha=0.7)
    valid_y_for_lim = v_84th_curve[np.isfinite(v_84th_curve)] if isinstance(v_84th_curve, np.ndarray) else [350]
    ylim_max_val = np.max(valid_y_for_lim) if len(valid_y_for_lim) > 0 and np.any(np.isfinite(valid_y_for_lim)) else 350
    ylim_max_plot = max(350, ylim_max_val * 1.1 if pd.notna(ylim_max_val) else 350)
    plt.ylim(0, ylim_max_plot)
    plt.xlim(0, np.max(R_plot_curve) if len(R_plot_curve) > 0 and np.any(np.isfinite(R_plot_curve)) else 25.0)
    plt.tight_layout(); plt.savefig(f"rotation_curve_fit_{ARGS_in.xi}.png", dpi=150)
    print(f"Saved rotation curve plot to rotation_curve_fit_{ARGS_in.xi}.png"); plt.close()

    aic_bic_text = "AIC/BIC: Not calculable (chain too small or error).\n"
    if len(chain_flat) > 0: 
        logL_samples_all = sampler.get_log_prob(discard=actual_burnin, thin=actual_thin, flat=True)
        if len(logL_samples_all) == len(chain_flat) and len(chain_flat) > 0 :
            max_log_prob_idx = np.argmax(logL_samples_all)
            best_fit_theta_mcmc = chain_flat[max_log_prob_idx]
            max_log_likelihood_val = log_likelihood(best_fit_theta_mcmc, R_obs_kpc, v_obs_kms, sigma_v_kms, ARGS_in.xi)
            if np.isfinite(max_log_likelihood_val):
                k_params = ndim; N_data = len(R_obs_kpc)
                AIC = 2 * k_params - 2 * max_log_likelihood_val
                BIC = k_params * np.log(N_data) - 2 * max_log_likelihood_val
                aic_bic_text = (f"Model: {ARGS_in.xi}\nMax Log-Likelihood: {max_log_likelihood_val:.2f}\n"
                                f"k: {k_params}, N: {N_data}\nAIC: {AIC:.2f}\nBIC: {BIC:.2f}\n\n")
                print(f"\n--- Model Comparison Metrics ---\n{aic_bic_text.strip()}")
            else: print("⚠️ Max log-likelihood is not finite. Cannot calculate AIC/BIC.")
        else: print("⚠️ Mismatch in chain and log_prob samples or empty chain for AIC/BIC.")
    else: print("⚠️ Empty chain, skipping AIC/BIC calculation.")

    print("\n--- RMS Residuals in Radial Bands ---")
    mask_inner = (R_obs_kpc < 5.0)
    mask_outer = (R_obs_kpc > 10.0) & (R_obs_kpc < 20.0)
    def calculate_rms_residuals(data_mask, params_for_rms_calc):
        if np.sum(data_mask) == 0 or len(params_for_rms_calc) != ndim: return np.nan
        v_pred_masked = v_model_for_emcee(R_obs_kpc[data_mask], params_for_rms_calc, xi_type_str=ARGS_in.xi)
        if np.all(np.isnan(v_pred_masked)): return np.nan
        return np.sqrt(np.nanmean((v_obs_kms[data_mask] - v_pred_masked)**2))
    rms_inner = calculate_rms_residuals(mask_inner, median_params)
    rms_outer = calculate_rms_residuals(mask_outer, median_params)
    print(f"RMS Δv (Median Params) for R < 5 kpc  : {rms_inner:.2f} km/s (N_stars={np.sum(mask_inner)})")
    print(f"RMS Δv (Median Params) for 10 < R < 20 kpc : {rms_outer:.2f} km/s (N_stars={np.sum(mask_outer)})")
    rms_text = (f"RMS inner (R < 5 kpc): {rms_inner:.2f} km/s\n"
                f"RMS outer (10 < R < 20 kpc): {rms_outer:.2f} km/s\n")
    
    info_filename = f"info_summary_{ARGS_in.xi}.txt"
    with open(info_filename, "w") as f:
        f.write(f"--- Fit Summary for xi_type = '{ARGS_in.xi}' ---\n"); f.write("Fitted Parameters (Median & 68% CI):\n")
        f.write(param_summary_text); f.write("\n--- Model Comparison ---\n"); f.write(aic_bic_text)
        f.write("\n--- RMS Residuals ---\n"); f.write(rms_text)
    print(f"\nSaved summary information to {info_filename}")

# --- Script Entry Point ---
if __name__ == '__main__':
    if platform.system() in ["Windows", "Darwin"]: 
        freeze_support() 
    
    # Moved ArgumentParser setup here so it's defined before being used by ARGS_global
    parser = argparse.ArgumentParser(description="Fit density-dependent metric model to Gaia data.")
    parser.add_argument('--xi', type=str, default='power', choices=['power', 'logistic'],
                        help='Type of xi(rho) function to use (power or logistic)')
    parser.add_argument('--nwalkers', type=int, default=64, help='Number of MCMC walkers')
    parser.add_argument('--nsteps', type=int, default=20000, help='Total MCMC steps')
    parser.add_argument('--burnin', type=int, default=5000, help='Initial estimate for burn-in steps')
    parser.add_argument('--thin', type=int, default=100, help='Initial estimate for thinning factor')
    parser.add_argument('--max_sample_gaia', type=int, default=80000, help='Max Gaia stars to load')
    parser.add_argument('--force_live_gaia', action='store_true',
                        help='Force a new LIVE Gaia ADQL query, ignoring any raw data CSV cache.')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of raw Gaia data, ignoring any processed Parquet cache.')
    parser.add_argument('--kill_existing', action='store_true', 
                        help='Attempt to kill other running instances of this script before starting.')
    try:
        num_cpus = cpu_count()
        default_cores = max(1, num_cpus - 2 if num_cpus and num_cpus > 2 else 1) # Ensure at least 1, leave 2 if many
    except NotImplementedError:
        default_cores = 1 
    parser.add_argument('--ncores', type=int, default=default_cores,
                        help=f'Number of CPU cores for MCMC parallelization (0 or 1 means serial). Default: {default_cores}')

    ARGS_global = parser.parse_args() 
    
    if ARGS_global.kill_existing:
        current_script_name = os.path.basename(__file__)
        kill_existing_instances(script_name_to_kill=current_script_name)
        print("Continuing with the current script run...")

    run_mcmc_analysis(ARGS_global) 

    print("\n--- main.py Script Finished ---")