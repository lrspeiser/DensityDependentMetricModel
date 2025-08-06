#!/usr/bin/env python3
"""
inspect_results.py - A simple script to load a Dynesty results pickle file
and inspect its contents, especially the shape of the samples array.
"""

import numpy as np
import pickle
import gzip

# --- IMPORTANT: UPDATE THIS PATH ---
RESULTS_FILE = 'chains_dynesty/NEWTONIAN_LIKE/dynesty_mw_power_Bf_DTf_DKf_Gf_results.pkl.gz'
# -----------------------------------

def main():
    print(f"--- Inspecting Dynesty Results File ---")
    print(f"File: {RESULTS_FILE}\n")

    try:
        with gzip.open(RESULTS_FILE, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{RESULTS_FILE}'")
        print("   Please make sure the path is correct.")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load or read the pickle file: {e}")
        return

    print("✅ File loaded successfully.")
    
    # --- Inspecting the contents ---
    
    if not hasattr(results, 'samples'):
        print("❌ ERROR: The loaded object does not have a 'samples' attribute.")
        return
        
    samples = results.samples
    
    print("\n--- Samples Array ---")
    print(f"  Shape of the samples array: {samples.shape}")
    
    num_samples = samples.shape[0]
    num_params = samples.shape[1]
    
    print(f"  This means you have {num_samples:,} samples for {num_params} fitted parameters.")

    print("\n--- Next Steps ---")
    print("Based on the number of fitted parameters, you must now update the")
    print("'param_names' list inside the 'load_dynesty_results' function in your")
    print("'visualization.py' script to have exactly this many names in the correct order.")
    
    # Provide a likely guess based on the number of parameters
    if num_params == 11:
        print("\n--- Likely Parameter List (for 11 parameters) ---")
        print("param_names = [")
        print("    'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',")
        print("    'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',")
        print("    'M_bulge_solar', 'a_bulge_kpc',")
        print("    'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'")
        print("]")
        print("\nNOTE: This run appears to have ONLY fitted the baryonic parameters.")

    elif num_params == 13:
        print("\n--- Likely Parameter List (for 13 parameters) ---")
        print("param_names = [")
        print("    'rho_c_solar_kpc3', 'n_exp',")
        print("    'M_disk_thin_solar', 'R_d_thin_kpc', 'h_z_thin_kpc',")
        print("    'M_disk_thick_solar', 'R_d_thick_kpc', 'h_z_thick_kpc',")
        print("    'M_bulge_solar', 'a_bulge_kpc',")
        print("    'M_gas_solar', 'R_d_gas_kpc', 'h_z_gas_kpc'")
        print("]")
        print("\nNOTE: This run appears to have fitted both gravity and baryonic parameters.")

    else:
        print(f"\nWARNING: An unusual number of parameters ({num_params}) was found.")
        print("You will need to manually determine the correct list of names.")

if __name__ == "__main__":
    main()