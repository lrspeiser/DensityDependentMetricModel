#!/usr/bin/env python3
"""
sparc_io.py - SPARC galaxy data loading utilities.
"""
import pandas as pd
import numpy as np
import logging
import pathlib
from scipy.interpolate import interp1d # For interpolating profiles

# Initialize logger for this module
logger = logging.getLogger("sparc_loader")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEFAULT_SPARC_DATA_DIR = pathlib.Path("data/sparc_data")
# Assumed M/L for 3.6 micron to get stellar mass surface density if not scaling it as a free param
# This is SPARC's default for V_disk. If stellar_ML_factor is fitted, this base M/L is important.
BASE_M_L_3_6_MICRON_DISK = 0.5  # Msun / Lsun,solar
BASE_M_L_3_6_MICRON_BULGE = 0.7 # Msun / Lsun,solar


def load_sparc_metadata(sparc_dir=DEFAULT_SPARC_DATA_DIR):
    meta_file = pathlib.Path(sparc_dir) / "MasterSheet_SPARC.csv"
    if not meta_file.exists():
        logger.error(f"SPARC MasterSheet {meta_file} not found.")
        return None
    try:
        df_meta = pd.read_csv(meta_file)
        logger.info(f"Successfully loaded SPARC MasterSheet with {len(df_meta)} galaxies.")
        return df_meta
    except Exception as e:
        logger.error(f"Error loading SPARC MasterSheet {meta_file}: {e}")
        return None

def load_single_sparc_galaxy(galaxy_id: str,
                             sparc_dir=DEFAULT_SPARC_DATA_DIR,
                             assume_gas_hz_kpc=0.1,
                             assume_stellar_hz_kpc=0.3
                             ):
    """
    Loads data for a single SPARC galaxy from _rotmod.dat, _HIrad.dat, and _SB.dat.
    Interpolates all data onto the radial grid of _rotmod.dat.
    Returns a dictionary including R_kpc, V_obs, e_V_obs, V_gas_comp_kms, V_disk_comp_kms, V_bulge_comp_kms,
    Sigma_gas_Msun_pc2, Sigma_star_Msun_pc2 (at base M/L), and derived rho_total_mid_Msun_kpc3.
    """
    if not isinstance(galaxy_id, str): galaxy_id = str(galaxy_id)

    rotmod_file = pathlib.Path(sparc_dir) / f"{galaxy_id}_rotmod.dat"
    hirad_file = pathlib.Path(sparc_dir) / f"{galaxy_id}_HIrad.dat" # HI surface density
    sb_file = pathlib.Path(sparc_dir) / f"{galaxy_id}_SB.dat"       # 3.6um Surface Brightness

    if not rotmod_file.exists():
        logger.error(f"Galaxy _rotmod.dat file {rotmod_file} not found for {galaxy_id}.")
        return None
    if not hirad_file.exists():
        logger.warning(f"Galaxy _HIrad.dat file {hirad_file} not found for {galaxy_id}. Gas density will be zero.")
    if not sb_file.exists():
        logger.warning(f"Galaxy _SB.dat file {sb_file} not found for {galaxy_id}. Stellar density will be zero.")

    try:
        df_rotmod = pd.read_csv(rotmod_file, delim_whitespace=True, comment='#',
                                names=['R_kpc', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'])
        logger.info(f"[{galaxy_id}] Loaded {len(df_rotmod)} radial points from {rotmod_file.name}.")
        # Use R_kpc from rotmod as the common radial grid
        common_R_kpc = df_rotmod['R_kpc'].values

        # --- Load and Interpolate Gas Surface Density (_HIrad.dat) ---
        # Columns: Radius (kpc), Sigma_HI (Msun/pc^2, already includes 1.33x for He)
        sigma_gas_interp_Msun_pc2 = np.zeros_like(common_R_kpc)
        if hirad_file.exists():
            df_hirad = pd.read_csv(hirad_file, delim_whitespace=True, comment='#',
                                   names=['R_HI_kpc', 'Sigma_HI_Msun_pc2'])
            if not df_hirad.empty and len(df_hirad['R_HI_kpc']) > 1:
                # Ensure radii are sorted for interpolation
                sort_idx = np.argsort(df_hirad['R_HI_kpc'].values)
                R_HI_sorted = df_hirad['R_HI_kpc'].values[sort_idx]
                Sigma_HI_sorted = df_hirad['Sigma_HI_Msun_pc2'].values[sort_idx]
                
                interp_func_gas = interp1d(R_HI_sorted, Sigma_HI_sorted,
                                           kind='linear', bounds_error=False, fill_value=0.0) # Extrapolate with 0
                sigma_gas_interp_Msun_pc2 = interp_func_gas(common_R_kpc)
                sigma_gas_interp_Msun_pc2 = np.maximum(sigma_gas_interp_Msun_pc2, 0) # Ensure non-negative
                logger.info(f"[{galaxy_id}] Interpolated Sigma_gas from {hirad_file.name}.")
            elif not df_hirad.empty and len(df_hirad['R_HI_kpc']) == 1: # Single point, use it if R matches
                 if np.isclose(df_hirad['R_HI_kpc'].iloc[0], common_R_kpc[0]): # Very basic check
                    sigma_gas_interp_Msun_pc2[:] = df_hirad['Sigma_HI_Msun_pc2'].iloc[0]
                 logger.warning(f"[{galaxy_id}] Only one point in _HIrad.dat. Applied if R matches first point.")
            else:
                logger.warning(f"[{galaxy_id}] _HIrad.dat is empty or has too few points for interpolation.")
        else:
            logger.warning(f"[{galaxy_id}] No _HIrad.dat file. Sigma_gas set to 0.")


        # --- Load and Interpolate Stellar Surface Brightness (_SB.dat) and convert to Mass Surface Density ---
        # Columns: Radius (kpc), SB_disk (Lsun/pc^2 at 3.6um), SB_bulge (Lsun/pc^2 at 3.6um)
        sigma_star_interp_Msun_pc2 = np.zeros_like(common_R_kpc)
        if sb_file.exists():
            df_sb = pd.read_csv(sb_file, delim_whitespace=True, comment='#',
                                names=['R_SB_kpc', 'SB_disk_Lsun_pc2', 'SB_bulge_Lsun_pc2'])
            if not df_sb.empty and len(df_sb['R_SB_kpc']) > 1:
                sort_idx_sb = np.argsort(df_sb['R_SB_kpc'].values)
                R_SB_sorted = df_sb['R_SB_kpc'].values[sort_idx_sb]
                SB_disk_sorted = df_sb['SB_disk_Lsun_pc2'].values[sort_idx_sb]
                SB_bulge_sorted = df_sb['SB_bulge_Lsun_pc2'].values[sort_idx_sb]

                interp_func_sb_disk = interp1d(R_SB_sorted, SB_disk_sorted,
                                               kind='linear', bounds_error=False, fill_value=0.0)
                interp_func_sb_bulge = interp1d(R_SB_sorted, SB_bulge_sorted,
                                                kind='linear', bounds_error=False, fill_value=0.0)
                
                sb_disk_interp_Lsun_pc2 = interp_func_sb_disk(common_R_kpc)
                sb_bulge_interp_Lsun_pc2 = interp_func_sb_bulge(common_R_kpc)

                # Convert SB to Sigma_star using BASE M/L. This Sigma_star will be scaled by stellar_ML_factor in main.py
                sigma_star_disk_Msun_pc2 = sb_disk_interp_Lsun_pc2 * BASE_M_L_3_6_MICRON_DISK
                sigma_star_bulge_Msun_pc2 = sb_bulge_interp_Lsun_pc2 * BASE_M_L_3_6_MICRON_BULGE
                sigma_star_interp_Msun_pc2 = np.maximum(sigma_star_disk_Msun_pc2 + sigma_star_bulge_Msun_pc2, 0)
                logger.info(f"[{galaxy_id}] Interpolated Sigma_star from {sb_file.name} using base M/L values.")
            elif not df_sb.empty and len(df_sb['R_SB_kpc']) == 1:
                # Basic single point handling
                sb_disk_val = df_sb['SB_disk_Lsun_pc2'].iloc[0] * BASE_M_L_3_6_MICRON_DISK
                sb_bulge_val = df_sb['SB_bulge_Lsun_pc2'].iloc[0] * BASE_M_L_3_6_MICRON_BULGE
                sigma_star_interp_Msun_pc2[:] = max(0, sb_disk_val + sb_bulge_val)
                logger.warning(f"[{galaxy_id}] Only one point in _SB.dat. Applied if R matches.")

            else:
                logger.warning(f"[{galaxy_id}] _SB.dat is empty or has too few points for interpolation.")
        else:
            logger.warning(f"[{galaxy_id}] No _SB.dat file. Sigma_star set to 0.")

        # Midplane volume densities
        kpc_per_pc_sq = (1e3)**2
        rho_star_mid_Msun_kpc3 = (sigma_star_interp_Msun_pc2 * kpc_per_pc_sq) / (2 * assume_stellar_hz_kpc) if assume_stellar_hz_kpc > 1e-9 else np.zeros_like(common_R_kpc)
        rho_gas_mid_Msun_kpc3 = (sigma_gas_interp_Msun_pc2 * kpc_per_pc_sq) / (2 * assume_gas_hz_kpc) if assume_gas_hz_kpc > 1e-9 else np.zeros_like(common_R_kpc)
        rho_total_mid_Msun_kpc3 = rho_star_mid_Msun_kpc3 + rho_gas_mid_Msun_kpc3 # This rho_star part will be scaled by M/L in main.py

        logger.info(f"[{galaxy_id}] Max stellar rho_mid (base M/L): {np.max(rho_star_mid_Msun_kpc3):.2e} Msun/kpc^3 (hz_star={assume_stellar_hz_kpc} kpc)")
        logger.info(f"[{galaxy_id}] Max gas rho_mid: {np.max(rho_gas_mid_Msun_kpc3):.2e} Msun/kpc^3 (hz_gas={assume_gas_hz_kpc} kpc)")

        df_meta = load_sparc_metadata(sparc_dir)
        galaxy_meta = None
        if df_meta is not None:
            # Robust matching for galaxy ID (e.g. "NGC0024" vs "NGC24")
            # Create a standardized ID for matching (lower, no spaces, remove leading zeros from numbers)
            def standardize_id(gid):
                import re
                gid_std = gid.lower().replace(" ", "")
                gid_std = re.sub(r"([a-zA-Z]+)0+(\d+)", r"\1\2", gid_std) # Remove leading zeros after letters
                return gid_std
            
            std_galaxy_id_arg = standardize_id(galaxy_id)
            df_meta['StdName'] = df_meta['Name'].apply(standardize_id)
            potential_matches = df_meta[df_meta['StdName'] == std_galaxy_id_arg]

            if not potential_matches.empty:
                galaxy_meta = potential_matches.iloc[0]
                logger.info(f"[{galaxy_id}] Found metadata: Dist={galaxy_meta.get('D_Mpc', 'N/A')} Mpc, M_HI={galaxy_meta.get('MHI', 'N/A')} Msun")
            else:
                logger.warning(f"[{galaxy_id}] Metadata not found in MasterSheet for ID '{galaxy_id}' (standardized to '{std_galaxy_id_arg}').")
        
        # V_newton_bary_kms will be constructed in main.py using these components and ML factor
        output_dict = {
            'galaxy_id': galaxy_id,
            'R_kpc': common_R_kpc,
            'V_obs': df_rotmod['V_obs'].values,
            'e_V_obs': df_rotmod['e_V_obs'].values,
            'V_gas_comp_kms': df_rotmod['V_gas'].values,
            'V_disk_comp_kms': df_rotmod['V_disk'].values, # Based on M/L=0.5
            'V_bulge_comp_kms': df_rotmod['V_bulge'].values,# Based on M/L=0.7
            'Sigma_star_Msun_pc2_baseML': sigma_star_interp_Msun_pc2, # Stellar surf. dens. at base M/L
            'Sigma_gas_Msun_pc2': sigma_gas_interp_Msun_pc2,
            'rho_star_mid_Msun_kpc3_baseML': rho_star_mid_Msun_kpc3, # Stellar vol. dens. at base M/L
            'rho_gas_mid_Msun_kpc3': rho_gas_mid_Msun_kpc3,
            # 'rho_total_mid_Msun_kpc3' will be calculated in main.py after M/L scaling of stellar part
            'assumed_hz_stellar_kpc': assume_stellar_hz_kpc,
            'assumed_hz_gas_kpc': assume_gas_hz_kpc,
            'distance_Mpc': galaxy_meta['D_Mpc'] if galaxy_meta is not None and 'D_Mpc' in galaxy_meta else np.nan,
            'M_HI_Msun': galaxy_meta['MHI'] if galaxy_meta is not None and 'MHI' in galaxy_meta else np.nan,
        }
        return output_dict

    except Exception as e:
        logger.error(f"Error processing SPARC galaxy {galaxy_id}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # DEBUG for sparc_io testing
    
    test_sparc_dir = pathlib.Path("temp_sparc_data_live")
    test_sparc_dir.mkdir(exist_ok=True)
    
    master_sheet_content = """Name,D_Mpc,MHI,Inc,HubbleT
NGC2403,3.2,2.70E+09,62.9,SABcd
UGC00128,67.6,1.01E+10,77,SABc
"""
    with open(test_sparc_dir / "MasterSheet_SPARC.csv", "w") as f: f.write(master_sheet_content)

    ngc2403_rotmod = """# R(kpc) Vobs e_Vobs Vgas Vdisk Vbul
0.11  15.2  11.2  0.0  25.0  0.0
0.55  50.0   5.0  2.5  60.0  0.0
1.10  75.0   3.0  5.0  80.0  0.0
5.50 125.0   2.0 20.0 100.0  0.0
11.0 135.0   2.0 30.0  70.0  0.0
"""
    with open(test_sparc_dir / "NGC2403_rotmod.dat", "w") as f: f.write(ngc2403_rotmod)
        
    ngc2403_hirad = """# R(kpc) Sigma_HI(Msun/pc^2)
0.2  1.1
1.0  5.5
5.0 10.2
10.0  7.5
15.0  2.1
"""
    with open(test_sparc_dir / "NGC2403_HIrad.dat", "w") as f: f.write(ngc2403_hirad)

    ngc2403_sb = """# R(kpc) SBdisk SBbulge (Lsun/pc^2 @3.6um)
0.1  1000.0  50.0
0.5   800.0  20.0
1.0   500.0   0.0
5.0   100.0   0.0
10.0   10.0   0.0
"""
    with open(test_sparc_dir / "NGC2403_SB.dat", "w") as f: f.write(ngc2403_sb)

    logger.info(f"--- Testing SPARC Metadata Loader ---")
    meta = load_sparc_metadata(sparc_dir=test_sparc_dir)
    if meta is not None: print(meta.head())

    logger.info(f"\n--- Testing Single SPARC Galaxy Loader (NGC2403) ---")
    # Test with a real SPARC ID format
    galaxy_data = load_single_sparc_galaxy("NGC2403", sparc_dir=test_sparc_dir)
    if galaxy_data:
        for key, val in galaxy_data.items():
            if isinstance(val, np.ndarray):
                print(f"  {key:<30}: array shape {val.shape}, e.g., {val[:min(3, len(val))]}")
            else:
                print(f"  {key:<30}: {val}")
        
        # Verify interpolated values
        print("\n  Example interpolated Sigma_gas_Msun_pc2:", galaxy_data['Sigma_gas_Msun_pc2'][:5])
        print("  Example interpolated Sigma_star_Msun_pc2_baseML:", galaxy_data['Sigma_star_Msun_pc2_baseML'][:5])
        print("  Example V_disk_comp_kms:", galaxy_data['V_disk_comp_kms'][:5])

    else:
        print("Failed to load NGC2403 test data.")
    
    # import shutil
    # shutil.rmtree(test_sparc_dir)
    # logger.info(f"Cleaned up {test_sparc_dir}")