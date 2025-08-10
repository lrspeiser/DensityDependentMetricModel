#!/usr/bin/env python3
"""
sparc_io.py - SPARC galaxy data loading utilities.
"""
import pandas as pd
import numpy as np
import logging
import pathlib
from scipy.interpolate import interp1d # For interpolating profiles
# Gas profile reconstruction helpers
try:
    from models.gas_profile import (
        reconstruct_gas_exponential,
        reconstruct_gas_exponential_truncated,
        reconstruct_gas_from_vgas,
    )
except Exception:
    reconstruct_gas_exponential = None
    reconstruct_gas_exponential_truncated = None
    reconstruct_gas_from_vgas = None

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
    # Try straightforward CSV read first
    try:
        df_meta = pd.read_csv(meta_file)
        logger.info(f"Successfully loaded SPARC MasterSheet with {len(df_meta)} galaxies.")
        return df_meta
    except Exception:
        # Fallback: skip MRT header preamble and parse only data block
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            start_idx = None
            for i, ln in enumerate(lines):
                if not ln.strip():
                    continue
                # Heuristic: data lines look like Name,T,D_Mpc,... where fields 2 and 3 are numbers
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) >= 5:
                    # parts[1] should be integer-like (Hubble T), parts[2] float-like (Distance)
                    try:
                        int(parts[1])
                        float(parts[2])
                        start_idx = i
                        break
                    except Exception:
                        continue
            if start_idx is None:
                raise ValueError("Could not locate data section in MasterSheet_SPARC.csv")
            # Define expected column names for the data block
            cols = [
                'Name','T','D_Mpc','e_D','f_D','Inc','e_Inc','L_3p6_1e9Lsun','e_L_3p6','Reff_kpc',
                'SBeff_Lsun_pc2','Rdisk_kpc','SBdisk0_Lsun_pc2','MHI_1e9Msun','RHI_kpc','Vflat_kms','e_Vflat_kms','Q','Ref'
            ]
            df_meta = pd.read_csv(meta_file, skiprows=start_idx, header=None, names=cols)
            logger.info(f"Successfully parsed SPARC MasterSheet data section with {len(df_meta)} galaxies (skipped header preamble).")
            return df_meta
        except Exception as e2:
            logger.error(f"Error loading SPARC MasterSheet {meta_file}: {e2}")
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

    # Resolve rotmod file robustly (handles leading zeros, underscores, aliases like M33)
    sparc_dir_path = pathlib.Path(sparc_dir)
    def resolve_rotmod_path(gid: str) -> pathlib.Path | None:
        # Direct canonical
        candidates = [
            sparc_dir_path / f"{gid}_rotmod.dat",
        ]
        # Variants: remove underscore, remove leading zeros in numeric part
        import re
        m = re.match(r"^([A-Za-z]+)0*(\d+)$", gid)
        if m:
            base = m.group(1); num = m.group(2)
            nozero = f"{base}{int(num)}"  # e.g., NGC0598 -> NGC598
            candidates += [
                sparc_dir_path / f"{nozero}_rotmod.dat",
                sparc_dir_path / f"{nozero}rotmod.dat",
                sparc_dir_path / f"{gid}rotmod.dat",
                sparc_dir_path / f"{gid.replace('_','')}_rotmod.dat",
            ]
            # Messier alias for common cases (e.g., NGC0598/M33)
            try:
                n_int = int(num)
                if base.upper() == 'NGC' and n_int == 598:
                    for alias in ['M033', 'M33']:
                        candidates += [
                            sparc_dir_path / f"{alias}_rotmod.dat",
                            sparc_dir_path / f"{alias}rotmod.dat",
                        ]
            except Exception:
                pass
        # Try exact candidates first
        for pth in candidates:
            if pth.exists():
                return pth
        # Glob fallback: try patterns around numeric or M33 tokens
        pats = []
        if m:
            pats.append(f"*{int(m.group(2))}*rotmod*.dat")
        if 'M33' not in ''.join([str(c) for c in candidates]):
            pats.append("*M33*rotmod*.dat")
        best = None
        for pat in pats:
            for pth in sparc_dir_path.glob(pat):
                if best is None or len(str(pth.name)) > len(str(best.name)):
                    best = pth
        return best

    rotmod_path_resolved = resolve_rotmod_path(galaxy_id)
    if rotmod_path_resolved is None:
        logger.error(f"Galaxy _rotmod.dat file not found for {galaxy_id} in {sparc_dir}")
        return None

    rotmod_file = rotmod_path_resolved
    hirad_file = pathlib.Path(sparc_dir) / f"{galaxy_id}_HIrad.dat" # HI surface density (optional)
    sb_file = pathlib.Path(sparc_dir) / f"{galaxy_id}_SB.dat"       # 3.6um Surface Brightness (optional)

    if not rotmod_file.exists():
        logger.error(f"Galaxy _rotmod.dat file {rotmod_file} not found for {galaxy_id}.")
        return None
    if not hirad_file.exists():
        logger.warning(f"Galaxy _HIrad.dat file {hirad_file} not found for {galaxy_id}. Gas surface density will be derived from rotmod gas curves only.")
    # SB may be provided by rotmod as SB columns; warn only if neither _SB.dat nor rotmod SB columns exist (checked later)

    try:
        # Many SPARC rotmod files include SBdisk and SBbulge columns. Read 8 columns; extra will be ignored, missing become NaN.
        df_rotmod = pd.read_csv(
            rotmod_file,
            sep='\s+',
            comment='#',
            names=['R_kpc', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge', 'SB_disk_Lsun_pc2', 'SB_bulge_Lsun_pc2']
        )
        logger.info(f"[{galaxy_id}] Loaded {len(df_rotmod)} radial points from {rotmod_file.name}.")
        # Use R_kpc from rotmod as the common radial grid
        common_R_kpc = df_rotmod['R_kpc'].values

        # --- Load and Interpolate Gas Surface Density (_HIrad.dat) ---
        # Columns: Radius (kpc), Sigma_HI (Msun/pc^2, already includes 1.33x for He)
        sigma_gas_interp_Msun_pc2 = np.zeros_like(common_R_kpc)
        need_gas_reconstruct = False
        if hirad_file.exists():
            df_hirad = pd.read_csv(hirad_file, sep='\s+', comment='#',
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
            logger.warning(f"[{galaxy_id}] No _HIrad.dat file. Will attempt reconstruction from SPARC metadata/rotmod.")
            need_gas_reconstruct = True


        # --- Load and Interpolate Stellar Surface Brightness (_SB.dat) and convert to Mass Surface Density ---
        # Columns: Radius (kpc), SB_disk (Lsun/pc^2 at 3.6um), SB_bulge (Lsun/pc^2 at 3.6um)
        sigma_star_interp_Msun_pc2 = np.zeros_like(common_R_kpc)
        used_sb_source = None
        if sb_file.exists():
            df_sb = pd.read_csv(sb_file, sep='\s+', comment='#',
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
                used_sb_source = sb_file.name
                logger.info(f"[{galaxy_id}] Interpolated Sigma_star from {sb_file.name} using base M/L values.")
            elif not df_sb.empty and len(df_sb['R_SB_kpc']) == 1:
                # Basic single point handling
                sb_disk_val = df_sb['SB_disk_Lsun_pc2'].iloc[0] * BASE_M_L_3_6_MICRON_DISK
                sb_bulge_val = df_sb['SB_bulge_Lsun_pc2'].iloc[0] * BASE_M_L_3_6_MICRON_BULGE
                sigma_star_interp_Msun_pc2[:] = max(0, sb_disk_val + sb_bulge_val)
                used_sb_source = sb_file.name
                logger.warning(f"[{galaxy_id}] Only one point in _SB.dat. Applied if R matches.")

        # If no _SB.dat or unusable, try SB columns from rotmod (already on common grid and in Lsun/pc^2)
        if used_sb_source is None and ('SB_disk_Lsun_pc2' in df_rotmod.columns):
            sb_disk = df_rotmod['SB_disk_Lsun_pc2'].values if 'SB_disk_Lsun_pc2' in df_rotmod.columns else None
            sb_bulge = df_rotmod['SB_bulge_Lsun_pc2'].values if 'SB_bulge_Lsun_pc2' in df_rotmod.columns else None
            if sb_disk is not None and np.any(np.isfinite(sb_disk)):
                sb_disk_interp_Lsun_pc2 = np.nan_to_num(sb_disk, nan=0.0)
                sb_bulge_interp_Lsun_pc2 = np.nan_to_num(sb_bulge, nan=0.0) if sb_bulge is not None else np.zeros_like(sb_disk_interp_Lsun_pc2)
                sigma_star_disk_Msun_pc2 = sb_disk_interp_Lsun_pc2 * BASE_M_L_3_6_MICRON_DISK
                sigma_star_bulge_Msun_pc2 = sb_bulge_interp_Lsun_pc2 * BASE_M_L_3_6_MICRON_BULGE
                sigma_star_interp_Msun_pc2 = np.maximum(sigma_star_disk_Msun_pc2 + sigma_star_bulge_Msun_pc2, 0)
                used_sb_source = f"{rotmod_file.name} (SB columns)"
                logger.info(f"[{galaxy_id}] Used SB from rotmod file columns to compute Sigma_star (base M/L).")

        if used_sb_source is None:
            logger.warning(f"[{galaxy_id}] No usable SB data (_SB.dat or rotmod SB columns). Sigma_star set to 0.")

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
                logger.info(f"[{galaxy_id}] Found metadata: Dist={galaxy_meta.get('D_Mpc', 'N/A')} Mpc, M_HI={galaxy_meta.get('MHI_1e9Msun', galaxy_meta.get('MHI', 'N/A'))} (1e9 Msun)")
            else:
                logger.warning(f"[{galaxy_id}] Metadata not found in MasterSheet for ID '{galaxy_id}' (standardized to '{std_galaxy_id_arg}').")

        # If we need gas reconstruction and helpers are available, try truncated Option A then B
        if need_gas_reconstruct and (reconstruct_gas_exponential_truncated is not None or reconstruct_gas_from_vgas is not None):
            try:
                MHI_1e9 = None
                RHI_kpc_val = None
                if galaxy_meta is not None:
                    # Accept several possible column names
                    if 'MHI_1e9Msun' in galaxy_meta: MHI_1e9 = float(galaxy_meta['MHI_1e9Msun'])
                    elif 'MHI' in galaxy_meta: MHI_1e9 = float(galaxy_meta['MHI'])
                    if 'RHI_kpc' in galaxy_meta: RHI_kpc_val = float(galaxy_meta['RHI_kpc'])
                    elif 'RHI' in galaxy_meta: RHI_kpc_val = float(galaxy_meta['RHI'])
                    elif 'R_HI' in galaxy_meta: RHI_kpc_val = float(galaxy_meta['R_HI'])
                used_option = None
                # Optional override: force use of Vgas-shaped profile
                import os as _os
                force_vgas = _os.environ.get("SPARC_GAS_FORCE_VGAS", "0").strip() in ("1","true","True")
                if force_vgas and reconstruct_gas_from_vgas is not None:
                    Vgas_col = df_rotmod['V_gas'].values if 'V_gas' in df_rotmod.columns else np.zeros_like(common_R_kpc)
                    gasB = reconstruct_gas_from_vgas(common_R_kpc, Vgas_col, MHI_1e9Msun=MHI_1e9, include_He=True)
                    sigma_gas_interp_Msun_pc2 = gasB['Sigma_gas']
                    used_option = 'B_shape(Vgas) [forced]'
                    logger.info(f"[{galaxy_id}] Forced Vgas-shaped Sigma_gas via env SPARC_GAS_FORCE_VGAS.")
                if used_option is None and (MHI_1e9 is not None) and (RHI_kpc_val is not None) and (reconstruct_gas_exponential_truncated is not None):
                    # Allow runtime override of truncation mode via environment variables
                    import os as _os
                    rmax_mode_env = _os.environ.get("SPARC_GAS_RMAX_MODE", "RHI").strip().upper()
                    if rmax_mode_env not in ("RHI", "KRD"):
                        rmax_mode_env = "RHI"
                    try:
                        krd_env = float(_os.environ.get("SPARC_GAS_KRD", "3.0"))
                    except Exception:
                        krd_env = 3.0
                    gasA = reconstruct_gas_exponential_truncated(
                        common_R_kpc,
                        MHI_1e9,
                        RHI_kpc_val,
                        include_He=True,
                        Rmax_mode=("RHI" if rmax_mode_env == "RHI" else "kRd"),
                        kRd=krd_env,
                        rd_bracket_kpc=(0.1, 30.0),
                        enforce_rd_bounds=True,
                        verbose=True,
                    )
                    sigma_gas_interp_Msun_pc2 = gasA['Sigma_gas']
                    used_option = f"A_trunc_exp(MHI,RHI); mode={rmax_mode_env} kRd={krd_env:.2f}"
                    rd_val = float(gasA['Rd_kpc'][0]) if 'Rd_kpc' in gasA else float('nan')
                    rmax_val = float(gasA['Rmax_kpc'][0]) if 'Rmax_kpc' in gasA else float('nan')
                    s0_val = float(gasA['Sigma0'][0]) if 'Sigma0' in gasA else float('nan')
                    mass_mismatch = float(gasA.get('mass_mismatch', [1.0])[0]) if isinstance(gasA.get('mass_mismatch', [1.0]), np.ndarray) else 1.0
                    penalty_mass = float(gasA.get('penalty_mass', [0.0])[0]) if isinstance(gasA.get('penalty_mass', [0.0]), np.ndarray) else 0.0
                    logger.info(f"[{galaxy_id}] Reconstructed Sigma_gas via Truncated Option A. Rd={rd_val:.3f} kpc; Σ0={s0_val:.2f}; Rmax={rmax_val:.2f} kpc; mode={rmax_mode_env} kRd={krd_env:.2f}; mass_mismatch={mass_mismatch:.3f}; penalty≈{penalty_mass:.2f}")
                    if rmax_mode_env == "RHI":
                        if not (0.5 <= rd_val <= 10.0):
                            logger.warning(f"[{galaxy_id}] Gas Rd={rd_val:.3f} kpc outside [0.5,10] under RHI truncation. Applying soft Σ(RHI)≈1 constraint; fallback to Vgas-shape only if still pathological.")
                            logger.warning(f"[{galaxy_id}] Gas Rd={rd_val:.3f} kpc outside [0.5,10] under RHI truncation. Check MHI/RHI metadata or consider Vgas-shaped fallback.")
                if used_option is None and reconstruct_gas_from_vgas is not None:
                    Vgas_col = df_rotmod['V_gas'].values if 'V_gas' in df_rotmod.columns else np.zeros_like(common_R_kpc)
                    gasB = reconstruct_gas_from_vgas(common_R_kpc, Vgas_col, MHI_1e9Msun=MHI_1e9, include_He=True)
                    sigma_gas_interp_Msun_pc2 = gasB['Sigma_gas']
                    used_option = 'B_shape(Vgas)'
                    logger.info(f"[{galaxy_id}] Reconstructed Sigma_gas via Option B (Vgas shape).")
                if used_option is None:
                    logger.warning(f"[{galaxy_id}] Gas reconstruction failed; Sigma_gas kept at zeros.")
            except Exception as e:
                logger.warning(f"[{galaxy_id}] Gas reconstruction error: {e}. Sigma_gas kept at zeros.")
        
        # Midplane volume densities
        kpc_per_pc_sq = (1e3)**2
        rho_star_mid_Msun_kpc3 = (sigma_star_interp_Msun_pc2 * kpc_per_pc_sq) / (2 * assume_stellar_hz_kpc) if assume_stellar_hz_kpc > 1e-9 else np.zeros_like(common_R_kpc)
        rho_gas_mid_Msun_kpc3 = (sigma_gas_interp_Msun_pc2 * kpc_per_pc_sq) / (2 * assume_gas_hz_kpc) if assume_gas_hz_kpc > 1e-9 else np.zeros_like(common_R_kpc)
        rho_total_mid_Msun_kpc3 = rho_star_mid_Msun_kpc3 + rho_gas_mid_Msun_kpc3 # This rho_star part will be scaled by M/L in main.py

        logger.info(f"[{galaxy_id}] Max stellar rho_mid (base M/L): {np.max(rho_star_mid_Msun_kpc3):.2e} Msun/kpc^3 (hz_star={assume_stellar_hz_kpc} kpc)")
        logger.info(f"[{galaxy_id}] Max gas rho_mid: {np.max(rho_gas_mid_Msun_kpc3):.2e} Msun/kpc^3 (hz_gas={assume_gas_hz_kpc} kpc)")

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
            'e_distance_Mpc': galaxy_meta['e_D'] if galaxy_meta is not None and 'e_D' in galaxy_meta else np.nan,
            'incl_deg': galaxy_meta['Inc'] if galaxy_meta is not None and 'Inc' in galaxy_meta else np.nan,
            'e_incl_deg': galaxy_meta['e_Inc'] if galaxy_meta is not None and 'e_Inc' in galaxy_meta else np.nan,
            'M_HI_Msun': (galaxy_meta['MHI_1e9Msun'] * 1e9) if galaxy_meta is not None and ('MHI_1e9Msun' in galaxy_meta) else (galaxy_meta['MHI'] if galaxy_meta is not None and 'MHI' in galaxy_meta else np.nan),
            # Gas reconstruction meta (optional)
            'gas_profile_mode': used_option if 'used_option' in locals() else None,
            'gas_Rd_kpc': (gasA['Rd_kpc'][0] if (locals().get('gasA') is not None and isinstance(gasA, dict) and 'Rd_kpc' in gasA) else np.nan),
            'gas_Sigma0': (gasA['Sigma0'][0] if (locals().get('gasA') is not None and isinstance(gasA, dict) and 'Sigma0' in gasA) else np.nan),
            'gas_Rmax_kpc': (gasA['Rmax_kpc'][0] if (locals().get('gasA') is not None and isinstance(gasA, dict) and 'Rmax_kpc' in gasA) else np.nan),
            'gas_mass_mismatch': (gasA['mass_mismatch'][0] if (locals().get('gasA') is not None and isinstance(gasA, dict) and 'mass_mismatch' in gasA) else np.nan),
'gas_penalty_mass': (gasA['penalty_mass'][0] if (locals().get('gasA') is not None and isinstance(gasA, dict) and 'penalty_mass' in gasA) else 0.0),
            # Resolved file paths for transparency/debugging
            'rotmod_path': str(rotmod_file),
            'hirad_path': (str(hirad_file) if hirad_file.exists() else None),
            'sb_path': (str(sb_file) if sb_file.exists() else None),
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