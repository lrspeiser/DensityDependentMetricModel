#!/usr/bin/env python3
"""
next_steps_from_run.py

Orchestrates the "Next-Step Analyses" for a given Dynesty run of the
RAR-gated metric model (here focusing on xi='rar_plateau'). It:

1) Loads the most recent dynesty run (NPZ/JSON) and extracts best-fit (and
   optional posterior) parameters.
2) Runs a SPARC external-galaxy check: with MW-tuned rar_plateau params fixed,
   fit a0 per galaxy (grid search) and report chi^2 improvements vs GR.
3) Computes Solar-System ΔG/G (worst-case vs gated rar_plateau) at 1–30 AU and
   writes a CSV + plot (see docs/cassini.md for scientific background).
4) Runs a small lensing pilot using a φ_env proxy calibrated from the MW xi(R)
   profile and tools/lensing_predict.py (see docs/lensing.md). If not
   available, logs a warning and skips.
5) Produces BTFR subset products from the same SPARC sample.
6) Writes docs/next_steps.md as an index with links to generated artifacts.

Notes
- This script is pure NumPy + Matplotlib; it does NOT depend on CuPy and does
  not perform any GPU work.
- It uses your in-repo SPARC loader if available (utils/Utilities/sparc_io.py).
- It never performs network requests; any web-based data retrieval (e.g. Gaia’s
  ADQL) must be done manually as documented in docs/gaia_slices_readme.md.

Security and Keys
- There are no API keys or web requests in this script.
- Per your rule, where web services would be relevant, this script includes
  comments pointing to the README explaining manual steps (e.g. Gaia ADQL).

Usage (examples)
- Minimal, best-fit only, small gold sample:
  python scripts/next_steps_from_run.py \
    --run-dir runs/rar_plateau_mw_full \
    --sparc-dir external_data/Rotmod_LTG \
    --posterior-samples 0

- Explicit galaxies with debugging:
  python scripts/next_steps_from_run.py \
    --run-dir runs/rar_plateau_mw_full \
    --sparc-dir external_data/Rotmod_LTG \
    --galaxies NGC3198 NGC2403 \
    --posterior-samples 25 --debug

Outputs
- results/next_steps/<run_name>/* (CSVs, small JSONs)
- images/next_steps/<run_name>/* (PNGs)
- docs/next_steps.md (index)

Author: Agent Mode (auto-generated)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# Constants
G_SI = 6.6743e-11              # m^3 kg^-1 s^-2
M_SUN = 1.98847e30             # kg
AU_M = 1.495978707e11          # m
ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14  # m/s^2 per [(km/s)^2/kpc]


# ---------- Utilities -----------------------------------------------------------------

def setup_logger(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def find_npz(run_dir: Path) -> Optional[Path]:
    """Find a plausible NPZ result file in the run directory.
    Looks for named outputs from run_dynesty_stellar_fit_cupy.py, or any *.npz with
    param_names/samples keys.
    """
    if not run_dir.exists():
        return None
    # Priority patterns
    candidates = []
    for pat in [
        '*stellar_fit_cupy*_results.npz',  # produced by run_dynesty_stellar_fit_cupy.py
        '*checkpoint*latest*.npz',         # checkpointer variants
        '*.npz',                           # fallback
    ]:
        candidates.extend(run_dir.glob(pat))
    # Heuristic: pick the largest NPZ expecting it to contain full arrays
    candidates = sorted(candidates, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return candidates[0] if candidates else None


def load_run_params(run_dir: Path) -> Dict[str, float]:
    """Load best-fit parameters for rar_plateau (and companions) from a run.
    Returns a dict with at least: a0_m_s2, zeta_env, rho_c, gamma_exp, T0, sigma_lnT, wmin.
    Missing entries are filled with sensible defaults documented inline.
    """
    npz_path = find_npz(run_dir)
    params: Dict[str, float] = {}
    if npz_path is None:
        logging.warning(f"No NPZ found in {run_dir}; using defaults for rar_plateau.")
    else:
        try:
            data = np.load(npz_path, allow_pickle=True)
            best_params = None
            if 'best_params' in data and 'param_names' in data:
                names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in data['param_names']]
                vals = data['best_params']
                best_params = {n: float(vals[i]) for i, n in enumerate(names)}
            elif 'samples' in data and 'logl' in data and 'param_names' in data:
                names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in data['param_names']]
                logl = data['logl']
                samples = data['samples']
                idx = int(np.argmax(logl))
                best_params = {names[i]: float(samples[idx, i]) for i in range(samples.shape[1])}
            else:
                logging.warning(f"NPZ present but lacks recognizable keys: {npz_path.name}")

            if best_params:
                # Map likely field names to rar_plateau semantic params
                # run_dynesty_stellar_fit_cupy.py uses 'a0_m_s2' for rar_plateau
                if 'a0_m_s2' in best_params:
                    params['a0_m_s2'] = best_params['a0_m_s2']
                # Optional companions (may not be present)
                for k, dflt in [
                    ('zeta_env', 0.0),
                    ('rho_c', None),
                    ('gamma_exp', 3.0),
                    ('T0', None),
                    ('sigma_lnT', None),
                    ('wmin', 0.0),
                ]:
                    if k in best_params and best_params[k] is not None and not (isinstance(best_params[k], float) and math.isnan(best_params[k])):
                        params[k] = float(best_params[k])
        except Exception as e:
            logging.warning(f"Failed to parse NPZ {npz_path}: {e}")

    # Defaults where missing
    if 'a0_m_s2' not in params:
        params['a0_m_s2'] = 1.2e-10
    params.setdefault('zeta_env', 0.0)
    # rho_c can be None (not used if zeta_env = 0)
    if 'rho_c' not in params:
        params['rho_c'] = None
    params.setdefault('gamma_exp', 3.0)
    params.setdefault('T0', None)
    params.setdefault('sigma_lnT', None)
    params.setdefault('wmin', 0.0)

    logging.info(f"RAR-Plateau parameters: {params}")
    return params


# ---------- Model: rar_plateau (NumPy version) -----------------------------------------

def tidal_window(T: np.ndarray, T0: Optional[float], sigma_lnT: Optional[float], wmin: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T0 is None or sigma_lnT is None or sigma_lnT <= 0:
        return np.ones_like(T)
    T0 = max(float(T0), 1e-30)
    s = max(float(sigma_lnT), 1e-6)
    u = (np.log(np.maximum(T, 1e-30)) - np.log(T0)) / s
    W = np.exp(-0.5 * u * u)
    return np.clip(float(wmin) + (1.0 - float(wmin)) * W, 0.0, 1.0)


def xi_rar_plateau_numpy(
    Vbar_kms: np.ndarray,
    R_kpc: np.ndarray,
    *,
    a0_m_s2: float,
    zeta_env: float = 0.0,
    rho: Optional[np.ndarray] = None,
    rho_c: Optional[float] = None,
    gamma_exp: float = 3.0,
    T0: Optional[float] = None,
    sigma_lnT: Optional[float] = None,
    wmin: float = 0.0,
) -> np.ndarray:
    """RAR-Plateau factor D ≡ g_eff/g_bar to multiply Vbar^2.

    D = 0.5 + sqrt(0.25 + a0_eff / g_bar)
    a0_eff = a0 * (1 + zeta_env * s_rho * W(T))
    s_rho = 1 / (1 + (rho/rho_c)^gamma) when rho and rho_c are defined.

    Inputs
    - Vbar_kms: baryonic circular speed in km/s
    - R_kpc: radius in kpc

    Returns
    - xi (== D), to be used as V_model^2 = Vbar^2 * xi
    """
    Vbar_kms = np.asarray(Vbar_kms, dtype=float)
    R_kpc = np.asarray(R_kpc, dtype=float)
    R_safe = np.maximum(R_kpc, 1e-12)
    # g_bar from Vbar and R
    g_bar = ACC_M_S2_PER_KMS2_PER_KPC * np.maximum(Vbar_kms, 0.0)**2 / R_safe
    # T proxy for optional tidal window
    T = np.maximum(Vbar_kms, 0.0)**2 / np.maximum(R_safe**2, 1e-18)

    # Density gate
    if zeta_env > 0.0 and rho is not None and rho_c is not None and rho_c > 0.0:
        rho_arr = np.asarray(rho, dtype=float)
        ratio = np.maximum(rho_arr, 1e-30) / max(float(rho_c), 1e-30)
        s_rho = 1.0 / (1.0 + np.power(ratio, float(gamma_exp)))
    else:
        s_rho = 0.0

    # Tidal window
    W = tidal_window(T, T0, sigma_lnT, wmin)

    a0_eff = float(a0_m_s2) * (1.0 + float(zeta_env) * s_rho * W)
    y = np.maximum(g_bar, 1e-30)
    D = 0.5 + np.sqrt(0.25 + np.maximum(a0_eff, 0.0) / y)
    D = np.where(np.isfinite(D), D, 1.0)
    D = np.maximum(D, 1.0)
    return D


# ---------- SPARC loader wrapper --------------------------------------------------------

def _import_by_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_sparc_galaxy(galaxy_id: str, sparc_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Use the repo’s SPARC loader to fetch a galaxy's rotation curve and components.
    Returns dict with arrays: R_kpc, V_obs, e_V_obs, V_gas, V_disk, V_bulge.
    """
    load_single_sparc_galaxy = None
    try:
        # Prefer package-style import if package init exists
        from utils.Utilities.sparc_io import load_single_sparc_galaxy as _ls
        load_single_sparc_galaxy = _ls
    except Exception:
        # Fallback: import by file path
        repo_root = Path.cwd()
        candidate = repo_root / 'utils' / 'Utilities' / 'sparc_io.py'
        if candidate.exists():
            try:
                mod = _import_by_path('sparc_io_runtime', candidate)
                load_single_sparc_galaxy = getattr(mod, 'load_single_sparc_galaxy', None)
            except Exception as e:
                logging.debug(f"Path import failed for sparc_io: {e}")
    if load_single_sparc_galaxy is None:
        logging.error("SPARC loader not available. Expected utils/Utilities/sparc_io.py")
        return None

    data = load_single_sparc_galaxy(galaxy_id, sparc_dir=str(sparc_dir))
    if not data:
        return None
    try:
        return {
            'R_kpc': np.asarray(data['R_kpc'], dtype=float),
            'V_obs': np.asarray(data['V_obs'], dtype=float),
            'e_V_obs': np.asarray(data['e_V_obs'], dtype=float),
            'V_gas': np.asarray(data['V_gas_comp_kms'], dtype=float),
            'V_disk': np.asarray(data['V_disk_comp_kms'], dtype=float),
            'V_bulge': np.asarray(data['V_bulge_comp_kms'], dtype=float),
        }
    except KeyError as e:
        logging.error(f"Missing expected SPARC key: {e}")
        return None


def compute_Vbar(V_gas: np.ndarray, V_disk: np.ndarray, V_bulge: np.ndarray, ML_disk: float = 1.0, ML_bulge: float = 1.0) -> np.ndarray:
    """Compute baryonic rotation curve from components and mass-to-light scalings.
    SPARC component velocities are typically at base M/L; apply sqrt(ML_factor) scaling.
    """
    Vd = float(np.sqrt(max(ML_disk, 0.0))) * np.asarray(V_disk, dtype=float)
    Vb = float(np.sqrt(max(ML_bulge, 0.0))) * np.asarray(V_bulge, dtype=float)
    Vg = np.asarray(V_gas, dtype=float)
    return np.sqrt(np.maximum(Vd, 0.0)**2 + np.maximum(Vb, 0.0)**2 + np.maximum(Vg, 0.0)**2)


# ---------- SPARC selection helpers ------------------------------------------------------

@dataclass
class SparcSelection:
    min_npts: int = 12
    min_rmax_kpc: float = 8.0
    max_quality: int = 2  # if metadata exists; otherwise ignored


def list_sparc_galaxies(sparc_dir: Path) -> List[str]:
    """List galaxy IDs discoverable under SPARC rotmod folder."""
    glx = set()
    for p in Path(sparc_dir).glob("**/*_rotmod.dat"):
        name = p.name
        g = name.split("_rotmod")[0]
        if g:
            glx.add(g)
    return sorted(glx)


def _std_id(gid: str) -> str:
    import re
    gid_std = gid.lower().replace(" ", "")
    gid_std = re.sub(r"([a-zA-Z]+)0+(\d+)", r"\1\2", gid_std)
    return gid_std


def filter_sparc_galaxies(sparc_dir: Path, selection: SparcSelection) -> List[str]:
    """Filter by simple heuristics: min_npts, min_rmax, and optional metadata quality (Q)."""
    # Try to get metadata for Q if available
    meta_df = None
    try:
        # import via file path to avoid import-time deps
        repo_root = Path.cwd()
        candidate = repo_root / 'utils' / 'Utilities' / 'sparc_io.py'
        if candidate.exists():
            mod = _import_by_path('sparc_io_runtime_sel', candidate)
            load_meta = getattr(mod, 'load_sparc_metadata', None)
            if load_meta is not None:
                meta_df = load_meta(sparc_dir=str(sparc_dir))
                if meta_df is not None and 'Name' in meta_df.columns:
                    meta_df = meta_df.copy()
                    meta_df['StdName'] = meta_df['Name'].apply(_std_id)
    except Exception:
        meta_df = None

    out: List[str] = []
    for gid in list_sparc_galaxies(sparc_dir):
        data = load_sparc_galaxy(gid, sparc_dir)
        if not data:
            continue
        R = np.asarray(data['R_kpc'], float)
        npts = int(np.isfinite(R).sum())
        rmax = float(np.nanmax(R)) if npts else 0.0
        if npts < int(selection.min_npts) or rmax < float(selection.min_rmax_kpc):
            continue
        # Optional Q filter if metadata present
        if meta_df is not None:
            std = _std_id(gid)
            row = meta_df[meta_df['StdName'] == std]
            if len(row) == 1 and 'Q' in row.columns:
                try:
                    Q = int(row.iloc[0]['Q'])
                    if Q > int(selection.max_quality):
                        continue
                except Exception:
                    pass
        out.append(gid)
    return out


# ---------- Fitting and metrics ---------------------------------------------------------

def chi2_velocity(V_obs: np.ndarray, V_model: np.ndarray, e_V_obs: np.ndarray, sigma_floor: float = 0.0) -> float:
    V_obs = np.asarray(V_obs, dtype=float)
    V_model = np.asarray(V_model, dtype=float)
    eV = np.asarray(e_V_obs, dtype=float)
    e_eff = np.sqrt(np.maximum(eV, 0.0)**2 + max(float(sigma_floor), 0.0)**2)
    e_eff = np.where(e_eff > 0, e_eff, 1.0)
    r = (V_obs - V_model) / e_eff
    return float(np.sum(r * r))


def fit_a0_grid(
    R_kpc: np.ndarray,
    Vbar_kms: np.ndarray,
    V_obs: np.ndarray,
    e_V_obs: np.ndarray,
    rar_params: Dict[str, float],
    grid_log10: Tuple[float, float] = (-10.5, -9.3),  # ~3e-11 to 5e-10 m/s^2
    ngrid: int = 60,
    sigma_floor: float = 0.0,
) -> Tuple[float, float]:
    """Grid-search a0 to minimize chi2 for rar_plateau with fixed other params.
    Returns (a0_best, chi2_best).
    """
    a0_vals = 10 ** np.linspace(grid_log10[0], grid_log10[1], ngrid)
    chi2_vals: List[float] = []
    for a0 in a0_vals:
        xi = xi_rar_plateau_numpy(
            Vbar_kms, R_kpc,
            a0_m_s2=float(a0),
            zeta_env=float(rar_params.get('zeta_env', 0.0)),
            rho=None,  # conservative: s_rho=0 for SPARC fits unless density grids exist
            rho_c=(None if rar_params.get('rho_c', None) in (None, 0.0) else float(rar_params['rho_c'])),
            gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=float(rar_params.get('wmin', 0.0)),
        )
        V_model = np.sqrt(np.maximum(Vbar_kms, 0.0)**2 * xi)
        chi2 = chi2_velocity(V_obs, V_model, e_V_obs, sigma_floor=sigma_floor)
        chi2_vals.append(chi2)
    idx = int(np.argmin(chi2_vals))
    return float(a0_vals[idx]), float(chi2_vals[idx])


def scan_a0_grid(
    R_kpc: np.ndarray,
    Vbar_kms: np.ndarray,
    V_obs: np.ndarray,
    e_V_obs: np.ndarray,
    rar_params: Dict[str, float],
    grid_log10: Tuple[float, float] = (-10.5, -9.3),
    ngrid: int = 60,
    sigma_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (a0_vals, chi2_vals) for the scan."""
    a0_vals = 10 ** np.linspace(grid_log10[0], grid_log10[1], ngrid)
    chi2_vals: List[float] = []
    for a0 in a0_vals:
        xi = xi_rar_plateau_numpy(
            Vbar_kms, R_kpc,
            a0_m_s2=float(a0),
            zeta_env=float(rar_params.get('zeta_env', 0.0)),
            rho=None,
            rho_c=(None if rar_params.get('rho_c', None) in (None, 0.0) else float(rar_params['rho_c'])),
            gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=float(rar_params.get('wmin', 0.0)),
        )
        V_model = np.sqrt(np.maximum(Vbar_kms, 0.0)**2 * xi)
        chi2_vals.append(chi2_velocity(V_obs, V_model, e_V_obs, sigma_floor=sigma_floor))
    return a0_vals, np.asarray(chi2_vals, dtype=float)


def fit_a0_err_from_grid(a0_vals: np.ndarray, chi2_vals: np.ndarray) -> Tuple[float, float]:
    """Quadratic approx near the min to get best a0 and 1σ error from Δχ²=1."""
    i = int(np.argmin(chi2_vals))
    a0_best = float(a0_vals[i])
    # Use neighbors if available
    lo = max(i-1, 0)
    hi = min(i+1, len(a0_vals)-1)
    x = a0_vals[lo:hi+1]
    y = chi2_vals[lo:hi+1]
    if len(x) >= 3:
        A = np.vstack([x**2, x, np.ones_like(x)]).T
        try:
            a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
            if a > 0:
                da = float(np.sqrt(1.0/a))
                return a0_best, da
        except Exception:
            pass
    return a0_best, float('nan')


# ---------- Solar System analysis -------------------------------------------------------

def solar_system_table(
    rar_params: Dict[str, float],
    radii_AU: List[float] = [1, 5, 10, 20, 30],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    a0 = float(rar_params.get('a0_m_s2', 1.2e-10))
    # Worst-case: no gating (s_rho=W=0 in a0_eff formula but that only reduces a0_eff here; for a
    # "worst-case" larger deviation, adopt s_rho=W=1 -> a0_eff = a0 * (1 + zeta_env). If zeta_env=0,
    # this reduces to standard rar-plateau.)
    zeta = float(rar_params.get('zeta_env', 0.0))
    T0 = rar_params.get('T0', None)
    sig = rar_params.get('sigma_lnT', None)
    wmin = float(rar_params.get('wmin', 0.0))

    for AU in radii_AU:
        r_m = float(AU) * AU_M
        # Kepler g_bar at radius r around the Sun
        g_bar = G_SI * M_SUN / max(r_m**2, 1.0)
        # Express g_bar using our D formula: need Vbar and R. For a circular orbit,
        # V^2/R = g_bar => pick an arbitrary R_kpc and V_kms that satisfy this relation.
        # Choose R_kpc from AU and Vbar from g_bar.
        R_kpc = r_m / (3.085677581491367e19)
        V_ms = math.sqrt(g_bar * (r_m))
        V_kms = V_ms / 1000.0
        # Worst-case: W=1, s_rho=1 -> a0_eff = a0*(1+zeta)
        xi_worst = 0.5 + math.sqrt(0.25 + (a0 * (1.0 + zeta)) / max(g_bar, 1e-30))
        # Gated: s_rho from a representative local density; W(T) from Kepler T
        # Adopt a nominal local midplane density ~ 0.04 Msun/pc^3 = 1.5e-21 kg/m^3. For rar_plateau, rho is only
        # used if zeta_env>0; otherwise effect is zero.
        rho_local = 1.5e-21
        if rar_params.get('rho_c', None) not in (None, 0.0) and zeta > 0.0:
            ratio = rho_local / max(float(rar_params['rho_c']), 1e-30)
            s_rho = 1.0 / (1.0 + ratio**float(rar_params.get('gamma_exp', 3.0)))
        else:
            s_rho = 0.0
        # Tidal window for Solar System: T = GM/r^3 in SI; convert to our T-units ≈ (km/s)^2/kpc^2
        T_si = G_SI * M_SUN / max(r_m**3, 1.0)
        # Convert SI to (km/s)^2/kpc^2: 1 (km/s)^2/kpc^2 = (1000 m/s)^2 / (kpc in m)^2
        KPC_M = 3.085677581491367e19
        unit = (1000.0**2) / (KPC_M**2)
        T_unit = T_si / unit
        if T0 is None or sig is None or sig <= 0:
            W = 1.0
        else:
            u = (math.log(max(T_unit, 1e-30)) - math.log(max(float(T0), 1e-30))) / max(float(sig), 1e-6)
            W = max(float(wmin) + (1.0 - float(wmin)) * math.exp(-0.5 * u * u), 0.0)
        a0_eff = a0 * (1.0 + zeta * s_rho * W)
        xi_gated = 0.5 + math.sqrt(0.25 + a0_eff / max(g_bar, 1e-30))

        rows.append({
            'AU': float(AU),
            'g_bar_m_s2': g_bar,
            'xi_worst': xi_worst,
            'dGoverG_worst': xi_worst - 1.0,
            'xi_gated': xi_gated,
            'dGoverG_gated': xi_gated - 1.0,
        })
    return rows


# ---------- Lensing baseline (anchored GR + SIS) ---------------------------------------

def _ang_dists(z_l: float, z_s: float, H0: float = 70.0, Om0: float = 0.3) -> Tuple[float, float, float]:
    """Angular diameter distances (Dl, Ds, Dls) in meters.
    Tries astropy; falls back to a simple flat-ΛCDM integral if not available.
    See docs/lensing.md and README (lensing baselines and cosmology setup).
    """
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
        D_s = cosmo.angular_diameter_distance(z_s).to(u.m).value
        D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value
        return D_l, D_s, D_ls
    except Exception:
        c = 299792.458  # km/s
        H0s = H0 / 3.085677581e19  # s^-1
        def Ez(z): return math.sqrt(Om0*(1+z)**3 + (1-Om0))
        def Dc(z, N=4096):
            zz = np.linspace(0.0, z, N)
            return (c*1000.0/H0s) * np.trapz(1.0/np.vectorize(Ez)(zz), zz)
        D_l = Dc(z_l)/(1+z_l)
        D_s = Dc(z_s)/(1+z_s)
        D_ls = (Dc(z_s)-Dc(z_l))/(1+z_s)
        return D_l, D_s, D_ls

def theta_E_pointmass_arcsec(M_Msun: float, z_l: float, z_s: float, H0: float = 70.0, Om0: float = 0.3) -> float:
    c = 299792458.0
    D_l, D_s, D_ls = _ang_dists(z_l, z_s, H0, Om0)
    if not (D_l > 0 and D_s > 0 and D_ls > 0):
        return 0.0
    M = float(M_Msun) * M_SUN
    term = (4.0 * G_SI * M / c**2) * (D_ls / (D_l * D_s))
    theta = math.sqrt(max(term, 0.0))  # radians
    return float(theta * 206265.0)

def theta_E_sis_arcsec(Vflat_kms: float, z_l: float, z_s: float, H0: float = 70.0, Om0: float = 0.3) -> float:
    # SIS: theta_E = 4π (σ^2 / c^2) D_ls / D_s with V_c ≈ sqrt(2) σ
    c_kms = 299792.458
    sigma = max(float(Vflat_kms), 0.0) / math.sqrt(2.0)
    D_l, D_s, D_ls = _ang_dists(z_l, z_s, H0, Om0)
    if not (D_l > 0 and D_s > 0 and D_ls > 0):
        return 0.0
    theta = 4.0 * math.pi * (sigma/c_kms)**2 * (D_ls / D_s)
    return float(theta * 206265.0)

def run_lensing_pilot(out_dir: Path, rar_params: Dict[str, float]) -> None:
    """Anchored lensing baselines: GR point-mass and SIS yardsticks.
    We intentionally avoid environment-coupling here and provide a physical GR baseline
    that cannot zero-out spuriously.
    """
    z_l, z_s = 0.2, 0.6
    Re_kpc = 5.0
    M_star = 10**11.2  # Msun (conservative stellar lens mass)
    th_gr = theta_E_pointmass_arcsec(M_star, z_l, z_s)
    th_sis_200 = theta_E_sis_arcsec(200.0, z_l, z_s)
    th_sis_250 = theta_E_sis_arcsec(250.0, z_l, z_s)

    table = out_dir / 'lensing_table.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('w', encoding='utf-8') as f:
        f.write('z_l,z_s,Re_kpc,log10M,theta_E_GR_arcsec,theta_E_SIS200_arcsec,theta_E_SIS250_arcsec\n')
        f.write(f"{z_l},{z_s},{Re_kpc},11.2,{th_gr:.3f},{th_sis_200:.3f},{th_sis_250:.3f}\n")

    if th_gr < 0.05:
        logging.warning(f"GR baryon-only θ_E unexpectedly tiny ({th_gr:.3f} arcsec) - check distances/units.")
    logging.info(f"Lensing pilot written: {table}")


# ---------- Main orchestrator ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Next steps orchestrator for rar_plateau runs')
    ap.add_argument('--run-dir', required=True, help='Path to the run folder (e.g., runs/rar_plateau_mw_full)')
    ap.add_argument('--sparc-dir', required=True, help='Path to SPARC rotmod folder (e.g., external_data/Rotmod_LTG)')
    ap.add_argument('--galaxies', nargs='*', default=None, help='Subset of SPARC galaxies to analyze (e.g., NGC3198 NGC2403)')
    ap.add_argument('--sample', default='gold', choices=['gold','all','q2plus'], help='SPARC sample selection if --galaxies not provided')
    ap.add_argument('--min-npts', type=int, default=12, help='Minimum RC points for inclusion')
    ap.add_argument('--min-rmax-kpc', type=float, default=8.0, help='Minimum R_max (kpc) for inclusion')
    ap.add_argument('--max-quality', type=int, default=2, help='Use Q <= max-quality if SPARC metadata is available')
    ap.add_argument('--sigma-floor', type=float, default=5.0, help='Velocity error floor (km/s) in chi2')
    ap.add_argument('--fit-global-a0', action='store_true', help='Also compute a global a0 across the sample')
    ap.add_argument('--posterior-samples', type=int, default=0, help='Optional number of posterior samples to propagate (0=best-fit only)')
    ap.add_argument('--out-root', default=None, help='Output results root, default results/next_steps/<run_name>')
    ap.add_argument('--images-root', default=None, help='Images root, default images/next_steps/<run_name>')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    setup_logger(args.debug)

    run_dir = Path(args.run_dir)
    sparc_dir = Path(args.sparc_dir)
    run_name = run_dir.name

    # Output roots
    results_root = Path(args.out_root) if args.out_root else Path('results') / 'next_steps' / run_name
    images_root = Path(args.images_root) if args.images_root else Path('images') / 'next_steps' / run_name
    docs_root = Path('docs')

    for p in [results_root, images_root]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) Load rar_plateau parameters
    rar_params = load_run_params(run_dir)

    # Save a metadata snapshot for reproducibility
    (results_root / 'run_metadata.json').write_text(json.dumps({'run_dir': str(run_dir), 'rar_plateau_params': rar_params}, indent=2), encoding='utf-8')

    # 2) SPARC a0 universality (initial or filtered subset)
    if args.galaxies:
        sample = args.galaxies
    else:
        if args.sample == 'gold':
            sample = ['M31', 'NGC3198', 'NGC2403', 'NGC2841', 'NGC5055']
        else:
            sel = SparcSelection(min_npts=args.min_npts, min_rmax_kpc=args.min_rmax_kpc, max_quality=args.max_quality)
            sample = filter_sparc_galaxies(sparc_dir, sel)
    logging.info(f"SPARC sample size: {len(sample)}")

    csv_path = results_root / 'sparc_a0_summary.csv'
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('galaxy,a0_best_m_s2,chi2_rar,chi2_gr,dof,notes\n')

    galaxy_store: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for gid in sample:
        logging.info(f"SPARC: {gid}")
        data = load_sparc_galaxy(gid, sparc_dir)
        if data is None:
            logging.warning(f"Skipping {gid}: missing data")
            continue
        R = data['R_kpc']
        Vobs = data['V_obs']
        eV = data['e_V_obs']
        Vbar = compute_Vbar(data['V_gas'], data['V_disk'], data['V_bulge'], ML_disk=1.0, ML_bulge=1.0)

        a0_vals, chi2_vals = scan_a0_grid(R, Vbar, Vobs, eV, rar_params, sigma_floor=float(args.sigma_floor))
        j = int(np.argmin(chi2_vals))
        a0_best = float(a0_vals[j])
        chi2_rar = float(chi2_vals[j])
        a0_best_fit, a0_err = fit_a0_err_from_grid(a0_vals, chi2_vals)
        V_model = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi_rar_plateau_numpy(
            Vbar, R,
            a0_m_s2=a0_best,
            zeta_env=rar_params.get('zeta_env', 0.0),
            rho=None,
            rho_c=rar_params.get('rho_c', None),
            gamma_exp=rar_params.get('gamma_exp', 3.0),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=rar_params.get('wmin', 0.0)
        ))
        chi2_gr = chi2_velocity(Vobs, np.maximum(Vbar, 0.0), eV, sigma_floor=float(args.sigma_floor))
        dof = max(len(R) - 1, 1)
        notes = f"a0_pm={a0_err:.2e}" if np.isfinite(a0_err) else ''

        # Save CSV row
        with csv_path.open('a', encoding='utf-8') as f:
            f.write(f"{gid},{a0_best_fit:.6e},{chi2_rar:.3f},{chi2_gr:.3f},{dof},{notes}\n")

        # Stash for possible global a0
        galaxy_store.append((gid, R, Vbar, Vobs, eV, a0_best))

        # Overlay plot
        plt.figure(figsize=(7.2, 5.0))
        plt.errorbar(R, Vobs, yerr=eV, fmt='o', color='k', ms=3, lw=1, alpha=0.8, label='Observed')
        plt.plot(R, np.maximum(Vbar, 0.0), 'b--', lw=2, label='Baryons (GR)')
        plt.plot(R, V_model, 'r-', lw=2, label='RAR-plateau (a0 fit)')
        plt.xlabel('R (kpc)')
        plt.ylabel('Vc (km/s)')
        plt.title(f"{gid} — RAR-plateau vs GR\n(a0={a0_best:.2e} m/s^2; Δχ²={chi2_gr-chi2_rar:.1f})")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        outpng = images_root / f"sparc_overlay_{gid.replace(' ','_')}.png"
        plt.tight_layout()
        plt.savefig(outpng, dpi=140)
        plt.close()
        logging.info(f"Saved {outpng}")

    logging.info(f"SPARC summary: {csv_path}")

    # 3) Solar-System ΔG/G
    solar_rows = solar_system_table(rar_params)
    solar_csv = results_root / 'solar_system_table.csv'
    with solar_csv.open('w', encoding='utf-8') as f:
        f.write('AU,g_bar_m_s2,xi_worst,dGoverG_worst,xi_gated,dGoverG_gated\n')
        for r in solar_rows:
            f.write(f"{r['AU']:.1f},{r['g_bar_m_s2']:.6e},{r['xi_worst']:.6e},{r['dGoverG_worst']:.6e},{r['xi_gated']:.6e},{r['dGoverG_gated']:.6e}\n")
    logging.info(f"Solar table: {solar_csv}")

    # Plot
    AUs = [r['AU'] for r in solar_rows]
    worst = [r['dGoverG_worst'] for r in solar_rows]
    gated = [r['dGoverG_gated'] for r in solar_rows]
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(AUs, worst, 'o-', label='worst-case (RAR-plateau, W=s_ρ=1)')
    plt.semilogy(AUs, gated, 's--', label='gated rar_plateau')
    # Cassini bound at Saturn (~9.6 AU): |γ-1| < 2.3e-5; we annotate at 10 AU for visibility
    plt.axhline(2.3e-5, color='k', ls=':', label='Cassini bound ~2.3e-5')
    plt.xlabel('Orbital distance (AU)')
    plt.ylabel('|ΔG/G| ≈ |ξ − 1|')
    plt.title('Solar-System constraints (RAR-plateau params)')
    plt.grid(alpha=0.3, which='both')
    plt.legend(frameon=False)
    outpng = images_root / 'solar_rar_plateau.png'
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close()
    logging.info(f"Saved {outpng}")

    # 4) Lensing pilot
    run_lensing_pilot(results_root, rar_params)

    # 5) BTFR subset (baryonic mass + observed V_flat)
    btfr_csv = results_root / 'btfr_summary.csv'
    if not btfr_csv.exists():
        with btfr_csv.open('w', encoding='utf-8') as f:
            f.write('galaxy,log10_Mb,log10_Vflat,source,notes\n')

    def estimate_vflat(R, Vobs):
        n = len(R)
        if n < 6:
            return float('nan'), 'too_few_points'
        idx0 = int(0.7*n)
        r = R[idx0:]
        v = Vobs[idx0:]
        if len(v) < 4:
            return float('nan'), 'too_few_outer'
        A = np.vstack([r, np.ones_like(r)]).T
        m, c = np.linalg.lstsq(A, v, rcond=None)[0]
        slope = abs(m) / max(np.nanmedian(v), 1e-6)
        note = f'slope_rel={slope:.2f}'
        if slope <= 0.10:
            note = 'flat_outer'
        return float(np.nanmedian(v)), note

    try:
        for gid in (args.galaxies or sample):
            d = load_sparc_galaxy(gid, sparc_dir)
            if not d:
                continue
            R = np.asarray(d['R_kpc'], float)
            Vobs = np.asarray(d['V_obs'], float)
            Vflat, note = estimate_vflat(R, Vobs)
            # gas mass (HI+He)
            MHI = float(d.get('M_HI_Msun', float('nan')))
            Mgas = 1.33 * MHI if np.isfinite(MHI) else float('nan')
            # stellar mass via Σ_* integration (base M/L)
            Sig_star = d.get('Sigma_star_Msun_pc2_baseML', None)
            Mstar = float('nan')
            if Sig_star is not None:
                R_pc = np.asarray(R) * 1000.0
                Sig_star = np.asarray(Sig_star, float)
                if np.all(np.isfinite(R_pc)) and np.all(np.isfinite(Sig_star)) and len(R_pc) > 2:
                    # M = 2π ∫ Σ(R) R dR (with R, dR in pc)
                    integrand = Sig_star * R_pc
                    Mstar = 2.0 * math.pi * float(np.trapz(integrand, R_pc))
            Mb = np.nansum([Mstar, Mgas])
            with btfr_csv.open('a', encoding='utf-8') as f:
                f.write(f"{gid},{(np.log10(Mb) if np.isfinite(Mb) and Mb>0 else 'nan')},{(np.log10(Vflat) if np.isfinite(Vflat) and Vflat>0 else 'nan')},obs_outer,{note}\n")
    except Exception as e:
        logging.warning(f"BTFR subset step skipped: {e}")

    # Optional: 5b) Global a0 across sample
    if args.fit_global_a0 and len(galaxy_store) > 0:
        a0_grid = 10 ** np.linspace(-10.5, -9.3, 80)
        totals: List[float] = []
        for a0 in a0_grid:
            tot = 0.0
            for (_gid, R, Vbar, Vobs, eV, _a0best) in galaxy_store:
                xi = xi_rar_plateau_numpy(Vbar, R, a0_m_s2=float(a0))
                Vmod = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi)
                tot += chi2_velocity(Vobs, Vmod, eV, sigma_floor=float(args.sigma_floor))
            totals.append(tot)
        a0_grid = np.asarray(a0_grid, float)
        totals = np.asarray(totals, float)
        a0_global, a0_sigma = fit_a0_err_from_grid(a0_grid, totals)
        (results_root/'global_a0.json').write_text(json.dumps({'a0_m_s2': a0_global, 'sigma': a0_sigma, 'n_gal': len(galaxy_store)}, indent=2), encoding='utf-8')
        logging.info(f"Global a0 ~ {a0_global:.3e} ± {a0_sigma if np.isfinite(a0_sigma) else float('nan'):.1e} m/s^2 over {len(galaxy_store)} galaxies")

    # 6) Write docs index stub
    ndx = Path('docs') / 'next_steps.md'
    try:
        lines = []
        lines.append('# Next-Step Analyses Index')
        lines.append('')
        lines.append(f"Run: `{run_name}`")
        lines.append('')
        lines.append('Artifacts:')
        lines.append(f"- SPARC summary: `{csv_path.as_posix()}`")
        lines.append(f"- Solar table: `{solar_csv.as_posix()}`, plot: `{(images_root / 'solar_rar_plateau.png').as_posix()}`")
        lines.append(f"- Lensing baseline table: `{(results_root / 'lensing_table.csv').as_posix()}` (if present)")
        lines.append(f"- BTFR subset: `{btfr_csv.as_posix()}`")
        lines.append(f"- Global a0: `{(results_root / 'global_a0.json').as_posix()}` (if present)")
        lines.append('')
        lines.append('Method Notes:')
        lines.append('- RAR-plateau: D = 0.5 + sqrt(0.25 + a0_eff/g_bar); xi == D multiplies Vbar^2')
        lines.append('- g_bar = (Vbar^2 / R) × 3.240779289e-14 in SI (m/s^2) for V in km/s and R in kpc')
        lines.append('- a0_eff = a0 × (1 + zeta_env × s_rho × W(T)); see docs/cassini.md and docs/lensing.md')
        lines.append('- Lensing baselines (GR point-mass, SIS) use Planck-like flat-ΛCDM distances; see docs/lensing.md')
        ndx.write_text('\n'.join(lines), encoding='utf-8')
        logging.info(f"Wrote {ndx}")
    except Exception as e:
        logging.warning(f"Could not write docs/next_steps.md: {e}")

    logging.info('Done.')


if __name__ == '__main__':
    main()
