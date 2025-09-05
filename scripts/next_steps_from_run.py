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


# ---------- Lensing pilot (optional) ---------------------------------------------------

def run_lensing_pilot(out_dir: Path, rar_params: Dict[str, float]) -> None:
    """Use tools/lensing_predict in a pilot mode by calibrating a crude φ_env.
    If the module is unavailable, attempt file-path import; otherwise skip.
    """
    Hernquist = PhiEnv = einstein_radius_arcsec = None
    try:
        from tools.lensing_predict import Hernquist as _H, PhiEnv as _P, einstein_radius_arcsec as _E
        Hernquist, PhiEnv, einstein_radius_arcsec = _H, _P, _E
    except Exception:
        repo_root = Path.cwd()
        cand = repo_root / 'tools' / 'lensing_predict.py'
        if cand.exists():
            try:
                mod = _import_by_path('lensing_predict_runtime', cand)
                Hernquist = getattr(mod, 'Hernquist', None)
                PhiEnv = getattr(mod, 'PhiEnv', None)
                einstein_radius_arcsec = getattr(mod, 'einstein_radius_arcsec', None)
            except Exception as e:
                logging.debug(f"Path import failed for lensing_predict: {e}")
    if any(x is None for x in (Hernquist, PhiEnv, einstein_radius_arcsec)):
        logging.warning("Lensing tools not available (tools/lensing_predict.py). Skipping lensing pilot.")
        return

    # Crude calibration: define xi at 10 kpc using a nominal Vbar=180 km/s, and map to φ_env amplitude
    R0 = 10.0  # kpc
    Vbar0 = 180.0  # km/s (nominal)
    xi0 = float(xi_rar_plateau_numpy(np.array([Vbar0]), np.array([R0]), **rar_params)[0])
    A_env = max(min(0.5 * math.log(max(xi0, 1.0)), 0.6), 0.05)  # keep amplitude modest

    penv = PhiEnv(A_env=A_env, p=1.2, r0_kpc=5.0)
    # Simple SLACS-like lens
    z_l, z_s = 0.2, 0.6
    Re_kpc = 5.0
    a_kpc = Re_kpc / 1.8153
    M_star = 10**11.2
    lens = Hernquist(M_star=M_star, a_kpc=a_kpc)

    th_gr = einstein_radius_arcsec(lens, penv, z_l, z_s, mode="gr")
    th_env = einstein_radius_arcsec(lens, penv, z_l, z_s, mode="tfr", a_env=1.0, b_env=1.0)

    table = out_dir / 'lensing_table.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('w', encoding='utf-8') as f:
        f.write('z_l,z_s,Re_kpc,log10M,theta_E_GR_arcsec,theta_E_TFR_arcsec,A_env,p,r0_kpc\n')
        f.write(f"{z_l},{z_s},{Re_kpc},11.2,{th_gr:.3f},{th_env:.3f},{A_env:.3f},1.2,5.0\n")
    logging.info(f"Lensing pilot written: {table}")


# ---------- Main orchestrator ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Next steps orchestrator for rar_plateau runs')
    ap.add_argument('--run-dir', required=True, help='Path to the run folder (e.g., runs/rar_plateau_mw_full)')
    ap.add_argument('--sparc-dir', required=True, help='Path to SPARC rotmod folder (e.g., external_data/Rotmod_LTG)')
    ap.add_argument('--galaxies', nargs='*', default=None, help='Subset of SPARC galaxies to analyze (e.g., NGC3198 NGC2403)')
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

    # 2) SPARC a0 universality (initial subset)
    if args.galaxies:
        sample = args.galaxies
    else:
        # Prefer Andromeda if present; else a small robust subset
        sample = ['M31', 'NGC3198', 'NGC2403', 'NGC2841', 'NGC5055']

    csv_path = results_root / 'sparc_a0_summary.csv'
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('galaxy,a0_best_m_s2,chi2_rar,chi2_gr,dof,notes\n')

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

        a0_best, chi2_rar = fit_a0_grid(R, Vbar, Vobs, eV, rar_params, sigma_floor=5.0)
        V_model = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi_rar_plateau_numpy(Vbar, R, a0_m_s2=a0_best, zeta_env=rar_params.get('zeta_env', 0.0), rho=None, rho_c=rar_params.get('rho_c', None), gamma_exp=rar_params.get('gamma_exp', 3.0), T0=rar_params.get('T0', None), sigma_lnT=rar_params.get('sigma_lnT', None), wmin=rar_params.get('wmin', 0.0)))
        chi2_gr = chi2_velocity(Vobs, np.maximum(Vbar, 0.0), eV, sigma_floor=5.0)
        dof = max(len(R) - 1, 1)
        notes = ''

        # Save CSV row
        with csv_path.open('a', encoding='utf-8') as f:
            f.write(f"{gid},{a0_best:.6e},{chi2_rar:.3f},{chi2_gr:.3f},{dof},{notes}\n")

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

    # 5) BTFR subset (simple)
    # For now, reuse last loaded galaxy per loop to compute V_flat; aggregate if files exist.
    # A simple implementation: parse overlays and compute V_flat as median model velocity for R>=0.8*R_max where R_max from data.
    btfr_csv = results_root / 'btfr_summary.csv'
    if not btfr_csv.exists():
        with btfr_csv.open('w', encoding='utf-8') as f:
            f.write('galaxy,log10_Mb,log10_Vflat,selection_note\n')
    try:
        # Re-iterate galaxies and compute a crude V_flat and M_b from SPARC metadata if available via sparc_io.
        load_single_sparc_galaxy = None
        try:
            from utils.Utilities.sparc_io import load_single_sparc_galaxy as _ls
            load_single_sparc_galaxy = _ls
        except Exception:
            repo_root = Path.cwd()
            candidate = repo_root / 'utils' / 'Utilities' / 'sparc_io.py'
            if candidate.exists():
                mod = _import_by_path('sparc_io_runtime_btfr', candidate)
                load_single_sparc_galaxy = getattr(mod, 'load_single_sparc_galaxy', None)
        if load_single_sparc_galaxy is None:
            raise ImportError('sparc_io not available')
        for gid in (args.galaxies or sample):
            data = load_single_sparc_galaxy(gid, sparc_dir=str(sparc_dir))
            if not data:
                continue
            R = np.asarray(data['R_kpc'], dtype=float)
            Vbar = compute_Vbar(np.asarray(data['V_gas_comp_kms']), np.asarray(data['V_disk_comp_kms']), np.asarray(data['V_bulge_comp_kms']))
            a0 = load_run_params(run_dir).get('a0_m_s2', 1.2e-10)  # reload to avoid mutation
            xi = xi_rar_plateau_numpy(Vbar, R, a0_m_s2=a0)
            Vmod = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi)
            # V_flat: outermost third
            n = len(R)
            sel = slice(max(0, int(2*n/3)), n)
            Vflat = float(np.median(Vmod[sel])) if n > 0 else float('nan')
            # M_b (crude): sum of stellar+gas masses from metadata when available
            Mb = 0.0
            try:
                Mb += float(data.get('M_HI_Msun', 0.0)) * 1.33  # include He
            except Exception:
                pass
            # Rough stellar mass proxy from SB integrals is not readily available; leave as NaN if absent
            Mb = Mb if Mb > 0 else float('nan')
            with btfr_csv.open('a', encoding='utf-8') as f:
                f.write(f"{gid},{(math.log10(Mb) if Mb>0 else 'nan')},{(math.log10(Vflat) if Vflat>0 else 'nan')},outer_third_median\n")
    except Exception as e:
        logging.warning(f"BTFR subset step skipped: {e}")

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
        lines.append(f"- Lensing pilot table: `{(results_root / 'lensing_table.csv').as_posix()}` (if present)")
        lines.append(f"- BTFR subset: `{btfr_csv.as_posix()}`")
        lines.append('')
        lines.append('Method Notes:')
        lines.append('- RAR-plateau: D = 0.5 + sqrt(0.25 + a0_eff/g_bar); xi == D multiplies Vbar^2')
        lines.append('- g_bar = (Vbar^2 / R) × 3.240779289e-14 in SI (m/s^2) for V in km/s and R in kpc')
        lines.append('- a0_eff = a0 × (1 + zeta_env × s_rho × W(T)); see docs/cassini.md and docs/lensing.md')
        ndx.write_text('\n'.join(lines), encoding='utf-8')
        logging.info(f"Wrote {ndx}")
    except Exception as e:
        logging.warning(f"Could not write docs/next_steps.md: {e}")

    logging.info('Done.')


if __name__ == '__main__':
    main()
