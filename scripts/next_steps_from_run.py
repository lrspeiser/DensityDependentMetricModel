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
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Placeholders; will populate after helper import function is defined
evaluate_ppn = None
check_c_T_guardrail = None

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# Constants
G_SI = 6.6743e-11              # m^3 kg^-1 s^-2
M_SUN = 1.98847e30             # kg
AU_M = 1.495978707e11          # m
ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14  # m/s^2 per [(km/s)^2/kpc]
KMPS2_PER_KPC_TO_M_S2 = ACC_M_S2_PER_KMS2_PER_KPC  # alias for clarity
TWOPI_G = 2.0 * np.pi * G_SI  # SI
CASSINI_BOUND_GAMMA = 2.3e-5


# ---------- Utilities -----------------------------------------------------------------

def setup_logger(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def write_source_data(path: str, **arrays) -> None:
    """Write a Source-Data CSV with named columns.
    Imports pandas locally to avoid hard dependency if unused.
    """
    try:
        import pandas as pd
        from os import makedirs
        d = os.path.dirname(path)
        if d:
            makedirs(d, exist_ok=True)
        df = pd.DataFrame({k: np.asarray(v) for k, v in arrays.items()})
        df.to_csv(path, index=False)
    except Exception:
        # Fallback: write raw CSV without pandas
        cols = list(arrays.keys())
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(','.join(cols) + '\n')
            # Infer length from first array
            n = len(np.asarray(arrays[cols[0]])) if cols else 0
            for i in range(n):
                row = [str(np.asarray(arrays[c])[i]) for c in cols]
                f.write(','.join(row) + '\n')


def find_npz(run_dir: Path) -> Optional[Path]:
    """Find a plausible NPZ result file in the run directory.
    Preference order:
    1) Any NPZ that contains both 'samples' and 'param_names' (posterior-friendly), largest first.
    2) Otherwise, fall back to the largest NPZ.
    """
    if not run_dir.exists():
        return None
    # Collect candidates
    cands = []
    for pat in [
        '*stellar_fit_cupy*_results.npz',  # produced by run_dynesty_stellar_fit_cupy.py
        '*checkpoint*latest*.npz',         # checkpointer variants
        '*.npz',                           # fallback
    ]:
        cands.extend(run_dir.glob(pat))
    if not cands:
        return None
    # Sort by size desc
    cands = sorted(cands, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    # Prefer those with posterior keys
    good = []
    for p in cands:
        try:
            with np.load(p, allow_pickle=False) as z:
                files = set(z.files)
                if 'samples' in files and 'param_names' in files:
                    good.append(p)
        except Exception:
            continue
    if good:
        # Largest among good
        good = sorted(good, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        return good[0]
    # Fallback: largest overall
    return cands[0]


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


def _load_npz_arrays(run_dir: Path):
    npz = find_npz(run_dir)
    if npz is None:
        return None
    try:
        data = np.load(npz, allow_pickle=True)
        return data
    except Exception:
        return None


def load_posterior_samples_npz(run_dir: Path, max_samples: int = 100) -> List[Dict[str, float]]:
    """Return a list of parameter dicts sampled from NPZ posterior if available.
    We derive weights from logl (or use 'weights' if present), then resample.
    """
    data = _load_npz_arrays(run_dir)
    if data is None:
        return []
    try:
        if 'samples' not in data or 'param_names' not in data:
            return []
        names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in data['param_names']]
        smp = np.asarray(data['samples'], float)
        if smp.ndim != 2 or smp.shape[0] < 2:
            return []
        if 'weights' in data:
            w = np.asarray(data['weights'], float)
        elif 'logl' in data:
            ll = np.asarray(data['logl'], float)
            m = float(np.max(ll))
            w = np.exp(ll - m)
        else:
            w = np.ones(smp.shape[0], float)
        w = np.maximum(w, 0.0)
        if float(np.sum(w)) <= 0:
            w = np.ones_like(w)
        w = w / float(np.sum(w))
        # Resample indices
        n = min(int(max_samples), smp.shape[0])
        idx = np.random.default_rng().choice(np.arange(smp.shape[0]), size=n, replace=True, p=w)
        out: List[Dict[str, float]] = []
        for i in idx:
            row = smp[i]
            pd = {names[j]: float(row[j]) for j in range(row.shape[0])}
            # Keep only known rar_plateau keys
            keep = {}
            for k in ('a0_m_s2','zeta_env','rho_c','gamma_exp','T0','sigma_lnT','wmin'):
                if k in pd:
                    keep[k] = pd[k]
            out.append(keep)
        return out
    except Exception:
        return []


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
    D_max: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """RAR gate (weak-field): returns (xi ≡ D, meta dict).

    V_model^2 = xi * Vbar^2
    D = 0.5 + sqrt(0.25 + a0_eff / g_bar)
    a0_eff = a0 * (1 + zeta_env * s_rho * W(T))
    s_rho = 1 / (1 + (rho/rho_c)^gamma) when rho and rho_c are defined.
    Optional finite plateau: D <= D_max if provided (>1).
    """
    Vbar_kms = np.asarray(Vbar_kms, dtype=float)
    R_kpc = np.asarray(R_kpc, dtype=float)
    R_safe = np.maximum(R_kpc, 1e-12)
    # g_bar from Vbar and R
    g_bar = ACC_M_S2_PER_KMS2_PER_KPC * np.maximum(Vbar_kms, 0.0)**2 / R_safe
    y = np.maximum(g_bar, 1e-30)
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

    # Effective a0 and boost
    a0_eff = float(a0_m_s2) * (1.0 + float(zeta_env) * s_rho * W)
    D = 0.5 + np.sqrt(0.25 + np.maximum(a0_eff, 0.0) / y)
    D = np.where(np.isfinite(D), D, 1.0)
    D = np.maximum(D, 1.0)

    # Optional finite plateau
    if (D_max is not None) and np.isfinite(D_max) and (float(D_max) > 1.0):
        D = np.minimum(D, float(D_max))

    meta = {
        'g_bar_m_s2': g_bar,
        'T': T,
        's_rho': s_rho,
        'W': W,
        'a0_eff': a0_eff,
        'D_max': D_max,
    }
    return D, meta


# ---------- SPARC loader wrapper --------------------------------------------------------

def _import_by_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Relativistic scaffolding (PPN export, c_T guardrail)
try:
    from theory.relativistic import evaluate_ppn as _eval_ppn_imp, check_c_T_guardrail as _ct_guard_imp
    evaluate_ppn = _eval_ppn_imp
    check_c_T_guardrail = _ct_guard_imp
except Exception:
    # Fallback: import by file path relative to repo root
    try:
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parent
        relativistic_path = repo_root / 'theory' / 'relativistic.py'
        if relativistic_path.exists():
            mod_rel = _import_by_path('relativistic_runtime', relativistic_path)
            evaluate_ppn = getattr(mod_rel, 'evaluate_ppn', None)
            check_c_T_guardrail = getattr(mod_rel, 'check_c_T_guardrail', None)
        else:
            evaluate_ppn = None
            check_c_T_guardrail = None
    except Exception:
        evaluate_ppn = None
        check_c_T_guardrail = None


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


def chi2_velocity_with_frac(V_obs: np.ndarray, V_model: np.ndarray, e_V_obs: np.ndarray,
                            sigma_floor: float = 0.0, obs_frac_sigma: float = 0.0) -> float:
    """Like chi2_velocity, but inflates errors by a fractional term f*V_obs to capture
    observational nuisances (distance, inclination, beam/non-circular) in aggregate.
    This provides a conservative nuisance treatment when full metadata are not available.
    """
    V_obs = np.asarray(V_obs, dtype=float)
    V_model = np.asarray(V_model, dtype=float)
    eV = np.asarray(e_V_obs, dtype=float)
    f = max(float(obs_frac_sigma), 0.0)
    e_eff = np.sqrt(np.maximum(eV, 0.0)**2 + max(float(sigma_floor), 0.0)**2 + (f * np.maximum(np.abs(V_obs), 0.0))**2)
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
        xi, _ = xi_rar_plateau_numpy(
            Vbar_kms, R_kpc,
            a0_m_s2=float(a0),
            zeta_env=float(rar_params.get('zeta_env', 0.0)),
            rho=None,  # conservative: s_rho=0 for SPARC fits unless density grids exist
            rho_c=(None if rar_params.get('rho_c', None) in (None, 0.0) else float(rar_params['rho_c'])),
            gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=float(rar_params.get('wmin', 0.0)),
            D_max=rar_params.get('D_max', None),
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
        xi, _ = xi_rar_plateau_numpy(
            Vbar_kms, R_kpc,
            a0_m_s2=float(a0),
            zeta_env=float(rar_params.get('zeta_env', 0.0)),
            rho=None,
            rho_c=(None if rar_params.get('rho_c', None) in (None, 0.0) else float(rar_params['rho_c'])),
            gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=float(rar_params.get('wmin', 0.0)),
            D_max=rar_params.get('D_max', None),
        )
        V_model = np.sqrt(np.maximum(Vbar_kms, 0.0)**2 * xi)
        chi2_vals.append(chi2_velocity(V_obs, V_model, e_V_obs, sigma_floor=sigma_floor))
    return a0_vals, np.asarray(chi2_vals, dtype=float)


def _gauss_ln_prior_ln_mult(ln_m: float, sigma: float) -> float:
    """Log prior density for ln(m) ~ N(0, sigma^2), normalized."""
    s = max(float(sigma), 1e-6)
    return -0.5 * (ln_m / s)**2 - np.log(s * np.sqrt(2.0 * np.pi))


def scan_a0_grid_marginalized(
    R_kpc: np.ndarray,
    V_gas: np.ndarray,
    V_disk: np.ndarray,
    V_bulge: np.ndarray,
    V_obs: np.ndarray,
    e_V_obs: np.ndarray,
    rar_params: Dict[str, float],
    *,
    grid_log10: Tuple[float, float] = (-10.5, -9.3),
    ngrid: int = 60,
    sigma_floor: float = 0.0,
    obs_frac_sigma: float = 0.0,
    ml_sigma: float = 0.15,
    ml_grid: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a0_vals, chi2_marg) where chi2_marg ≡ -2 ln ∫ L(a0, η) π(η) dη across nuisance η.

    Nuisances handled:
    - Stellar mass-to-light multipliers for disk and bulge: m_d, m_b with ln m ~ N(0, ml_sigma^2).
    - Observational fractional noise inflation f on V_obs is absorbed into error model via obs_frac_sigma.
    """
    a0_vals = 10 ** np.linspace(grid_log10[0], grid_log10[1], ngrid)
    # Build ln m grids (±2σ) for disk and bulge
    span = 2.0 * float(ml_sigma)
    if int(ml_grid) < 2:
        ml_grid = 2
    ln_grid = np.linspace(-span, span, int(ml_grid))
    # Precompute multipliers and log-priors
    m_d = np.exp(ln_grid)
    m_b = np.exp(ln_grid)
    lp_d = np.array([_gauss_ln_prior_ln_mult(lnv, float(ml_sigma)) for lnv in ln_grid])
    lp_b = lp_d.copy()

    chi2_vals: List[float] = []
    # For stability, use log-sum-exp over nuisance combinations
    for a0 in a0_vals:
        ll_list: List[float] = []
        for i_d, md in enumerate(m_d):
            # scale disk
            Vd = float(np.sqrt(max(md, 0.0))) * np.asarray(V_disk, dtype=float)
            for i_b, mb in enumerate(m_b):
                Vb = float(np.sqrt(max(mb, 0.0))) * np.asarray(V_bulge, dtype=float)
                Vbar = np.sqrt(np.maximum(V_gas, 0.0)**2 + np.maximum(Vd, 0.0)**2 + np.maximum(Vb, 0.0)**2)
                xi, _ = xi_rar_plateau_numpy(
                    Vbar, R_kpc,
                    a0_m_s2=float(a0),
                    zeta_env=float(rar_params.get('zeta_env', 0.0)),
                    rho=None,
                    rho_c=(None if rar_params.get('rho_c', None) in (None, 0.0) else float(rar_params['rho_c'])),
                    gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
                    T0=rar_params.get('T0', None),
                    sigma_lnT=rar_params.get('sigma_lnT', None),
                    wmin=float(rar_params.get('wmin', 0.0)),
                    D_max=rar_params.get('D_max', None),
                )
                V_model = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi)
                # likelihood with fractional noise inflation
                chi2 = chi2_velocity_with_frac(V_obs, V_model, e_V_obs,
                                               sigma_floor=float(sigma_floor),
                                               obs_frac_sigma=float(obs_frac_sigma))
                ll = -0.5 * float(chi2) + float(lp_d[i_d]) + float(lp_b[i_b])
                ll_list.append(ll)
        # log-sum-exp over nuisances
        ll_arr = np.asarray(ll_list, dtype=float)
        m = np.nanmax(ll_arr)
        int_ll = m + np.log(np.sum(np.exp(ll_arr - m)))
        # Effective marginalized chi2: -2 ln ∫ exp(ll) dη
        chi2_marg = -2.0 * float(int_ll)
        chi2_vals.append(chi2_marg)

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
    write_csv_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Compute Solar-System dG/G under RAR-plateau and write Source-Data CSV.

    In the adopted metric-only subclass (Φ=Ψ), PPN γ=1 exactly; Cassini is satisfied.
    """
    rows: List[Dict[str, float]] = []
    a0 = float(rar_params.get('a0_m_s2', 1.2e-10))
    zeta = float(rar_params.get('zeta_env', 0.0))
    T0 = rar_params.get('T0', None)
    sig = rar_params.get('sigma_lnT', None)
    wmin = float(rar_params.get('wmin', 0.0))

    for AU in radii_AU:
        r_m = float(AU) * AU_M
        g_bar = G_SI * M_SUN / max(r_m**2, 1.0)  # Kepler g_N
        R_kpc = r_m / (3.085677581491367e19)
        V_ms = math.sqrt(g_bar * r_m)
        V_kms = V_ms / 1000.0

        # Worst-case: W=1, s_rho=1 -> a0_eff = a0*(1+zeta)
        xi_worst = 0.5 + math.sqrt(0.25 + a0 * (1.0 + zeta) / max(g_bar, 1e-30))

        # Gated branch
        if (T0 is not None) and (sig is not None):
            T_si = G_SI * M_SUN / max(r_m**3, 1.0)
            KPC_M = 3.085677581491367e19
            unit = (1000.0**2) / (KPC_M**2)
            T_unit = T_si / unit
            u = (math.log(max(T_unit, 1e-30)) - math.log(max(float(T0), 1e-30))) / max(float(sig), 1e-6)
            W = max(float(wmin) + (1.0 - float(wmin)) * math.exp(-0.5 * u * u), 0.0)
        else:
            W = 1.0
        a0_eff = a0 * (1.0 + zeta * 1.0 * W)
        xi_gated = 0.5 + math.sqrt(0.25 + a0_eff / max(g_bar, 1e-30))

        rows.append({
            'AU': float(AU),
            'g_bar_m_s2': float(g_bar),
            'xi_worst': float(xi_worst),
            'dGoverG_worst': float(xi_worst - 1.0),
            'xi_gated': float(xi_gated),
            'dGoverG_gated': float(xi_gated - 1.0),
            'gamma_minus_1': 0.0,
            'cassini_bound': CASSINI_BOUND_GAMMA,
        })

    if write_csv_path:
        try:
            import pandas as pd  # local import to avoid hard dependency if unused
            df = pd.DataFrame(rows)
            Path(write_csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(write_csv_path, index=False)
        except Exception:
            # Fallback: write simple CSV
            with open(write_csv_path, 'w', encoding='utf-8') as f:
                cols = list(rows[0].keys()) if rows else []
                f.write(','.join(cols) + '\n')
                for r in rows:
                    f.write(','.join(str(r[c]) for c in cols) + '\n')

    return rows


def solar_system_posterior_bands(
    run_dir: Path,
    radii_AU: List[float],
    n_samples: int,
    base_params: Dict[str, float],
) -> Optional[np.ndarray]:
    """Compute posterior bands for dG/G (gated) across radii using NPZ samples.
    Returns array with columns [AU, p16, p50, p84], or None if unavailable.
    """
    if n_samples <= 0:
        return None
    samples = load_posterior_samples_npz(run_dir, max_samples=n_samples)
    if not samples:
        return None
    AUs = np.asarray(radii_AU, float)
    vals = []  # shape: (nsamples, nAU)
    for sp in samples:
        rp = dict(base_params)
        rp.update(sp)
        rows = solar_system_table(rp, radii_AU=radii_AU)
        vals.append([r['dGoverG_gated'] for r in rows])
    V = np.asarray(vals, float)
    p16 = np.nanpercentile(V, 16, axis=0)
    p50 = np.nanpercentile(V, 50, axis=0)
    p84 = np.nanpercentile(V, 84, axis=0)
    return np.vstack([AUs, p16, p50, p84]).T


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


# ---------- RAR lensing via 'phantom mass' mapping -------------------------------------

def _sersic_deprojected_density_prugniel_simien(r_kpc: np.ndarray, M_star_Msun: float, Re_kpc: float, n: float = 4.0) -> np.ndarray:
    """Approximate 3D density for a Sersic profile using Prugniel & Simien (1997) form.
    ρ(r) = ρ0 (r/Re)^(-p_n) exp(-b_n (r/Re)^(1/n)), with p_n ≈ 1 - 0.6097/n + 0.05463/n^2.
    ρ0 is fixed by requiring 4π∫ ρ r^2 dr = M_star.
    Returns ρ in Msun/kpc^3 for r_kpc grid.
    """
    r = np.asarray(r_kpc, float)
    n = float(n)
    Re = max(float(Re_kpc), 1e-6)
    # Coefficients (Ciotti & Bertin/Prugniel & Simien approximations)
    p_n = 1.0 - 0.6097/n + 0.05463/(n*n)
    b_n = 2.0*n - 1.0/3.0 + 0.009876/n
    x = np.maximum(r / Re, 1e-9)
    # Unnormalized shape
    rho_shape = np.power(x, -p_n) * np.exp(-b_n * np.power(x, 1.0/n))
    # Normalize ρ0 so that total mass matches M_star over a wide radial extent
    # Integrate from ~0 to ~100 Re for practical convergence
    r_int = np.logspace(np.log10(Re/200.0), np.log10(100.0*Re), 1200)
    x_int = r_int / Re
    rho_shape_int = np.power(x_int, -p_n) * np.exp(-b_n * np.power(x_int, 1.0/n))
    M_shape = 4.0 * math.pi * np.trapezoid(rho_shape_int * (r_int**2), r_int)
    if M_shape <= 0:
        return np.full_like(r, np.nan)
    rho0 = float(M_star_Msun) / float(M_shape)
    return rho0 * rho_shape


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = np.zeros_like(x, dtype=float)
    if len(x) >= 2:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out


def _enclosed_mass_from_density(r_kpc: np.ndarray, rho_Msun_per_kpc3: np.ndarray) -> np.ndarray:
    r = np.asarray(r_kpc, float)
    rho = np.asarray(rho_Msun_per_kpc3, float)
    # Cumulative integral 4π ∫ ρ r^2 dr
    return 4.0 * math.pi * _cumtrapz(rho * (r**2), r)


def _project_surface_density_abel(r_kpc: np.ndarray, rho_Msun_per_kpc3: np.ndarray, R_eval_kpc: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project a spherical density via Abel integral: Σ(R) = 2∫_R^∞ ρ(r) r dr / sqrt(r^2 - R^2).
    Returns (R_grid, Sigma_Msun_per_kpc2).
    """
    r = np.asarray(r_kpc, float)
    rho = np.asarray(rho_Msun_per_kpc3, float)
    if R_eval_kpc is None:
        R_eval = r.copy()
    else:
        R_eval = np.asarray(R_eval_kpc, float)
    Sigma = np.zeros_like(R_eval)
    for i, R in enumerate(R_eval):
        mask = r > max(R, 1e-12)
        rr = r[mask]
        if len(rr) < 2:
            Sigma[i] = 0.0
            continue
        rh = rho[mask]
        kern = rr / np.sqrt(np.maximum(rr*rr - R*R, 1e-30))
        Sigma[i] = 2.0 * float(np.trapezoid(rh * kern, rr))
    return R_eval, Sigma

# ---------- NFW (DM baseline) -----------------------------------------------------------

def _H_of_z(H0: float, Om0: float, z: float) -> float:
    H0s = H0 / 3.085677581e19  # s^-1
    Ez = math.sqrt(Om0*(1.0+z)**3 + (1.0-Om0))
    return H0s * Ez

def _rho_crit_z(H0: float, Om0: float, z: float) -> float:
    G = G_SI
    Hz = _H_of_z(H0, Om0, z)  # s^-1
    return 3.0 * Hz*Hz / (8.0 * math.pi * G)  # kg/m^3

def _nfw_rho_kpc(M200_Msun: float, c: float, z_l: float, r_kpc: np.ndarray, H0: float = 70.0, Om0: float = 0.3) -> np.ndarray:
    """Return ρ_NFW(r) in Msun/kpc^3 for given M200, c at redshift z_l."""
    M200 = float(M200_Msun) * M_SUN  # kg
    c = max(float(c), 1.0)
    # ρ_crit at z_l
    rho_crit = _rho_crit_z(H0, Om0, z_l)  # kg/m^3
    # r200 where mean density is 200 ρ_crit
    r200_m = (3.0 * M200 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0/3.0)  # meters
    KPC_M = 3.085677581491367e19
    r200_kpc = r200_m / KPC_M
    rs_kpc = r200_kpc / c
    # NFW normalization ρ_s from M200 = 4π ρ_s r_s^3 [ln(1+c) - c/(1+c)]
    g_c = math.log(1.0 + c) - c / (1.0 + c)
    rho_s_kg_m3 = M200 / (4.0 * math.pi * (rs_kpc*KPC_M)**3 * g_c)
    # ρ(r) = ρ_s / [(r/r_s)(1+r/r_s)^2]
    r = np.asarray(r_kpc, float)
    x = np.maximum(r / max(rs_kpc, 1e-12), 1e-9)
    rho_si = rho_s_kg_m3 / (x * (1.0 + x)**2)  # kg/m^3
    # Convert to Msun/kpc^3
    rho_msun_kpc3 = rho_si * (KPC_M**3) / M_SUN
    return rho_msun_kpc3


def _nfw_vcirc_kms(R_kpc: np.ndarray, M200_Msun: float, c: float, z: float = 0.0, H0: float = 70.0, Om0: float = 0.3) -> np.ndarray:
    """Return circular speed contribution (km/s) from an NFW halo at radii R_kpc.
    Uses enclosed mass M(<r) = M200 * g(x)/g(c) with x = r/rs.
    """
    KPC_M = 3.085677581491367e19
    G = G_SI
    M200 = float(M200_Msun) * M_SUN
    c = max(float(c), 1.0)
    # r200 and r_s
    rho_crit = _rho_crit_z(H0, Om0, z)
    r200_m = (3.0 * M200 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0/3.0)
    r200_kpc = r200_m / KPC_M
    rs_kpc = r200_kpc / c
    # g(u)
    def g(u):
        return np.log(1.0 + u) - u / (1.0 + u)
    gc = g(c)
    gc = np.maximum(gc, 1e-12)
    r = np.asarray(R_kpc, float)
    x = np.maximum(r / max(rs_kpc, 1e-12), 1e-12)
    Menc = M200 * g(x) / gc  # kg
    # V^2 = G M(<r) / r
    r_m = r * KPC_M
    V2 = G * np.maximum(Menc, 0.0) / np.maximum(r_m, 1.0)
    V_kms = np.sqrt(np.maximum(V2, 0.0)) / 1000.0
    return V_kms


def _einstein_radius_from_surface_density(R_kpc: np.ndarray, Sigma_Msun_per_kpc2: np.ndarray, z_l: float, z_s: float, *, sigma_cr_scale: float = 1.0) -> Tuple[float, float]:
    """Solve for R_E (kpc) where mean surface density <Σ>(<R) = Σ_cr.
    Returns (R_E_kpc, theta_E_arcsec). If no solution found, returns (nan, nan).

    sigma_cr_scale: optional multiplicative factor on Σ_cr for controlled distance sensitivity tests.
    """
    # Critical surface density Σ_cr in Msun/kpc^2
    D_l, D_s, D_ls = _ang_dists(z_l, z_s)
    if not (D_l > 0 and D_s > 0 and D_ls > 0):
        return float('nan'), float('nan')
    KPC_M = 3.085677581491367e19
    c = 299792458.0
    Sigma_cr_SI = (c*c) / (4.0 * math.pi * G_SI) * (D_s / (D_l * D_ls))  # kg/m^2
    Sigma_cr = float(sigma_cr_scale) * (Sigma_cr_SI * (KPC_M*KPC_M) / M_SUN)  # Msun/kpc^2

    R = np.asarray(R_kpc, float)
    Sig = np.asarray(Sigma_Msun_per_kpc2, float)

    # 2D cumulative mass and mean surface density
    M2D = 2.0 * math.pi * _cumtrapz(Sig * R, R)
    mean_Sig = np.where(R > 0, M2D / (math.pi * R * R), np.nan)

    # --- NEW: enforce monotone decrease to kill tiny-bin wiggles ---
    # (max-accumulate on the reversed array -> monotone non-increasing)
    mean_Sig_mon = np.maximum.accumulate(mean_Sig[::-1])[::-1]
    if np.any(np.diff(mean_Sig) > 0):
        logging.warning("Einstein solver: non-monotone <Sigma>(R) detected; applying monotone envelope.")

    f = mean_Sig_mon - Sigma_cr

    # Guardrails
    if not np.all(np.isfinite(f)):
        return float('nan'), float('nan')
    if f[0] < 0:
        # Even the innermost mean Σ is below Σ_cr -> no strong-lensing solution on this grid
        return float('nan'), float('nan')
    if f[-1] > 0:
        # Still above at outer edge -> extend the grid outward and retry (caller may handle)
        return float('nan'), float('nan')

    # --- NEW: bracket using the LAST index where f >= 0 ---
    idx_nonneg = np.where(f >= 0)[0]
    if len(idx_nonneg) == 0:
        return float('nan'), float('nan')
    j = int(idx_nonneg[-1])
    # Ensure we have a bracket (j, j+1) with f[j] >= 0 and f[j+1] <= 0
    if j >= len(R) - 1:
        return float('nan'), float('nan')
    x0, x1 = R[j], R[j+1]
    y0, y1 = f[j], f[j+1]
    # Linear interpolation (f is smooth and monotone after the step above)
    if (y1 - y0) == 0:
        R_E = x0
    else:
        R_E = x0 - y0 * (x1 - x0) / (y1 - y0)

    # Convert to arcsec
    theta_rad = (R_E * KPC_M) / D_l
    theta_arcsec = theta_rad * 206265.0
    return float(R_E), float(theta_arcsec)


def _hernquist_deprojected_density(r_kpc: np.ndarray, M_star_Msun: float, Re_kpc: float) -> np.ndarray:
    """Hernquist 3D density (Msun/kpc^3) with total mass M_star and scale a from Re.
    a ≈ Re/1.8153 for projected half-light at Re.
    ρ(r) = (M/(2π)) a / [r (r + a)^3]
    """
    r = np.asarray(r_kpc, float)
    Re = max(float(Re_kpc), 1e-9)
    a = Re / 1.8153
    rr = np.maximum(r, 1e-12)
    rho = (M_star_Msun / (2.0 * math.pi)) * (a / (rr * (rr + a)**3))
    return rho


def _jaffe_deprojected_density(r_kpc: np.ndarray, M_star_Msun: float, Re_kpc: float) -> np.ndarray:
    """Jaffe 3D density (Msun/kpc^3) with total mass M_star and scale a from Re.
    a ≈ Re/0.7447 gives Re as projected half-light radius.
    ρ(r) = (M/(4π)) a / [r^2 (r + a)^2]
    """
    r = np.asarray(r_kpc, float)
    Re = max(float(Re_kpc), 1e-9)
    a = Re / 0.7447
    rr = np.maximum(r, 1e-12)
    rho = (M_star_Msun / (4.0 * math.pi)) * (a / (rr**2 * (rr + a)**2))
    return rho


def _baryon_density_profile(r_kpc: np.ndarray, M_star_Msun: float, Re_kpc: float, profile: str = 'sersic', n: float = 4.0) -> np.ndarray:
    p = (profile or 'sersic').lower()
    if p == 'hernquist':
        return _hernquist_deprojected_density(r_kpc, M_star_Msun, Re_kpc)
    if p == 'jaffe':
        return _jaffe_deprojected_density(r_kpc, M_star_Msun, Re_kpc)
    # default: Sersic (Prugniel–Simien deprojection)
    return _sersic_deprojected_density_prugniel_simien(r_kpc, M_star_Msun, Re_kpc, n=max(float(n), 0.5))


def _theta_E_from_profile_with_xi(log10M_star: float, Re_kpc: float, z_l: float, z_s: float,
                                  rar_params: Dict[str, float], *,
                                  n: float = 4.0, use_rar: bool = True,
                                  density_profile: str = 'sersic',
                                  sigma_cr_scale: float = 1.0) -> Tuple[float, float]:
    """Compute Einstein radius for a sphericalized stellar lens with chosen profile.
    If use_rar=True, builds an effective 'phantom' mass via xi(R) from xi_rar_plateau_numpy.
    Returns (R_E_kpc, theta_E_arcsec).
    """
    M_star = 10.0 ** float(log10M_star)
    Re = float(Re_kpc)
    # Radial grid covering deep core to large halo extent
    rmin = max(Re/200.0, 0.01)
    rmax = max(50.0*Re, Re + 100.0)
    r = np.logspace(np.log10(rmin), np.log10(rmax), 600)
    rho_b = _baryon_density_profile(r, M_star, Re, profile=density_profile, n=n)
    if not np.all(np.isfinite(rho_b)):
        return float('nan'), float('nan')
    M_b_encl = _enclosed_mass_from_density(r, rho_b)

    # Compute effective enclosed mass
    if use_rar:
        # Build Vbar_kms from M_b(<r): V^2 = G M(<r) / r
        KPC_M = 3.085677581491367e19
        V_ms = np.sqrt(np.maximum(G_SI * (M_b_encl * M_SUN) / (r * KPC_M), 0.0))
        V_kms = V_ms / 1000.0
        xi, _ = xi_rar_plateau_numpy(V_kms, r, a0_m_s2=float(rar_params.get('a0_m_s2', 1.2e-10)),
                                  zeta_env=float(rar_params.get('zeta_env', 0.0)),
                                  rho=None, rho_c=rar_params.get('rho_c', None),
                                  gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
                                  T0=rar_params.get('T0', None),
                                  sigma_lnT=rar_params.get('sigma_lnT', None),
                                  wmin=float(rar_params.get('wmin', 0.0)),
                                  D_max=rar_params.get('D_max', None))
        M_eff_encl = np.maximum(xi, 0.0) * np.maximum(M_b_encl, 0.0)
    else:
        M_eff_encl = np.maximum(M_b_encl, 0.0)

    # Guardrail: effective mass must not dip below baryonic mass
    if np.any(M_eff_encl < M_b_encl - 1e-9):
        logging.warning("RAR lensing guardrail: M_eff(<r) dipped below M_star(<r); check xi/profile.")

    # 3D density from dM/dr: ρ = (1/(4π r^2)) dM/dr
    dM_dr = np.gradient(M_eff_encl, r, edge_order=2)
    rho_eff = np.maximum(dM_dr, 0.0) / (4.0 * math.pi * np.maximum(r, 1e-9)**2)

    # Project and solve for Einstein radius
    R_eval, Sigma = _project_surface_density_abel(r, rho_eff)
    R_E_kpc, theta_arcsec = _einstein_radius_from_surface_density(R_eval, Sigma, z_l, z_s, sigma_cr_scale=float(sigma_cr_scale))
    return R_E_kpc, theta_arcsec


def run_lensing_rar_from_csv(out_dir: Path, images_dir: Path, csv_path: Path, rar_params: Dict[str, float],
                              alpha_lens_ph: float = 1.0,
                              zeta_env_lens: float = 0.0,
                              env_profile: str = 'constant',
                              density_profile: str = 'sersic',
                              sigma_cr_scale: float = 1.0,
                              metric_only: bool = False,
                              *,
                              nfw_enable: bool = False,
                              nfw_mass_ratio: float = 50.0,
                              nfw_c: float = 8.0) -> None:
    """Compute lensing for a CSV lens list using spherical baryons and RAR 'phantom mass' from xi.
    CSV columns (header required): lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec]

    Two paths:
    - metric_only=True: strictly metric predictions (GR stars vs RAR stars+phantom via xi). Writes
      lensing_metric_table.csv and per-lens radial profiles (Σ, ⟨Σ⟩, ΔΣ) without any α or ζ scaling.
    - metric_only=False (default): also computes an optional lensing-only scaled combination using α_lens_ph
      and ζ_env_lens for internal pilot studies; not for manuscript figures.

    Phantom-lensing scalars (internal pilot only):
    - alpha_lens_ph: scales Σ_ph in lensing only (default 1.0).
    - zeta_env_lens: additional amplitude on Σ_ph via (1 + zeta_env_lens f(R)).
    - env_profile: 'constant' (f=1) or 'tapered' (f = [1 + (R/Re)^2]^(-1/2)).

    Outputs:
    - results/.../lensing_metric_table.csv (metric-only path) OR results/.../lensing_rar_table.csv (pilot path)
    - images/.../lensing_rar_<lens_id>.png
    - results/.../lensing_metric_profiles/<lens_id>_profiles.csv (R, Σ, ⟨Σ⟩, ΔΣ, Σ_cr)
    """
    if not csv_path.exists():
        logging.warning(f"RAR lensing: CSV not found: {csv_path}")
        return
    images_dir.mkdir(parents=True, exist_ok=True)
    # Choose output table depending on path
    out_csv = out_dir / ('lensing_metric_table.csv' if metric_only else 'lensing_rar_table.csv')
    with out_csv.open('w', encoding='utf-8') as f:
        if metric_only:
            f.write('lens_id,z_l,z_s,Re_kpc,log10M_star,n_sersic,theta_E_obs_arcsec,theta_E_GR_arcsec,theta_E_RAR_arcsec,theta_E_NFW_arcsec,theta_E_SIS200_arcsec,theta_E_SIS250_arcsec\n')
        else:
            f.write('lens_id,z_l,z_s,Re_kpc,log10M_star,n_sersic,theta_E_obs_arcsec,theta_E_GR_arcsec,theta_E_RAR_arcsec,theta_E_RAR_phscaled_arcsec,alpha_lens_ph_used,zeta_env_lens_used,env_profile,alpha_req_at_thetaE_obs,theta_E_NFW_arcsec,theta_E_SIS200_arcsec,theta_E_SIS250_arcsec\n')

    # Read CSV quickly (no pandas dependency)
    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        cols = [h.strip() for h in header]
        # Validate required columns
        required = ["lens_id", "z_l", "z_s", "log10M_star", "Re_kpc"]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"Lens table missing columns: {missing}")
        for line in f:
            if not line.strip():
                continue
            fr = [s.strip() for s in line.strip().split(',')]
            if len(fr) < len(cols):
                # pad with empties
                fr = fr + [''] * (len(cols) - len(fr))
            data = {cols[i]: fr[i] for i in range(len(cols))}
            rows.append(data)
    # Validate z_s > z_l
    for r in rows:
        try:
            if float(r['z_s']) <= float(r['z_l']):
                raise ValueError(f"Invalid lens row (z_s<=z_l) for lens_id={r.get('lens_id','?')}")
        except Exception:
            raise ValueError(f"Invalid redshift values (z_l,z_s) in lens table for lens_id={r.get('lens_id','?')}")

    for row in rows:
        try:
            lens_id = row.get('lens_id', 'lens')
            z_l = float(row['z_l']); z_s = float(row['z_s'])
            log10M = float(row['log10M_star'])
            Re = float(row['Re_kpc'])
            n = float(row.get('n_sersic', '4') or 4.0)
            # Optional halo inputs for NFW baseline
            halo_logM = row.get('halo_M200_log10', '')
            halo_c = row.get('halo_c', '')
            profile_override = (row.get('profile', '') or '').strip().lower()
            th_obs = row.get('theta_E_obs_arcsec', '')
            th_obs_f = float(th_obs) if th_obs not in ('', 'nan', 'NaN') else float('nan')

            # GR and RAR (spherical Sersic)
            prof_use = profile_override if profile_override in ('sersic','hernquist','jaffe') else density_profile
            _, th_gr = _theta_E_from_profile_with_xi(log10M, Re, z_l, z_s, rar_params, n=n, use_rar=False,
                                                      density_profile=prof_use, sigma_cr_scale=float(sigma_cr_scale))
            _, th_rar = _theta_E_from_profile_with_xi(log10M, Re, z_l, z_s, rar_params, n=n, use_rar=True,
                                                      density_profile=prof_use, sigma_cr_scale=float(sigma_cr_scale))
            # SIS yardsticks
            th_sis_200 = theta_E_sis_arcsec(200.0, z_l, z_s)
            th_sis_250 = theta_E_sis_arcsec(250.0, z_l, z_s)

            # NFW yardstick (DM baseline)
            th_nfw = float('nan')
            if nfw_enable:
                # Determine M200 and c
                try:
                    if halo_logM not in ('', 'nan', 'NaN') and halo_c not in ('', 'nan', 'NaN'):
                        M200 = 10**float(halo_logM); cval = float(halo_c)
                    else:
                        # Yardstick from stellar mass
                        Mstar = 10**log10M
                        M200 = max(float(nfw_mass_ratio) * Mstar, 1e10)
                        cval = float(nfw_c)
                    # Build NFW profile and project
                    Rk = np.logspace(np.log10(max(Re/200.0, 0.01)), np.log10(max(50.0*Re, Re+100.0)), 600)
                    rho_nfw = _nfw_rho_kpc(M200, cval, z_l, Rk)
                    Rn, Sig_nfw = _project_surface_density_abel(Rk, rho_nfw)
                    _, th_nfw = _einstein_radius_from_surface_density(Rn, Sig_nfw, z_l, z_s, sigma_cr_scale=float(sigma_cr_scale))
                except Exception:
                    th_nfw = float('nan')

            # Build profiles and metric-consistent lensing quantities (no scaling)
            M_star = 10**log10M
            r = np.logspace(np.log10(max(Re/200.0, 0.01)), np.log10(max(50.0*Re, Re+100.0)), 600)
            rho_b = _baryon_density_profile(r, M_star, Re, profile=density_profile, n=n)
            M_b = _enclosed_mass_from_density(r, rho_b)
            # GR projection (stars only)
            Rg, Sig_g = _project_surface_density_abel(r, rho_b)
            # RAR projection (stars + phantom via xi)
            KPC_M = 3.085677581491367e19
            V_ms = np.sqrt(np.maximum(G_SI * (M_b * M_SUN) / (r * KPC_M), 0.0)); V_kms = V_ms/1000.0
            xi, _ = xi_rar_plateau_numpy(V_kms, r, a0_m_s2=float(rar_params.get('a0_m_s2', 1.2e-10)),
                                      zeta_env=float(rar_params.get('zeta_env', 0.0)),
                                      rho=None, rho_c=rar_params.get('rho_c', None),
                                      gamma_exp=float(rar_params.get('gamma_exp', 3.0)),
                                      T0=rar_params.get('T0', None), sigma_lnT=rar_params.get('sigma_lnT', None),
                                      wmin=float(rar_params.get('wmin', 0.0)),
                                      D_max=rar_params.get('D_max', None))
            M_eff = np.maximum(xi, 0.0) * np.maximum(M_b, 0.0)
            dM = np.gradient(M_eff, r, edge_order=2)
            rho_eff = np.maximum(dM, 0.0) / (4.0 * math.pi * np.maximum(r, 1e-9)**2)
            Rr, Sig_r = _project_surface_density_abel(r, rho_eff)
            # Σ_cr and mean profiles
            D_l, D_s, D_ls = _ang_dists(z_l, z_s)
            Sigma_cr_SI = (299792458.0**2) / (4.0 * math.pi * G_SI) * (D_s / (D_l * D_ls))
            Sigma_cr = float(sigma_cr_scale) * (Sigma_cr_SI * (KPC_M*KPC_M) / M_SUN)
            M2D_g = 2.0 * math.pi * _cumtrapz(Sig_g * Rg, Rg)
            M2D_r = 2.0 * math.pi * _cumtrapz(Sig_r * Rr, Rr)
            mean_g = np.where(Rg>0, M2D_g/(math.pi*Rg*Rg), np.nan)
            mean_r = np.where(Rr>0, M2D_r/(math.pi*Rr*Rr), np.nan)
            # Phantom components
            # Ensure grids match; if not, resample Sig_g to Rr grid for difference
            if not np.allclose(Rg, Rr):
                # Simple linear interpolation
                Sig_g_r = np.interp(Rr, Rg, Sig_g)
                M2D_g_r = 2.0 * math.pi * _cumtrapz(Sig_g_r * Rr, Rr)
                mean_g_r = np.where(Rr>0, M2D_g_r/(math.pi*Rr*Rr), np.nan)
                mean_star = mean_g_r
                Sigma_star = Sig_g_r
                Rgrid = Rr
            else:
                mean_star = mean_g
                Sigma_star = Sig_g
                Rgrid = Rg
            mean_tot = mean_r
            Sigma_tot = Sig_r
            mean_ph = np.maximum(mean_tot - mean_star, 0.0)
            Sigma_ph = np.maximum(Sigma_tot - Sigma_star, 0.0)
            # Environment profile f(R)
            if env_profile == 'tapered':
                fR = 1.0 / np.sqrt(1.0 + (np.maximum(Rgrid, 1e-9)/max(Re, 1e-9))**2)
            else:
                fR = np.ones_like(Rgrid)
            # Prepare optional pilot scaling (internal only)
            th_mod = float('nan')
            if not metric_only:
                # Allow signed zeta_env_lens but do not subtract phantom mass: clamp (1 + zeta f(R)) >= 0
                # See docs/lensing.md for notes on lensing-only scaling and environment profiles.
                scale_env = np.maximum(1.0 + float(zeta_env_lens) * fR, 0.0)
                Sigma_lens = Sigma_star + np.maximum(alpha_lens_ph, 0.0) * scale_env * Sigma_ph
                # Solve θE for phantom-weighted lensing
                _R_E_mod_kpc, th_mod = _einstein_radius_from_surface_density(Rgrid, Sigma_lens, z_l, z_s, sigma_cr_scale=float(sigma_cr_scale))

            # alpha_req at observed θE if provided (pilot only)
            alpha_req = ''
            if not metric_only and np.isfinite(th_obs_f):
                R_E_obs_kpc = th_obs_f/206265.0 * D_l / KPC_M
                # Interpolate mean_* and mean_ph at R_E_obs
                mg_obs = np.interp(R_E_obs_kpc, Rgrid, mean_star)
                mph_obs = np.interp(R_E_obs_kpc, Rgrid, mean_ph)
                if mph_obs > 1e-30:
                    alpha_req_val = max((Sigma_cr - mg_obs) / mph_obs, 0.0)
                    alpha_req = f"{alpha_req_val:.3f}"
                else:
                    alpha_req = 'nan'

            # Write row
            with out_csv.open('a', encoding='utf-8') as f:
                if metric_only:
                    f.write(f"{lens_id},{z_l},{z_s},{Re},{log10M},{n},{(th_obs if th_obs else '')},{(th_gr if np.isfinite(th_gr) else 'nan')},{(th_rar if np.isfinite(th_rar) else 'nan')},{(th_nfw if np.isfinite(th_nfw) else 'nan')},{th_sis_200:.3f},{th_sis_250:.3f}\n")
                else:
                    f.write(f"{lens_id},{z_l},{z_s},{Re},{log10M},{n},{(th_obs if th_obs else '')},{(th_gr if np.isfinite(th_gr) else 'nan')},{(th_rar if np.isfinite(th_rar) else 'nan')},{(th_mod if np.isfinite(th_mod) else 'nan')},{alpha_lens_ph:.3f},{zeta_env_lens:.3f},{env_profile},{alpha_req},{(th_nfw if np.isfinite(th_nfw) else 'nan')},{th_sis_200:.3f},{th_sis_250:.3f}\n")

            # Save radial profiles (metric) and plot
            # ΔΣ(R) = ⟨Σ⟩(R) - Σ(R) for total (RAR) and stars (GR)
            DeltaSigma_star = np.maximum(mean_star - Sigma_star, 0.0)
            DeltaSigma_tot = np.maximum(mean_tot - Sigma_tot, 0.0)
            prof_dir = out_dir / 'lensing_metric_profiles'
            prof_dir.mkdir(parents=True, exist_ok=True)
            prof_csv = prof_dir / f"{lens_id}_profiles.csv"
            with prof_csv.open('w', encoding='utf-8') as pf:
                pf.write('R_kpc,Sigma_star,Sigma_tot,mean_star,mean_tot,DeltaSigma_tot,Sigma_cr\n')
                for Ri, s_star, s_tot, m_star, m_tot, d_tot in zip(Rgrid, Sigma_star, Sigma_tot, mean_star, mean_tot, DeltaSigma_tot):
                    pf.write(f"{Ri:.6e},{s_star:.6e},{s_tot:.6e},{m_star:.6e},{m_tot:.6e},{d_tot:.6e},{Sigma_cr:.6e}\n")

            # Plot mean profiles and Σ_cr; mark GR, RAR, and scaled intersections
            plt.figure(figsize=(7.0, 4.6))
            plt.loglog(Rgrid, mean_star, 'b--', lw=2, label='⟨Σ⟩(R) stars (GR)')
            plt.loglog(Rgrid, mean_tot, 'r-', lw=2, alpha=0.7, label='⟨Σ⟩(R) RAR total')
            if not metric_only:
                # Build mean_lens corresponding to Sigma_lens
                M2D_l = 2.0 * math.pi * _cumtrapz(Sigma_lens * Rgrid, Rgrid)
                mean_l = np.where(Rgrid>0, M2D_l/(math.pi*Rgrid*Rgrid), np.nan)
                plt.loglog(Rgrid, mean_l, 'g-', lw=2, label='⟨Σ⟩(R) RAR lens (scaled)')
            plt.axhline(Sigma_cr, color='k', ls=':', label='Σ_cr')
            if np.isfinite(th_gr):
                R_E_gr = th_gr/206265.0 * D_l / KPC_M
                plt.axvline(R_E_gr, color='b', ls='--', alpha=0.5, label='θ_E, GR')
            if np.isfinite(th_rar):
                R_E_r = th_rar/206265.0 * D_l / KPC_M
                plt.axvline(R_E_r, color='r', ls='-', alpha=0.5, label='θ_E, RAR')
            if np.isfinite(th_mod):
                R_E_m = th_mod/206265.0 * D_l / KPC_M
                plt.axvline(R_E_m, color='g', ls='-', alpha=0.6, label='θ_E, scaled (pilot)')
            if np.isfinite(th_nfw):
                R_E_n = th_nfw/206265.0 * D_l / KPC_M
                plt.axvline(R_E_n, color='tab:green', ls='--', alpha=0.6, label='θ_E, NFW yardstick')
            # Observed θE marker if available
            if np.isfinite(th_obs_f):
                R_E_obs = th_obs_f/206265.0 * D_l / KPC_M
                plt.axvline(R_E_obs, color='k', ls='-.', alpha=0.7, label='θ_E, obs')
            plt.xlabel('R (kpc)')
            plt.ylabel('Mean surface density ⟨Σ⟩ (Msun/kpc^2)')
            if metric_only:
                ttl = f"{lens_id}: θ_E (GR={th_gr:.3f}″, RAR={th_rar:.3f}″) — metric"
            else:
                ttl = f"{lens_id}: θ_E (GR={th_gr:.3f}″, RAR={th_rar:.3f}″, scaled={th_mod:.3f}″)"
                if alpha_req:
                    ttl += f"\nα_req@θE_obs={alpha_req} (α={alpha_lens_ph:.2f}, ζ_env={zeta_env_lens:.2f}, {env_profile})"
            plt.title(ttl)
            plt.grid(alpha=0.3, which='both')
            plt.legend(frameon=False, loc='best')
            figp = images_dir / f"lensing_rar_{lens_id}.png"
            plt.tight_layout(); plt.savefig(figp, dpi=140); plt.close()
            if metric_only:
                logging.info(f"Metric lensing: {lens_id} θE_GR={th_gr:.3f} arcsec, θE_RAR={th_rar:.3f} arcsec → {figp}")
            else:
                logging.info(f"RAR lensing: {lens_id} θE_GR={th_gr:.3f} arcsec, θE_RAR={th_rar:.3f} arcsec, θE_scaled={th_mod:.3f} arcsec → {figp}")
        except Exception as e:
            logging.warning(f"RAR lensing: failed for row {row}: {e}")

    logging.info(f"Lensing table written: {out_csv}")

    # If observed θE are provided, compute quantitative metrics and a scatter plot
    try:
        # Read the table we just wrote
        obs = []
        pred_rar = []
        pred_gr = []
        rows2 = []
        with out_csv.open('r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            cm = {h:i for i,h in enumerate(header)}
            for line in f:
                if not line.strip():
                    continue
                parts = [s.strip() for s in line.strip().split(',')]
                def getf(col):
                    try:
                        return float(parts[cm[col]]) if parts[cm[col]] not in ('', 'nan', 'NaN') else float('nan')
                    except Exception:
                        return float('nan')
                th_obs = getf('theta_E_obs_arcsec') if 'theta_E_obs_arcsec' in cm else float('nan')
                th_gr = getf('theta_E_GR_arcsec') if 'theta_E_GR_arcsec' in cm else float('nan')
                th_rar = getf('theta_E_RAR_arcsec') if 'theta_E_RAR_arcsec' in cm else float('nan')
                if np.isfinite(th_obs) and np.isfinite(th_rar):
                    obs.append(th_obs); pred_rar.append(th_rar)
                if np.isfinite(th_obs) and np.isfinite(th_gr):
                    pred_gr.append(th_gr)
                rows2.append((parts[cm.get('lens_id',0)], th_obs, th_gr, th_rar))
        if len(obs) >= 2:
            obs = np.asarray(obs, float)
            pr = np.asarray(pred_rar, float)
            # Metrics
            resid = pr - obs
            rel = resid / np.where(obs!=0, obs, np.nan)
            metrics = {
                'N': int(np.isfinite(resid).sum()),
                'RMSE_abs_arcsec': float(np.sqrt(np.nanmean(resid**2))),
                'MAE_abs_arcsec': float(np.nanmean(np.abs(resid))),
                'Bias_abs_arcsec': float(np.nanmean(resid)),
                'RMSE_rel': float(np.sqrt(np.nanmean(rel**2))),
                'MAE_rel': float(np.nanmean(np.abs(rel))),
                'Bias_rel': float(np.nanmean(rel)),
            }
            (out_dir / 'lensing_thetaE_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            logging.info(f"θE metrics (RAR vs obs): {metrics}")
            # Scatter plot
            plt.figure(figsize=(5.2, 5.2))
            lim = [0.0, max(1.05*np.nanmax(obs), 1.05*np.nanmax(pr), 0.5)]
            plt.plot(lim, lim, 'k:', label='1:1')
            plt.scatter(obs, pr, c='tab:red', label='RAR metric', s=45, alpha=0.8)
            # If we have GR and NFW preds for same rows, scatter those too
            # Re-read to align arrays
            ogr = []
            gpr = []
            onfw = []
            pnfw = []
            with out_csv.open('r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
                cm = {h:i for i,h in enumerate(header)}
                for line in f:
                    parts = [s.strip() for s in line.strip().split(',')]
                    try:
                        tobs = float(parts[cm['theta_E_obs_arcsec']]) if parts[cm['theta_E_obs_arcsec']] not in ('','nan','NaN') else float('nan')
                        tgr = float(parts[cm['theta_E_GR_arcsec']]) if parts[cm['theta_E_GR_arcsec']] not in ('','nan','NaN') else float('nan')
                        tnfw = float(parts[cm['theta_E_NFW_arcsec']]) if 'theta_E_NFW_arcsec' in cm and parts[cm['theta_E_NFW_arcsec']] not in ('','nan','NaN') else float('nan')
                    except Exception:
                        tobs, tgr, tnfw = float('nan'), float('nan'), float('nan')
                    if np.isfinite(tobs) and np.isfinite(tgr):
                        ogr.append(tobs); gpr.append(tgr)
                    if np.isfinite(tobs) and np.isfinite(tnfw):
                        onfw.append(tobs); pnfw.append(tnfw)
            if len(ogr) >= 2:
                plt.scatter(np.asarray(ogr,float), np.asarray(gpr,float), c='tab:blue', label='GR (baryons)', s=45, alpha=0.8)
            if len(onfw) >= 2:
                plt.scatter(np.asarray(onfw,float), np.asarray(pnfw,float), c='tab:green', label='NFW yardstick', s=45, alpha=0.8)
            plt.xlabel('θ_E observed [arcsec]'); plt.ylabel('θ_E predicted [arcsec]')
            plt.title('Einstein radius: predicted vs observed')
            plt.grid(alpha=0.3); plt.legend(frameon=False)
            sp = images_dir / 'lensing_thetaE_pred_vs_obs.png'
            plt.tight_layout(); plt.savefig(sp, dpi=140); plt.close()
            logging.info(f"Scatter plot written: {sp}")
    except Exception as e:
        logging.warning(f"θE metrics step skipped: {e}")

    # Aggregate ΔΣ stack (metric-only inputs are always available regardless of flag)
    try:
        prof_dir = out_dir / 'lensing_metric_profiles'
        profiles = sorted([p for p in prof_dir.glob('*_profiles.csv') if p.is_file()])
        if len(profiles) >= 1:
            # Read all profiles and build a common R grid
            Rs = []
            Ds = []
            for pth in profiles:
                Rloc = []
                DSloc = []
                with pth.open('r', encoding='utf-8') as pf:
                    header = pf.readline().strip().split(',')
                    colmap = {h:i for i,h in enumerate(header)}
                    for line in pf:
                        parts = line.strip().split(',')
                        Rloc.append(float(parts[colmap['R_kpc']]))
                        DSloc.append(float(parts[colmap['DeltaSigma_tot']]))
                if len(Rloc) > 8:
                    Rs.append(np.asarray(Rloc, float))
                    Ds.append(np.asarray(DSloc, float))
            if len(Rs) >= 1:
                Rmin = max([r.min() for r in Rs])
                Rmax = min([r.max() for r in Rs])
                if Rmax > Rmin:
                    Rgrid = np.logspace(np.log10(Rmin), np.log10(Rmax), 80)
                    DSstack = []
                    for Rr, Dd in zip(Rs, Ds):
                        DSstack.append(np.interp(Rgrid, Rr, Dd))
                    DSarr = np.vstack(DSstack)
                    mean = np.nanmean(DSarr, axis=0)
                    p16 = np.nanpercentile(DSarr, 16, axis=0)
                    p84 = np.nanpercentile(DSarr, 84, axis=0)
                    stack_csv = out_dir / 'lensing_metric_stack.csv'
                    with stack_csv.open('w', encoding='utf-8') as sf:
                        sf.write('R_kpc,DeltaSigma_mean,DeltaSigma_p16,DeltaSigma_p84,N\n')
                        for Rv, m, l, u in zip(Rgrid, mean, p16, p84):
                            sf.write(f"{Rv:.6e},{m:.6e},{l:.6e},{u:.6e},{DSarr.shape[0]}\n")
                    # Also write Source-Data via helper
                    try:
                        write_source_data(
                            (out_dir / 'lensing_metric_stack_source.csv').as_posix(),
                            R_kpc=Rgrid, DeltaSigma_mean=mean, DeltaSigma_p16=p16, DeltaSigma_p84=p84
                        )
                    except Exception as _e:
                        logging.debug(f"Lensing stack Source-Data write failed: {_e}")
                    # Plot
                    plt.figure(figsize=(6.8, 4.4))
                    plt.loglog(Rgrid, mean, 'k-', lw=2, label='ΔΣ (RAR metric) mean')
                    plt.fill_between(Rgrid, p16, p84, color='gray', alpha=0.3, label='16–84%')
                    plt.xlabel('R (kpc)'); plt.ylabel('ΔΣ (Msun/kpc^2)')
                    plt.title('Stacked ΔΣ from metric predictions (per-lens average)')
                    plt.grid(alpha=0.3, which='both'); plt.legend(frameon=False)
                    figp = images_dir / 'lensing_metric_stack.png'
                    plt.tight_layout(); plt.savefig(figp, dpi=140); plt.close()
                    logging.info(f"Lensing stack written: {stack_csv} and {figp}")
    except Exception as e:
        logging.warning(f"Lensing stack step skipped: {e}")


# ---------- Milky Way Kz/Sigma full 3D helpers ----------------------------------------

@dataclass
class GridSpec:
    R_min: float = 0.5
    R_max: float = 20.0
    Z_max: float = 3.0
    nR: int = 256
    nZ: int = 256

def cylindrical_grid(gs: GridSpec):
    R = np.linspace(gs.R_min, gs.R_max, gs.nR)
    Z = np.linspace(-gs.Z_max, gs.Z_max, gs.nZ)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    return R, Z, dR, dZ

def grad_cylindrical_scalar(F: np.ndarray, dR: float, dZ: float):
    dF_dR = np.zeros_like(F)
    dF_dZ = np.zeros_like(F)
    dF_dR[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2.0 * dR)
    dF_dZ[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dZ)
    dF_dR[0, :] = (F[1, :] - F[0, :]) / max(dR, 1e-12)
    dF_dR[-1, :] = (F[-1, :] - F[-2, :]) / max(dR, 1e-12)
    dF_dZ[:, 0] = (F[:, 1] - F[:, 0]) / max(dZ, 1e-12)
    dF_dZ[:, -1] = (F[:, -1] - F[:, -2]) / max(dZ, 1e-12)
    return dF_dR, dF_dZ

def _laplacian_cylindrical(Phi: np.ndarray, R_vec: np.ndarray, dR: float, dZ: float) -> np.ndarray:
    """Axisymmetric cylindrical Laplacian: ∇²Φ = 1/R ∂/∂R (R ∂Φ/∂R) + ∂²Φ/∂Z².
    Phi is 2D over (R,Z), R_vec is 1D of R values.
    Units track those of Phi and coordinates.
    """
    dPhi_dR, _ = grad_cylindrical_scalar(Phi, dR, dZ)
    R2D = np.repeat(R_vec[:, None], Phi.shape[1], axis=1)
    # ∂/∂R (R ∂Φ/∂R)
    A = R2D * dPhi_dR
    dA_dR = np.zeros_like(A)
    dA_dR[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2.0 * dR)
    dA_dR[0, :] = (A[1, :] - A[0, :]) / max(dR, 1e-12)
    dA_dR[-1, :] = (A[-1, :] - A[-2, :]) / max(dR, 1e-12)
    term_R = np.where(R2D > 0, dA_dR / R2D, 0.0)
    # ∂²Φ/∂Z²
    term_Z = np.zeros_like(Phi)
    term_Z[:, 1:-1] = (Phi[:, 2:] - 2.0 * Phi[:, 1:-1] + Phi[:, :-2]) / max(dZ, 1e-12)**2
    term_Z[:, 0] = (Phi[:, 2] - 2.0 * Phi[:, 1] + Phi[:, 0]) / max(dZ, 1e-12)**2 if Phi.shape[1] > 2 else 0.0
    term_Z[:, -1] = (Phi[:, -1] - 2.0 * Phi[:, -2] + Phi[:, -3]) / max(dZ, 1e-12)**2 if Phi.shape[1] > 2 else 0.0
    return term_R + term_Z

# Component potentials in (km/s)^2 units
G_KPC = 4.300917270e-6  # kpc (km/s)^2 / Msun

def _mn_phi_kms2(R: np.ndarray, Z: np.ndarray, M: float, a: float, b: float) -> np.ndarray:
    B = np.sqrt(Z*Z + b*b)
    return -G_KPC * M / np.sqrt(R*R + (a + B)**2)

def _hern_phi_kms2(R: np.ndarray, Z: np.ndarray, M: float, a: float) -> np.ndarray:
    r = np.sqrt(R*R + Z*Z)
    return -G_KPC * M / (r + a)

def build_baryon_grids(R_vec: np.ndarray, Z_vec: np.ndarray, mw: dict):
    """Return (Phi_kms2, gR_SI, gZ_SI, rho_b_SI, Vbar_kms_grid) on (R,Z) grid."""
    R2D, Z2D = np.meshgrid(R_vec, Z_vec, indexing='ij')
    Phi = np.zeros_like(R2D)
    if mw.get('M_disk_thin', 0.0) > 0:
        Phi += _mn_phi_kms2(R2D, Z2D, mw['M_disk_thin'], mw['R_d_thin'], mw['h_z_thin'])
    if mw.get('M_disk_thick', 0.0) > 0:
        Phi += _mn_phi_kms2(R2D, Z2D, mw['M_disk_thick'], mw['R_d_thick'], mw['h_z_thick'])
    if mw.get('M_gas', 0.0) > 0:
        Phi += _mn_phi_kms2(R2D, Z2D, mw['M_gas'], mw['R_d_gas'], mw['h_z_gas'])
    if mw.get('M_bulge', 0.0) > 0:
        Phi += _hern_phi_kms2(R2D, Z2D, mw['M_bulge'], mw['a_bulge'])

    dR = R_vec[1] - R_vec[0]
    dZ = Z_vec[1] - Z_vec[0]
    dPhi_dR_kms2_per_kpc, dPhi_dZ_kms2_per_kpc = grad_cylindrical_scalar(Phi, dR, dZ)
    # Accelerations in SI
    KPC_M = 3.085677581491367e19
    gR_SI = -dPhi_dR_kms2_per_kpc * (1000.0**2) / KPC_M
    gZ_SI = -dPhi_dZ_kms2_per_kpc * (1000.0**2) / KPC_M

    # ρ_b from Poisson
    lap_kms2_per_kpc2 = _laplacian_cylindrical(Phi, R_vec, dR, dZ)
    lap_SI = lap_kms2_per_kpc2 * (1000.0**2) / (KPC_M**2)
    rho_b_SI = lap_SI / (4.0 * np.pi * G_SI)

    # Vbar from gR
    R_m = R2D * KPC_M
    V_ms = np.sqrt(np.maximum(np.abs(gR_SI) * np.maximum(R_m, 1.0), 0.0))
    Vbar_kms_grid = V_ms / 1000.0

    return Phi, gR_SI, gZ_SI, rho_b_SI, Vbar_kms_grid

def phantom_density_from_xi(R_kpc: np.ndarray, Z_kpc: np.ndarray, gR_bar_SI: np.ndarray, gZ_bar_SI: np.ndarray, xi: np.ndarray, rho_b_SI: np.ndarray) -> np.ndarray:
    dR = R_kpc[1] - R_kpc[0]
    dZ = Z_kpc[1] - Z_kpc[0]
    dxi_dR, dxi_dZ = grad_cylindrical_scalar(xi, dR, dZ)
    dot_term = dxi_dR * gR_bar_SI + dxi_dZ * gZ_bar_SI
    rho_ph = (xi - 1.0) * rho_b_SI - (1.0 / (4.0 * np.pi * G_SI)) * dot_term
    return rho_ph

def kz_sigma_from_grid(
    R: np.ndarray, Z: np.ndarray, rho_b_SI: np.ndarray, gR_bar_SI: np.ndarray, gZ_bar_SI: np.ndarray, Vbar_kms_grid: np.ndarray, *,
    a0_m_s2: float, gate_kwargs: dict, R0_kpc: float, z_list_kpc: Tuple[float, ...], D_max: Optional[float] = None
):
    """Compute (Kz, Sigma) at R0 for |z| in z_list using full phantom density."""
    nR, nZ = len(R), len(Z)
    xi = np.zeros((nR, nZ))
    a0_eff_grid = np.zeros_like(xi)
    for iR, Rval in enumerate(R):
        Vbar_line = Vbar_kms_grid[iR, :]
        R_line = np.full_like(Vbar_line, Rval)
        xi_line, meta = xi_rar_plateau_numpy(Vbar_line, R_line, a0_m_s2=a0_m_s2, D_max=D_max, **gate_kwargs)
        xi[iR, :] = xi_line
        a0_eff_grid[iR, :] = meta['a0_eff']
    rho_ph_SI = phantom_density_from_xi(R, Z, gR_bar_SI, gZ_bar_SI, xi, rho_b_SI)
    rho_tot_SI = rho_b_SI + rho_ph_SI

    # Integrate Σ and compute Kz ≈ 2πG Σ
    iR0 = int(np.argmin(np.abs(R - float(R0_kpc))))
    KPC_M = 3.085677581491367e19
    dZ_m = (Z[1] - Z[0]) * KPC_M
    mid = int(np.argmin(np.abs(Z - 0.0)))

    out = []
    for z_abs in z_list_kpc:
        jmax = int(np.argmin(np.abs(Z - float(z_abs))))
        lo = min(mid, jmax)
        hi = max(mid, jmax)
        sl = slice(lo, hi + 1)
        Sigma = np.trapz(rho_tot_SI[iR0, sl], dx=dZ_m)
        Kz = TWOPI_G * Sigma
        out.append({'z_kpc': float(z_abs), 'Sigma_SI': float(Sigma), 'Kz_m_s2': float(Kz)})

    try:
        import pandas as pd  # local import
        df = pd.DataFrame(out)
    except Exception:
        df = out  # caller can handle list of dicts fallback
    return df, {'xi_grid': xi, 'a0_eff_grid': a0_eff_grid}

# ---------- Main orchestrator ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Next steps orchestrator for rar_plateau runs')
    ap.add_argument('--preset', choices=['paper', 'pilot', 'custom'], default='custom',
                    help='Preset configuration. "paper" enforces metric-only lensing, posterior bands, Q≤2 SPARC cuts.')
    ap.add_argument('--run-dir', required=True, help='Path to the run folder (e.g., runs/rar_plateau_mw_full)')
    ap.add_argument('--sparc-dir', required=True, help='Path to SPARC rotmod folder (e.g., external_data/Rotmod_LTG)')
    ap.add_argument('--galaxies', nargs='*', default=None, help='Subset of SPARC galaxies to analyze (e.g., NGC3198 NGC2403)')
    ap.add_argument('--sample', default='gold', choices=['gold','all','q2plus'], help='SPARC sample selection if --galaxies not provided')
    ap.add_argument('--min-npts', type=int, default=12, help='Minimum RC points for inclusion')
    ap.add_argument('--min-rmax-kpc', type=float, default=8.0, help='Minimum R_max (kpc) for inclusion')
    ap.add_argument('--max-quality', type=int, default=2, help='Use Q <= max-quality if SPARC metadata is available')
    ap.add_argument('--sigma-floor', type=float, default=5.0, help='Velocity error floor (km/s) in chi2')
    # Nuisance marginalization (stellar M/L and observational fractional noise)
    ap.add_argument('--nuisance-enable', action='store_true', help='Enable nuisance marginalization for per-galaxy a0 grids (stellar M/L and fractional observational noise).')
    ap.add_argument('--nuisance-ml-sigma', type=float, default=0.15, help='Sigma for ln M/L (disk and bulge) Gaussian prior (dimensionless).')
    ap.add_argument('--nuisance-ml-grid', type=int, default=5, help='Grid points per axis for ln M/L integration (≥2).')
    ap.add_argument('--obs-frac-sigma', type=float, default=0.05, help='Fractional observational noise added in quadrature to e_V_obs (captures distance/inc/beam in aggregate).')
    ap.add_argument('--fit-global-a0', action='store_true', help='Also compute a global a0 across the sample')
    ap.add_argument('--fit-nfw', action='store_true', help='Also fit an NFW halo per galaxy (M200, c) and record chi2 and BIC.')
    ap.add_argument('--nfw-logm200-range', type=float, nargs=2, default=(11.2, 13.6), help='Range for log10 M200 [Msun] grid inclusive (e.g., 11.2 13.6).')
    ap.add_argument('--nfw-c-range', type=float, nargs=2, default=(5.0, 20.0), help='Range for concentration c grid inclusive (e.g., 5 20).')
    ap.add_argument('--nfw-grid', type=int, nargs=2, default=(18, 16), help='Grid sizes for (logM200, c).')
    ap.add_argument('--posterior-samples', type=int, default=0, help='Optional number of posterior samples to propagate (0=best-fit only)')
    ap.add_argument('--out-root', default=None, help='Output results root, default results/next_steps/<run_name>')
    ap.add_argument('--images-root', default=None, help='Images root, default images/next_steps/<run_name>')
    ap.add_argument('--hierarchical-a0', action='store_true', help='After per-galaxy scans, perform a two-stage hierarchical fit for a0 across the sample (lognormal prior).')
    ap.add_argument('--lensing-sample-csv', default=None, help='CSV with lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec] for RAR lensing prediction')
    ap.add_argument('--alpha-lens-ph', type=float, default=1.0, help='Lensing-only scale on Σ_ph (phantom)')
    ap.add_argument('--zeta-env-lens', type=float, default=0.0, help='Lensing-only environment amplitude on Σ_ph via (1+ζ_env f(R))')
    ap.add_argument('--env-profile', choices=['constant','tapered'], default='constant', help='Environment radial profile f(R) for lensing-only phantom scaling')
    ap.add_argument('--density-profile', choices=['sersic','hernquist','jaffe'], default='sersic', help='Stellar 3D profile for lensing deprojection (lensing-only)')
    ap.add_argument('--sigma-cr-scale', type=float, default=1.0, help='Multiplicative factor on Σ_cr (lensing-only, distances sensitivity)')
    ap.add_argument('--metric-lensing-only', action='store_true', default=True, help='If set, compute lensing strictly from the metric (Φ+Ψ via xi) and disable any α_lens_ph or ζ_env_lens scaling in outputs (paper build path).')
    ap.add_argument('--allow-pilot-lensing', action='store_true', default=False, help='Allow lensing-only phantom scaling (α_lens_ph, ζ_env_lens) for pilot studies. Default False for paper builds.')
    ap.add_argument('--rar-dmax', type=float, default=None, help='Optional finite plateau cap D_max for xi (nu). If unset, no cap is applied.')
    # NFW yardstick (DM baseline)
    ap.add_argument('--nfw-enable', action='store_true', help='If set, compute an NFW yardstick per lens (dark-matter baseline) and include θE_NFW and ΔΣ_NFW overlays.')
    ap.add_argument('--nfw-mass-ratio', type=float, default=50.0, help='If no halo mass provided in CSV, set M200 = ratio * M_star (yardstick).')
    ap.add_argument('--nfw-c', type=float, default=8.0, help='If no halo concentration provided in CSV, use this c for NFW yardstick.')
    ap.add_argument('--write-ppn-table', action='store_true', help='Write PPN table (γ, β, α1, α2) for Solar-System radii under adopted subclass (Φ=Ψ, c_T=1).')
    # Milky Way vertical force Kz and Sigma_1.1
    ap.add_argument('--mw-kz', action='store_true', help='Compute Milky Way Kz(R0,z) and Σ_1.1 from a simple MN+Hernquist baryon model; also provide a D-scaled (RAR) approximation.')
    ap.add_argument('--mw-R0-kpc', type=float, default=8.2, help='Solar radius R0 (kpc) for Kz/Σ_1.1 calculation.')
    ap.add_argument('--mw-zmax-kpc', type=float, default=3.0, help='Max height (kpc) for Kz/Σ_1.1 grid.')
    ap.add_argument('--mw-nz', type=int, default=181, help='Number of z samples for Kz/Σ_1.1 (including z=0).')
    ap.add_argument('--mw-kz-overlay-csv', type=str, default='', help='Optional CSV with comparison bands for Kz (columns: z_kpc,Kz_min,Kz_max) to overlay.')
    # Hierarchical Bayesian posterior (dynesty nested sampling) over (mu, sigma) for ln a0
    ap.add_argument('--hierarchical-a0-bayes', action='store_true', help='Run full Bayesian hierarchical posterior over (mu, sigma) using dynesty and precomputed per-galaxy chi2 grids.')
    ap.add_argument('--hierarchical-a0-live', type=int, default=400, help='Number of live points for dynesty nested sampling (Bayesian hierarchical step).')
    ap.add_argument('--hierarchical-a0-sigma-bounds', type=float, nargs=2, default=(0.05, 1.2), help='Bounds for sigma prior in ln a0 space: [lo, hi].')
    ap.add_argument('--hierarchical-a0-seed', type=int, default=0, help='Random seed for dynesty (0=auto).')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    # Apply preset configurations
    if args.preset == 'paper':
        # Paper defaults: ensure reproducible manuscript figures
        args.metric_lensing_only = True
        args.allow_pilot_lensing = False
        args.posterior_samples = max(args.posterior_samples, 400)
        args.sample = 'q2plus' if args.galaxies is None else args.sample
        args.min_npts = max(args.min_npts, 12)
        args.min_rmax_kpc = max(args.min_rmax_kpc, 8.0)
        args.max_quality = min(args.max_quality, 2)
        if args.rar_dmax in (None, 0.0):
            args.rar_dmax = 50.0
        logging.info("Applied paper preset: metric-only lensing, 400+ posterior samples, Q≤2 SPARC selection, D_max=%.1f" % float(args.rar_dmax))
elif args.preset == 'pilot':
        # Pilot mode: enable experimental features
        args.allow_pilot_lensing = True
        logging.info("Applied pilot preset: experimental lensing scalars enabled")

    # Paper preset: write PPN table if mapping present
    if args.preset == 'paper':
        args.write_ppn_table = True

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
    # Thread preset D_max into rar_params for consistent xi usage
    if getattr(args, 'rar_dmax', None) not in (None, 0.0):
        rar_params['D_max'] = float(args.rar_dmax)

    # Save a metadata snapshot for reproducibility
    import sys
    import datetime
    run_meta = {
        'run_dir': str(run_dir),
        'rar_plateau_params': rar_params,
        'flags': {
            'preset': args.preset,
            'metric_lensing_only': bool(args.metric_lensing_only),
            'allow_pilot_lensing': bool(args.allow_pilot_lensing),
            'posterior_samples': int(args.posterior_samples),
            'D_max': rar_params.get('D_max', None),
            'sample': args.sample,
            'min_npts': args.min_npts,
            'min_rmax_kpc': args.min_rmax_kpc,
            'max_quality': args.max_quality,
        },
        'env': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
        },
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    (results_root / 'run_metadata.json').write_text(json.dumps(run_meta, indent=2), encoding='utf-8')

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
        f.write('galaxy,a0_best_m_s2,a0_lo_m_s2,a0_hi_m_s2,chi2_rar,chi2_gr,dof,notes\n')
    # Model comparison CSV (will append rows as we go)
    mc_csv = results_root / 'model_comparison_bic.csv'
    with mc_csv.open('w', encoding='utf-8') as fmc:
        fmc.write('galaxy,npts,k_gr,k_rar,k_nfw,chi2_gr,chi2_rar,chi2_nfw,bic_gr,bic_rar,bic_nfw,delta_logZ_rar_vs_gr,delta_logZ_nfw_vs_gr,delta_logZ_rar_vs_nfw\n')
    grids_dir = results_root / 'sparc_a0_grids'
    grids_dir.mkdir(parents=True, exist_ok=True)

    galaxy_store: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    selection_info = []
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

        if args.nuisance_enable:
            # Use nuisance-marginalized grid over M/L and fractional observational noise
            a0_vals, chi2_vals = scan_a0_grid_marginalized(
                R, data['V_gas'], data['V_disk'], data['V_bulge'], Vobs, eV, rar_params,
                sigma_floor=float(args.sigma_floor), obs_frac_sigma=float(args.obs_frac_sigma),
                ml_sigma=float(args.nuisance_ml_sigma), ml_grid=int(args.nuisance_ml_grid))
        else:
            a0_vals, chi2_vals = scan_a0_grid(R, Vbar, Vobs, eV, rar_params, sigma_floor=float(args.sigma_floor))
        # Save per-galaxy grid for hierarchical stage (CSV with log10a0, chi2)
        grid_csv = grids_dir / f"{gid.replace(' ','_')}.csv"
        with grid_csv.open('w', encoding='utf-8') as gf:
            gf.write('log10_a0,chi2\n')
            for a0v, c2 in zip(a0_vals, chi2_vals):
                gf.write(f"{np.log10(a0v):.9f},{float(c2):.6f}\n")
        j = int(np.argmin(chi2_vals))
        a0_best = float(a0_vals[j])
        chi2_rar = float(chi2_vals[j])
        # 1σ profile-likelihood bounds (Δχ²=1)
        mask = chi2_vals <= (chi2_rar + 1.0)
        a0_lo = float(a0_vals[mask].min()) if np.any(mask) else float('nan')
        a0_hi = float(a0_vals[mask].max()) if np.any(mask) else float('nan')
        a0_best_fit, a0_err = fit_a0_err_from_grid(a0_vals, chi2_vals)
        xi_best, _ = xi_rar_plateau_numpy(
            Vbar, R,
            a0_m_s2=a0_best,
            zeta_env=rar_params.get('zeta_env', 0.0),
            rho=None,
            rho_c=rar_params.get('rho_c', None),
            gamma_exp=rar_params.get('gamma_exp', 3.0),
            T0=rar_params.get('T0', None),
            sigma_lnT=rar_params.get('sigma_lnT', None),
            wmin=rar_params.get('wmin', 0.0),
            D_max=rar_params.get('D_max', None)
        )
        V_model = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi_best)
        chi2_gr = chi2_velocity(Vobs, np.maximum(Vbar, 0.0), eV, sigma_floor=float(args.sigma_floor))
        dof = max(len(R) - 1, 1)
        notes = f"a0_pm~{a0_err:.2e}" if np.isfinite(a0_err) else ''

        # Save CSV row
        with csv_path.open('a', encoding='utf-8') as f:
            f.write(f"{gid},{a0_best_fit:.6e},{a0_lo:.6e},{a0_hi:.6e},{chi2_rar:.3f},{chi2_gr:.3f},{dof},{notes}\n")

        # Optional NFW fit (grid) for model comparison
        chi2_nfw = float('nan'); M200_best = float('nan'); c_best = float('nan')
        if args.fit_nfw:
            try:
                logM_lo, logM_hi = float(args.nfw_logm200_range[0]), float(args.nfw_logm200_range[1])
                c_lo, c_hi = float(args.nfw_c_range[0]), float(args.nfw_c_range[1])
                nM, nC = int(args.nfw_grid[0]), int(args.nfw_grid[1])
                logM_grid = np.linspace(logM_lo, logM_hi, nM)
                c_grid = np.linspace(c_lo, c_hi, nC)
                best = (float('inf'), float('nan'), float('nan'))
                for logM in logM_grid:
                    M200 = 10**logM
                    V_nfw = _nfw_vcirc_kms(R, M200, c_grid[0])  # init vector for shape
                    for cval in c_grid:
                        Vn = _nfw_vcirc_kms(R, M200, cval)
                        V_model_nfw = np.sqrt(np.maximum(Vbar, 0.0)**2 + np.maximum(Vn, 0.0)**2)
                        c2 = chi2_velocity(Vobs, V_model_nfw, eV, sigma_floor=float(args.sigma_floor))
                        if c2 < best[0]:
                            best = (c2, M200, cval)
                chi2_nfw, M200_best, c_best = best
            except Exception as e:
                logging.warning(f"NFW fit failed for {gid}: {e}")

        # Compute BICs and delta log Z (BIC ≈ -2 ln Z + k ln n; ΔlogZ ≈ -0.5 ΔBIC)
        npts = len(R)
        k_gr = 0
        k_rar = 1  # a0 per galaxy
        k_nfw = 2  # M200, c
        bic_gr = chi2_gr + k_gr * math.log(max(npts,1))
        bic_rar = chi2_rar + k_rar * math.log(max(npts,1))
        bic_nfw = (chi2_nfw + k_nfw * math.log(max(npts,1))) if np.isfinite(chi2_nfw) else float('nan')
        dlogZ_rar_vs_gr = -0.5 * (bic_rar - bic_gr)
        dlogZ_nfw_vs_gr = -0.5 * (bic_nfw - bic_gr) if np.isfinite(bic_nfw) else float('nan')
        dlogZ_rar_vs_nfw = -0.5 * (bic_rar - bic_nfw) if np.isfinite(bic_nfw) else float('nan')
        with mc_csv.open('a', encoding='utf-8') as fmc:
            fmc.write(f"{gid},{npts},{k_gr},{k_rar},{k_nfw},{chi2_gr:.3f},{chi2_rar:.3f},{(chi2_nfw if np.isfinite(chi2_nfw) else float('nan'))},{bic_gr:.3f},{bic_rar:.3f},{(bic_nfw if np.isfinite(bic_nfw) else float('nan'))},{dlogZ_rar_vs_gr:.3f},{(dlogZ_nfw_vs_gr if np.isfinite(dlogZ_nfw_vs_gr) else float('nan'))},{(dlogZ_rar_vs_nfw if np.isfinite(dlogZ_rar_vs_nfw) else float('nan'))}\n")

        # Stash for possible global a0
        galaxy_store.append((gid, R, Vbar, Vobs, eV, a0_best))
        # Record selection metrics for disclosure
        selection_info.append({
            'galaxy': gid,
            'npts': int(len(R)),
            'rmax_kpc': float(np.nanmax(R)) if len(R) else float('nan')
        })

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
        # Source-Data for this overlay
        try:
            write_source_data(
                (results_root / f"sparc_overlay_{gid.replace(' ','_')}_source.csv").as_posix(),
                R_kpc=R,
                V_obs_kms=Vobs,
                e_V_obs_kms=eV,
                Vbar_kms=Vbar,
                V_model_kms=V_model,
            )
        except Exception as _e:
            logging.debug(f"Source-Data write failed for {gid}: {_e}")

    logging.info(f"SPARC summary: {csv_path}")
    # Write SPARC selection disclosure
    try:
        (results_root / 'sparc_selection.json').write_text(json.dumps(selection_info, indent=2), encoding='utf-8')
        logging.info(f"SPARC selection info: {results_root/'sparc_selection.json'}")
    except Exception as e:
        logging.debug(f"SPARC selection disclosure skipped: {e}")

    # Model comparison distributions (histograms)
    try:
        import pandas as pd
        mdf = pd.read_csv(results_root / 'model_comparison_bic.csv')
        # Drop NaNs in the relevant columns
        cols = ['delta_logZ_rar_vs_gr','delta_logZ_nfw_vs_gr','delta_logZ_rar_vs_nfw']
        for c in cols:
            if c not in mdf.columns:
                raise ValueError('model_comparison_bic.csv missing required columns')
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
        bins = 20
        mdf['delta_logZ_rar_vs_gr'].replace([np.inf, -np.inf], np.nan, inplace=True)
        mdf['delta_logZ_nfw_vs_gr'].replace([np.inf, -np.inf], np.nan, inplace=True)
        mdf['delta_logZ_rar_vs_nfw'].replace([np.inf, -np.inf], np.nan, inplace=True)
        ax.hist(mdf['delta_logZ_rar_vs_gr'].dropna(), bins=bins, alpha=0.6, label='ΔlogZ (DGG−GR)', color='tab:red')
        ax.hist(mdf['delta_logZ_nfw_vs_gr'].dropna(), bins=bins, alpha=0.6, label='ΔlogZ (NFW−GR)', color='tab:blue')
        ax.hist(mdf['delta_logZ_rar_vs_nfw'].dropna(), bins=bins, alpha=0.6, label='ΔlogZ (DGG−NFW)', color='tab:green')
        ax.set_xlabel('Δ log Z'); ax.set_ylabel('Count')
        ax.set_title('Model comparison distributions (BIC approximation)')
        ax.legend(frameon=False)
        outp = images_root / 'model_comparison' / 'delta_logZ_hist.png'
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(outp, dpi=140); plt.close(fig)
        logging.info(f"Model comparison histogram written: {outp}")
    except Exception as e:
        logging.warning(f"Model comparison histogram skipped: {e}")

    # 3) Solar-System ΔG/G and optional PPN table
    solar_csv = results_root / 'solar_system_table.csv'
    solar_rows = solar_system_table(rar_params, write_csv_path=str(solar_csv))
    logging.info(f"Solar table: {solar_csv}")

    # Plot
    AUs = [r['AU'] for r in solar_rows]
    worst = [r['dGoverG_worst'] for r in solar_rows]
    gated = [r['dGoverG_gated'] for r in solar_rows]
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(AUs, worst, 'o-', label='worst-case (RAR-plateau, W=s_ρ=1)')
    plt.semilogy(AUs, gated, 's--', label='gated rar_plateau')
    # Optional posterior bands
    bands = solar_system_posterior_bands(run_dir, AUs, int(args.posterior_samples), rar_params)
    if bands is not None:
        bA, p16, p50, p84 = bands[:,0], bands[:,1], bands[:,2], bands[:,3]
        plt.fill_between(bA, np.maximum(p16, 1e-16), np.maximum(p84, 1e-16), color='tab:red', alpha=0.20, label='posterior 16–84% (gated)')
        plt.semilogy(bA, np.maximum(p50, 1e-16), 'r-', lw=1.8, alpha=0.8, label='posterior median (gated)')
        bands_csv = results_root / 'solar_system_posterior_bands.csv'
        with bands_csv.open('w', encoding='utf-8') as bf:
            bf.write('AU,p16,p50,p84\n')
            for a, l, m, u in zip(bA, p16, p50, p84):
                bf.write(f'{a:.2f},{l:.6e},{m:.6e},{u:.6e}\n')
        logging.info(f"Solar posterior bands: {bands_csv}")
    # Cassini bound at Saturn (~9.6 AU): |γ-1| < 2.3e-5; annotate at 10 AU for visibility
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

    # Optional PPN export for Solar-System radii (γ, β, α1, α2 with explicit theory assumption)
    if args.write_ppn_table:
        if check_c_T_guardrail is not None and not check_c_T_guardrail(rar_params):
            logging.error('PPN export aborted: c_T guardrail failed (c_T != 1).')
        elif evaluate_ppn is None:
            logging.warning('PPN export unavailable: theory.relativistic not importable.')
        else:
            radii_AU = [r['AU'] for r in solar_rows]
            ppn_list = evaluate_ppn(rar_params, radii_AU)
            ppn_csv = results_root / 'ppn_table.csv'
            with ppn_csv.open('w', encoding='utf-8') as f:
                f.write('AU,gamma,beta,alpha1,alpha2,theory_assumption,note\n')
                for AU, res in zip(radii_AU, ppn_list):
                    f.write(f"{AU:.1f},{res.gamma:.6f},{res.beta:.6f},{res.alpha1:.6f},{res.alpha2:.6f}," \
                            f"{res.theory_assumption.replace(',', ';')},{res.note.replace(',', ';')}\n")
            logging.info(f"PPN table: {ppn_csv}")

    # 4) Lensing pilot
    run_lensing_pilot(results_root, rar_params)

    # 4b) Metric lensing from CSV sample (stars + phantom via xi); optional pilot scaling suppressed if --metric-lensing-only
    if args.lensing_sample_csv:
        try:
            # Enforce metric-only unless explicitly allowed for pilot scaling
            enforce_metric = bool(args.metric_lensing_only) and (not bool(args.allow_pilot_lensing))
            alpha_lens_ph = 1.0 if enforce_metric else float(args.alpha_lens_ph)
            zeta_env_lens = 0.0 if enforce_metric else float(args.zeta_env_lens)
            run_lensing_rar_from_csv(
                results_root, images_root, Path(args.lensing_sample_csv), rar_params,
                alpha_lens_ph=alpha_lens_ph,
                zeta_env_lens=zeta_env_lens,
                env_profile=str(args.env_profile),
                density_profile=str(args.density_profile),
                sigma_cr_scale=float(args.sigma_cr_scale),
                metric_only=bool(args.metric_lensing_only),
                nfw_enable=bool(args.nfw_enable),
                nfw_mass_ratio=float(args.nfw_mass_ratio),
                nfw_c=float(args.nfw_c),
            )
        except Exception as e:
            logging.warning(f"Lensing step skipped: {e}")

    # 5) BTFR: baryonic mass (M_star + 1.33 M_HI) and observed V_flat with flatness checks
    btfr_csv = results_root / 'btfr_summary.csv'
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

    # Use the full sparc_io loader to access Sigma_* and M_HI metadata
    full_loader = None
    try:
        from utils.Utilities.sparc_io import load_single_sparc_galaxy as _ls_full
        full_loader = _ls_full
    except Exception:
        repo_root = Path.cwd()
        candidate = repo_root / 'utils' / 'Utilities' / 'sparc_io.py'
        if candidate.exists():
            try:
                mod = _import_by_path('sparc_io_runtime_btfr_full', candidate)
                full_loader = getattr(mod, 'load_single_sparc_galaxy', None)
            except Exception as e:
                logging.warning(f"BTFR: could not import SPARC loader by path ({e}); skipping BTFR metadata join.")
                full_loader = None

    used = []
    x = []  # log10 V_flat
    y = []  # log10 M_b
    for gid in (args.galaxies or sample):
        try:
            d = full_loader(gid, sparc_dir=str(sparc_dir)) if full_loader else None
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
                    integrand = Sig_star * R_pc  # Σ * R
            Mstar = 2.0 * math.pi * float(np.trapezoid(integrand, R_pc))
            Mb = np.nansum([Mstar, Mgas])
            logV = (np.log10(Vflat) if np.isfinite(Vflat) and Vflat > 0 else float('nan'))
            logM = (np.log10(Mb) if np.isfinite(Mb) and Mb > 0 else float('nan'))
            with btfr_csv.open('a', encoding='utf-8') as f:
                f.write(f"{gid},{(logM if np.isfinite(logM) else 'nan')},{(logV if np.isfinite(logV) else 'nan')},obs_outer,{note}\n")
            if np.isfinite(logV) and np.isfinite(logM):
                x.append(logV); y.append(logM); used.append(gid)
        except Exception as e:
            logging.debug(f"BTFR: {gid} error: {e}")
            continue

    n_btfr = len(x)
    logging.info(f"BTFR: usable galaxies = {n_btfr}")
    # Source-Data for BTFR points
    try:
        write_source_data(
            (results_root / 'btfr_source.csv').as_posix(),
            galaxy=np.array(used, dtype=object),
            log10Vflat=np.array(x, dtype=float),
            log10Mb=np.array(y, dtype=float),
        )
    except Exception as _e:
        logging.debug(f"BTFR Source-Data write failed: {_e}")

    # Fit log M_b = a + b log V; report slope, R^2, RMS scatter; test curvature
    btfr_png = images_root / 'btfr_baryonic.png'
    if n_btfr >= 10:
        xv = np.asarray(x, float); yv = np.asarray(y, float)
        p, cov = np.polyfit(xv, yv, 1, cov=True)
        a_lin, b_lin = p[1], p[0]
        b_err = float(np.sqrt(cov[0,0])) if cov.shape == (2,2) else float('nan')
        # Bootstrap slope uncertainty
        try:
            rng = np.random.default_rng(42)
            B = 1000
            boots = []
            n = len(xv)
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                xb = xv[idx]; yb = yv[idx]
                pb = np.polyfit(xb, yb, 1)
                boots.append(pb[0])
            boots = np.asarray(boots)
            p16, p50, p84 = np.percentile(boots, [16, 50, 84])
        except Exception:
            p16 = p50 = p84 = float('nan')
        yhat = a_lin + b_lin * xv
        resid = yv - yhat
        rss = float(np.sum(resid**2))
        tss = float(np.sum((yv - np.mean(yv))**2))
        r2 = 1.0 - rss / tss if tss > 0 else float('nan')
        rms = float(np.sqrt(np.mean(resid**2)))
        # Curvature test via quadratic fit
        pq = np.polyfit(xv, yv, 2)
        a2, b2, c2 = pq[0], pq[1], pq[2]
        yhat2 = a2 * xv**2 + b2 * xv + c2
        rss2 = float(np.sum((yv - yhat2)**2))
        n = len(xv)
        bic_lin = n * np.log(rss / n + 1e-30) + 2 * np.log(n)
        bic_quad = n * np.log(rss2 / n + 1e-30) + 3 * np.log(n)
        delta_bic = bic_quad - bic_lin  # > 0 favors linear
        # Save a small JSON summary
        btfr_summary = {
            'N': n_btfr,
            'slope': b_lin,
            'slope_err': b_err,
            'intercept': a_lin,
            'R2': r2,
            'rms_dex': rms,
            'quad_c': a2,
'delta_BIC_quad_minus_lin': delta_bic,
            'slope_bootstrap': {'p16': float(p16), 'p50': float(p50), 'p84': float(p84)},
        }
        (results_root / 'btfr_fit_summary.json').write_text(json.dumps(btfr_summary, indent=2), encoding='utf-8')
        logging.info(f"BTFR fit: slope={b_lin:.3f}±{(b_err if np.isfinite(b_err) else float('nan')):.3f}, R2={r2:.3f}, rms={rms:.3f} dex, ΔBIC={delta_bic:.2f}")

        # Figure
        plt.figure(figsize=(7.2, 5.0))
        plt.scatter(xv, yv, s=16, alpha=0.7, label=f'BTFR sample (N={n_btfr})')
        xs = np.linspace(min(xv)-0.05, max(xv)+0.05, 200)
        y_fit = a_lin + b_lin*xs
        plt.plot(xs, y_fit, 'r-', lw=2, label=f"fit: slope={b_lin:.2f}±{(b_err if np.isfinite(b_err) else float('nan')):.2f}")
        # Bootstrap CI shading if available
        try:
            y_lo = a_lin + p16*xs
            y_hi = a_lin + p84*xs
            import numpy as _np
            m = _np.isfinite(y_lo) & _np.isfinite(y_hi)
            if _np.any(m):
                plt.fill_between(xs[m], y_lo[m], y_hi[m], color='red', alpha=0.12, label='bootstrap 16–84%')
        except Exception:
            pass
        plt.xlabel('log10 V_flat [km/s]')
        plt.ylabel('log10 M_b [M_sun]')
        plt.title('Baryonic Tully–Fisher Relation (observed V_flat)')
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(btfr_png, dpi=150); plt.close()
        logging.info(f"Saved {btfr_png}")
    else:
        logging.warning("BTFR: insufficient usable galaxies (<10); figure and fit not generated.")

    # Optional: 5b) Global a0 across sample
    if args.fit_global_a0 and len(galaxy_store) > 0:
        a0_grid = 10 ** np.linspace(-10.5, -9.3, 80)
        totals: List[float] = []
        for a0 in a0_grid:
            tot = 0.0
            for (_gid, R, Vbar, Vobs, eV, _a0best) in galaxy_store:
                xi, _ = xi_rar_plateau_numpy(Vbar, R, a0_m_s2=float(a0), D_max=rar_params.get('D_max', None))
                Vmod = np.sqrt(np.maximum(Vbar, 0.0)**2 * xi)
                tot += chi2_velocity(Vobs, Vmod, eV, sigma_floor=float(args.sigma_floor))
            totals.append(tot)
        a0_grid = np.asarray(a0_grid, float)
        totals = np.asarray(totals, float)
        a0_global, a0_sigma = fit_a0_err_from_grid(a0_grid, totals)
        (results_root/'global_a0.json').write_text(json.dumps({'a0_m_s2': a0_global, 'sigma': a0_sigma, 'n_gal': len(galaxy_store)}, indent=2), encoding='utf-8')
        logging.info(f"Global a0 ~ {a0_global:.3e} ± {a0_sigma if np.isfinite(a0_sigma) else float('nan'):.1e} m/s^2 over {len(galaxy_store)} galaxies")

    # Optional: two-stage hierarchical a0 across sample (MLE grid search)
    if args.hierarchical_a0 and len(list((results_root/'sparc_a0_grids').glob('*.csv'))) > 0:
        try:
            grids = sorted((results_root/'sparc_a0_grids').glob('*.csv'))
            # Build a common ln a0 grid from min/max across galaxies
            lnmins = []
            lnmaxs = []
            per = []
            for g in grids:
                xs = []
                with g.open('r', encoding='utf-8') as f:
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split(',')
                        xs.append(float(parts[0]))
                if len(xs) > 3:
                    lnmins.append(min(xs)); lnmaxs.append(max(xs))
            if lnmins and lnmaxs:
                lnlo = max(lnmins); lnhi = min(lnmaxs)
                if lnhi > lnlo:
                    ln_a0 = np.linspace(lnlo, lnhi, 160)
                    # For each galaxy, read chi2 and compute log-likelihood over ln_a0 via interp
                    ll_list = []
                    for g in grids:
                        xa = []
                        c2 = []
                        with g.open('r', encoding='utf-8') as f:
                            f.readline()
                            for line in f:
                                a, v = line.strip().split(',')
                                xa.append(float(a)); c2.append(float(v))
                        xa = np.asarray(xa, float); c2 = np.asarray(c2, float)
                        # Interpolate chi2 onto ln_a0; guard large values
                        c2i = np.interp(ln_a0, xa, c2, left=np.nan, right=np.nan)
                        mask = np.isfinite(c2i)
                        # log likelihood up to additive const: -0.5 * chi2
                        lli = -0.5 * c2i[mask]
                        # normalize by subtracting max to avoid underflow later
                        lli = lli - np.nanmax(lli)
                        # Store as array matching ln_a0 size (nan where not covered)
                        ll = np.full_like(ln_a0, -np.inf)
                        ll[mask] = lli
                        ll_list.append(ll)
                    # Hyperparameter grid for µ, σ
                    mu_grid = np.linspace(lnlo, lnhi, 80)
                    sigma_grid = np.linspace(0.05, 1.2, 60)
                    log_like = np.full((len(mu_grid), len(sigma_grid)), -np.inf)
                    dln = (ln_a0[1]-ln_a0[0]) if len(ln_a0)>1 else 1.0
                    for i, mu in enumerate(mu_grid):
                        for j, sg in enumerate(sigma_grid):
                            # prior over ln a0: Normal(mu, sg)
                            prior = np.exp(-0.5*((ln_a0-mu)/sg)**2) / (sg*np.sqrt(2*np.pi))
                            # For each galaxy, integral over a0: sum exp(ll_i) * prior * dln
                            tot = 0.0
                            for lli in ll_list:
                                y = lli + np.log(np.maximum(prior, 1e-300))
                                m = np.nanmax(y)
                                # log integral using log-sum-exp for stability
                                s = np.sum(np.exp(y - m)) * dln
                                tot += (m + np.log(max(s, 1e-300)))
                            log_like[i, j] = tot
                    # Find max and report
                    idx = np.unravel_index(int(np.nanargmax(log_like)), log_like.shape)
                    mu_hat = float(mu_grid[idx[0]]); sigma_hat = float(sigma_grid[idx[1]])
                    hier_json = results_root / 'hierarchical_a0_summary.json'
                    hier_json.write_text(json.dumps({'mu': mu_hat, 'sigma': sigma_hat, 'n_gal': len(ll_list)}, indent=2), encoding='utf-8')
                    # Heatmap plot
                    plt.figure(figsize=(6.4, 5.0))
                    plt.imshow(log_like.T - np.nanmax(log_like), origin='lower', aspect='auto',
                               extent=[mu_grid[0], mu_grid[-1], sigma_grid[0], sigma_grid[-1]], cmap='viridis')
                    plt.colorbar(label='log ℒ(µ,σ) − max')
                    plt.scatter([mu_hat], [sigma_hat], c='r', s=40, label='MLE')
                    plt.xlabel('µ = ln a0'); plt.ylabel('σ (lognormal)'); plt.legend(frameon=False)
                    hm = images_root / 'hierarchical_a0_heatmap.png'
                    plt.tight_layout(); plt.savefig(hm, dpi=140); plt.close()
                    logging.info(f"Hierarchical a0 MLE: mu={mu_hat:.3f}, sigma={sigma_hat:.3f} → {hier_json}; heatmap: {hm}")
        except Exception as e:
            logging.warning(f"Hierarchical a0 step skipped: {e}")

    # Optional: Full Bayesian hierarchical posterior over (mu, sigma) using dynesty
    if args.hierarchical_a0_bayes and len(list((results_root/'sparc_a0_grids').glob('*.csv'))) > 0:
        try:
            # Build ln a0 grid coverage from per-galaxy grids
            grids = sorted((results_root/'sparc_a0_grids').glob('*.csv'))
            lnmins = []
            lnmaxs = []
            for g in grids:
                xs = []
                with g.open('r', encoding='utf-8') as f:
                    f.readline()
                    for line in f:
                        parts = line.strip().split(',')
                        xs.append(float(parts[0]))
                if len(xs) > 3:
                    lnmins.append(min(xs)); lnmaxs.append(max(xs))
            if lnmins and lnmaxs:
                lnlo = max(lnmins); lnhi = min(lnmaxs)
                if lnhi > lnlo:
                    ln_a0 = np.linspace(lnlo, lnhi, 220)
                    dln = (ln_a0[1]-ln_a0[0]) if len(ln_a0)>1 else 1.0
                    # Build per-galaxy log-likelihood arrays ll_i(ln a0)
                    ll_list = []
                    for g in grids:
                        xa = []
                        c2 = []
                        with g.open('r', encoding='utf-8') as f:
                            f.readline()
                            for line in f:
                                a, v = line.strip().split(',')
                                xa.append(float(a)); c2.append(float(v))
                        xa = np.asarray(xa, float); c2 = np.asarray(c2, float)
                        c2i = np.interp(ln_a0, xa, c2, left=np.nan, right=np.nan)
                        lli = -0.5 * c2i
                        # Normalize each galaxy's ll by subtracting max over available range to aid stability
                        m = np.nanmax(lli)
                        ll = np.full_like(ln_a0, -np.inf)
                        mask = np.isfinite(lli)
                        if np.any(mask):
                            ll[mask] = lli[mask] - m
                        ll_list.append(ll)

                    # Dynesty nested sampling
                    try:
                        import dynesty  # type: ignore
                        from dynesty import utils as dyutils  # type: ignore
                    except Exception as e:
                        logging.warning(f"Hierarchical Bayes: dynesty not available ({e}); skipping.")
                        dynesty = None  # type: ignore

                    if dynesty is not None:
                        sig_lo, sig_hi = float(args.hierarchical_a0_sigma_bounds[0]), float(args.hierarchical_a0_sigma_bounds[1])
                        sig_lo = max(sig_lo, 1e-3)

                        def prior_transform(u):
                            u = np.asarray(u, float)
                            mu = lnlo + u[0] * (lnhi - lnlo)
                            sg = sig_lo + u[1] * (max(sig_hi - sig_lo, 1e-6))
                            return np.array([mu, sg], float)

                        def loglike(theta):
                            mu, sg = float(theta[0]), float(theta[1])
                            if not (np.isfinite(mu) and np.isfinite(sg) and sg > 0):
                                return -np.inf
                            # Normal prior over ln a0 for each galaxy
                            prior_pdf = np.exp(-0.5*((ln_a0-mu)/sg)**2) / (sg*np.sqrt(2.0*np.pi))
                            prior_pdf = np.maximum(prior_pdf, 1e-300)
                            tot = 0.0
                            for lli in ll_list:
                                y = lli + np.log(prior_pdf)
                                m = np.nanmax(y)
                                s = np.sum(np.exp(y - m)) * dln
                                if s <= 0 or not np.isfinite(s):
                                    return -np.inf
                                tot += (m + np.log(s))
                            return float(tot)

                        # Note: dynesty versions may not accept a seed parameter; to keep compatibility, we do not pass it here.
                        sampler = dynesty.NestedSampler(loglike, prior_transform, 2, nlive=int(args.hierarchical_a0_live), sample='rwalk', bound='multi')
                        sampler.run_nested()
                        res = sampler.results
                        # Equal-weight samples
                        ws = np.exp(res['logwt'] - res['logz'][-1])
                        smp = res['samples']
                        try:
                            eq = dyutils.resample_equal(smp, ws)
                        except Exception:
                            # fallback simple normalization
                            w = ws / np.sum(ws)
                            idx = np.random.default_rng(seed).choice(np.arange(len(smp)), size=min(2000, len(smp)), replace=True, p=w)
                            eq = smp[idx]
                        # Summaries
                        mu_s = eq[:,0]; sg_s = eq[:,1]
                        def pct(v):
                            return float(np.nanpercentile(v, 16)), float(np.nanpercentile(v, 50)), float(np.nanpercentile(v, 84))
                        mu_p = pct(mu_s); sg_p = pct(sg_s)
                        post_json = results_root / 'hierarchical_a0_posterior_summary.json'
                        post_json.write_text(json.dumps({
                            'mu_ln_a0': {'p16': mu_p[0], 'p50': mu_p[1], 'p84': mu_p[2]},
                            'sigma_ln_a0': {'p16': sg_p[0], 'p50': sg_p[1], 'p84': sg_p[2]},
                            'nlive': int(args.hierarchical_a0_live),
                            'n_gal': len(ll_list),
                            'ln_a0_grid': {'lo': lnlo, 'hi': lnhi},
                        }, indent=2), encoding='utf-8')
                        np.savez(results_root / 'hierarchical_a0_posterior_samples.npz', samples=smp, logwt=res['logwt'], logz=res['logz'])
                        # Heatmap
                        plt.figure(figsize=(6.4, 5.0))
                        H, xedges, yedges = np.histogram2d(mu_s, sg_s, bins=60)
                        plt.imshow(H.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='magma')
                        plt.colorbar(label='posterior density (arb)')
                        plt.xlabel('μ = ln a0'); plt.ylabel('σ (ln a0)')
                        ttl = f"Hierarchical a0 posterior (nlive={int(args.hierarchical_a0_live)}, Ngal={len(ll_list)})"
                        plt.title(ttl)
                        hm = images_root / 'hierarchical_a0_posterior_heatmap.png'
                        plt.tight_layout(); plt.savefig(hm, dpi=150); plt.close()
                        logging.info(f"Hierarchical a0 posterior: {post_json}; heatmap: {hm}")
        except Exception as e:
            logging.warning(f"Hierarchical Bayesian step skipped: {e}")

    # 5c) Milky Way vertical force Kz and Σ_1.1 (baryons-only + D-scaled approximation)
    if args.mw_kz:
        try:
            # Build (R,Z) grid and baryonic fields
            gs = GridSpec(R_min=0.5, R_max=20.0, Z_max=float(args.mw_zmax_kpc), nR=192, nZ=int(max(args.mw_nz, 129)))
            Rg, Zg, dR, dZ = cylindrical_grid(gs)
            mw = {
                'M_disk_thin': 4.0e10,
                'M_disk_thick': 1.5e10,
                'M_bulge': 1.2e10,
                'M_gas': 3.0e10,
                'R_d_thin': 2.6,
                'R_d_thick': 4.5,
                'R_d_gas': 7.0,
                'a_bulge': 0.7,
                'h_z_thin': 0.3,
                'h_z_thick': 0.9,
                'h_z_gas': 0.15,
            }
            Phi_kms2, gR_SI, gZ_SI, rho_b_SI, Vbar_kms_grid = build_baryon_grids(Rg, Zg, mw)
            gate_kwargs = dict(
                zeta_env=rar_params.get('zeta_env', 0.0),
                rho=None, rho_c=None, gamma_exp=3.0,
                T0=None, sigma_lnT=None, wmin=0.0,
            )
            df_kz, meta_kz = kz_sigma_from_grid(
                Rg, Zg, rho_b_SI, gR_SI, gZ_SI, Vbar_kms_grid,
                a0_m_s2=float(rar_params.get('a0_m_s2', 1.2e-10)),
                gate_kwargs=gate_kwargs,
                R0_kpc=float(args.mw_R0_kpc),
                z_list_kpc=(0.5, 0.8, 1.1, 1.5, 2.0),
                D_max=rar_params.get('D_max', None)
            )
            out_csv = results_root / 'mw_kz_sigma_full3d.csv'
            # Support both pandas DataFrame and list fallback
            try:
                import pandas as pd
                if isinstance(df_kz, pd.DataFrame):
                    df_kz.to_csv(out_csv, index=False)
                else:
                    df = pd.DataFrame(df_kz)
                    df.to_csv(out_csv, index=False)
            except Exception:
                with out_csv.open('w', encoding='utf-8') as f:
                    if isinstance(df_kz, list) and df_kz:
                        cols = list(df_kz[0].keys())
                        f.write(','.join(cols) + '\n')
                        for r in df_kz:
                            f.write(','.join(str(r[c]) for c in cols) + '\n')
            # Plot
            try:
                import pandas as pd
                if not isinstance(df_kz, pd.DataFrame):
                    df_kz = pd.read_csv(out_csv)
            except Exception:
                df_kz = None
            if df_kz is not None:
                plt.figure(figsize=(6.8, 4.6))
                plt.plot(df_kz['z_kpc'], df_kz['Kz_m_s2'], 'r-', lw=2, label='Kz (full 3D)')
                # Optional overlay bands
                if args.mw_kz_overlay_csv:
                    import csv as _csv
                    ovp = Path(args.mw_kz_overlay_csv)
                    if ovp.exists():
                        zz = []; lo = []; hi = []
                        with ovp.open('r', encoding='utf-8') as fh:
                            rdr = _csv.DictReader(fh)
                            for row in rdr:
                                try:
                                    zz.append(float(row['z_kpc']))
                                    lo.append(float(row['Kz_min']))
                                    hi.append(float(row['Kz_max']))
                                except Exception:
                                    continue
                        if len(zz) >= 2:
                            import numpy as _np
                            zz = _np.asarray(zz); lo = _np.asarray(lo); hi = _np.asarray(hi)
                            order = _np.argsort(zz)
                            zz = zz[order]; lo = lo[order]; hi = hi[order]
                            plt.fill_between(zz, lo, hi, color='gray', alpha=0.25, label='Reference band')
                plt.xlabel('z (kpc)'); plt.ylabel('Kz (m s$^{-2}$)')
                plt.title(f'MW Kz at R0={args.mw_R0_kpc} kpc (full 3D phantom)')
                plt.grid(alpha=0.3); plt.legend(frameon=False)
                outpng = images_root / 'mw_kz_sigma_full3d.png'
                plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close()
            logging.info(f"MW Kz/Σ (full 3D) written: {out_csv}")
        except Exception as e:
            logging.warning(f"MW Kz/Σ full-3D step skipped: {e}")

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
        lines.append(f"- Lensing RAR table: `{(results_root / 'lensing_rar_table.csv').as_posix()}` (if present)")
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
