#!/usr/bin/env python3
"""
Batch-generate SPARC rotation curve overlays using MW-fitted tidal models.

- For each galaxy ID, loads SPARC components, builds v_bar(R)
- Computes tidal curves via environment-based xi_env using parameters mapped from
  MW runs' posterior medians for three variants: tidal_band2, tidal_ratio, tidal_noisyor
- Also plots GR (baryons) and NFW (if evidence JSON is available)
- Writes PNGs to the specified output directory without overwriting existing images

Usage example:
  python scripts/plot_sparc_overlays_from_mw_tidal.py \
    --galaxies NGC3198 NGC2403 NGC2841 NGC5055 NGC6946 \
    --sparc-dir external_data/Rotmod_LTG \
    --band2-run runs/tidal_band2_20250814_104418 \
    --ratio-run runs/tidal_ratio_20250814_104554 \
    --noisyor-run runs/tidal_noisyor_20250814_104602 \
    --out-dir images/sparc_overlays_new
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "utils", REPO_ROOT / "tools"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from utils.Utilities.sparc_io import load_single_sparc_galaxy, BASE_M_L_3_6_MICRON_DISK  # type: ignore
from models.er_sparc import v_bar_from_components  # type: ignore
from models.nfw import v_model_nfw  # type: ignore
from models.er_env import xi_env, tidal_proxy_from_vbar, finite_diff  # type: ignore

import json

def _weighted_medians_from_npz(npz_path: Path) -> Dict[str, float]:
    data = np.load(str(npz_path))
    names = None
    if "names" in data:
        names = [str(n) for n in data["names"]]
    elif "param_names" in data:
        names = [str(n) for n in data["param_names"]]
    samples = None
    for key in ("samples", "posterior_samples", "xs"):
        if key in data:
            samples = np.asarray(data[key])
            break
    weights = None
    for key in ("weights", "w"):
        if key in data:
            weights = np.asarray(data[key], dtype=float)
            break
    if samples is None:
        return {}
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    N = samples.shape[0]
    if weights is None or weights.size != N:
        weights = np.ones(N, dtype=float) / float(N)
    else:
        s = float(np.sum(weights)); weights = weights / s if s > 0 else np.ones(N, dtype=float)/float(N)
    if names is None or len(names) != samples.shape[1]:
        names = [f"param_{i}" for i in range(samples.shape[1])]
    order = np.argsort(samples, axis=0)
    med: Dict[str, float] = {}
    for j, name in enumerate(names):
        idx = order[:, j]
        xs = samples[idx, j]
        ws = weights[idx]
        cdf = np.cumsum(ws); cdf /= cdf[-1]
        p50 = float(np.interp(0.5, cdf, xs))
        med[name] = p50
    return med


def _map_to_env_params(med: Dict[str, float], variant: str) -> Dict[str, float]:
    """Map MW tidal run median params to env xi parameters.
    We use available keys when present; fill missing with reasonable defaults.
    """
    out = {}
    # Common keys
    out['lambda_max'] = float(med.get('lambda_max', 3.0))
    out['rho_c'] = float(med.get('rho_c_solar_kpc3', 1e16))
    # gamma
    if 'gamma_exp' in med:
        out['gamma_exp'] = float(med['gamma_exp'])
    elif variant == 'ratio' and 'eta' in med:
        out['gamma_exp'] = float(med['eta'])
    else:
        out['gamma_exp'] = 2.5
    # T0 and sigma
    T0 = float(med.get('T0', med.get('T_0', 10.0)))
    out['lnT0'] = float(np.log(max(T0, 1e-6)))
    out['sigma_lnT'] = float(med.get('sigma_lnT', 0.8))
    # w_min
    out['w_min'] = float(med.get('wmin', med.get('w_min', 0.02)))
    return out


def _compute_tidal_proxy(R: np.ndarray, vbar: np.ndarray, mode: str = "curvature", norm: str = "robust") -> np.ndarray:
    R_safe = np.clip(R.astype(float), 1e-6, None)
    vb = np.maximum(vbar.astype(float), 0.0)
    if mode == "curvature":
        g = (vb**2) / R_safe
        dgdR = finite_diff(g, R_safe)
        T_raw = np.abs(dgdR)
    elif mode == "shear":
        Omega = vb / R_safe
        dO_dR = finite_diff(Omega, R_safe)
        T_raw = np.abs(dO_dR)
    elif mode == "epicyclic":
        Omega = vb / R_safe
        dO_dR = finite_diff(Omega, R_safe)
        kappa2 = 2.0 * Omega * (2.0 * Omega + R_safe * dO_dR)
        T_raw = np.sqrt(np.clip(np.abs(kappa2), 0.0, None))
    else:
        T_raw = tidal_proxy_from_vbar(R_safe, vb)
    # robust norm
    med = float(np.nanmedian(T_raw))
    scale = med if med > 0 else (float(np.nanmax(T_raw)) if np.nanmax(T_raw) > 0 else 1.0)
    return np.clip(T_raw/scale, 1e-12, None)


def _try_load_nfw(gr_json_dir: Path, galid: str, R: np.ndarray, vbar: np.ndarray) -> Optional[np.ndarray]:
    path = gr_json_dir / f'sparc_nfw_evidence_{galid.lower()}.json'
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        p = meta.get('params', {})
        if 'V200' in p and 'c' in p:
            V200 = float(p['V200']); c = float(p['c'])
            return v_model_nfw(R, vbar, V200, c)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--sparc-dir', required=True)
    ap.add_argument('--band2-run', required=True)
    ap.add_argument('--ratio-run', required=True)
    ap.add_argument('--noisyor-run', required=True)
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'images' / 'sparc_overlays_new'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = REPO_ROOT / 'images'

    # Load MW run medians
    def load_medians(run_path: str) -> Dict[str, float]:
        npz = Path(run_path) / 'posterior_samples.npz'
        if not npz.exists():
            print(f"WARNING: missing {npz}")
            return {}
        return _weighted_medians_from_npz(npz)

    med_band2 = load_medians(args.band2_run)
    med_ratio = load_medians(args.ratio_run)
    med_noisyor = load_medians(args.noisyor_run)

    env_band2 = _map_to_env_params(med_band2, 'band2')
    env_ratio = _map_to_env_params(med_ratio, 'ratio')
    env_noisyor = _map_to_env_params(med_noisyor, 'noisyor')

    for galid in args.galaxies:
        data = load_single_sparc_galaxy(galid, sparc_dir=args.sparc_dir)
        if data is None:
            print(f"Failed to load {galid}")
            continue
        R = np.asarray(data['R_kpc'], dtype=float)
        Vobs = np.asarray(data['V_obs'], dtype=float)
        eV = np.asarray(data['e_V_obs'], dtype=float)
        Vgas = np.asarray(data['V_gas_comp_kms'], dtype=float)
        Vdisk = np.asarray(data['V_disk_comp_kms'], dtype=float)
        Vbul = np.asarray(data['V_bulge_comp_kms'], dtype=float)
        rho_mid_base = np.asarray(data['rho_star_mid_Msun_kpc3_baseML'], dtype=float) + np.asarray(data['rho_gas_mid_Msun_kpc3'], dtype=float)

        # Adopt ER defaults for M/L if not provided per galaxy
        ups_d = 0.5
        ups_b = 0.7
        vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
        v_gr = vbar.copy()

        # Approximate M/L scaling for rho (consistent with ER fitter approach)
        f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
        rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)
        Tn = _compute_tidal_proxy(R, vbar, mode='curvature', norm='robust')

        curves: Dict[str, np.ndarray] = {}
        # Compute xi and curves for each variant
        for label, env in (
            ('tidal_band2', env_band2),
            ('tidal_ratio', env_ratio),
            ('tidal_noisyor', env_noisyor),
        ):
            try:
                xi = xi_env(rho_mid, Tn, env['lambda_max'], env['rho_c'], env['gamma_exp'], np.exp(env['lnT0']), env['sigma_lnT'], env['w_min'])
                curves[label] = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar, 0.0)
            except Exception as e:
                print(f"{galid}: failed to compute {label}: {e}")

        v_nfw = _try_load_nfw(images_dir, galid, R, vbar)

        # Plot
        plt.figure(figsize=(10, 7))
        plt.errorbar(R, Vobs, yerr=eV, fmt='o', ms=4, color='k', lw=1, alpha=0.85, label='Observed (SPARC)')
        plt.plot(R, v_gr, 'b--', lw=2.0, label='GR (baryons)')
        if v_nfw is not None:
            plt.plot(R, v_nfw, color='green', ls='-.', lw=2.0, label='NFW (ΛCDM)')
        for label, arr in curves.items():
            color = {'tidal_band2':'#D62728','tidal_ratio':'#9467BD','tidal_noisyor':'#FF7F0E'}.get(label, None)
            plt.plot(R, arr, color=color, lw=2.2, label=label.replace('_',' '))
        plt.xlabel('R (kpc)'); plt.ylabel('Vc (km s$^{-1}$)')
        plt.title(f'{galid}: GR, NFW, and MW-fitted Tidal variants')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(R.max()*1.2, R.max()+5))
        ymax = max(np.nanmax(Vobs+eV), np.nanmax(v_gr)*1.2)
        for arr in curves.values():
            ymax = max(ymax, float(np.nanmax(arr))*1.1)
        plt.ylim(0, max(300, ymax + 40))
        plt.legend(frameon=False)
        out_png = out_dir / f'{galid.lower()}_overlay_5way_mw_tidal.png'
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f'Saved: {out_png}')

if __name__ == '__main__':
    main()

