#!/usr/bin/env python3
"""
fit_sparc_rar_gate_grid.py

Batch-run a simple RAR-gate grid over SPARC galaxies that have midplane
density data, producing per-galaxy plots and JSON summaries plus a
summary.csv per (gamma, lambda) slice.

The model used here mirrors the core CuPy implementation conceptually:
  xi(R) = 1 + lambda_max * S_g(g_bar) * W(T)
    where
      T   = v_bar(R)^2 / R^2                in (km/s)^2 / kpc^2
      g_bar = T * R * K                     in m/s^2, K = 3.240779289e-14
      S_g = 1 / (1 + (g_bar / a0)^gamma)
      W(T) = wmin + (1-wmin) * exp(-(ln(T/T0))^2 / (2 sigma_lnT^2))

This script does NOT fit parameters; it sweeps a small grid to generate
plots and lightweight diagnostics suitable for quick comparisons.

Usage examples:
  # Default SPARC dir and out dir, gold sample autodetected, modest grid
  python tools/fit_sparc_rar_gate_grid.py \
    --gold-sample \
    --gammas 1 2 \
    --lambdas 0.6 0.8 1.0 1.2

  # Explicit galaxy subset and custom T0/sigma/wmin
  python tools/fit_sparc_rar_gate_grid.py \
    --galaxies NGC3198 NGC2403 \
    --T0 15 --sigma-lnT 0.9 --wmin 0.02

Outputs layout (relative to out dir):
  sparc_rargate_gold_plots/
    gamma{g}_lambda{lam}/
      summary.csv
      plots/{galaxy}.png
      json/{galaxy}.json
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# Repo root on path (tools/ lives one level below repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SPARC loader and helper used elsewhere in repo
from utils.Utilities.sparc_io import (
    load_single_sparc_galaxy,
    BASE_M_L_3_6_MICRON_DISK as BASE_UPS_DISK,
    BASE_M_L_3_6_MICRON_BULGE as BASE_UPS_BULGE,
)
from models.er_sparc import v_bar_from_components

ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14  # m/s^2 per [(km/s)^2/kpc]


def w_t_log_normal(T: np.ndarray, T0: float, sigma_lnT: float, wmin: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    T0 = float(max(T0, 1e-30))
    sig = float(max(sigma_lnT, 1e-6))
    u = (np.log(np.maximum(T, 1e-30)) - math.log(T0)) / sig
    W = np.exp(-0.5 * u * u)
    wmin_c = float(np.clip(wmin, 0.0, 0.2))
    return wmin_c + (1.0 - wmin_c) * W


def xi_rar_gate(vbar_kms: np.ndarray, R_kpc: np.ndarray,
                a0_m_s2: float, gamma_exp: float, lambda_max: float,
                T0: float, sigma_lnT: float, wmin: float) -> np.ndarray:
    vbar_kms = np.asarray(vbar_kms, dtype=float)
    R_kpc = np.asarray(R_kpc, dtype=float)
    R_safe = np.maximum(R_kpc, 1e-12)
    T = np.maximum(vbar_kms, 0.0)**2 / np.maximum(R_safe**2, 1e-18)
    gbar = ACC_M_S2_PER_KMS2_PER_KPC * T * R_safe
    x = gbar / max(a0_m_s2, 1e-30)
    Sg = 1.0 / (1.0 + np.power(x, float(gamma_exp)))
    W = w_t_log_normal(T, T0, sigma_lnT, wmin)
    lam = float(lambda_max)
    xi = 1.0 + lam * Sg * W
    return np.clip(xi, 1.0, 1.0 + lam)


def chi2_with_floor(v_obs: np.ndarray, v_mod: np.ndarray, e_v: np.ndarray, sigma_floor: float) -> float:
    v_obs = np.asarray(v_obs, dtype=float)
    v_mod = np.asarray(v_mod, dtype=float)
    e_v = np.asarray(e_v, dtype=float)
    e_eff = np.sqrt(np.maximum(e_v, 0.0)**2 + float(max(sigma_floor, 0.0))**2)
    e_eff = np.where(e_eff > 0, e_eff, 1.0)
    r = (v_obs - v_mod) / e_eff
    return float(np.sum(r * r))


def discover_galaxies(sparc_dir: Path) -> List[str]:
    ids = []
    for p in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = p.name.replace('_rotmod.dat', '')
        if name:
            ids.append(name)
    return ids


def has_midplane_densities(galaxy_id: str, sparc_dir: Path,
                           hz_star: float, hz_gas: float, hz_alpha: float) -> bool:
    data = load_single_sparc_galaxy(
        galaxy_id,
        sparc_dir=str(sparc_dir),
        assume_gas_hz_kpc=float(hz_gas),
        assume_stellar_hz_kpc=float(hz_star),
        assume_hz_alpha=float(hz_alpha),
    )
    if data is None:
        return False
    rho_star = data.get('rho_star_mid_Msun_kpc3_baseML')
    rho_gas = data.get('rho_gas_mid_Msun_kpc3')
    if rho_star is None or rho_gas is None:
        return False
    try:
        arr = np.asarray(rho_star) + np.asarray(rho_gas)
        return np.isfinite(arr).any() and arr.size > 0
    except Exception:
        return False


def write_summary_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            f.write('galaxy,chi2,chi2_dof,gamma_exp,lambda_max,a0_m_s2,T0,sigma_lnT,wmin,'
                    'R_data_max_kpc,n_points,ups_disk,ups_bulge,sigma_floor,xi_med,xi_max\n')


def append_summary_row(path: Path, row: Dict[str, object]):
    with path.open('a', encoding='utf-8') as f:
        vals = [
            row.get('galaxy'),
            f"{row.get('chi2', float('nan'))}",
            f"{row.get('chi2_dof', float('nan'))}",
            f"{row.get('gamma_exp')}",
            f"{row.get('lambda_max')}",
            f"{row.get('a0_m_s2')}",
            f"{row.get('T0')}",
            f"{row.get('sigma_lnT')}",
            f"{row.get('wmin')}",
            f"{row.get('R_data_max_kpc')}",
            f"{row.get('n_points')}",
            f"{row.get('ups_disk')}",
            f"{row.get('ups_bulge')}",
            f"{row.get('sigma_floor')}",
            f"{row.get('xi_med')}",
            f"{row.get('xi_max')}",
        ]
        f.write(','.join(map(str, vals)) + '\n')


def save_plot(R: np.ndarray, Vobs: np.ndarray, eV: np.ndarray,
              vbar: np.ndarray, vmod: np.ndarray,
              out_png: Path, title: str,
              R_data_max: float):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,7))
    plt.errorbar(R, Vobs, yerr=eV, fmt='o', color='k', ms=4, lw=1, alpha=0.8, label='Observed (SPARC)')
    plt.plot(R, vbar, 'b--', lw=2, label='GR (baryons)')

    # Grid for split curve drawing
    R_grid = np.linspace(max(1e-3, float(np.min(R))), max(float(np.max(R))*1.2, float(np.max(R))+5), 400)
    vbar_g = np.interp(R_grid, R, vbar)
    vmod_g = np.interp(R_grid, R, vmod)
    m_in = R_grid <= R_data_max
    m_out = ~m_in
    if np.any(m_in):
        plt.plot(R_grid[m_in], vmod_g[m_in], 'r-', lw=2.5, label='RAR gate — constrained')
    if np.any(m_out):
        plt.plot(R_grid[m_out], vmod_g[m_out], color='#FF8C00', ls='--', lw=2.5, label='RAR gate — extrapolation')
        plt.axvspan(R_data_max, R_grid.max(), color='#FFA500', alpha=0.08)
    plt.axvline(R_data_max, color='k', ls=':', alpha=0.6, label=f"Max data R ≈ {R_data_max:.1f} kpc")

    plt.xlabel('R (kpc)')
    plt.ylabel('Vc (km s^{-1})')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.xlim(0, max(float(np.max(R))*1.2, float(np.max(R))+5))
    ymax = max(np.nanmax(Vobs+eV), np.nanmax(vbar)*1.2)
    plt.ylim(0, max(300, float(ymax)+40))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_one_galaxy(gid: str, sparc_dir: Path,
                        a0: float, gamma_exp: float, lambda_max: float,
                        T0: float, sigma_lnT: float, wmin: float,
                        ups_disk: float, ups_bulge: float,
                        sigma_floor: float,
                        hz_star: float, hz_gas: float, hz_alpha: float,
                        out_slice: Path) -> Optional[Dict[str, object]]:
    data = load_single_sparc_galaxy(
        gid,
        sparc_dir=str(sparc_dir),
        assume_gas_hz_kpc=float(hz_gas),
        assume_stellar_hz_kpc=float(hz_star),
        assume_hz_alpha=float(hz_alpha),
    )
    if data is None:
        print(f"  [skip] could not load {gid}")
        return None

    R = np.asarray(data['R_kpc'], dtype=float)
    Vobs = np.asarray(data['V_obs'], dtype=float)
    eV = np.asarray(data['e_V_obs'], dtype=float)
    Vgas = np.asarray(data['V_gas_comp_kms'], dtype=float)
    Vdisk = np.asarray(data['V_disk_comp_kms'], dtype=float)
    Vbul = np.asarray(data['V_bulge_comp_kms'], dtype=float)

    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_disk, ups_bulge)
    xi = xi_rar_gate(vbar, R, a0, gamma_exp, lambda_max, T0, sigma_lnT, wmin)
    vmod = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar, 0.0)

    chi2 = chi2_with_floor(Vobs, vmod, eV, sigma_floor)
    dof = max(1, R.size - 2)  # no parameter fit in this script; use a small dof floor

    # Write JSON
    jdir = out_slice / 'json'
    jdir.mkdir(parents=True, exist_ok=True)
    jpath = jdir / f"{gid}.json"
    meta = {
        'galaxy_id': gid,
        'model': 'rar_gate_grid',
        'params': {
            'a0_m_s2': float(a0),
            'gamma_exp': float(gamma_exp),
            'lambda_max': float(lambda_max),
            'T0': float(T0),
            'sigma_lnT': float(sigma_lnT),
            'wmin': float(wmin),
            'ups_disk': float(ups_disk),
            'ups_bulge': float(ups_bulge),
            'sigma_floor': float(sigma_floor),
            'hz_star_kpc': float(hz_star),
            'hz_gas_kpc': float(hz_gas),
            'hz_alpha': float(hz_alpha),
        },
        'chi2': float(chi2),
        'chi2_dof': float(chi2/dof),
        'n_points': int(R.size),
        'R_data_max_kpc': float(np.max(R) if R.size else float('nan')),
        'xi_stats': {
            'xi_min': float(np.nanmin(xi)),
            'xi_med': float(np.nanmedian(xi)),
            'xi_max': float(np.nanmax(xi)),
        }
    }
    with jpath.open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Plot PNG
    pdir = out_slice / 'plots'
    pdir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / f"{gid}.png"
    title = f"{gid}: RAR gate (γ={gamma_exp}, λ={lambda_max}, a0={a0:.2e})"
    save_plot(R, Vobs, eV, vbar, vmod, ppath, title, float(np.max(R)))

    row = {
        'galaxy': gid,
        'chi2': chi2,
        'chi2_dof': chi2/dof,
        'gamma_exp': gamma_exp,
        'lambda_max': lambda_max,
        'a0_m_s2': a0,
        'T0': T0,
        'sigma_lnT': sigma_lnT,
        'wmin': wmin,
        'R_data_max_kpc': float(np.max(R) if R.size else float('nan')),
        'n_points': int(R.size),
        'ups_disk': ups_disk,
        'ups_bulge': ups_bulge,
        'sigma_floor': sigma_floor,
        'xi_med': float(np.nanmedian(xi)),
        'xi_max': float(np.nanmax(xi)),
    }
    return row


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc_dir', type=str, default=str(REPO_ROOT / 'external_data' / 'Rotmod_LTG'))
    ap.add_argument('--out_dir', type=str, default='sparc_rargate_gold_plots')
    ap.add_argument('--galaxies', nargs='*', default=None, help='Explicit list of galaxy IDs (e.g., NGC3198 NGC2403)')
    ap.add_argument('--gold-sample', action='store_true', help='Autodetect galaxies that have midplane density data available')
    ap.add_argument('--gammas', nargs='+', type=float, default=[1.0, 2.0])
    ap.add_argument('--lambdas', nargs='+', type=float, default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument('--a0_m_s2', type=float, default=1.2e-10)
    ap.add_argument('--T0', type=float, default=10.0)
    ap.add_argument('--sigma-lnT', dest='sigma_lnT', type=float, default=0.8)
    ap.add_argument('--wmin', type=float, default=0.02)
    ap.add_argument('--ups-disk', dest='ups_disk', type=float, default=BASE_UPS_DISK)
    ap.add_argument('--ups-bulge', dest='ups_bulge', type=float, default=BASE_UPS_BULGE)
    ap.add_argument('--sigma-floor', type=float, default=5.0)
    ap.add_argument('--hz-star', type=float, default=0.30)
    ap.add_argument('--hz-gas', type=float, default=0.10)
    ap.add_argument('--hz-alpha', type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    sparc_dir = Path(args.sparc_dir)
    out_root = (REPO_ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not sparc_dir.exists():
        print(f"SPARC directory not found: {sparc_dir}")
        sys.exit(2)

    # Build galaxy list
    if args.galaxies:
        galaxies = list(args.galaxies)
    else:
        candidates = discover_galaxies(sparc_dir)
        if args.gold_sample:
            print(f"Discovered {len(candidates)} SPARC candidates; filtering for midplane densities...")
            galaxies = [g for g in candidates if has_midplane_densities(g, sparc_dir, args.hz_star, args.hz_gas, args.hz_alpha)]
            print(f"Gold sample (with densities): {len(galaxies)}")
        else:
            galaxies = candidates
            print(f"Using all discovered galaxies: {len(galaxies)}")

    if not galaxies:
        print("No galaxies to process.")
        sys.exit(0)

    # Sweep grid
    for gamma_exp in args.gammas:
        for lam in args.lambdas:
            slice_dir = out_root / f"gamma{gamma_exp}_lambda{lam}"
            summary_csv = slice_dir / 'summary.csv'
            write_summary_header(summary_csv)
            print(f"\n=== Sweep: gamma={gamma_exp}, lambda={lam} ===")
            for gid in galaxies:
                try:
                    row = process_one_galaxy(
                        gid, sparc_dir,
                        args.a0_m_s2, gamma_exp, lam,
                        args.T0, args.sigma_lnT, args.wmin,
                        args.ups_disk, args.ups_bulge,
                        args.sigma_floor,
                        args.hz_star, args.hz_gas, args.hz_alpha,
                        slice_dir,
                    )
                    if row is not None:
                        append_summary_row(summary_csv, row)
                        print(f"  {gid}: chi2={row['chi2']:.1f}, chi2/dof={row['chi2_dof']:.2f}, xi_med={row['xi_med']:.2f}")
                except Exception as e:
                    print(f"  [error] {gid}: {e}")

    print(f"\nComplete. Results under: {out_root}")


if __name__ == '__main__':
    main()

