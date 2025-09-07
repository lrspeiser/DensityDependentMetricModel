#!/usr/bin/env python3
"""
Plot rar_plateau (acceleration-based RAR bridge) vs GR for the Milky Way.

- Reads best-fit parameters from a rar_plateau run directory (npz file saved by the dynesty GPU runner).
- Uses the comprehensive CuPy backend to compute GR (baryons-only) and rar_plateau curves on a smooth grid.
- Saves a two-panel PNG: velocities and (RAR - GR) delta.

Usage:
  python scripts/plot_rar_plateau_mw_comparison.py --run_dir runs/rar_plateau_mw_full --out images/rar_plateau_analysis/rar_plateau_mw_comparison.png

If --run_dir is omitted or does not contain results, it falls back to runs/rar_plateau_mw_dryrun.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cupy as cp  # type: ignore
from core.density_metric_cupy import (
    v_baryon_comprehensive_kms_cupy,
    v_total_kms_cupy,
)

# Simple NFW helper (uses baryon curve for total)
import numpy as _np

def compute_nfw_velocity(R_kpc: _np.ndarray, v_bar: _np.ndarray, M_200=1.5e12, c=12.0, R_200=230.0) -> _np.ndarray:
    G = 4.301e-6  # km^2 kpc / (M_sun s^2)
    Rs = float(R_200) / float(c)
    R = _np.asarray(R_kpc, dtype=float)
    def M_enc(r):
        x = r / Rs
        denom = _np.log(1.0 + c) - c/(1.0 + c)
        return M_200 * (_np.log(1.0 + x) - x/(1.0 + x)) / denom
    M_enclosed = _np.array([M_enc(r) if r > 0 else 0.0 for r in R])
    R_safe = _np.maximum(R, 1e-6)
    v_dm = _np.sqrt(_np.maximum(G * M_enclosed / R_safe, 0.0))
    return _np.sqrt(_np.maximum(v_bar, 0.0)**2 + _np.maximum(v_dm, 0.0)**2)

# Gaia processing (local-only)
import pandas as pd
try:
    from core.data_io import process_gaia_data
except Exception as e:
    raise RuntimeError(f"Could not import core.data_io.process_gaia_data: {e}")


def _load_best_from_npz(npz_path: Path) -> dict:
    data = np.load(str(npz_path))
    # param names
    names = None
    if "param_names" in data:
        names = [str(x) for x in data["param_names"]]
    elif "names" in data:
        names = [str(x) for x in data["names"]]
    # best params vector
    best = None
    for key in ("best_params", "best", "x_best"):
        if key in data:
            best = np.asarray(data[key])
            break
    if best is None and "samples" in data and "logl" in data:
        # Fallback: take sample with max logl
        logl = np.asarray(data["logl"])  # (Ns,)
        idx = int(np.argmax(logl))
        samples = np.asarray(data["samples"])  # (Ns, ndim)
        best = samples[idx]
    if best is None or names is None:
        raise RuntimeError("Could not locate best_params and/or param_names in npz")
    if best.ndim == 0:
        best = best.reshape(1)
    return dict(zip(names, best.tolist()))


def _default_baryon_params() -> dict:
    # Mirror runner fixed_params for consistency
    return {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }


def build_params_for_model(a0_params: dict) -> dict:
    p = _default_baryon_params()
    # Add rar_plateau keys
    p.update({
        'a0_m_s2': float(a0_params.get('a0_m_s2', 1.2e-10)),
        'allow_experimental': True,
    })
    # Optional gates (kept off by default)
    for k in ('zeta_env', 'rho_c_solar_kpc3', 'gamma_exp'):
        if k in a0_params:
            p[k] = float(a0_params[k])
    return p


def _load_gaia_local_df(repo_root: Path, override: Path | None = None) -> pd.DataFrame:
    # Prefer override path if provided
    if override is not None and override.exists():
        df = pd.read_csv(override)
        return process_gaia_data(df)
    # Fall back to common locations
    candidates = [
        repo_root / 'external_data' / 'gaia_sky_slices' / 'all_sky_gaia.csv',
        repo_root / 'gaia_sky_slices' / 'all_sky_gaia.csv',
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            return process_gaia_data(df)
    raise FileNotFoundError('Could not find local Gaia cache all_sky_gaia.csv; pass --gaia-csv to specify a file.')


def _star_stats_at_grid(df: pd.DataFrame, grid: np.ndarray, half_width: float = 0.25):
    Rk = df['R_kpc'].values
    V = df['v_obs'].values
    radii = np.asarray(grid, dtype=float)
    v_med = np.full_like(radii, np.nan)
    v_lo = np.full_like(radii, np.nan)
    v_hi = np.full_like(radii, np.nan)
    n = np.zeros_like(radii)
    for i, R0 in enumerate(radii):
        m = (Rk >= R0 - half_width) & (Rk < R0 + half_width)
        if np.any(m):
            vv = V[m]
            v_med[i] = np.median(vv)
            v_lo[i] = np.percentile(vv, 16)
            v_hi[i] = np.percentile(vv, 84)
            n[i] = np.sum(m)
    return radii, v_med, v_lo, v_hi, n


def make_plot(params: dict, out_path: Path, *, gaia_csv: Path | None = None, median_start: float = 1.0, median_step: float = 1.0, median_halfwidth: float = 0.25, rmax_cap: float = 30.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Radii (1..30 kpc log grid)
    R = np.logspace(0.0, np.log10(float(rmax_cap)), 320).astype(np.float32)
    R_cp = cp.asarray(R, dtype=cp.float32)

    # GR (baryons-only)
    v_gr = cp.asnumpy(v_baryon_comprehensive_kms_cupy(R_cp, params))

    # rar_plateau (experimental)
    v_rar = cp.asnumpy(v_total_kms_cupy(R_cp, dict(params), xi_type='rar_plateau'))

    # NFW (ΛCDM baseline) on top of baryons
    v_nfw = compute_nfw_velocity(R, v_gr, M_200=1.5e12, c=12.0, R_200=230.0)

    # Load Gaia and compute star medians at requested grid
    df = _load_gaia_local_df(REPO_ROOT, override=gaia_csv)
    Rk = df['R_kpc'].values
    # Build grid up to observed max, capped by rmax_cap
    rmax_obs = float(np.nanmax(Rk)) if Rk.size else float(rmax_cap)
    r_hi = min(float(rmax_cap), rmax_obs)
    grid = np.arange(float(median_start), float(r_hi) + 0.5*float(median_step), float(median_step))
    R_int, v_med_obs, v_lo_obs, v_hi_obs, n_obs = _star_stats_at_grid(df, grid, half_width=float(median_halfwidth))

    # Build figure
    fig = plt.figure(figsize=(12, 9))

    # Panel 1: velocities
    ax1 = plt.subplot(2, 1, 1)
    # Observational medians at grid radii with error bars
    ok = np.isfinite(v_med_obs)
    ax1.errorbar(R_int[ok], v_med_obs[ok], yerr=[v_med_obs[ok]-v_lo_obs[ok], v_hi_obs[ok]-v_med_obs[ok]], 
                 fmt='ko', ms=4, mec='k', mfc='k', alpha=0.8, ecolor='gray', elinewidth=1.0, capsize=3, label='Gaia medians @ integers')

    ax1.plot(R, v_gr, 'b--', lw=2.5, alpha=0.85, label='GR (baryons only)')
    ax1.plot(R, v_nfw, 'g:', lw=2.8, alpha=0.9, label='GR + NFW')
    ax1.plot(R, v_rar, 'r-', lw=2.8, alpha=0.95, label='RAR-Plateau')
    ax1.axvline(8.5, color='orange', ls='--', alpha=0.5, lw=2)
    ax1.text(8.5, max(v_rar.max(), v_gr.max())*0.95, '☉', ha='center', va='top', color='orange')
    ax1.set_ylabel('Circular velocity (km/s)')
    ax1.set_xlim(0, 31)
    ax1.set_ylim(0, max(v_rar.max(), v_nfw.max(), v_gr.max(), np.nanmax(v_hi_obs))*1.10)
    # Mark 14 kpc and shade beyond as extrapolation region (if desired)
    ax1.axvline(14.0, color='#444444', ls=':', alpha=0.6)
    ax1.axvspan(14.0, 31.0, color='#FFD580', alpha=0.10, label='Extrapolation beyond 14 kpc')
    ax1.set_title('Milky Way: GR vs NFW vs RAR-Plateau with Gaia medians @ integer radii')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Panel 2: delta vs GR
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(R, v_rar - v_gr, 'r-', lw=2.5, alpha=0.95, label='RAR-Plateau - GR')
    ax2.plot(R, v_nfw - v_gr, 'g:', lw=2.2, alpha=0.9, label='(GR+NFW) - GR')
    # Observed minus GR at integer radii
    # Interpolate GR at integer radii for delta
    v_gr_int = np.interp(R_int, R, v_gr)
    ok2 = ok & np.isfinite(v_gr_int)
    ax2.errorbar(R_int[ok2], (v_med_obs - v_gr_int)[ok2], 
                 yerr=[(v_med_obs - v_gr_int - (v_lo_obs - v_gr_int))[ok2], (v_hi_obs - v_gr_int - (v_med_obs - v_gr_int))[ok2]],
                 fmt='ks', ms=3.5, alpha=0.8, ecolor='gray', elinewidth=1.0, capsize=3, label='Gaia median - GR @ integers')

    ax2.axhline(0.0, color='k', lw=0.8)
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Δv (km/s)')
    ax2.set_xlim(0, 31)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, default='runs/rar_plateau_mw_full', help='Path to rar_plateau run directory')
    ap.add_argument('--out', type=str, default='images/rar_plateau_analysis/rar_plateau_mw_comparison.png', help='Output PNG path')
    ap.add_argument('--gaia-csv', type=str, default='', help='Optional explicit path to merged Gaia CSV (all_sky_gaia.csv)')
    ap.add_argument('--median-start', type=float, default=1.0, help='Start radius (kpc) for Gaia median bins')
    ap.add_argument('--median-step', type=float, default=1.0, help='Step (kpc) for Gaia median bins (use 0.1 for fine)')
    ap.add_argument('--median-halfwidth', type=float, default=0.25, help='Half-width (kpc) of the bin around each grid radius')
    ap.add_argument('--rmax-cap', type=float, default=30.0, help='Maximum radius (kpc) for model curves and plotting')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)

    npz = run_dir / 'stellar_fit_cupy_rar_plateau_results.npz'
    if not npz.exists():
        # Fallback to dryrun
        alt = REPO_ROOT / 'runs' / 'rar_plateau_mw_dryrun' / 'stellar_fit_cupy_rar_plateau_results.npz'
        if alt.exists():
            npz = alt
        else:
            raise FileNotFoundError(f"No results npz found in {run_dir} or fallback {alt}")

    best = _load_best_from_npz(npz)
    params = build_params_for_model(best)

    gaia_path = Path(args.gaia_csv) if args.gaia_csv else None
    make_plot(
        params,
        out_path,
        gaia_csv=gaia_path,
        median_start=float(args.median_start),
        median_step=float(args.median_step),
        median_halfwidth=float(args.median_halfwidth),
        rmax_cap=float(args.rmax_cap),
    )


if __name__ == '__main__':
    main()

