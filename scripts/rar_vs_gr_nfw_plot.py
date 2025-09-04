#!/usr/bin/env python3
"""
Generate overlay plot: Gaia DR3 (144k) vs GR (baryon), NFW (baryon+halo, from confirm_nfw_144k), and RAR gate (best-fit from prior run).
Output: images/rar_vs_gr_nfw_gaia.png
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cupy as cp
import pandas as pd
import sys

# Ensure repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.density_metric_cupy import (
    v_baryon_comprehensive_kms_cupy,
    v_total_kms_cupy as v_total_core,
    DEFAULT_DTYPE,
)
from core.data_io import process_gaia_data


def load_gaia_processed(repo_root: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load raw Gaia CSV and compute R_kpc, v_obs using process_gaia_data."""
    candidates = [
        repo_root / 'external_data' / 'gaia_sky_slices' / 'all_sky_gaia.csv',
        repo_root / 'gaia_sky_slices' / 'all_sky_gaia.csv',
    ]
    for p in candidates:
        if p.exists():
            df_raw = pd.read_csv(p)
            df = process_gaia_data(df_raw)
            m = (
                np.isfinite(df['R_kpc'])
                & np.isfinite(df['v_obs'])
                & (df['R_kpc'] > 0.5)
                & (df['R_kpc'] < 30)
            )
            df = df.loc[m, ['R_kpc', 'v_obs']].copy()
            R_kpc = df['R_kpc'].to_numpy(dtype=float)
            v_obs = df['v_obs'].to_numpy(dtype=float)
            R_data_max = float(np.nanmax(R_kpc)) if R_kpc.size else 0.0
            return R_kpc, v_obs, R_data_max
    raise FileNotFoundError('No local Gaia CSV found at external_data/gaia_sky_slices/all_sky_gaia.csv')


def bin_median(R: np.ndarray, V: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.digitize(R, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    v_med = np.full_like(centers, np.nan, dtype=float)
    v_lo = np.full_like(centers, np.nan, dtype=float)
    v_hi = np.full_like(centers, np.nan, dtype=float)
    for i in range(1, len(bins)):
        sel = idx == i
        if np.any(sel):
            vv = V[sel]
            v_med[i - 1] = np.median(vv)
            v_lo[i - 1] = np.percentile(vv, 16)
            v_hi[i - 1] = np.percentile(vv, 84)
    return centers, v_med, v_lo, v_hi


def load_rar_best(repo_root: Path) -> dict:
    p = repo_root / 'runs' / 'rar_gate_20250818_164443' / 'run_summary_enhanced.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    best = data['parameter_estimates']['best_fit']
    best['allow_experimental'] = True
    return best


def load_nfw_best(repo_root: Path) -> tuple[float, float]:
    candidates = [
        repo_root / 'runners' / 'runs' / 'confirm_nfw_144k' / 'stellar_fit_cupy_nfw_results.npz',
        repo_root / 'runners' / 'confirm_nfw' / 'stellar_fit_cupy_nfw_results.npz',
    ]
    for p in candidates:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            names = [str(n) for n in z['param_names'].tolist()]
            vals = z['best_params']
            d = {n: float(v) for n, v in zip(names, vals)}
            return float(d.get('M_vir', 1.0e12)), float(d.get('c_vir', 12.0))
    raise FileNotFoundError('NFW results NPZ not found; run confirm_nfw_144k first.')


def main():
    repo = REPO_ROOT
    # Gaia
    R_kpc, v_obs, R_data_max = load_gaia_processed(repo)
    centers, v_med, v_lo, v_hi = bin_median(R_kpc, v_obs, np.linspace(2.0, 30.0, 29))

    # RAR best-fit params
    p_best = load_rar_best(repo)

    # NFW params
    M_vir, c_vir = load_nfw_best(repo)

    # Build baryon dict from RAR best fit
    baryon_keys = [
        'M_thin_disk_solar', 'R_thin_disk_kpc', 'hz_thin_disk_kpc',
        'M_thick_disk_solar', 'R_thick_disk_kpc', 'hz_thick_disk_kpc',
        'M_bulge_solar', 'R_bulge_kpc', 'M_gas_solar', 'R_gas_kpc', 'hz_gas_kpc',
    ]
    P_b = {k: float(p_best[k]) for k in baryon_keys if k in p_best}
    P_b.update({
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    })

    # R grid
    R_grid = np.linspace(2.0, 30.0, 400)
    Rg = cp.asarray(R_grid, dtype=DEFAULT_DTYPE)

    # GR (baryon-only)
    v_baryon = v_baryon_comprehensive_kms_cupy(Rg, P_b)
    v_gr = v_baryon

    # RAR total
    P_rar = dict(P_b)
    for k in ['a0_m_s2', 'gamma_exp', 'lambda_max', 'T0', 'sigma_lnT', 'wmin']:
        if k in p_best:
            P_rar[k] = float(p_best[k])
    P_rar['allow_experimental'] = True
    v_rar = v_total_core(Rg, P_rar, xi_type='rar_gate')

    # NFW total
    G = 4.30091e-6  # kpc (km/s)^2 / Msun
    rho_crit = cp.asarray(100.0, dtype=DEFAULT_DTYPE)
    Mvir = cp.asarray(M_vir, dtype=DEFAULT_DTYPE)
    cvir = cp.asarray(max(c_vir, 1e-6), dtype=DEFAULT_DTYPE)
    Rvir = cp.power(Mvir / (200.0 * rho_crit * (4.0 * cp.pi / 3.0)), 1.0 / 3.0)
    rs = Rvir / cvir
    x = cp.clip(Rg / cp.maximum(rs, cp.asarray(1e-6, dtype=DEFAULT_DTYPE)), 1e-8, cp.inf)
    gx = cp.log1p(x) - x / (1.0 + x)
    gc = cp.log1p(cvir) - cvir / (1.0 + cvir)
    gc = cp.maximum(gc, cp.asarray(1.0e-12, dtype=DEFAULT_DTYPE))
    Menc = Mvir * gx / gc
    v_nfw_halo_sq = G * Menc / cp.maximum(Rg, cp.asarray(1e-6, dtype=DEFAULT_DTYPE))
    v_nfw_total = cp.sqrt(cp.maximum(v_baryon**2 + v_nfw_halo_sq, 0.0))

    # Plot
    out_dir = repo / 'images'
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'rar_vs_gr_nfw_gaia.png'

    plt.figure(figsize=(11, 8))
    valid = np.isfinite(v_med)
    plt.plot(centers[valid], v_med[valid], color='#4D4D4D', lw=2, label='Gaia: median stellar speed')
    band_valid = np.isfinite(v_lo) & np.isfinite(v_hi)
    plt.fill_between(centers[band_valid], v_lo[band_valid], v_hi[band_valid], color='#A6A6A6', alpha=0.25, label='Gaia: 16–84 percentile')

    plt.plot(R_grid, cp.asnumpy(v_gr), 'b--', lw=2, label='GR (baryon-only)')
    plt.plot(R_grid, cp.asnumpy(v_nfw_total), color='green', ls='-', lw=2, label='LCDM/NFW (fit, baryon+halo)')
    plt.plot(R_grid, cp.asnumpy(v_rar), color='red', ls='-', lw=2.2, label='RAR gate (best-fit)')

    if R_data_max > 0:
        plt.axvline(R_data_max, color='k', ls=':', alpha=0.6, label=f'Max Gaia R ≈ {R_data_max:.1f} kpc')
        plt.axvspan(R_data_max, R_grid.max(), color='#FFA500', alpha=0.08)

    plt.xlabel('Galactocentric radius R (kpc)')
    plt.ylabel('Circular speed v (km/s)')
    plt.title('Milky Way Rotation Curve: Gaia vs GR, NFW, and RAR gate')
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 30)
    ymax = float(np.nanmax([
        np.nanmax(v_med),
        np.nanmax(cp.asnumpy(v_gr)),
        np.nanmax(cp.asnumpy(v_rar)),
        np.nanmax(cp.asnumpy(v_nfw_total)),
    ]))
    plt.ylim(0, max(300, ymax + 40))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f'Saved: {out_file}')


if __name__ == '__main__':
    main()

