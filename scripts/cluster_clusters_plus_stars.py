#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

G_CGS = 6.67430e-8
MSUN_CGS = 1.98847e33
KPC_CM = 3.0856775814913673e21


def hernquist_accel_cgs(M_solar: float, Re_kpc: float, r_kpc: float) -> float:
    """Hernquist sphere radial acceleration g(r) in cgs for mass M and scale Re.
    a = Re / 1.8153; M(<r) = M r^2/(r+a)^2; g = GM(<r)/r^2 = G M / (r+a)^2
    """
    if M_solar <= 0 or Re_kpc <= 0 or r_kpc < 0:
        return 0.0
    a_kpc = Re_kpc / 1.8153
    denom_cm = (r_kpc + a_kpc) * KPC_CM
    return G_CGS * (M_solar * MSUN_CGS) / (denom_cm * denom_cm)


def xi_rar_plateau(gbar_cgs: np.ndarray, a0_cgs: float, dmax: float) -> np.ndarray:
    # xi = min( 0.5 + sqrt(0.25 + a0/gbar), Dmax)
    x = 0.5 + np.sqrt(0.25 + a0_cgs / np.clip(gbar_cgs, 1e-99, None))
    if np.isfinite(dmax) and dmax > 0:
        x = np.minimum(x, dmax)
    return x


def load_r200_by_cluster(diagnostics_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with diagnostics_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = row['cluster']
            out[name] = float(row['r200c_kpc'])
    return out


def load_points(points_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    clusters = []
    z = []
    r_kpc = []
    loggbar = []
    loggNFW = []
    loggRAR = []  # not used; present for reference
    with points_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            clusters.append(row['cluster'])
            z.append(float(row['z']))
            r_kpc.append(float(row['r_kpc']))
            loggbar.append(float(row['log10_gbar_cgs']))
            loggNFW.append(float(row['log10_gNFWtot_cgs']))
            loggRAR.append(float(row.get('log10_gRAR_cgs', 'nan')))
    return (
        np.array(z),
        np.array(r_kpc),
        np.array(loggbar),
        np.array(loggNFW),
        np.array(loggRAR),
        clusters,
    )


def load_stars_csv(stars_csv: Optional[Path]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not stars_csv:
        return out
    if not stars_csv.exists():
        raise FileNotFoundError(f"Stars CSV not found: {stars_csv}")
    with stars_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = row['cluster'].strip()
            # Primary BCG
            bcg_logM = row.get('log10Mstar_BCG')
            bcg_Re = row.get('Re_kpc') or row.get('Re_BCG_kpc')
            prof = (row.get('profile') or 'hernquist').lower()
            # Optional ICL
            icl_logM = row.get('log10Mstar_ICL')
            icl_Re = row.get('Re_ICL_kpc')
            icl_prof = (row.get('profile_ICL') or prof).lower()
            out[name] = {
                'bcg_logM': float(bcg_logM) if bcg_logM not in (None, '') else None,
                'bcg_Re_kpc': float(bcg_Re) if bcg_Re not in (None, '') else None,
                'bcg_profile': prof,
                'icl_logM': float(icl_logM) if icl_logM not in (None, '') else None,
                'icl_Re_kpc': float(icl_Re) if icl_Re not in (None, '') else None,
                'icl_profile': icl_prof,
            }
    return out


def compute_with_optional_stars(
    clusters: list,
    r_kpc: np.ndarray,
    loggbar_cgs: np.ndarray,
    stars_map: Dict[str, dict],
) -> np.ndarray:
    gbar = np.power(10.0, loggbar_cgs)
    if not stars_map:
        return gbar
    g_add = np.zeros_like(gbar)
    for i, cl in enumerate(clusters):
        st = stars_map.get(cl)
        if not st:
            continue
        # BCG
        if st.get('bcg_logM') is not None and st.get('bcg_Re_kpc') is not None:
            if st.get('bcg_profile', 'hernquist') in ('hernquist', 'sersic4', 'de_vaucouleurs'):
                g_add[i] += hernquist_accel_cgs(10.0 ** st['bcg_logM'], st['bcg_Re_kpc'], r_kpc[i])
        # ICL (optional)
        if st.get('icl_logM') is not None and st.get('icl_Re_kpc') is not None:
            if st.get('icl_profile', 'hernquist') in ('hernquist', 'sersic4', 'de_vaucouleurs'):
                g_add[i] += hernquist_accel_cgs(10.0 ** st['icl_logM'], st['icl_Re_kpc'], r_kpc[i])
    return gbar + g_add


def fit_global_a0(gbar_cgs: np.ndarray, gtot_log_cgs: np.ndarray, dmax: float,
                  log10_min=-9.5, log10_max=-6.0, n=701) -> Tuple[float, float, np.ndarray]:
    # Grid in a0 (cgs)
    grid = np.logspace(log10_min, log10_max, n)
    loggbar = np.log10(np.clip(gbar_cgs, 1e-99, None))
    target = gtot_log_cgs
    best_a0 = None
    best_rms = float('inf')
    best_pred = None
    for a0 in grid:
        xi = xi_rar_plateau(gbar_cgs, a0, dmax)
        pred = np.log10(xi * gbar_cgs)
        rms = np.sqrt(np.mean((target - pred) ** 2))
        if rms < best_rms:
            best_rms = rms
            best_a0 = a0
            best_pred = pred
    return best_a0, best_rms, best_pred


def compute_supplementary_metrics(r_over_r200: np.ndarray, residuals: np.ndarray) -> dict:
    abs_res = np.abs(residuals)
    frac_0p1 = float(np.mean(abs_res <= 0.1))
    frac_0p2 = float(np.mean(abs_res <= 0.2))
    pos_frac = float(np.mean(residuals > 0))
    inner = residuals[r_over_r200 <= 0.2]
    outer = residuals[r_over_r200 > 0.2]
    med_in = float(np.median(inner)) if inner.size else float('nan')
    med_out = float(np.median(outer)) if outer.size else float('nan')
    # Linear fit residual = a + b*x
    x = r_over_r200
    if x.size >= 2:
        b, a = np.polyfit(x, residuals, 1)
        x_zero = float(-a / b) if b != 0 else float('nan')
    else:
        a = b = x_zero = float('nan')
    return {
        'fraction_within_0p1_dex': frac_0p1,
        'fraction_within_0p2_dex': frac_0p2,
        'positive_fraction': pos_frac,
        'median_inner_le_0p2_dex': med_in,
        'median_outer_gt_0p2_dex': med_out,
        'linreg': {'a': float(a), 'b': float(b), 'x_zero': float(x_zero)},
    }


def make_plots(
    out_images: Path,
    loggbar_total: np.ndarray,
    loggNFW: np.ndarray,
    best_a0: float,
    dmax: float,
    r_over_r200: np.ndarray,
    pred_log: np.ndarray,
):
    out_images.mkdir(parents=True, exist_ok=True)

    # Scatter with GR and RAR lines
    xmin = float(np.nanmin(loggbar_total)) - 0.2
    xmax = float(np.nanmax(loggbar_total)) + 0.2
    xs = np.linspace(xmin, xmax, 400)
    gscan = 10 ** xs
    rar_line = np.log10(xi_rar_plateau(gscan, best_a0, dmax) * gscan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(loggbar_total, loggNFW, s=10, c='tab:blue', alpha=0.7, label='CLASH NFW points')
    ax.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1, label='GR (baryons only)')
    ax.plot(xs, rar_line, 'r-', lw=1.5, label=f'RAR plateau (a0={best_a0:.2e}, Dmax={dmax:.0f})')
    ax.set_xlabel('log10 g_bar [cgs]')
    ax.set_ylabel('log10 g_tot (NFW) [cgs]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_images / 'cluster_rar_scatter_plus.png', dpi=150)
    plt.close(fig)

    # Residuals vs r/R200
    residuals = loggNFW - pred_log
    b, a = np.polyfit(r_over_r200, residuals, 1)
    xline = np.linspace(0, float(np.nanmax(r_over_r200)), 200)
    yline = a + b * xline
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.scatter(r_over_r200, residuals, s=10, c='tab:blue', alpha=0.6)
    ax.plot(xline, yline, 'r-', lw=1.5, label=f'fit: {a:.3f} + {b:.3f} x')
    ax.set_xlabel('r / R200c')
    ax.set_ylabel('Δ log10 g (NFW − RAR)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_images / 'cluster_rar_residuals_vs_r200_plus.png', dpi=150)
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30, color='tab:blue', alpha=0.8)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Δ log10 g (NFW − RAR)')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(out_images / 'cluster_rar_residual_hist_plus.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Cluster RAR post-process + optional stars (BCG/ICL).')
    ap.add_argument('--accept', type=Path, required=False,
                    help='Path to ACCEPT .dat (not used here; kept for interface symmetry).')
    ap.add_argument('--points', type=Path, required=True,
                    help='Path to existing cluster_rar_points.csv from the base pipeline run.')
    ap.add_argument('--diagnostics', type=Path, required=True,
                    help='Path to diagnostics.csv to read R200c per cluster.')
    ap.add_argument('--stars', type=Path, default=None,
                    help='Optional stars CSV with columns: cluster,log10Mstar_BCG,Re_kpc,[log10Mstar_ICL,Re_ICL_kpc].')
    ap.add_argument('--outdir', type=Path, required=True,
                    help='Directory to write CSV/JSON outputs to.')
    ap.add_argument('--images', type=Path, required=True,
                    help='Directory to write plot images to.')
    ap.add_argument('--dmax', type=float, default=50.0, help='RAR plateau cap Dmax.')
    ap.add_argument('--a0_log10_min', type=float, default=-9.5)
    ap.add_argument('--a0_log10_max', type=float, default=-6.0)
    ap.add_argument('--a0_grid_n', type=int, default=701)
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    r200_map = load_r200_by_cluster(args.diagnostics)
    z, r_kpc, loggbar, loggNFW, _, clusters = load_points(args.points)

    stars_map = load_stars_csv(args.stars)
    gbar_total = compute_with_optional_stars(clusters, r_kpc, loggbar, stars_map)

    # Fit a0 on the full set
    best_a0, best_rms, pred_log = fit_global_a0(
        gbar_total, loggNFW, args.dmax, args.a0_log10_min, args.a0_log10_max, args.a0_grid_n
    )

    # r/R200 per point
    r_over_r200 = np.array([rk / r200_map[cl] for rk, cl in zip(r_kpc, clusters)], dtype=float)

    # Supplementary metrics
    residuals = loggNFW - pred_log
    supp = compute_supplementary_metrics(r_over_r200, residuals)

    # Write points CSV (plus)
    pts_out = outdir / 'cluster_rar_points_plus.csv'
    with pts_out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster', 'z', 'r_kpc', 'r_over_r200', 'log10_gbar_plus_cgs', 'log10_gNFWtot_cgs', 'log10_gRAR_plus_cgs'])
        for cl, zi, rk, x, gb, gN, gP in zip(clusters, z, r_kpc, r_over_r200, np.log10(gbar_total), loggNFW, pred_log):
            w.writerow([cl, f"{zi:.6f}", f"{rk:.6f}", f"{x:.6f}", f"{gb:.9f}", f"{gN:.9f}", f"{gP:.9f}"])

    # Write metrics JSON (plus)
    metrics = {
        'counts': int(len(loggNFW)),
        'rar_plateau': {
            'a0_cgs': float(best_a0),
            'Dmax': float(args.dmax),
            'rms_dex': float(best_rms),
        },
        'coverage': supp,
    }
    with (outdir / 'metrics_plus.json').open('w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    make_plots(args.images, np.log10(gbar_total), loggNFW, best_a0, args.dmax, r_over_r200, pred_log)

    # Console summary
    print(f"Fitted a0 (RAR plateau): {best_a0:.3e} cgs; RMS = {best_rms:.3f} dex")
    print(f"Coverage: ±0.1 dex = {supp['fraction_within_0p1_dex']*100:.1f}%, ±0.2 dex = {supp['fraction_within_0p2_dex']*100:.1f}%")
    print(f"Radial trend: Δlog g ≈ {supp['linreg']['a']:.3f} + {supp['linreg']['b']:.2f} (r/R200), zero at x ≈ {supp['linreg']['x_zero']:.3f}")


if __name__ == '__main__':
    main()
