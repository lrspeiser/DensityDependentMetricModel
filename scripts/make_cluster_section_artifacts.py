#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cluster_section_artifacts.py

Consume the cluster RAR CSVs produced by scripts/cluster_rar_pipeline.py and emit
paper-ready metrics and figures:

Outputs (default locations):
- results/cluster_rar/cluster_section_metrics.json
- results/cluster_rar/cluster_section_per_cluster.csv
- results/cluster_rar/cluster_section_jackknife.csv
- images/cluster_rar/cluster_rar_scatter.(png|svg)
- images/cluster_rar/cluster_rar_residual_hist.(png|svg)
- images/cluster_rar/cluster_rar_residuals_vs_r200.(png|svg)

Requires only numpy, matplotlib, and the standard library.

Usage example:
    python make_cluster_section_artifacts.py \
      --points results/cluster_rar/cluster_rar_points.csv \
      --diag   results/cluster_rar/diagnostics.csv \
      --outdir results/cluster_rar \
      --imgdir images/cluster_rar \
      --dmax 50 \
      --refit-a0 \
      --bootstrap 200
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def xi_rar_plateau(gb: np.ndarray, a0: float, dmax: float = 50.0) -> np.ndarray:
    gb_safe = np.maximum(gb, 1e-99)
    xi = 0.5 + np.sqrt(0.25 + (a0 / gb_safe))
    if dmax is not None and np.isfinite(dmax):
        xi = np.minimum(xi, dmax)
    return xi


def g_rar(gb: np.ndarray, a0: float, dmax: float = 50.0) -> np.ndarray:
    return xi_rar_plateau(gb, a0, dmax=dmax) * gb


def fit_a0(gb: np.ndarray, gt: np.ndarray, dmax: float = 50.0,
           a0_min: float = 1e-12, a0_max: float = 1e-6, n_grid: int = 6000) -> Tuple[float, float]:
    mask = np.isfinite(gb) & np.isfinite(gt) & (gb > 0) & (gt > 0)
    if not np.any(mask):
        return float('nan'), float('nan')
    gb = gb[mask]
    gt = gt[mask]
    a0s = np.logspace(math.log10(a0_min), math.log10(a0_max), n_grid)
    best, best_rms = None, 1e9
    lg_gt = np.log10(gt)
    for a0 in a0s:
        pred = g_rar(gb, a0, dmax=dmax)
        rms = float(np.sqrt(np.mean((lg_gt - np.log10(pred))**2)))
        if rms < best_rms:
            best_rms, best = rms, a0
    return best if best is not None else float('nan'), best_rms


def load_points(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    # Expect header: cluster,z,r_kpc,log10_gbar_cgs,log10_gNFWtot_cgs,log10_gGR_cgs,log10_gRAR_cgs
    data_by_cluster: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row['cluster']
            data_by_cluster[c]['z'].append(float(row['z']))
            data_by_cluster[c]['r_kpc'].append(float(row['r_kpc']))
            data_by_cluster[c]['lg_gb'].append(float(row['log10_gbar_cgs']))
            data_by_cluster[c]['lg_gt'].append(float(row['log10_gNFWtot_cgs']))
            if 'log10_gGR_cgs' in row:
                data_by_cluster[c]['lg_ggr'].append(float(row['log10_gGR_cgs']))
            if 'log10_gRAR_cgs' in row and row['log10_gRAR_cgs'] not in (None, ''):
                try:
                    data_by_cluster[c]['lg_grar_csv'].append(float(row['log10_gRAR_cgs']))
                except Exception:
                    pass
    # convert to numpy
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c, d in data_by_cluster.items():
        out[c] = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    return out


def load_diag(path: str) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            info[row['cluster']] = {
                'z': float(row['z']),
                'r200c_kpc': float(row['r200c_kpc']) if row['r200c_kpc'] else float('nan'),
                'fgas_R200': float(row['fgas_R200']) if row['fgas_R200'] else float('nan'),
                'n_used': float(row['n_used']) if row['n_used'] else float('nan'),
            }
    return info


def compute_global_metrics(points_by_cluster: Dict[str, Dict[str, np.ndarray]],
                           r200_by_cluster: Dict[str, float],
                           dmax: float,
                           refit_a0: bool,
                           bootstrap: int) -> Tuple[dict, dict, List[dict]]:
    # Join arrays
    clusters = list(points_by_cluster.keys())
    all_lg_gb = np.concatenate([points_by_cluster[c]['lg_gb'] for c in clusters])
    all_lg_gt = np.concatenate([points_by_cluster[c]['lg_gt'] for c in clusters])
    all_gb = 10**all_lg_gb
    all_gt = 10**all_lg_gt
    all_lg_gr = np.concatenate([points_by_cluster[c]['lg_ggr'] for c in clusters if 'lg_ggr' in points_by_cluster[c]])

    if refit_a0:
        a0, rms_rar = fit_a0(all_gb, all_gt, dmax=dmax)
        lg_grar = np.log10(g_rar(all_gb, a0, dmax=dmax))
    else:
        # use CSV rar if present; otherwise fit
        if all('lg_grar_csv' in points_by_cluster[c] for c in clusters):
            lg_grar = np.concatenate([points_by_cluster[c]['lg_grar_csv'] for c in clusters])
            a0, rms_rar = float('nan'), float(np.sqrt(np.mean((all_lg_gt - lg_grar)**2)))
        else:
            a0, rms_rar = fit_a0(all_gb, all_gt, dmax=dmax)
            lg_grar = np.log10(g_rar(all_gb, a0, dmax=dmax))

    # Jackknife for a0
    jk_rows: List[dict] = []
    a0_vals = []
    for leave in clusters:
        mask = np.ones_like(all_gb, dtype=bool)
        # build mask to exclude cluster 'leave'
        cnt = 0
        offset = 0
        for c in clusters:
            n = len(points_by_cluster[c]['lg_gb'])
            if c == leave:
                mask[offset:offset+n] = False
            offset += n
        a0_j, rms_j = fit_a0(all_gb[mask], all_gt[mask], dmax=dmax)
        jk_rows.append({'omit_cluster': leave, 'a0_cgs': a0_j, 'rms_dex': rms_j})
        a0_vals.append(a0_j)
    a0_vals = np.array(a0_vals, dtype=float) if a0_vals else np.array([a0])

    # Bootstrap RMS for RAR
    bs_rms = []
    rng = np.random.default_rng(12345)
    idx_all = np.arange(len(all_gb))
    B = int(max(0, bootstrap))
    for _ in range(B):
        samp = rng.choice(idx_all, size=idx_all.size, replace=True)
        a0_b, rms_b = fit_a0(all_gb[samp], all_gt[samp], dmax=dmax)
        bs_rms.append(rms_b)
    bs_rms = np.array(bs_rms, dtype=float) if B > 0 else np.array([])

    # Per-cluster RMS/bias
    per_cluster_rows = []
    offset = 0
    for c in clusters:
        n = len(points_by_cluster[c]['lg_gb'])
        lg_gb_c = all_lg_gb[offset:offset+n]
        lg_gt_c = all_lg_gt[offset:offset+n]
        lg_gr_c = all_lg_gr[offset:offset+n]
        lg_grar_c = lg_grar[offset:offset+n]
        offset += n
        rms_gr = float(np.sqrt(np.mean((lg_gt_c - lg_gr_c)**2)))
        rms_rar_c = float(np.sqrt(np.mean((lg_gt_c - lg_grar_c)**2)))
        med_res = float(np.median(lg_gt_c - lg_grar_c))
        bias = float(np.mean(lg_gt_c - lg_grar_c))
        per_cluster_rows.append({
            'cluster': c,
            'n_points': int(n),
            'rms_rar_dex': rms_rar_c,
            'rms_gr_dex': rms_gr,
            'median_residual_dex': med_res,
            'mean_residual_dex': bias,
        })

    # Global metrics
    metrics = {
        'counts': int(len(all_gb)),
        'rar_plateau': {
            'a0_cgs': float(a0),
            'Dmax': float(dmax),
            'rms_dex': float(rms_rar),
            'jackknife': {
                'mean_a0_cgs': float(np.nanmean(a0_vals)) if a0_vals.size else float('nan'),
                'std_a0_cgs': float(np.nanstd(a0_vals, ddof=1)) if a0_vals.size > 1 else float('nan'),
                'n': int(len(jk_rows)),
            },
            'bootstrap_rms': {
                'mean_rms_dex': float(np.mean(bs_rms)) if bs_rms.size else float('nan'),
                'std_rms_dex': float(np.std(bs_rms, ddof=1)) if bs_rms.size > 1 else float('nan'),
                'n': int(bs_rms.size),
            }
        },
        'gr': {
            'rms_dex': float(np.sqrt(np.mean((all_lg_gt - all_lg_gr)**2))),
            'median_residual_dex': float(np.median(all_lg_gt - all_lg_gr)),
            'mean_residual_dex': float(np.mean(all_lg_gt - all_lg_gr)),
        }
    }

    # Residuals and radii for plots
    # Need r/R200; assemble from per-cluster r200c_kpc
    all_r_kpc = np.concatenate([points_by_cluster[c]['r_kpc'] for c in clusters])
    r200_vec = np.concatenate([np.full_like(points_by_cluster[c]['r_kpc'], r200_by_cluster.get(c, np.nan)) for c in clusters])
    residuals = (all_lg_gt - lg_grar)
    aux = {
        'lg_gb': all_lg_gb,
        'lg_gt': all_lg_gt,
        'lg_grar': lg_grar,
        'lg_gr': all_lg_gr,
        'r_over_r200': all_r_kpc / np.maximum(r200_vec, 1e-9),
        'residuals': residuals,
    }
    return metrics, aux, per_cluster_rows, jk_rows


def plot_scatter(aux: dict, imgdir: str):
    os.makedirs(imgdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.scatter(aux['lg_gb'], aux['lg_gt'], s=8, alpha=0.5, label='CLASH NFW (tot)')
    # GR (y=x)
    xmin, xmax = float(np.nanmin(aux['lg_gb'])), float(np.nanmax(aux['lg_gb']))
    x = np.linspace(xmin, xmax, 300)
    ax.plot(x, x, color='k', lw=1.8, label='GR (baryons)')
    # RAR (from aux)
    ax.plot(aux['lg_gb'], aux['lg_grar'], '.', ms=1, color='crimson', alpha=0.25, label='RAR pred (pts)')
    ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}$ [cgs]')
    ax.set_ylabel(r'$\log_{10}\,g_{\rm tot}$ [cgs]')
    ax.set_title('Cluster RAR: CLASH totals vs GR and RAR')
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(imgdir, 'cluster_rar_scatter.png'))
    fig.savefig(os.path.join(imgdir, 'cluster_rar_scatter.svg'))


def plot_residual_hist(aux: dict, imgdir: str):
    os.makedirs(imgdir, exist_ok=True)
    res = aux['residuals']
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.hist(res[np.isfinite(res)], bins=30, color='steelblue', alpha=0.8)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel(r'$\Delta \log_{10} g \equiv \log g_{\rm NFW}-\log g_{\rm RAR}$')
    ax.set_ylabel('Counts')
    ax.set_title('Residuals histogram (RAR)')
    fig.tight_layout()
    fig.savefig(os.path.join(imgdir, 'cluster_rar_residual_hist.png'))
    fig.savefig(os.path.join(imgdir, 'cluster_rar_residual_hist.svg'))


def plot_residuals_vs_r200(aux: dict, imgdir: str):
    os.makedirs(imgdir, exist_ok=True)
    rrat = aux['r_over_r200']
    res = aux['residuals']
    m = np.isfinite(rrat) & np.isfinite(res)
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.scatter(rrat[m], res[m], s=8, color='tab:gray', alpha=0.5)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel(r'$r/R_{200c}$')
    ax.set_ylabel(r'$\log g_{\rm NFW} - \log g_{\rm RAR}$')
    ax.set_title('Residuals vs radius')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(imgdir, 'cluster_rar_residuals_vs_r200.png'))
    fig.savefig(os.path.join(imgdir, 'cluster_rar_residuals_vs_r200.svg'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', required=True, help='results/cluster_rar/cluster_rar_points.csv')
    ap.add_argument('--diag', required=True, help='results/cluster_rar/diagnostics.csv')
    ap.add_argument('--outdir', default='results/cluster_rar')
    ap.add_argument('--imgdir', default='images/cluster_rar')
    ap.add_argument('--dmax', type=float, default=50.0)
    ap.add_argument('--refit-a0', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=200)
    args = ap.parse_args()

    pts = load_points(args.points)
    diag = load_diag(args.diag)
    r200_map = {c: diag.get(c, {}).get('r200c_kpc', float('nan')) for c in pts.keys()}

    metrics, aux, per_cluster_rows, jk_rows = compute_global_metrics(
        pts, r200_map, dmax=args.dmax, refit_a0=args.refit_a0, bootstrap=args.bootstrap
    )

    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.outdir, 'cluster_section_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.outdir, 'cluster_section_per_cluster.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_cluster_rows[0].keys()))
        w.writeheader(); w.writerows(per_cluster_rows)

    with open(os.path.join(args.outdir, 'cluster_section_jackknife.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jk_rows[0].keys()))
        w.writeheader(); w.writerows(jk_rows)

    plot_scatter(aux, args.imgdir)
    plot_residual_hist(aux, args.imgdir)
    plot_residuals_vs_r200(aux, args.imgdir)

    # Console summary
    a0 = metrics['rar_plateau']['a0_cgs']
    rms_rar = metrics['rar_plateau']['rms_dex']
    rms_gr = metrics['gr']['rms_dex']
    print(f"RAR a0 = {a0:.3e} cgs, RMS={rms_rar:.3f} dex; GR RMS={rms_gr:.3f} dex; N={metrics['counts']}")


if __name__ == '__main__':
    main()
