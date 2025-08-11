#!/usr/bin/env python3
"""
compare_models_accuracy.py

Phase 1: Compute empirical accuracy metrics and plots for SPARC galaxies using
baryons-only GR predictions derived from rotmod components (V_model^2 = V_gas^2 + V_disk^2 + V_bulge^2).

Outputs:
- data/accuracy_leaderboard_sparc.csv (per galaxy metrics for full/outer/inner)
- images/accuracy_sparc/<GAL>.png (obs vs model and residuals)
- docs/accuracy_summary_sparc.md (aggregate stats)

Notes:
- Outer disk default: R >= 0.7 * Rmax. This can be changed via --outer-frac.
- This is a first step focusing on GR (no halo). We will extend to GR+halo and TFR when
  per-point model predictions are available or computed.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib default style tweaks (non-invasive)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

logger = logging.getLogger("compare_models_accuracy")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_rotmod(rotmod_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        rotmod_path,
        sep='\s+',
        comment='#',
        names=['R_kpc', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'],
        engine='python'
    )
    # Drop rows with NaNs in essential columns
    keep = df[['R_kpc', 'V_obs', 'e_V_obs']].apply(np.isfinite).all(axis=1)
    df = df.loc[keep].reset_index(drop=True)
    return df


def compute_baryons_only_curve(df: pd.DataFrame) -> np.ndarray:
    vg = np.nan_to_num(df['V_gas'].values, nan=0.0)
    vd = np.nan_to_num(df['V_disk'].values, nan=0.0)
    vb = np.nan_to_num(df['V_bulge'].values, nan=0.0)
    v_model = np.sqrt(np.maximum(vg**2 + vd**2 + vb**2, 0.0))
    return v_model


def region_masks(R: np.ndarray, outer_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(R) == 0:
        z = np.zeros(0, dtype=bool)
        return z, z, z
    rmax = np.nanmax(R)
    thr = outer_frac * rmax
    mask_outer = R >= thr
    mask_inner = R < thr
    mask_all = np.isfinite(R)
    return mask_all, mask_outer, mask_inner


def metrics_from_residuals(resid: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    m = {}
    if resid.size == 0:
        return {
            'n': 0,
            'rmse': np.nan,
            'mae': np.nan,
            'chi2': np.nan,
            'chi2_red': np.nan,
            'cover_1sigma': np.nan,
            'median_bias': np.nan,
        }
    w = 1.0 / np.maximum(sigma, 1e-9)
    w2 = 1.0 / np.maximum(sigma**2, 1e-18)
    m['n'] = resid.size
    m['rmse'] = float(np.sqrt(np.mean(resid**2)))
    m['mae'] = float(np.mean(np.abs(resid)))
    m['chi2'] = float(np.sum((resid / np.maximum(sigma, 1e-9))**2))
    dof = max(resid.size - 0, 1)  # no free params counted here for baryons-only diagnostic
    m['chi2_red'] = float(m['chi2'] / dof)
    m['cover_1sigma'] = float(np.mean(np.abs(resid) <= sigma))
    m['median_bias'] = float(np.median(resid))
    return m


def plot_obs_vs_model(gal: str, df: pd.DataFrame, v_model: np.ndarray, out_png: Path, outer_frac: float):
    R = df['R_kpc'].values
    v = df['V_obs'].values
    ev = df['e_V_obs'].values

    mask_all, mask_outer, mask_inner = region_masks(R, outer_frac)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, height_ratios=[2.0, 1.0])

    # Top: rotation curve
    ax1.errorbar(R, v, yerr=ev, fmt='o', color='tab:blue', ms=3, lw=0.8, alpha=0.9, label='Observed')
    ax1.plot(R, v_model, '-', color='tab:orange', lw=2.0, label='GR (baryons-only)')
    ax1.set_ylabel('v (km/s)')
    ax1.set_title(f'{gal} — Observed vs GR (baryons-only)')
    ax1.legend(loc='best')

    # Bottom: residuals
    resid = v - v_model
    ax2.axhline(0.0, color='k', lw=1.0, alpha=0.6)
    ax2.fill_between(R, -ev, ev, color='gray', alpha=0.15, step='mid', label='±1σ obs')
    ax2.plot(R, resid, 'o-', color='tab:purple', ms=3, lw=0.8, alpha=0.9, label='Residual (obs - model)')
    # Outer region shading
    rmax = np.nanmax(R) if len(R) else np.nan
    thr = outer_frac * rmax if np.isfinite(rmax) else np.nan
    if np.isfinite(thr):
        ax2.axvspan(thr, rmax, color='tab:green', alpha=0.08, label=f'Outer (≥ {outer_frac:.0%} Rmax)')
    ax2.set_xlabel('R (kpc)')
    ax2.set_ylabel('Δv (km/s)')
    ax2.legend(loc='best')

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def process_sparc(sparc_dir: Path, outer_frac: float, limit: int = 0) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rows = []
    per_gal_metrics: Dict[str, Dict[str, float]] = {}
    rotmods = sorted(sparc_dir.glob('*_rotmod.dat'))
    if limit > 0:
        rotmods = rotmods[:limit]

    images_dir = Path('images') / 'accuracy_sparc'
    images_dir.mkdir(parents=True, exist_ok=True)

    for rotmod in rotmods:
        gal = rotmod.name.replace('_rotmod.dat', '')
        try:
            df = load_rotmod(rotmod)
            if df.empty:
                logger.warning(f"{gal}: rotmod empty or unreadable, skipping")
                continue
            v_model = compute_baryons_only_curve(df)
            R = df['R_kpc'].values
            v = df['V_obs'].values
            ev = df['e_V_obs'].values
            resid = v - v_model
            mask_all, mask_outer, mask_inner = region_masks(R, outer_frac)

            m_all = metrics_from_residuals(resid[mask_all], ev[mask_all])
            m_outer = metrics_from_residuals(resid[mask_outer], ev[mask_outer])
            m_inner = metrics_from_residuals(resid[mask_inner], ev[mask_inner])

            # Record
            row = {
                'galaxy_id': gal,
                'model': 'GR_baryons_only',
                'n_all': m_all['n'], 'rmse_all': m_all['rmse'], 'mae_all': m_all['mae'], 'chi2_all': m_all['chi2'], 'chi2_red_all': m_all['chi2_red'], 'cover1s_all': m_all['cover_1sigma'], 'median_bias_all': m_all['median_bias'],
                'n_outer': m_outer['n'], 'rmse_outer': m_outer['rmse'], 'mae_outer': m_outer['mae'], 'chi2_outer': m_outer['chi2'], 'chi2_red_outer': m_outer['chi2_red'], 'cover1s_outer': m_outer['cover_1sigma'], 'median_bias_outer': m_outer['median_bias'],
                'n_inner': m_inner['n'], 'rmse_inner': m_inner['rmse'], 'mae_inner': m_inner['mae'], 'chi2_inner': m_inner['chi2'], 'chi2_red_inner': m_inner['chi2_red'], 'cover1s_inner': m_inner['cover_1sigma'], 'median_bias_inner': m_inner['median_bias'],
                'Rmax_kpc': float(np.nanmax(R)) if len(R) else np.nan,
            }
            rows.append(row)

            # Plot
            out_png = images_dir / f'{gal}.png'
            plot_obs_vs_model(gal, df, v_model, out_png, outer_frac)

            per_gal_metrics[gal] = {
                'outer_frac': outer_frac,
                **{f'all_{k}': v for k, v in m_all.items()},
                **{f'outer_{k}': v for k, v in m_outer.items()},
                **{f'inner_{k}': v for k, v in m_inner.items()},
            }
            logger.info(f"{gal}: RMSE_all={m_all['rmse']:.2f} km/s, RMSE_outer={m_outer['rmse']:.2f} km/s")
        except Exception as e:
            logger.error(f"Error processing {gal}: {e}", exc_info=True)
            continue

    df_leader = pd.DataFrame(rows)
    return df_leader, per_gal_metrics


def write_summary(df_leader: pd.DataFrame, per_gal_metrics: Dict[str, Dict[str, float]], outer_frac: float, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_leader.to_csv(out_csv, index=False)

    # Aggregate summaries
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f"# SPARC Accuracy Summary (GR baryons-only)\n\n")
        f.write(f"Outer region defined as R >= {outer_frac:.0%} Rmax.\n\n")
        if len(df_leader) == 0:
            f.write("No galaxies processed.\n")
            return
        def agg(col):
            return df_leader[col].replace([np.inf, -np.inf], np.nan).dropna().values
        rmse_all = agg('rmse_all'); rmse_outer = agg('rmse_outer'); rmse_inner = agg('rmse_inner')
        mae_all = agg('mae_all'); mae_outer = agg('mae_outer'); mae_inner = agg('mae_inner')
        chi2r_all = agg('chi2_red_all'); chi2r_outer = agg('chi2_red_outer'); chi2r_inner = agg('chi2_red_inner')
        f.write("Aggregate RMSE (km/s):\n")
        f.write(f"- All:   mean={np.nanmean(rmse_all):.2f}, median={np.nanmedian(rmse_all):.2f}\n")
        f.write(f"- Outer: mean={np.nanmean(rmse_outer):.2f}, median={np.nanmedian(rmse_outer):.2f}\n")
        f.write(f"- Inner: mean={np.nanmean(rmse_inner):.2f}, median={np.nanmedian(rmse_inner):.2f}\n\n")
        f.write("Aggregate MAE (km/s):\n")
        f.write(f"- All:   mean={np.nanmean(mae_all):.2f}, median={np.nanmedian(mae_all):.2f}\n")
        f.write(f"- Outer: mean={np.nanmean(mae_outer):.2f}, median={np.nanmedian(mae_outer):.2f}\n")
        f.write(f"- Inner: mean={np.nanmean(mae_inner):.2f}, median={np.nanmedian(mae_inner):.2f}\n\n")
        f.write("Aggregate reduced chi^2:\n")
        f.write(f"- All:   mean={np.nanmean(chi2r_all):.2f}, median={np.nanmedian(chi2r_all):.2f}\n")
        f.write(f"- Outer: mean={np.nanmean(chi2r_outer):.2f}, median={np.nanmedian(chi2r_outer):.2f}\n")
        f.write(f"- Inner: mean={np.nanmean(chi2r_inner):.2f}, median={np.nanmedian(chi2r_inner):.2f}\n")


def main():
    ap = argparse.ArgumentParser(description="Compare empirical accuracy using SPARC rotmod (GR baryons-only)")
    ap.add_argument('--sparc-dir', type=str, default='external_data/Rotmod_LTG', help='Directory with *_rotmod.dat files')
    ap.add_argument('--outer-frac', type=float, default=0.7, help='Threshold as fraction of Rmax for outer region')
    ap.add_argument('--limit', type=int, default=0, help='Process only first N galaxies (for quick test)')
    args = ap.parse_args()

    sparc_dir = Path(args.sparc_dir)
    out_csv = Path('data') / 'accuracy_leaderboard_sparc.csv'
    out_md = Path('docs') / 'accuracy_summary_sparc.md'

    if not sparc_dir.exists():
        logger.error(f"SPARC directory not found: {sparc_dir}")
        return

    df_leader, per_gal_metrics = process_sparc(sparc_dir, args.outer_frac, limit=args.limit)
    write_summary(df_leader, per_gal_metrics, args.outer_frac, out_csv, out_md)
    logger.info(f"Wrote {out_csv} and {out_md}")
    logger.info("Per-galaxy plots saved under images/accuracy_sparc/")


if __name__ == '__main__':
    main()

