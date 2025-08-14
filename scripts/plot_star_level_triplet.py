#!/usr/bin/env python3
"""
Plot star-level comparison for the N most distant Gaia stars:
- Observed star speeds (points with optional error bars)
- GR prediction at those radii
- TFR (tidal_band) prediction at those radii
- NFW (ΛCDM) prediction at those radii

This uses the same Gaia loader/processors and model evaluators as the existing
rotation overlays, and pulls parameters from your provided run directories.

Usage:
  python scripts/plot_star_level_triplet.py \
    --gr runs/gr_YYYYMMDD_HHMMSS \
    --tfr runs/tidal_band_YYYYMMDD_HHMMSS \
    --nfw runs/nfw_YYYYMMDD_HHMMSS \
    [--n 15] [--rmin 20] [--out images/mw_star_level_triplet.png]
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo path setup
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT, REPO_ROOT / "core", REPO_ROOT / "tools", REPO_ROOT / "scripts"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Import shared helpers and evaluators
from tools.plot_rotation_comparison import (
    load_gaia_local_or_fail, process_gaia_data, default_baryon_params,
    default_tidal_band_params
)
from scripts.plot_mw_overlay_triplet import _params_from_runs, v_total_kms, v_baryon_total_newtonian_kms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", required=True, help="Path to GR run directory")
    ap.add_argument("--tfr", required=True, help="Path to TFR (tidal_band) run directory")
    ap.add_argument("--nfw", required=True, help="Path to NFW run directory")
    ap.add_argument("--n", type=int, default=15, help="Number of most distant stars to plot")
    ap.add_argument("--rmin", type=float, default=0.0, help="Minimum R_kpc for selection (optional)")
    ap.add_argument("--out", default=str(REPO_ROOT / "images" / "mw_star_level_triplet.png"))
    args = ap.parse_args()

    # Load Gaia data (local cache only) and process
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)

    R_all = df["R_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs"].to_numpy(dtype=float)
    e_v = df.get("sigma_v", None)
    e_obs = e_v.to_numpy(dtype=float) if e_v is not None else None

    # Select N most distant stars with optional rmin filter
    finite = np.isfinite(R_all) & np.isfinite(v_obs)
    if args.rmin > 0:
        finite &= (R_all >= float(args.rmin))
    idx_sorted = np.argsort(R_all[finite])[::-1]
    if idx_sorted.size == 0:
        raise SystemExit("No finite Gaia stars after filtering; try lowering --rmin")
    # Map sorted indices back to full array indices
    full_idx = np.nonzero(finite)[0][idx_sorted]
    sel_idx = full_idx[: max(1, int(args.n))]

    R = R_all[sel_idx]
    V = v_obs[sel_idx]
    EV = e_obs[sel_idx] if e_obs is not None else None

    # Load parameters from run dirs
    bary, tfr, nfw_params = _params_from_runs(args.gr, args.tfr, args.nfw)
    params_tfr = dict(bary)
    params_tfr.update(tfr)

    # Compute predictions at star radii
    v_gr = v_baryon_total_newtonian_kms(R, bary)
    v_tfr = v_total_kms(R, params_tfr, xi_type='tidal_band')
    v_nfw = v_total_kms(R, {**bary, **nfw_params}, xi_type='nfw')

    # Plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))

    # Observations with error bars if available
    if EV is not None and np.all(np.isfinite(EV)):
        plt.errorbar(R, V, yerr=EV, fmt='o', ms=5, color='k', alpha=0.8, label='Stars: observed (±σ)')
    else:
        plt.plot(R, V, 'ko', ms=5, alpha=0.8, label='Stars: observed')

    # Model predictions at star radii
    plt.plot(R, v_gr, 'b^', ms=6, label='GR prediction @ star R')
    plt.plot(R, v_nfw, color='#2CA02C', marker='s', linestyle='None', ms=6, label='NFW prediction @ star R')
    plt.plot(R, v_tfr, 'rD', ms=6, label='TFR prediction @ star R')

    # Aesthetics
    plt.xlabel('Galactocentric radius R (kpc)')
    plt.ylabel('Speed v (km/s)')
    plt.title('Most distant Gaia stars: Observed vs GR / NFW / TFR predictions')
    plt.grid(True, alpha=0.3)
    xmin = max(2.0, float(np.nanmin(R) - 1.0))
    xmax = float(np.nanmax(R) + 1.0)
    plt.xlim(xmin, xmax)
    ymax = np.nanmax([np.nanmax(V), np.nanmax(v_gr), np.nanmax(v_tfr), np.nanmax(v_nfw)])
    plt.ylim(0, max(300, float(ymax) + 40))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
