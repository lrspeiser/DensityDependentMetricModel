#!/usr/bin/env python3
"""
Check distant Gaia stars against TFR (tidal_band) predicted speeds and the plotted band.

- Loads Gaia DR3 (local cache only), computes binned median and 16–84% bands
- Loads fitted parameters from provided run directories
- Selects at least 10 of the most distant stars
- For each star: reports R_kpc, observed speed v_obs, TFR predicted speed v_tfr(R), and whether v_obs lies within the bin's 16–84% range used in the chart

Usage:
  python scripts/check_tfr_distant_stars.py \
    --tfr_run runs/tidal_band_YYYYMMDD_HHMMSS \
    [--gr_run runs/gr_YYYYMMDD_HHMMSS] \
    [--n N]

Outputs:
  - Writes CSV and Markdown summary into <tfr_run>/analysis/
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# Matplotlib not required; we won't plot here

# Repo imports reusing the plotting utilities to ensure consistency
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
for p in [REPO_ROOT, REPO_ROOT / "core", REPO_ROOT / "tools", REPO_ROOT / "runners"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from tools.plot_rotation_comparison import (
    load_gaia_local_or_fail, process_gaia_data, bin_by_radius,
    default_baryon_params, default_tidal_band_params,
    build_param_dict, _maybe_update_params_from_runs, v_total_kms
)

@dataclass
class StarCheck:
    idx: int
    R_kpc: float
    v_obs: float
    v_tfr: float
    bin_center: float
    v_lo: float
    v_med: float
    v_hi: float
    within_band: bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfr_run", required=True, help="Path to TFR (tidal_band) run directory")
    ap.add_argument("--gr_run", default=None, help="Optional GR run directory to source baryonic params")
    ap.add_argument("--n", type=int, default=10, help="Number of distant stars to check (default: 10)")
    args = ap.parse_args()

    tfr_run = Path(args.tfr_run)
    gr_run = Path(args.gr_run) if args.gr_run else None

    # 1) Load Gaia data (local only) and process
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)

    # 2) Build bin statistics used in plots
    R_kpc = df["R_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs"].to_numpy(dtype=float)
    bins = np.linspace(2.0, 30.0, 29)
    R_centers, v_med, v_lo, v_hi = bin_by_radius(R_kpc, v_obs, bins=bins)

    # 3) Build TFR model parameters from runs (reuse same helpers as plots)
    baryon = default_baryon_params()
    xi = default_tidal_band_params()
    baryon, xi = _maybe_update_params_from_runs(baryon, xi, str(gr_run) if gr_run else None, str(tfr_run))
    params_tfr = build_param_dict(baryon, xi)

    # 4) Select N most distant stars (by R_kpc). If ties/NaNs, handle safely
    order = np.argsort(R_kpc)
    order = order[::-1]  # descending
    # mask finite
    finite_mask = np.isfinite(R_kpc) & np.isfinite(v_obs)
    order = order[finite_mask[order]]
    if order.size == 0:
        raise SystemExit("No finite Gaia stars found.")
    N = max(1, int(args.n))
    sel_idx = order[:N]

    # 5) For each selected star, compute TFR predicted speed and bin band membership
    checks: List[StarCheck] = []
    # TFR evaluator accepts arrays; we'll vectorize for efficiency
    R_sel = R_kpc[sel_idx].astype(float)
    v_tfr_sel = v_total_kms(R_sel, params_tfr, xi_type='tidal_band')

    # Determine bin for each star
    bin_indices = np.digitize(R_sel, bins)  # returns 1..len(bins)
    for j, (i_row, Rj, vj, vtfr) in enumerate(zip(sel_idx, R_sel, v_obs[sel_idx], v_tfr_sel)):
        bi = bin_indices[j]
        # Convert to zero-based index for centers/bands
        k = bi - 1
        if k < 0 or k >= len(R_centers):
            # Out of defined bin range; mark band as NaN
            bc = float('nan'); vlo = float('nan'); vmd = float('nan'); vhi_ = float('nan')
            within = False
        else:
            bc = float(R_centers[k])
            vlo = float(v_lo[k])
            vmd = float(v_med[k])
            vhi_ = float(v_hi[k])
            within = (np.isfinite(vlo) and np.isfinite(vhi_) and (vj >= vlo) and (vj <= vhi_))
        checks.append(StarCheck(
            idx=int(i_row), R_kpc=float(Rj), v_obs=float(vj), v_tfr=float(vtfr),
            bin_center=bc, v_lo=vlo, v_med=vmd, v_hi=vhi_, within_band=bool(within)
        ))

    # 6) Write results into tfr_run/analysis
    out_dir = tfr_run / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "check_tfr_distant_stars.csv"
    md_path = out_dir / "check_tfr_distant_stars.md"

    # CSV
    df_out = pd.DataFrame([{
        "row_index": c.idx,
        "R_kpc": c.R_kpc,
        "v_obs": c.v_obs,
        "v_tfr": c.v_tfr,
        "bin_center": c.bin_center,
        "band_lo": c.v_lo,
        "band_med": c.v_med,
        "band_hi": c.v_hi,
        "in_band": c.within_band,
    } for c in checks])
    df_out.to_csv(csv_path, index=False)

    # Markdown summary
    n_in = int(sum(1 for c in checks if c.within_band))
    n_total = len(checks)
    frac = (n_in / n_total) if n_total else 0.0
    lines: List[str] = []
    lines.append("# TFR distant-star spot check")
    lines.append("")
    lines.append(f"Run: {tfr_run}")
    lines.append(f"Checked top {n_total} most distant stars by R_kpc")
    lines.append(f"Within 16–84% band: {n_in}/{n_total} ({frac:.1%})")
    lines.append("")
    lines.append("Columns: row_index, R_kpc, v_obs, v_tfr, bin_center, band_lo, band_med, band_hi, in_band")
    lines.append("")
    # Show a small table excerpt
    head = df_out.copy()
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        pass
    # Render as a simple markdown table (first 10 rows)
    head = head.head(10)
    lines.append("| row_index | R_kpc | v_obs | v_tfr | bin_center | band_lo | band_med | band_hi | in_band |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for _, r in head.iterrows():
        lines.append(
            f"| {int(r['row_index'])} | {r['R_kpc']:.2f} | {r['v_obs']:.1f} | {r['v_tfr']:.1f} | "
            f"{r['bin_center']:.2f} | {r['band_lo']:.1f} | {r['band_med']:.1f} | {r['band_hi']:.1f} | {'yes' if r['in_band'] else 'no'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Within-band count: {n_in}/{n_total} ({frac:.1%})")


if __name__ == "__main__":
    main()
