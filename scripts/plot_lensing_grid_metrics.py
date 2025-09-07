#!/usr/bin/env python3
"""
plot_lensing_grid_metrics.py

Make simple RMSE vs alpha and RMSE vs zeta (at fixed alpha) plots from the
metrics CSVs produced by scripts/metrics_from_combined.py.

Usage examples:
  python -u scripts/plot_lensing_grid_metrics.py \
    --alpha-metrics results/next_steps/btfr_fix_20250906_lastcross/combined_grid_alpha_only_metrics_by_run.csv \
    --grid-metrics  results/next_steps/btfr_fix_20250906_lastcross/combined_grid_metrics_by_run.csv \
    --out-dir       results/next_steps/btfr_fix_20250906_lastcross/plots

Outputs:
  rmse_vs_alpha.png, rmse_vs_zeta_alpha2.png
"""
from __future__ import annotations
import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

def read_metrics_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def try_float(s: str) -> float:
    try:
        v = float(s)
        return v if math.isfinite(v) else float('nan')
    except Exception:
        return float('nan')


def parse_alpha_from_label(label: str) -> float:
    # Expect forms like 'alpha_2', 'alpha_2p5', 'alpha_1p5' etc.
    m = re.search(r"alpha_([mp\d]+)", label)
    if not m:
        return float('nan')
    token = m.group(1)
    token = token.replace('p', '.')
    token = token.replace('m', '-')
    return try_float(token)


def parse_zeta_from_label(label: str) -> float:
    # Expect 'alpha_2_zeta_...'
    m = re.search(r"zeta_([mp\d]+)", label)
    if not m:
        return float('nan')
    token = m.group(1)
    token = token.replace('p', '.')
    token = token.replace('m', '-')
    return try_float(token)


def rmse_vs_alpha(alpha_rows: List[Dict[str,str]], out_dir: Path) -> None:
    pts: List[Tuple[float, float]] = []
    for r in alpha_rows:
        lab = r.get('run_label', '')
        a = parse_alpha_from_label(lab)
        rmse = try_float(r.get('RMSE_arcsec', 'nan'))
        if math.isfinite(a) and math.isfinite(rmse):
            pts.append((a, rmse))
    pts.sort(key=lambda t: t[0])
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(xs, ys, 'o-', lw=2)
    plt.xlabel('alpha_lens_ph')
    plt.ylabel('RMSE [arcsec]')
    plt.title('RMSE vs alpha (zeta=0, constant)')
    plt.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_dir / 'rmse_vs_alpha.png', dpi=140); plt.close()


def rmse_vs_zeta_at_alpha(grid_rows: List[Dict[str,str]], alpha_target: float, out_dir: Path) -> None:
    pts: List[Tuple[float, float]] = []
    for r in grid_rows:
        lab = r.get('run_label', '')
        a = parse_alpha_from_label(lab)
        if not math.isfinite(a) or abs(a - alpha_target) > 1e-6:
            continue
        z = parse_zeta_from_label(lab)
        rmse = try_float(r.get('RMSE_arcsec', 'nan'))
        if math.isfinite(z) and math.isfinite(rmse):
            pts.append((z, rmse))
    pts.sort(key=lambda t: t[0])
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(xs, ys, 'o-', lw=2)
    plt.xlabel('zeta_env (tapered)')
    plt.ylabel('RMSE [arcsec]')
    plt.title(f'RMSE vs zeta (alpha={alpha_target}, tapered)')
    plt.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_dir / f'rmse_vs_zeta_alpha{alpha_target}.png', dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha-metrics', required=True, help='CSV from alpha-only sweep metrics (combined_grid_alpha_only_metrics_by_run.csv)')
    ap.add_argument('--grid-metrics', required=True, help='CSV from full grid metrics (combined_grid_metrics_by_run.csv)')
    ap.add_argument('--alpha-target', type=float, default=2.0, help='Alpha to slice for zeta plot')
    ap.add_argument('--out-dir', required=True, help='Directory to write plots')
    args = ap.parse_args()

    alpha_rows = read_metrics_csv(args.alpha_metrics)
    grid_rows = read_metrics_csv(args.grid_metrics)
    out_dir = Path(args.out_dir)

    rmse_vs_alpha(alpha_rows, out_dir)
    rmse_vs_zeta_at_alpha(grid_rows, float(args.alpha_target), out_dir)

if __name__ == '__main__':
    main()

