#!/usr/bin/env python3
"""
scan_kappa_bao.py — quick kappa scan to summarize SN (full-cov anchored) and BAO rd metrics.

Writes: tariff/results/bao_kappa_scan.json

Usage:
  python tariff/scripts/scan_kappa_bao.py \
      --pantheon external_data/pantheon/Pantheon+SH0ES.dat \
      --bao tariff/data/bao_compilation.csv \
      --kappa0 1e-5 --factors 13 --span 0.6 --mode frw_overlay

Notes:
- Uses the overlay_hubble_unified and bao_shape_overlay functions to avoid
  overwriting the main metrics file for each scan step.
- sigma_int_mag is set to 0.0 so the Pantheon+ STAT+SYS covariance is the sole
  error model when available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np

# Local imports (script is inside tariff/)
from tariff.analysis_unified_gate import overlay_hubble_unified, bao_shape_overlay, GateParams


def make_factors(n: int, span: float) -> np.ndarray:
    """Geometric factors across +/- span in log10 space: 10^linspace(-span, +span, n)."""
    return 10.0 ** np.linspace(-span, span, int(n))


def run_scan(pantheon_path: str, bao_csv_path: str, kappa0: float, n_factors: int, span: float, mode: str) -> dict:
    # Default GateParams matching example entry point
    base = GateParams(eta=3.0, p=1.5, q=0.0, rho_star_evcm3=0.26, kappa_per_Mpc=kappa0)
    factors = make_factors(n_factors, span)
    rows = []
    best = None
    for f in factors:
        k = float(kappa0 * f)
        params = GateParams(eta=base.eta, p=base.p, q=base.q, rho_star_evcm3=base.rho_star_evcm3,
                            kappa_per_Mpc=k, sigma=0.0, enable_backreaction=False)
        # One pass overlay + BAO metrics without touching the main JSON
        hubble, (r_grid, z_of_r, tau_of_r) = overlay_hubble_unified(pantheon_path, params, mode=mode, sigma_int_mag=0.0)
        bao = bao_shape_overlay(z_of_r, r_grid, bao_csv_path, mode=mode)
        row = {
            'kappa_per_Mpc': k,
            'sn_chi2': hubble.get('chi2'),
            'sn_red_chi2': hubble.get('red_chi2'),
            'sn_dof': hubble.get('dof'),
            'anchor_delta_mu_mag': hubble.get('anchor_delta_mu_mag'),
            'rd_best_Mpc': bao.get('rd_best_Mpc'),
            'bao_chi2': bao.get('bao_chi2'),
            'bao_red_chi2': bao.get('bao_red_chi2'),
            'bao_dof': bao.get('bao_dof'),
        }
        rows.append(row)
        # Track best by BAO red chi2 (if finite), else by SN red chi2
        key = math.inf
        if isinstance(row['bao_red_chi2'], (int, float)) and math.isfinite(row['bao_red_chi2']):
            key = row['bao_red_chi2']
        elif isinstance(row['sn_red_chi2'], (int, float)) and math.isfinite(row['sn_red_chi2']):
            key = row['sn_red_chi2'] + 1e3  # deprioritize vs BAO
        if best is None or key < best.get('_score', math.inf):
            best = dict(row)
            best['_score'] = key
    return {
        'base_kappa_per_Mpc': kappa0,
        'factors': factors.tolist(),
        'mode': mode,
        'rows': rows,
        'best': {k: v for k, v in (best or {}).items() if k != '_score'}
    }


def main():
    ap = argparse.ArgumentParser(description='Scan kappa for SN+BAO metrics')
    ap.add_argument('--pantheon', type=str, default=os.path.join('external_data', 'pantheon', 'Pantheon+SH0ES.dat'))
    ap.add_argument('--bao', type=str, default=os.path.join('tariff', 'data', 'bao_compilation.csv'))
    ap.add_argument('--kappa0', type=float, default=1e-5)
    ap.add_argument('--factors', type=int, default=13)
    ap.add_argument('--span', type=float, default=0.6, help='log10 span; 0.6 => ~x0.25..x4')
    ap.add_argument('--mode', type=str, default='frw_overlay', choices=['frw_overlay', 'tariff_only'])
    args = ap.parse_args()

    out = run_scan(args.pantheon, args.bao, args.kappa0, args.factors, args.span, args.mode)
    out_path = Path(__file__).resolve().parents[1] / 'results' / 'bao_kappa_scan.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()