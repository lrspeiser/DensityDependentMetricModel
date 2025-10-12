#!/usr/bin/env python3
"""
scan_params_sn_bao.py — grid scan over (eta, p, kappa) to summarize SN (full-cov anchored) and BAO rd metrics.

Writes: tariff/results/sn_bao_grid.json

Usage:
  python tariff/scripts/scan_params_sn_bao.py \
      --pantheon external_data/pantheon/Pantheon+SH0ES.dat \
      --bao tariff/data/bao_compilation.csv \
      --etas 2.0,3.0,4.0 --ps 1.2,1.5,1.8 --kappa0 1e-5 --ksteps 7 --kspan 0.3 \
      --sn-red-target 1.0 --sn-red-tol 0.2 --mode frw_overlay

Notes:
- sigma_int_mag is set to 0.0 so the Pantheon+ STAT+SYS covariance is the sole
  error model when available.
- Uses overlay_hubble_unified and bao_shape_overlay to reuse existing code paths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np

# Import relative to repo root or script invocation
try:
    from tariff.analysis_unified_gate import overlay_hubble_unified, bao_shape_overlay, GateParams
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from analysis_unified_gate import overlay_hubble_unified, bao_shape_overlay, GateParams


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def run_grid(pantheon_path: str, bao_csv_path: str, etas: list[float], ps: list[float], kappa0: float, ksteps: int, kspan: float, mode: str) -> list[dict]:
    rows = []
    factors = 10.0 ** np.linspace(-kspan, kspan, int(ksteps))
    for eta in etas:
        for pval in ps:
            for f in factors:
                k = float(kappa0 * f)
                params = GateParams(eta=eta, p=pval, q=0.0, rho_star_evcm3=0.26, kappa_per_Mpc=k, sigma=0.0, enable_backreaction=False)
                hubble, (r_grid, z_of_r, tau_of_r) = overlay_hubble_unified(pantheon_path, params, mode=mode, sigma_int_mag=0.0)
                bao = bao_shape_overlay(z_of_r, r_grid, bao_csv_path, mode=mode)
                rows.append({
                    'eta': eta,
                    'p': pval,
                    'kappa_per_Mpc': k,
                    'sn_chi2': hubble.get('chi2'),
                    'sn_red_chi2': hubble.get('red_chi2'),
                    'sn_dof': hubble.get('dof'),
                    'rd_best_Mpc': bao.get('rd_best_Mpc'),
                    'bao_chi2': bao.get('bao_chi2'),
                    'bao_red_chi2': bao.get('bao_red_chi2'),
                    'bao_dof': bao.get('bao_dof'),
                })
    return rows


def select_best(rows: list[dict], sn_target: float, sn_tol: float) -> dict | None:
    best = None
    for r in rows:
        red = r.get('bao_red_chi2')
        snr = r.get('sn_red_chi2')
        if not isinstance(snr, (int, float)) or not math.isfinite(snr):
            continue
        if abs(snr - sn_target) > sn_tol:
            continue
        if isinstance(red, (int, float)) and math.isfinite(red):
            if (best is None) or (red < best.get('bao_red_chi2', math.inf)):
                best = r
    return best


def main():
    ap = argparse.ArgumentParser(description='Grid scan over (eta, p, kappa) for SN+BAO')
    ap.add_argument('--pantheon', type=str, default=os.path.join('external_data', 'pantheon', 'Pantheon+SH0ES.dat'))
    ap.add_argument('--bao', type=str, default=os.path.join('tariff', 'data', 'bao_compilation.csv'))
    ap.add_argument('--etas', type=str, default='2.0,3.0,4.0')
    ap.add_argument('--ps', type=str, default='1.2,1.5,1.8')
    ap.add_argument('--kappa0', type=float, default=1e-5)
    ap.add_argument('--ksteps', type=int, default=7)
    ap.add_argument('--kspan', type=float, default=0.3)
    ap.add_argument('--sn-red-target', type=float, default=1.0)
    ap.add_argument('--sn-red-tol', type=float, default=0.2)
    ap.add_argument('--mode', type=str, default='frw_overlay', choices=['frw_overlay', 'tariff_only'])
    args = ap.parse_args()

    etas = parse_list(args.etas)
    ps = parse_list(args.ps)

    rows = run_grid(args.pantheon, args.bao, etas, ps, args.kappa0, args.ksteps, args.kspan, args.mode)
    best = select_best(rows, args.__dict__['sn_red_target'], args.__dict__['sn_red_tol'])

    out = {
        'grid': rows,
        'best_filtered_by_sn': best,
        'sn_red_target': args.__dict__['sn_red_target'],
        'sn_red_tol': args.__dict__['sn_red_tol'],
        'mode': args.mode,
    }
    out_path = Path(__file__).resolve().parents[1] / 'results' / 'sn_bao_grid.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()