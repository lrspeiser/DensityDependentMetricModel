#!/usr/bin/env python3
"""
build_bao_compilation.py — Create a BAO CSV in the expected schema from simple key-value inputs.

Usage:
  python tariff/scripts/build_bao_compilation.py --out tariff/data/bao_compilation.csv --template boss_dr12

Templates:
- boss_dr12: three anisotropic points (z≈0.38,0.51,0.61) with D_M/rd and D_H/rd and diagonal errors.

Notes:
- This script is a convenience to get you started; for publication use the latest curated compilations
  (e.g., BOSS/eBOSS/DESI releases) and include per-point correlations when available.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd


def boss_dr12_default() -> pd.DataFrame:
    # Values adapted from Alam et al. 2017 (illustrative; please replace with your curated table)
    z = np.array([0.38, 0.51, 0.61], float)
    DM_over_rd = np.array([1512.0, 1975.0, 2307.0], float)  # illustrative
    DH_over_rd = np.array([81.2, 90.9, 98.9], float)        # illustrative
    D_M_err = np.array([24.0, 30.0, 37.0], float)
    D_H_err = np.array([2.4, 2.3, 2.3], float)
    # No per-point correlation provided in this quick template; set rho=0
    rho = np.zeros_like(z)
    df = pd.DataFrame({'z': z, 'D_M_over_rd': DM_over_rd, 'D_H_over_rd': DH_over_rd,
                       'D_M_err': D_M_err, 'D_H_err': D_H_err, 'rho': rho})
    return df


def main():
    ap = argparse.ArgumentParser(description='Build a BAO compilation CSV in the expected schema.')
    ap.add_argument('--out', type=str, default=os.path.join('tariff', 'data', 'bao_compilation.csv'))
    ap.add_argument('--template', type=str, default='boss_dr12', choices=['boss_dr12'])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.template == 'boss_dr12':
        df = boss_dr12_default()
    else:
        print(f"[ERROR] Unknown template {args.template}")
        sys.exit(1)

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with columns: {list(df.columns)} and N={len(df)}")


if __name__ == '__main__':
    main()
