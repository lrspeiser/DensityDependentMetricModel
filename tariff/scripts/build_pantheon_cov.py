#!/usr/bin/env python3
"""
build_pantheon_cov.py — Build or validate Pantheon+ full covariance next to the data table.

Usage:
  python tariff/scripts/build_pantheon_cov.py \
    --base-dir external_data/pantheon \
    --basename Pantheon+SH0ES

What it does:
- Looks for Pantheon+ table (basename.dat or .txt)
- Looks for a combined STAT+SYS covariance in common formats (.npy/.csv/.txt)
- If separate STAT and SYS covariances are found, sums them
- Ensures the covariance is SPD (adds small jitter if necessary)
- Trims/aligns to the number of SNe in the table if needed
- Writes Pantheon+SH0ES_cov.npy (and optionally .csv) in the same directory

Notes:
- This script does NOT download data. Place the STAT+SYS covariance files in the base directory.
- Common sources: Pantheon+ data release (Brout et al. 2022) — STAT+SYS covariance.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np


def _ensure_spd(C: np.ndarray, jitter_frac: float = 1e-10, max_trials: int = 5) -> np.ndarray:
    C = 0.5 * (C + C.T)
    diag = np.diag(C)
    scale = float(np.median(diag)) if np.all(np.isfinite(diag)) and np.any(diag > 0) else 1.0
    for _ in range(max_trials + 1):
        try:
            np.linalg.cholesky(C)
            return C
        except np.linalg.LinAlgError:
            C = C + np.eye(C.shape[0]) * (jitter_frac * scale)
    # Fallback: project to SPD via eigen cleanup
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-15, None)
    return (evecs * evals) @ evecs.T


def _load_matrix(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)
    rows = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            rows.append([float(x) for x in s.replace(',', ' ').split()])
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser(description='Build Pantheon+ full covariance next to the data table.')
    ap.add_argument('--base-dir', type=str, default=os.path.join('external_data', 'pantheon'))
    ap.add_argument('--basename', type=str, default='Pantheon+SH0ES')
    ap.add_argument('--write-csv', action='store_true', help='Also write a CSV alongside the .npy file.')
    args = ap.parse_args()

    base = os.path.abspath(args.base_dir)
    data_candidates = [os.path.join(base, f'{args.basename}.dat'), os.path.join(base, f'{args.basename}.txt')]
    data_path = next((p for p in data_candidates if os.path.exists(p)), None)
    if data_path is None:
        print(f"[ERROR] Data table not found in {base}. Expected {args.basename}.dat or .txt", file=sys.stderr)
        sys.exit(1)

    # Count rows in table
    n_rows = 0
    with open(data_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('CID'):
                continue
            n_rows += 1
    print(f"Found table: {data_path} with N={n_rows} SNe")

    # Candidates for covariance
    comb_candidates = [
        os.path.join(base, f'{args.basename}_cov.npy'),
        os.path.join(base, f'{args.basename}_STAT+SYS_cov.npy'),
        os.path.join(base, f'{args.basename}_cov.csv'),
        os.path.join(base, f'{args.basename}_STAT+SYS_cov.csv'),
        os.path.join(base, f'{args.basename}_cov.txt'),
        os.path.join(base, f'{args.basename}_STAT+SYS_cov.txt'),
    ]
    stat_candidates = [
        os.path.join(base, f'{args.basename}_STAT_cov.npy'),
        os.path.join(base, f'{args.basename}_STAT_cov.csv'),
        os.path.join(base, f'{args.basename}_STAT_cov.txt'),
    ]
    sys_candidates = [
        os.path.join(base, f'{args.basename}_SYS_cov.npy'),
        os.path.join(base, f'{args.basename}_SYS_cov.csv'),
        os.path.join(base, f'{args.basename}_SYS_cov.txt'),
    ]

    C = None
    # Try combined first
    for cp in comb_candidates:
        if os.path.exists(cp):
            try:
                C = _load_matrix(cp)
                print(f"Loaded combined covariance: {cp}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load {cp}: {e}")
                C = None
    # Try sum of STAT + SYS
    if C is None:
        stat_path = next((p for p in stat_candidates if os.path.exists(p)), None)
        sys_path = next((p for p in sys_candidates if os.path.exists(p)), None)
        if stat_path and sys_path:
            try:
                C_stat = _load_matrix(stat_path)
                C_sys = _load_matrix(sys_path)
                C = C_stat + C_sys
                print(f"Loaded and summed STAT+SYS: {stat_path} + {sys_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load STAT+SYS covariances: {e}", file=sys.stderr)
                sys.exit(2)

    if C is None:
        print("[ERROR] No covariance files found. Please download Pantheon+ STAT+SYS covariance into:")
        for cp in comb_candidates + stat_candidates + sys_candidates:
            print("  ", cp)
        sys.exit(3)

    # Align to n_rows if necessary
    if C.shape[0] != n_rows:
        n = min(C.shape[0], n_rows)
        print(f"[WARN] Covariance shape {C.shape} != N={n_rows}; trimming to {n}")
        C = C[:n, :n]

    C = _ensure_spd(C)
    out_npy = os.path.join(base, f'{args.basename}_cov.npy')
    np.save(out_npy, C)
    print(f"Wrote {out_npy} ({C.shape[0]}x{C.shape[1]})")
    if args.write_csv:
        out_csv = os.path.join(base, f'{args.basename}_cov.csv')
        np.savetxt(out_csv, C, delimiter=',')
        print(f"Wrote {out_csv}")


if __name__ == '__main__':
    main()
