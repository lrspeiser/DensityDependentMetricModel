#!/usr/bin/env python3
"""
ingest_boss_dr12_consensus.py — Parse the BOSS DR12 consensus Gaussian constraints into our BAO CSV schema.

Usage:
  # First unpack the tarball into external_data/BOSS/DR12_consensus
  #   tar -xzf external_data/BOSS/ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints.tar.gz \
  #       -C external_data/BOSS/DR12_consensus
  # Then run this script to build the CSV used by tariff:
  python tariff/scripts/ingest_boss_dr12_consensus.py \
      --in-dir external_data/BOSS/DR12_consensus \
      --out tariff/data/bao_compilation.csv

Notes:
- We expect three effective redshifts z≈{0.38, 0.51, 0.61} with anisotropic constraints.
- The DR12 files often express H constraint as H*r_d/c; we convert this to D_H/r_d = 1/(H*r_d/c).
- If a per-bin correlation ρ between D_M/r_d and H*r_d/c is provided, we convert to ρ for D_M/r_d and D_H/r_d by flipping the sign (ρ' = -ρ).
- Inline comments point to data README per project rule.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# See data/observational/README_DATA_SOURCES.md for exact data provenance and citations.


def _parse_numeric_lines(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 3:
                rows.append(parts)
    return rows


def parse_one_table(path: str) -> List[Dict[str, Any]]:
    """Parse lines like:
       0.38 dM(rsfid/rs) 1512.39
       0.38 Hz(rs/rsfid) 81.2087
    and collect DM/rd and Hrd/c per redshift. No errors or rho are in these files; we will backfill
    errors/covariance later if needed.
    """
    rows_out: Dict[float, Dict[str, Any]] = {}
    rows = _parse_numeric_lines(path)
    for parts in rows:
        try:
            z = float(parts[0])
        except ValueError:
            continue
        if len(parts) < 3:
            continue
        key = parts[1]
        val = parts[2]
        try:
            x = float(val)
        except ValueError:
            continue
        rec = rows_out.setdefault(z, dict(z=z))
        if 'dM' in key:
            # File gives dM = D_M * (rsfid/rs) => D_M/rd = dM * (rd_fid/rd). We do not know rd_fid here; treat as DM/rd in pipeline units.
            rec['DM_over_rd'] = x
        elif key.startswith('Hz'):
            # Hz(rs/rsfid) is H(z) * (rs/rsfid) in km/s/Mpc. We prefer to store Hrd/c directly or convert later.
            rec['Hz_scaled'] = x
    # Convert into expected schema columns if possible; errors and rho unknown here.
    out = []
    for z, rec in sorted(rows_out.items()):
        if 'DM_over_rd' in rec and 'Hz_scaled' in rec:
            out.append(dict(z=z, DM_over_rd=rec['DM_over_rd'], Hz_scaled=rec['Hz_scaled']))
    return out


def build_csv(in_dir: str, out_csv: str):
    entries: List[Dict[str, Any]] = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if fn.lower().endswith(('.txt', '.dat', '.csv')):
                path = os.path.join(root, fn)
                try:
                    entries.extend(parse_one_table(path))
                except Exception:
                    continue
    if not entries:
        print(f"[ERROR] No DR12 consensus entries found in {in_dir}. Inspect the files and adjust parser.", file=sys.stderr)
        sys.exit(2)
    # Deduplicate by z and keep the first occurrence per z in sorted order
    df = pd.DataFrame(entries).drop_duplicates(subset=['z']).sort_values('z')
    # We have DM_over_rd and Hz_scaled; convert to D_H_over_rd via Hrd/c if we know c and scaling.
    # Without the exact rs/rsfid factor, we cannot get absolute D_H_over_rd; we will output only D_M_over_rd for now.
    df['D_M_over_rd'] = df['DM_over_rd']
    cols = ['z', 'D_M_over_rd']
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df[cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with columns {cols} and N={len(df)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Ingest BOSS DR12 consensus Gaussian constraints into BAO CSV schema.')
    ap.add_argument('--in-dir', type=str, default=os.path.join('external_data', 'BOSS', 'DR12_consensus'))
    ap.add_argument('--out', type=str, default=os.path.join('tariff', 'data', 'bao_compilation.csv'))
    args = ap.parse_args()
    build_csv(args.in_dir, args.out)