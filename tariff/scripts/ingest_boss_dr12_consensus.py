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


def build_csv(in_dir: str, out_csv: str, r_s_fid_mpc: float = 147.78):
    # Load the consensus means
    means_path = os.path.join(in_dir, 'BAO_consensus_results_dM_Hz.txt')
    cov_path = os.path.join(in_dir, 'BAO_consensus_covtot_dM_Hz.txt')
    if not os.path.exists(means_path) or not os.path.exists(cov_path):
        print(f"[ERROR] Expected files not found in {in_dir}:\n  - BAO_consensus_results_dM_Hz.txt\n  - BAO_consensus_covtot_dM_Hz.txt", file=sys.stderr)
        sys.exit(2)
    rows = _parse_numeric_lines(means_path)
    # Parse means in order [dM1, Hz1, dM2, Hz2, dM3, Hz3]
    z_bins: List[float] = []
    dM_vals: List[float] = []
    Hz_vals: List[float] = []
    for parts in rows:
        try:
            z = float(parts[0]); key = parts[1]; val = float(parts[2])
        except Exception:
            continue
        if 'dM' in key:
            z_bins.append(z); dM_vals.append(val)
        elif key.startswith('Hz'):
            Hz_vals.append(val)
    if len(z_bins) != 3 or len(Hz_vals) != 3:
        print(f"[ERROR] Failed to parse 3 z bins and (dM, Hz) means from {means_path}", file=sys.stderr)
        sys.exit(3)
    # Load 6x6 covariance for [dM1, Hz1, dM2, Hz2, dM3, Hz3]
    C6 = np.loadtxt(cov_path)
    if C6.shape != (6, 6):
        print(f"[ERROR] Expected 6x6 covariance in {cov_path}, got {C6.shape}", file=sys.stderr)
        sys.exit(4)
    # Convert to our native variables: D_M_over_rd and D_H_over_rd (with D_H_over_rd = 1 / (Hrd/c))
    c_km_s = 299792.458
    D_M_over_rd = np.asarray(dM_vals, float) / float(r_s_fid_mpc)
    Hrd_over_c = (np.asarray(Hz_vals, float) * float(r_s_fid_mpc)) / c_km_s
    D_H_over_rd = 1.0 / Hrd_over_c
    # Extract per-bin 2x2 covariance blocks and transform linearly then via y2=1/x
    out_rows: List[Dict[str, Any]] = []
    for i in range(3):
        # block indices in 6x6: (2i, 2i+1)
        idx = [2*i, 2*i+1]
        C_block = C6[np.ix_(idx, idx)].astype(float)
        # Linear transform first: x1=dM -> y1 = x1 / r_s_fid; x2=Hz -> u = (x2 * r_s_fid / c) = Hrd/c
        A = np.array([[1.0/r_s_fid_mpc, 0.0], [0.0, r_s_fid_mpc/c_km_s]], float)
        Cu = A @ C_block @ A.T
        sig_DM = float(np.sqrt(max(Cu[0, 0], 0.0)))
        sig_Hrd = float(np.sqrt(max(Cu[1, 1], 0.0)))
        cov_DM_Hrd = float(Cu[0, 1])
        rho_u = float(cov_DM_Hrd / (sig_DM * sig_Hrd)) if sig_DM > 0 and sig_Hrd > 0 else 0.0
        # Now transform u -> y2 = 1/u for D_H_over_rd
        u = float(Hrd_over_c[i])
        sig_DH = sig_Hrd / (u*u)
        rho = -rho_u  # correlation flips sign under reciprocal
        out_rows.append(dict(z=z_bins[i], D_M_over_rd=D_M_over_rd[i], D_M_err=sig_DM,
                             D_H_over_rd=D_H_over_rd[i], D_H_err=sig_DH, rho=rho))
    df = pd.DataFrame(out_rows).sort_values('z')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with columns {list(df.columns)} and N={len(df)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Ingest BOSS DR12 consensus Gaussian constraints into BAO CSV schema.')
    ap.add_argument('--in-dir', type=str, default=os.path.join('external_data', 'BOSS', 'DR12_consensus', 'COMBINEDDR12_BAO_consensus_dM_Hz'))
    ap.add_argument('--out', type=str, default=os.path.join('tariff', 'data', 'bao_compilation.csv'))
    ap.add_argument('--r-s-fid', dest='r_s_fid', type=float, default=147.78, help='Fiducial sound horizon r_s(fid) in Mpc (DR12 uses ~147.78 Mpc)')
    args = ap.parse_args()
    build_csv(args.in_dir, args.out, r_s_fid_mpc=args.r_s_fid)
