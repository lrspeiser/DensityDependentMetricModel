#!/usr/bin/env python3
# Build rho_proxy_v1.csv using disk scale-length fit and a flaring scale-height model.
# rho_mid(R) ≈ (Sigma_disk(R)*MLd + Sigma_bul(R)*MLb) / [2 h_z(R)] ;
# h_z(R) = f_hz * h_R * (1 + eta * R/h_R)
# Units: SB in L/pc^2 -> Sigma in Msun/pc^2 -> Msun/pc^3 -> g/cm^3.

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import numpy as np

MSUN = 1.98847e33
PC_CM = 3.0856775814913673e18


def read_rotmod_dat(p: Path):
    R, SBd, SBb = [], [], []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = re.split(r"\s+", s)
            if len(parts) < 8: continue
            R.append(float(parts[0]))
            SBd.append(float(parts[6]))
            SBb.append(float(parts[7]))
    return np.array(R), np.array(SBd), np.array(SBb)


def fit_exponential_scale_length(R, SB):
    mask = (SB > 0) & np.isfinite(SB)
    if mask.sum() < 3:
        return np.nan
    x = R[mask]
    y = np.log(SB[mask])
    # Fit ln SB = ln SB0 - R/h_R  => slope = -1/h_R
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[1]
    if slope <= 0:
        h_R = 1.0  # fallback 1 kpc
    else:
        h_R = 1.0 / slope
    return float(h_R)


def main():
    ap = argparse.ArgumentParser(description='rho_proxy_v1 from rotmod .dat with flaring h_z(R)')
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--mld', type=float, default=0.5)
    ap.add_argument('--mlb', type=float, default=0.7)
    ap.add_argument('--f-hz', type=float, default=0.1, help='h_z(0)/h_R factor')
    ap.add_argument('--eta', type=float, default=0.15, help='flaring coefficient in h_z(R)=f*h_R*(1+eta*R/h_R)')
    ap.add_argument('--limit', type=int, default=50)
    args = ap.parse_args()

    srcd = Path(args.src_dir)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['galaxy_id','R_kpc','rho_cgs'])
        for p in sorted(srcd.glob('*_rotmod.dat')):
            gid = p.name.replace('_rotmod.dat','')
            R, SBd, SBb = read_rotmod_dat(p)
            if R.size == 0:
                continue
            h_R = fit_exponential_scale_length(R, SBd)
            if not np.isfinite(h_R) or h_R <= 0:
                h_R = 3.0  # fallback
            hz0 = args.f_hz * h_R  # kpc
            hz_R = hz0 * (1.0 + args.eta * (R / h_R))  # kpc
            # convert to pc
            hz_pc = hz_R * 1e3
            # surface densities in Msun/pc^2
            Sigma = args.mld * SBd + args.mlb * SBb
            rho_Msun_pc3 = Sigma / (2.0 * np.maximum(hz_pc, 10.0))  # avoid tiny hz
            rho_cgs = rho_Msun_pc3 * MSUN / (PC_CM**3)
            for r, rh in zip(R, rho_cgs):
                w.writerow([gid, f"{r:.6f}", f"{rh:.6e}"])
            count += 1
            if args.limit and count >= args.limit:
                break
    print('Wrote', out)

if __name__ == '__main__':
    main()
