#!/usr/bin/env python3
# Build a simple SPARC rho_proxy.csv from rotmod .dat files using midplane
# approximation: rho ≈ (Sigma_disk*M/L_disk + Sigma_bul*M/L_bul) / (2 h_z).
# Units: SB in L/pc^2 -> Sigma in Msun/pc^2 via M/L; then convert to g/cm^3.

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re

MSUN = 1.98847e33             # g
PC_CM = 3.0856775814913673e18 # cm


def parse_rotmod_dat(p: Path):
    rows = []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = re.split(r"\s+", s)
            if len(parts) < 8: continue
            R_kpc = float(parts[0])
            SBdisk = float(parts[6])  # L/pc^2
            SBbul  = float(parts[7])  # L/pc^2
            rows.append((R_kpc, SBdisk, SBbul))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Build SPARC rho proxy CSV from rotmod .dat')
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--mld', type=float, default=0.5, help='M/L for disk (Msun/Lsun)')
    ap.add_argument('--mlb', type=float, default=0.7, help='M/L for bulge (Msun/Lsun)')
    ap.add_argument('--hz-pc', type=float, default=300.0, help='Scale height (pc)')
    ap.add_argument('--limit', type=int, default=20)
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
            data = parse_rotmod_dat(p)
            for R_kpc, SBdisk, SBbul in data:
                Sigma_Msun_pc2 = args.mld * SBdisk + args.mlb * SBbul
                rho_Msun_pc3 = Sigma_Msun_pc2 / (2.0 * args.hz_pc)
                rho_cgs = rho_Msun_pc3 * MSUN / (PC_CM**3)
                w.writerow([gid, f"{R_kpc:.6f}", f"{rho_cgs:.6e}"])
            count += 1
            if args.limit and count >= args.limit:
                break
    print('Wrote', out)

if __name__ == '__main__':
    main()
