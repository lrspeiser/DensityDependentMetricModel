#!/usr/bin/env python3
# Extract SPARC observed velocities (Vobs, errV) from rotmod .dat files to CSV

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re

def parse_rotmod_dat(p: Path):
    rows = []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = re.split(r"\s+", s)
            if len(parts) < 6: continue
            R_kpc = float(parts[0])
            Vobs  = float(parts[1])
            errV  = float(parts[2])
            rows.append((R_kpc, Vobs, errV))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Build SPARC observed velocity CSV from rotmod .dat')
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args()

    srcd = Path(args.src_dir)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['galaxy_id','R_kpc','Vobs_kms','err_kms'])
        for p in sorted(srcd.glob('*_rotmod.dat')):
            gid = p.name.replace('_rotmod.dat','')
            data = parse_rotmod_dat(p)
            for R_kpc, Vobs, err in data:
                w.writerow([gid, f"{R_kpc:.6f}", f"{Vobs:.6f}", f"{err:.6f}"])
            count += 1
            if args.limit and count >= args.limit:
                break
    print('Wrote', out)

if __name__ == '__main__':
    main()
