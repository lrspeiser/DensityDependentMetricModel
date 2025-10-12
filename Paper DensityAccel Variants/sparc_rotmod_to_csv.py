#!/usr/bin/env python3
# Convert SPARC rotmod .dat files to simple CSV with R_kpc and Vbar_kms (quadrature of components)

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import math

def convert_one(src: Path, dst: Path) -> None:
    R, Vgas, Vdisk, Vbul = [], [], [], []
    with src.open() as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts = re.split(r"\s+", line)
            if len(parts) < 6: continue
            r_kpc = float(parts[0]); vgas = float(parts[3]); vdisk = float(parts[4]); vbul = float(parts[5])
            R.append(r_kpc); Vgas.append(vgas); Vdisk.append(vdisk); Vbul.append(vbul)
    R_kpc = R
    Vbar = [math.sqrt(max(0.0, vg**2 + vd**2 + vb**2)) for vg,vd,vb in zip(Vgas,Vdisk,Vbul)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['R_kpc','Vbar_kms'])
        for r, v in zip(R_kpc, Vbar):
            w.writerow([f"{r:.6f}", f"{v:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pattern', default='*_rotmod.dat')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    srcd = Path(args.src_dir)
    outd = Path(args.out_dir)
    count = 0
    for p in sorted(srcd.glob(args.pattern)):
        gid = p.name.replace('_rotmod.dat','')
        convert_one(p, outd / f"{gid}.csv")
        count += 1
        if args.limit and count >= args.limit:
            break
    print(f"Converted {count} files to", outd)

if __name__ == '__main__':
    main()
