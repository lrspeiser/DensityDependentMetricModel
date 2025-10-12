#!/usr/bin/env python3
"""
Build a per-point RAR table from SPARC Rotmod files.

Inputs:
  --sparc-dir ../external_data/Rotmod_LTG
Outputs:
  tariff/data/rar_points.csv with columns:
    galaxy, R_kpc, gbar_m_s2, gobs_m_s2

We compute:
  gobs = (Vobs_kms*1e3)^2 / (R_kpc * 3.085677581e19)
  gbar = (Vbar_kms*1e3)^2 / (R_kpc * 3.085677581e19)
where Vbar_kms^2 = Vdisk^2 + Vbul^2 + Vgas^2.

Filters:
  - drop rows with non-finite values
  - optional Q filtering could be added using SPARC_Lelli2016c.mrt if desired
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse the rotmod loader already present in repo
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data_loaders.sparc_loader import load_rotmod

KPC_TO_M = 3.085677581e19


def process_galaxy(path: Path) -> pd.DataFrame:
    d = load_rotmod(path)
    R_kpc = d["R_kpc"]
    Vobs = d["Vobs_kms"]
    Vgas = d["Vgas_kms"]
    Vdisk = d["Vdisk_kms"]
    Vbul = d["Vbul_kms"]
    Vbar2 = np.maximum(0.0, Vgas**2 + Vdisk**2 + Vbul**2)
    Vbar = np.sqrt(Vbar2)
    # accelerations in m/s^2
    gobs = (Vobs*1e3)**2 / (np.clip(R_kpc, 1e-12, None) * KPC_TO_M)
    gbar = (Vbar*1e3)**2 / (np.clip(R_kpc, 1e-12, None) * KPC_TO_M)
    out = pd.DataFrame({
        "galaxy": [path.name.replace('_rotmod.dat','')]*len(R_kpc),
        "R_kpc": R_kpc,
        "gbar_m_s2": gbar,
        "gobs_m_s2": gobs,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def main():
    ap = argparse.ArgumentParser(description="Build RAR per-point table from SPARC Rotmod files")
    ap.add_argument("--sparc-dir", type=str, default=str(Path('..')/ 'external_data' / 'Rotmod_LTG'))
    ap.add_argument("--out", type=str, default=str(Path('tariff')/ 'data' / 'rar_points.csv'))
    args = ap.parse_args()

    sparc_dir = Path(args.sparc_dir)
    if not sparc_dir.exists():
        raise FileNotFoundError(f"SPARC directory not found: {sparc_dir}")

    files = sorted(sparc_dir.glob('*_rotmod.dat'))
    if not files:
        print("No *_rotmod.dat files found.")
        return 1

    rows = []
    for f in files:
        try:
            df = process_galaxy(f)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")
    if not rows:
        print("No usable rotmod files parsed.")
        return 1
    full = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print("Wrote", out_path, f"(N={len(full)})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())