#!/usr/bin/env python3
# Compute GR (baryons-only) SPARC metrics using Vbar_kms from rotmod CSVs vs observed Vobs.

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np


def load_obs(obs_csv: Path, galaxies: list[str]):
    data = {}
    with obs_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gid = row['galaxy_id']
            if gid not in galaxies: continue
            data.setdefault(gid, {'R': [], 'Vobs': [], 'err': []})
            data[gid]['R'].append(float(row['R_kpc']))
            data[gid]['Vobs'].append(float(row['Vobs_kms']))
            data[gid]['err'].append(float(row['err_kms']))
    for gid in list(data.keys()):
        for k in data[gid]:
            data[gid][k] = np.array(data[gid][k])
    return data


def load_vbar(rotmod_csv: Path):
    R, Vbar = [], []
    with rotmod_csv.open() as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        norm = {c.lower(): c for c in cols}
        col_R = norm.get('r_kpc') or norm.get('r') or norm.get('radius') or 'R_kpc'
        col_Vbar = norm.get('vbar_kms') or norm.get('vbar') or 'Vbar_kms'
        for row in rdr:
            R.append(float(row[col_R]))
            Vbar.append(float(row[col_Vbar]))
    return np.array(R), np.array(Vbar)


def metrics_one(gid: str, obs: dict, rotmod_csv: Path) -> dict:
    Rm, Vbar = load_vbar(rotmod_csv)
    Ro, Vo, eo = obs['R'], obs['Vobs'], obs['err']
    Vbar_at_R = np.interp(Ro, Rm, Vbar, left=Vbar[0], right=Vbar[-1])
    resid = Vo - Vbar_at_R
    rms = float(np.sqrt(np.mean(resid**2)))
    frac_10 = float(np.mean(np.abs(resid) <= 10.0))
    frac_20 = float(np.mean(np.abs(resid) <= 20.0))
    return {
        'galaxy': gid,
        'n': int(len(Ro)),
        'rms_kms': rms,
        'coverage_10kms': frac_10,
        'coverage_20kms': frac_20,
    }


def main():
    ap = argparse.ArgumentParser(description='SPARC GR (Vbar) metrics')
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--rotmods-dir', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    obs = load_obs(Path(args.obs_csv), args.galaxies)
    rows = []
    for gid in args.galaxies:
        rotmod_csv = Path(args.rotmods_dir) / f"{gid}.csv"
        if not rotmod_csv.exists():
            continue
        rows.append(metrics_one(gid, obs[gid], rotmod_csv))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['galaxy','n','rms_kms','coverage_10kms','coverage_20kms'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote GR metrics to', args.out_csv)


if __name__ == '__main__':
    main()
