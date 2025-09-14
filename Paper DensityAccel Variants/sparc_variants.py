#!/usr/bin/env python3
# Sandbox SPARC variant: compute model curves using accel/density/hybrid gate.
# Does not modify main code; requires SPARC rotmod CSVs and (for density/hybrid)
# a per-galaxy rho proxy CSV.

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from xi_registry_variants import build_gate

ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14  # m/s^2 per [(km/s)^2/kpc]


def load_rotmod(path: Path) -> Dict[str, np.ndarray]:
    # Minimal loader for SPARC rotmod-like CSV: R (kpc), Vbar (km/s)
    R_kpc, Vbar = [], []
    with path.open() as f:
        rdr = csv.DictReader(f)
        cols = [c.lower() for c in rdr.fieldnames or []]
        # heuristics for column names
        col_R = 'r_kpc' if 'r_kpc' in cols else ('r' if 'r' in cols else None)
        col_Vbar = 'vbar_kms' if 'vbar_kms' in cols else ('vbar' if 'vbar' in cols else None)
        if col_R is None or col_Vbar is None:
            raise ValueError(f"Rotmod CSV must contain R_kpc and Vbar columns: {path}")
        for row in rdr:
            R_kpc.append(float(row[col_R]))
            Vbar.append(float(row[col_Vbar]))
    return {'R_kpc': np.array(R_kpc), 'Vbar_kms': np.array(Vbar)}


def load_rho_proxy(rho_csv: Path) -> Dict[str, Dict[str, np.ndarray]]:
    # rho CSV columns: galaxy_id, R_kpc, rho_cgs
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with rho_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gid = row['galaxy_id']
            R = float(row['R_kpc'])
            rho = float(row['rho_cgs'])
            rec = out.setdefault(gid, {'R_kpc': [], 'rho_cgs': []})
            rec['R_kpc'].append(R)
            rec['rho_cgs'].append(rho)
    for gid in out:
        out[gid]['R_kpc'] = np.array(out[gid]['R_kpc'])
        out[gid]['rho_cgs'] = np.array(out[gid]['rho_cgs'])
    return out


def main():
    ap = argparse.ArgumentParser(description='SPARC sandbox: gate variants')
    ap.add_argument('--sparc-rotmods', required=True, help='Dir with per-galaxy rotmod CSVs')
    ap.add_argument('--galaxies', nargs='+', required=True, help='Galaxy IDs (filenames without .csv)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--gate', default='accel', choices=['accel','density','hybrid'])
    ap.add_argument('--rho-csv', default=None, help='CSV with columns: galaxy_id,R_kpc,rho_cgs (required for density/hybrid)')
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--gamma', type=float, default=1.5)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rotdir = Path(args.sparc_rotmods)
    rho_map = load_rho_proxy(Path(args.rho_csv)) if args.rho_csv else {}

    gate = build_gate(args.gate, a0=args.a0, rho_c=args.rho_c, gamma=args.gamma, zeta=args.zeta, Dmax=args.Dmax)

    for gid in args.galaxies:
        rotmod_path = rotdir / f"{gid}.csv"
        data = load_rotmod(rotmod_path)
        R = data['R_kpc']
        Vbar = data['Vbar_kms']
        # gbar in cgs: (km/s)^2/kpc * C -> m/s^2; convert to cgs (×100); but ratios cancel in xi.
        gbar_SI = (Vbar**2 / R) * ACC_M_S2_PER_KMS2_PER_KPC
        gbar_cgs = gbar_SI * 100.0

        if args.gate in ('density','hybrid'):
            if gid not in rho_map:
                raise ValueError(f"No rho profile found in {args.rho_csv} for galaxy_id={gid}")
            # Interpolate rho(R) to rotmod radii
            R_rho = rho_map[gid]['R_kpc']
            rho = rho_map[gid]['rho_cgs']
            rho_at_R = np.interp(R, R_rho, rho, left=rho[0], right=rho[-1])
        else:
            rho_at_R = np.zeros_like(R)

        xi = gate(gbar_cgs, rho_at_R, R)
        V_model = np.sqrt(np.clip(xi, 1.0, None)) * Vbar

        of = outdir / f"{gid}_gate_model.csv"
        with of.open('w') as f:
            f.write('R_kpc,Vbar_kms,xi,V_model_kms\n')
            for r, vb, x, vm in zip(R, Vbar, xi, V_model):
                f.write(f"{r:.6f},{vb:.6f},{x:.6f},{vm:.6f}\n")

    meta = {
        'gate': args.gate,
        'params': {'a0': args.a0, 'rho_c': args.rho_c, 'gamma': args.gamma, 'zeta': args.zeta, 'Dmax': args.Dmax},
        'galaxies': args.galaxies,
    }
    (outdir/'sparc_variants_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Wrote models to', outdir)


if __name__ == '__main__':
    main()
