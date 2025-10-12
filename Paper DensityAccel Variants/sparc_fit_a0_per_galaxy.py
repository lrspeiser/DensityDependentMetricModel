#!/usr/bin/env python3
# Fit per-galaxy a0 (accel gate) using a simple RMS loss with optional floors/masks.

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np

ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14


def load_rotmod(path: Path):
    R, Vbar = [], []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            R.append(float(row['R_kpc']))
            Vbar.append(float(row['Vbar_kms']))
    return np.array(R), np.array(Vbar)


def load_obs(obs_csv: Path, gid: str):
    R, Vobs, err = [], [], []
    with obs_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row['galaxy_id'] != gid: continue
            R.append(float(row['R_kpc']))
            Vobs.append(float(row['Vobs_kms']))
            err.append(float(row['err_kms']))
    return np.array(R), np.array(Vobs), np.array(err)


def xi_accel(gbar_cgs: np.ndarray, a0: float, Dmax: float) -> np.ndarray:
    gb = np.maximum(gbar_cgs, 1e-99)
    xi = 0.5 + np.sqrt(0.25 + a0/gb)
    if np.isfinite(Dmax):
        xi = np.minimum(xi, Dmax)
    return xi


def fit_a0_for_galaxy(gid: str, rotdir: Path, obs_csv: Path, Dmax: float,
                      a0_grid: np.ndarray, sigma_floor: float = 6.0,
                      xmin: float | None = None, xmax: float | None = None) -> dict:
    Rm, Vbar = load_rotmod(rotdir / f"{gid}.csv")
    Ro, Vobs, eobs = load_obs(obs_csv, gid)
    # masks
    mask = np.ones_like(Ro, dtype=bool)
    if xmin is not None:
        mask &= (Ro >= xmin)
    if xmax is not None:
        mask &= (Ro <= xmax)
    R, V = Ro[mask], Vobs[mask]
    # errors with floor
    err = np.maximum(eobs[mask], sigma_floor)
    # model gbar
    gbar_SI = (Vbar**2 / Rm) * ACC_M_S2_PER_KMS2_PER_KPC
    gbar_cgs_at_R = np.interp(R, Rm, gbar_SI, left=gbar_SI[0], right=gbar_SI[-1]) * 100.0

    best = None
    best_chi2 = np.inf
    for a0 in a0_grid:
        xi = xi_accel(gbar_cgs_at_R, a0, Dmax)
        Vmod = np.sqrt(np.clip(xi, 1.0, None)) * np.interp(R, Rm, Vbar)
        chi2 = np.sum(((V - Vmod)/err)**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = (a0, Vmod)
    a0_best, Vmod_best = best
    rms = float(np.sqrt(np.mean((V - Vmod_best)**2)))
    return {'galaxy': gid, 'a0': float(a0_best), 'rms_kms': rms, 'n': int(len(R))}


def main():
    ap = argparse.ArgumentParser(description='Per-galaxy a0 grid-fit (accel gate)')
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--rotmods-dir', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--Dmax', type=float, default=50.0)
    ap.add_argument('--a0-min', type=float, default=1e-9)
    ap.add_argument('--a0-max', type=float, default=1e-6)
    ap.add_argument('--a0-N', type=int, default=201)
    ap.add_argument('--sigma-floor', type=float, default=6.0)
    ap.add_argument('--xmin', type=float, default=None)
    ap.add_argument('--xmax', type=float, default=None)
    args = ap.parse_args()

    a0_grid = np.logspace(math.log10(args.a0_min), math.log10(args.a0_max), args.a0_N)
    rows = []
    for gid in args.galaxies:
        rows.append(fit_a0_for_galaxy(gid, Path(args.rotmods_dir), Path(args.obs_csv),
                                      args.Dmax, a0_grid, args.sigma_floor, args.xmin, args.xmax))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['galaxy','n','a0','rms_kms'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote a0 fits to', args.out_csv)


if __name__ == '__main__':
    main()
