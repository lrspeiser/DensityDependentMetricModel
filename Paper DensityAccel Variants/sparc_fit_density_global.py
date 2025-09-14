#!/usr/bin/env python3
# Fit global parameters for density or hybrid gate across a SPARC subset.

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14


def load_rotmod_csv(rotmods_dir: Path, gid: str):
    R, Vbar = [], []
    with (rotmods_dir / f"{gid}.csv").open() as f:
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


def load_rho_proxy(proxy_csv: Path, gid: str):
    R, rho = [], []
    with proxy_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row['galaxy_id'] != gid: continue
            R.append(float(row['R_kpc']))
            rho.append(float(row['rho_cgs']))
    return np.array(R), np.array(rho)


def xi_density(rho_cgs: np.ndarray, rho_c: float, gamma: float, xi_max: float) -> np.ndarray:
    r = np.maximum(rho_cgs, 1e-99)
    env = 1.0 / (1.0 + np.power(r / rho_c, gamma))
    return 1.0 + (xi_max - 1.0) * env


def xi_hybrid(gbar_cgs: np.ndarray, rho_cgs: np.ndarray, a0: float, rho_c: float, gamma: float, zeta: float, xi_max: float) -> np.ndarray:
    r = np.maximum(rho_cgs, 1e-99)
    env = 1.0 / (1.0 + np.power(r / rho_c, gamma))
    a0_eff = a0 * (1.0 + zeta * env)
    gb = np.maximum(gbar_cgs, 1e-99)
    xi = 0.5 + np.sqrt(0.25 + a0_eff / gb)
    return np.minimum(xi, xi_max)


def rms_with_floor(resid: np.ndarray, err: np.ndarray, sigma_floor: float) -> float:
    # Report unweighted RMS in km/s; err used for chi2 if needed
    return float(np.sqrt(np.mean(resid**2)))


def evaluate_gate_for_galaxy(kind: str, params: dict, rotmods_dir: Path, obs_csv: Path, rho_proxy_csv: Path, gid: str, sigma_floor: float) -> float:
    Rm, Vbar = load_rotmod_csv(rotmods_dir, gid)
    Ro, Vobs, eobs = load_obs(obs_csv, gid)
    Rrho, rho = load_rho_proxy(rho_proxy_csv, gid)
    # interpolate to observed radii
    Vbar_at = np.interp(Ro, Rm, Vbar, left=Vbar[0], right=Vbar[-1])
    gbar_SI = (Vbar**2 / Rm) * ACC_M_S2_PER_KMS2_PER_KPC
    gbar_cgs_at = np.interp(Ro, Rm, gbar_SI, left=gbar_SI[0], right=gbar_SI[-1]) * 100.0
    rho_at = np.interp(Ro, Rrho, rho, left=rho[0], right=rho[-1])
    if kind == 'density':
        xi = xi_density(rho_at, params['rho_c'], params['gamma'], params['xi_max'])
    else:
        xi = xi_hybrid(gbar_cgs_at, rho_at, params['a0'], params['rho_c'], params['gamma'], params['zeta'], params['xi_max'])
    Vmod = np.sqrt(np.clip(xi, 1.0, None)) * Vbar_at
    resid = Vobs - Vmod
    err = np.maximum(eobs, sigma_floor)
    return rms_with_floor(resid, err, sigma_floor)


def main():
    ap = argparse.ArgumentParser(description='Fit global DG/hybrid parameters over SPARC subset')
    ap.add_argument('--kind', choices=['density','hybrid'], required=True)
    ap.add_argument('--rotmods-dir', required=True)
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--rho-proxy-csv', required=True)
    ap.add_argument('--galaxies', nargs='+', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--sigma-floor', type=float, default=6.0)
    # Grid params
    ap.add_argument('--rho-c-grid', default='1e-28,3e-28,1e-27,3e-27,1e-26')
    ap.add_argument('--gamma-grid', default='0.8,1.0,1.2,1.5')
    ap.add_argument('--xi-max', type=float, default=50.0)
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--zeta-grid', default='0.5,1.0,2.0')
    args = ap.parse_args()

    rho_c_grid = [float(x) for x in args.rho_c_grid.split(',') if x]
    gamma_grid = [float(x) for x in args.gamma_grid.split(',') if x]
    zeta_grid = [float(x) for x in args.zeta_grid.split(',') if x]

    best = None
    best_metric = np.inf
    per_params = []

    for rho_c in rho_c_grid:
        for gamma in gamma_grid:
            if args.kind == 'density':
                params = {'rho_c': rho_c, 'gamma': gamma, 'xi_max': args.xi_max}
                rms_list = []
                for gid in args.galaxies:
                    rms_list.append(evaluate_gate_for_galaxy('density', params, Path(args.rotmods_dir), Path(args.obs_csv), Path(args.rho_proxy_csv), gid, args.sigma_floor))
                med_rms = float(np.median(rms_list))
                per_params.append({'params': params, 'median_rms': med_rms})
                if med_rms < best_metric:
                    best_metric = med_rms
                    best = ('density', params)
            else:
                for zeta in zeta_grid:
                    params = {'rho_c': rho_c, 'gamma': gamma, 'zeta': zeta, 'xi_max': args.xi_max, 'a0': args.a0}
                    rms_list = []
                    for gid in args.galaxies:
                        rms_list.append(evaluate_gate_for_galaxy('hybrid', params, Path(args.rotmods_dir), Path(args.obs_csv), Path(args.rho_proxy_csv), gid, args.sigma_floor))
                    med_rms = float(np.median(rms_list))
                    per_params.append({'params': params, 'median_rms': med_rms})
                    if med_rms < best_metric:
                        best_metric = med_rms
                        best = ('hybrid', params)

    out = {'kind': args.kind, 'best': {'kind': best[0], 'params': best[1], 'median_rms': best_metric}, 'grid_results': per_params, 'galaxies': args.galaxies, 'sigma_floor': args.sigma_floor}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Wrote global fit to', args.out_json)


if __name__ == '__main__':
    main()
