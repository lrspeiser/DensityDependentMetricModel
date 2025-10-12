#!/usr/bin/env python3
# Sandbox variant of the cluster pipeline wired to a density/hybrid gate.
# This does not modify scripts/cluster_rar_pipeline.py; it allows you to
# compare accel vs density gating with identical data plumbing.

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

from xi_registry_variants import build_gate

G = 6.67430e-8
M_sun = 1.98847e33
kpc = 3.0856775814913673e21
Mpc = 3.0856775814913673e24
m_p = 1.67262192369e-24


def hernquist_rho_cgs(M_solar: float, Re_kpc: float, r_kpc: np.ndarray) -> np.ndarray:
    # Hernquist density: rho(r) = (M / (2π)) * a / [ r (r + a)^3 ]; a = Re/1.8153
    a_kpc = Re_kpc / 1.8153
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    a = a_kpc
    rho_kpc = (M_solar / (2.0 * math.pi)) * (a) / (r * np.power(r + a, 3))
    # Convert Msun/kpc^3 to g/cm^3
    return rho_kpc * (M_sun / (kpc**3))


def parse_accept(path: str) -> Dict[str, List[Tuple[float,float,float]]]:
    clusters: Dict[str, List[Tuple[float,float,float]]] = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#') or line.startswith('###'):
                continue
            parts = line.split()
            try:
                name = parts[0]
                Rin = float(parts[1])
                Rout = float(parts[2])
                ne = float(parts[3])
            except Exception:
                continue
            clusters.setdefault(name, []).append((Rin, Rout, ne))
    for k in clusters:
        clusters[k].sort(key=lambda t: t[1])
    return clusters


def build_profiles(shells: List[Tuple[float,float,float]], mu_e: float = 1.17):
    r_edges_cm = []
    Mgas_cum = []
    rho_gas_edge = []
    total = 0.0
    last_rout = 0.0
    for Rin, Rout, ne in shells:
        if Rin >= Rout or ne <= 0:
            continue
        rin = Rin * Mpc
        rout = Rout * Mpc
        if rout <= last_rout:
            continue
        vol = (4.0 * math.pi / 3.0) * (rout**3 - rin**3)
        rho_g = mu_e * m_p * ne
        dM = rho_g * vol
        total += dM
        r_edges_cm.append(rout)
        Mgas_cum.append(total)
        rho_gas_edge.append(rho_g)
        last_rout = rout
    r_edges_cm = np.array(r_edges_cm, dtype=float)
    Mgas_cum = np.array(Mgas_cum, dtype=float)
    rho_gas_edge = np.array(rho_gas_edge, dtype=float)
    gbar = G * Mgas_cum / (r_edges_cm**2)
    return r_edges_cm, gbar, rho_gas_edge


def main():
    ap = argparse.ArgumentParser(description="Cluster DG/AG sandbox pipeline")
    ap.add_argument('--accept', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--mu-e', type=float, default=1.17)
    ap.add_argument('--gate', default='hybrid', choices=['accel','density','hybrid'])
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--rho-gamma', type=float, default=1.5)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)
    ap.add_argument('--stars-csv', default=None)
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.images, exist_ok=True)

    stars: Dict[str, Dict[str, float]] = {}
    if args.stars_csv and Path(args.stars_csv).exists():
        with open(args.stars_csv) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = row['cluster']
                rec = {}
                for k in ['log10Mstar_BCG','Re_kpc','log10Mstar_ICL','Re_ICL_kpc']:
                    if k in row and row[k]:
                        rec[k] = float(row[k])
                stars[name] = rec

    accept = parse_accept(args.accept)

    # Build gate
    gate = build_gate(args.gate, a0=args.a0, rho_c=args.rho_c,
                      gamma=args.rho_gamma, zeta=args.zeta, Dmax=args.Dmax)

    # Simple run without NFW: we just export xi diagnostics vs radius
    out_rows = []
    for name, shells in accept.items():
        r_cm, gbar, rho_g = build_profiles(shells, mu_e=args.mu_e)
        r_kpc = r_cm / kpc
        rho_star = np.zeros_like(r_kpc)
        if name in stars:
            st = stars[name]
            if 'log10Mstar_BCG' in st and 'Re_kpc' in st:
                rho_star += hernquist_rho_cgs(10.0**st['log10Mstar_BCG'], st['Re_kpc'], r_kpc)
            if 'log10Mstar_ICL' in st and 'Re_ICL_kpc' in st:
                rho_star += hernquist_rho_cgs(10.0**st['log10Mstar_ICL'], st['Re_ICL_kpc'], r_kpc)
        rho_bary = rho_g + rho_star
        xi = gate(gbar, rho_bary, r_kpc)
        for rk, gb, rh, x in zip(r_kpc, gbar, rho_bary, xi):
            out_rows.append((name, rk, math.log10(gb), math.log10(max(rh,1e-99)), float(x)))

    with open(Path(args.results)/'cluster_gate_diagnostics.csv', 'w') as f:
        f.write('cluster,r_kpc,log10_gbar_cgs,log10_rho_cgs,xi\n')
        for row in out_rows:
            f.write(','.join(map(str,row))+'\n')

    summary = {
        'gate': args.gate,
        'params': {'a0': args.a0, 'rho_c': args.rho_c, 'gamma': args.rho_gamma,
                   'zeta': args.zeta, 'Dmax': args.Dmax},
        'mu_e': args.mu_e,
    }
    with open(Path(args.results)/'summary_variants.json','w') as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", (Path(args.results)/'cluster_gate_diagnostics.csv').as_posix())


if __name__ == '__main__':
    main()
