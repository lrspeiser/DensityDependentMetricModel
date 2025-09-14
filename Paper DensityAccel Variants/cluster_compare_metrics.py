#!/usr/bin/env python3
# Compare GR (baryons-only) and optional hybrid/density gates against CLASH NFW totals
# using ACCEPT shells, with r/R200c masks and equal-cluster weighting.

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util
import sys
import numpy as np

# Gate registry (sandbox)
try:
    from xi_registry_variants import build_gate
except Exception:
    build_gate = None

G = 6.67430e-8
M_sun = 1.98847e33
kpc = 3.0856775814913673e21
Mpc = 3.0856775814913673e24
m_p = 1.67262192369e-24


def load_cluster_module(repo_root: Path):
    mod_path = repo_root / 'scripts' / 'cluster_rar_pipeline.py'
    spec = importlib.util.spec_from_file_location('cluster_rar_pipeline', str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    # Ensure module is registered so dataclasses/typing introspection works
    sys.modules[spec.name] = mod  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def parse_accept(path: Path) -> Dict[str, List[Tuple[float,float,float]]]:
    clusters: Dict[str, List[Tuple[float,float,float]]] = {}
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('###'): continue
            parts = s.split()
            if len(parts) < 5: continue
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
    r_edges_cm, Mgas_cum, rho_gas_edge = [], [], []
    total = 0.0
    last_rout = 0.0
    for Rin, Rout, ne in shells:
        if Rin >= Rout or ne <= 0: continue
        rin = Rin * Mpc
        rout = Rout * Mpc
        if rout <= last_rout: continue
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
    ap = argparse.ArgumentParser(description='Cluster GR vs (optional) hybrid/density metrics')
    ap.add_argument('--accept', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--mu-e', type=float, default=1.17)
    ap.add_argument('--xmin', type=float, default=0.05, help='Min r/R200c')
    ap.add_argument('--xmax', type=float, default=0.8, help='Max r/R200c')
    ap.add_argument('--equal-cluster-weight', action='store_true')
    # Optional gate
    ap.add_argument('--gate', choices=['hybrid','density','density-plateau','accel'], default=None)
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--gamma', type=float, default=1.5)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--n', type=float, default=2.0)
    ap.add_argument('--xi-max', type=float, default=50.0)
    args = ap.parse_args()

    repo_root = Path.cwd()
    mod = load_cluster_module(repo_root)
    # CLASH params and helpers
    CLASH_PARAMS = mod.CLASH_PARAMS
    def canon(name: str) -> str | None:
        return mod._canonicalize_clash_key(name)

    accept = parse_accept(Path(args.accept))
    rows = []
    per_cluster = {}

    # Build optional gate
    gate = None
    if args.gate:
        if build_gate is None:
            raise RuntimeError('xi_registry_variants not available')
        gate = build_gate(args.gate, a0=args.a0, rho_c=args.rho_c, gamma=args.gamma, zeta=args.zeta, Dmax=args.xi_max, n=args.n)

    for raw_name, shells in accept.items():
        cname = canon(raw_name)
        if cname is None or cname not in CLASH_PARAMS:
            continue
        pars = CLASH_PARAMS[cname]
        nfw = mod.NFW(M200c_Msun_h70=pars['M200c'], c200c=pars['c200c'], z=pars['z'])
        r_cm, gbar, rho_g = build_profiles(shells, mu_e=args.mu_e)
        if r_cm.size == 0: continue
        gtot = nfw.g_tot(r_cm)
        x = (r_cm / kpc) / nfw.r200c_kpc
        mask = (x >= args.xmin) & (x <= args.xmax) & np.isfinite(gbar) & np.isfinite(gtot) & (gbar>0) & (gtot>0)
        if not mask.any():
            continue
        lgNFW = np.log10(gtot[mask])
        lgGR = np.log10(gbar[mask])
        resid_gr = lgNFW - lgGR
        # Optional hybrid/density
        resid_gate = None
        if gate is not None:
            xi = gate(gbar[mask], rho_g[mask], (r_cm[mask]/kpc))
            lgG = np.log10(np.clip(xi * gbar[mask], 1e-99, None))
            resid_gate = lgNFW - lgG
        # Equal cluster weight via per-point weights 1/N
        if args.equal_cluster_weight:
            w = np.full(resid_gr.shape, 1.0/resid_gr.size)
        else:
            w = np.ones_like(resid_gr)
        rms_gr = float(np.sqrt(np.sum(w * resid_gr**2) / np.sum(w)))
        out = {
            'cluster': cname,
            'n_points': int(resid_gr.size),
            'rms_gr_dex': rms_gr,
        }
        if resid_gate is not None:
            rms_gate = float(np.sqrt(np.sum(w * resid_gate**2) / np.sum(w)))
            out['rms_gate_dex'] = rms_gate
        per_cluster[cname] = out

    # Aggregate
    agg = {
        'config': {
            'xmin': args.xmin,
            'xmax': args.xmax,
            'equal_cluster_weight': args.equal_cluster_weight,
            'gate': args.gate,
            'params': {'a0': args.a0, 'rho_c': args.rho_c, 'gamma': args.gamma, 'zeta': args.zeta, 'xi_max': args.xi_max} if args.gate else None,
        },
        'per_cluster': list(per_cluster.values()),
    }
    # Global medians
    if per_cluster:
        rms_grs = [c['rms_gr_dex'] for c in per_cluster.values()]
        agg['global'] = {'median_rms_gr_dex': float(np.median(rms_grs))}
        if args.gate:
            rms_gates = [c.get('rms_gate_dex') for c in per_cluster.values() if 'rms_gate_dex' in c]
            if rms_gates:
                agg['global']['median_rms_gate_dex'] = float(np.median(rms_gates))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(agg, indent=2), encoding='utf-8')
    print('Wrote cluster comparison to', args.out_json)


if __name__ == '__main__':
    main()
