#!/usr/bin/env python3
"""
fit_sparc_nfw_evidence.py - NFW evidence for a SPARC galaxy using dynesty.
Non-invasive: matches sigma-floor/masks/priors with ER pipeline; does not touch original runners.
Outputs JSON with logZ, logZ_err, and dynesty controls.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy
from models.nfw import v_model_nfw

try:
    import dynesty  # type: ignore
    _HAS_DYNASTY = True
except Exception:
    _HAS_DYNASTY = False


def build_vbar(gal):
    Vg = np.asarray(gal['V_gas_comp_kms'], dtype=float)
    Vd = np.asarray(gal['V_disk_comp_kms'], dtype=float)
    Vb = np.asarray(gal['V_bulge_comp_kms'], dtype=float)
    return np.sqrt(np.clip(Vg**2 + Vd**2 + Vb**2, 0.0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy_id', required=True)
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--sigma-floor', type=float, default=0.0)
    ap.add_argument('--use-master-priors', action='store_true')
    ap.add_argument('--mode', choices=['evidence'], default='evidence')
    # NFW priors (uniform): V200 in [40, 400], c in [2, 40]
    ap.add_argument('--V200_min', type=float, default=40.0)
    ap.add_argument('--V200_max', type=float, default=400.0)
    ap.add_argument('--c_min', type=float, default=2.0)
    ap.add_argument('--c_max', type=float, default=40.0)
    # Dynesty knobs
    ap.add_argument('--nlive', type=int, default=1000)
    ap.add_argument('--maxcall', type=int, default=200000)
    ap.add_argument('--dlogz-target', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    gal = load_single_sparc_galaxy(args.galaxy_id, sparc_dir=args.sparc_dir)
    if gal is None:
        print(f"Failed to load galaxy {args.galaxy_id}")
        sys.exit(2)

    R = np.asarray(gal['R_kpc'], dtype=float)
    Vobs = np.asarray(gal['V_obs'], dtype=float)
    eV = np.asarray(gal['e_V_obs'], dtype=float)
    vbar = build_vbar(gal)

    # Likelihood utilities
    def e_eff():
        e = np.sqrt(eV**2 + float(max(0.0, args.sigma_floor))**2)
        return np.where(e > 0, e, 1.0)

    def loglike(theta):
        V200, c = float(theta[0]), float(theta[1])
        vmod = v_model_nfw(R, vbar, V200, c)
        r = (Vobs - vmod) / e_eff()
        return -0.5 * float(np.sum(r*r))

    lo = np.array([args.V200_min, args.c_min], dtype=float)
    hi = np.array([args.V200_max, args.c_max], dtype=float)

    def prior_transform(u):
        u = np.asarray(u, dtype=float)
        return lo + u * (hi - lo)

    if not _HAS_DYNASTY:
        print(json.dumps({'galaxy_id': args.galaxy_id, 'error': 'dynesty not available'}))
        sys.exit(0)

    rng = np.random.default_rng(args.seed)
    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=2, bound='multi', sample='rslice', rstate=rng)
    dsampler.run_nested(maxcall=args.maxcall, dlogz_init=args.dlogz_target, nlive_init=args.nlive)
    res = dsampler.results

    out_dir = REPO_ROOT / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'sparc_nfw_evidence_{args.galaxy_id.lower()}.json'
    meta = {
        'galaxy_id': args.galaxy_id,
        'model': 'nfw',
        'sigma_floor': float(args.sigma_floor),
        'priors': {
            'use_master_priors': bool(args.use_master_priors),
            'V200': {'min': float(args.V200_min), 'max': float(args.V200_max)},
            'c': {'min': float(args.c_min), 'max': float(args.c_max)},
        },
        'evidence': {
            'logZ': float(res.logz[-1]),
            'logZ_err': float(res.logzerr[-1]) if hasattr(res, 'logzerr') else None,
            'nlive': int(args.nlive),
            'maxcall': int(args.maxcall),
            'dlogz_target': float(args.dlogz_target),
            'seed': int(args.seed),
        },
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_json}")

if __name__ == '__main__':
    main()

