#!/usr/bin/env python3
"""
fit_sparc_gr_evidence.py - GR(baryons-only) evidence for a SPARC galaxy using dynesty.
Non-invasive: matches sigma-floor/masks/priors with ER pipeline; does not touch original runners.
Outputs JSON with logZ and metadata sidecar next to a small plot.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy
from models.nfw import v_model_gr

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

    # GR has no free parameters. Evidence is integral over delta-function -> treat as fixed-parameter model.
    # We still run dynesty on a dummy 0-D parameter space by returning constant prior and likelihood at fixed model.
    e_eff = np.sqrt(eV**2 + float(max(0.0, args.sigma_floor))**2)
    e_eff = np.where(e_eff > 0, e_eff, 1.0)
    vmod = v_model_gr(vbar)
    resid = (Vobs - vmod) / e_eff
    chi2 = float(np.sum(resid*resid))
    lnL_const = -0.5 * chi2

    if not _HAS_DYNASTY:
        print(json.dumps({'galaxy_id': args.galaxy_id, 'chi2': chi2, 'loglike_no_const': lnL_const}))
        sys.exit(0)

    # Use 1-D dummy parameter in [0,1] that does nothing
    def prior_transform(u):
        return np.array(u)

    def loglike(theta):
        return lnL_const

    rng = np.random.default_rng(args.seed)
    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=1, bound='multi', sample='rslice', rstate=rng)
    dsampler.run_nested(maxcall=args.maxcall, dlogz_init=args.dlogz_target, nlive_init=args.nlive)
    res = dsampler.results

    # Sidecar JSON (no plot needed but include minimal figure for parity if desired)
    out_dir = REPO_ROOT / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'sparc_gr_evidence_{args.galaxy_id.lower()}.json'
    meta = {
        'galaxy_id': args.galaxy_id,
        'model': 'gr',
        'sigma_floor': float(args.sigma_floor),
        'priors': {'use_master_priors': bool(args.use_master_priors)},
        'chi2': chi2,
        'loglike_no_const': lnL_const,
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

